#!/usr/bin/env python
"""Stage 2 — train the scalar-porosity-conditioned base LDM.

Implements §3.2 Algorithm 1 of main.tex: train a 3D latent diffusion
denoiser from scratch on 256^3 subvolumes of a single rock, conditioned
on the scalar porosity (mean of the subvolume), with classifier-free
condition dropout so the same checkpoint can be sampled unconditionally
via the null input.

This stage was **not** present in the legacy scripts/ tree — the four
<rock>_pcond.ckpt files under savedmodels/pore/production/ were produced
by the separate Naiff2026 repo. This file is the intended local home
for re-running that stage when needed.

It is derived from:
  - scripts/legacy/0009-unconditional-training-3d.py   (training loop)
  - scripts/legacy/0003c-porosity-field-training-vol-correction.py
                                                        (LDM + VAE wiring)
and deliberately kept editable — the expected usage is that Danilo
verifies the network-construction block against the Naiff2026 code and
adjusts before first training run.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import lightning
import lightning.pytorch.callbacks as pl_callbacks
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import diffsci2.data  # noqa: E402
import diffsci2.models  # noqa: E402
import diffsci2.nets  # noqa: E402

from pipelines._common import (  # noqa: E402
    load_config, model_root, raw_volume_path, parse_devices, published_checkpoints,
)


# ---------------------------------------------------------------------------
# Dataset: random 256^3 subvolumes + scalar porosity + condition dropout
# ---------------------------------------------------------------------------

class VolumeSubvolumeScalarPorosityDataset(Dataset):
    """Random-crop subvolume with on-the-fly scalar porosity conditioning.

    Mirrors the dataset in scripts/legacy/0003c but conditions on the
    scalar mean of the subvolume rather than a spatial field. Condition
    dropout replaces the scalar with a null sentinel with probability
    `cond_dropout_p` — this is the hook that lets the same checkpoint be
    used as the Uncond baseline at inference time.
    """

    NULL_POROSITY = -1.0  # sentinel for "no conditioning"; any value outside [0,1] works.

    def __init__(
        self,
        volume: np.ndarray,
        dataset_size: int,
        subvolume_size: int = 256,
        cube_symmetry: diffsci2.data.CubeSymmetry | None = None,
        cond_dropout_p: float = 0.1,
    ):
        self.volume = volume
        self.dataset_size = dataset_size
        self.subvolume_size = subvolume_size
        self.cube_symmetry = cube_symmetry
        self.cond_dropout_p = cond_dropout_p
        self.rng = np.random.default_rng()

        self.max_d = volume.shape[0] - subvolume_size
        self.max_h = volume.shape[1] - subvolume_size
        self.max_w = volume.shape[2] - subvolume_size
        assert self.max_d >= 0 and self.max_h >= 0 and self.max_w >= 0, (
            f"Volume {volume.shape} smaller than subvolume {subvolume_size}"
        )

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        sd = self.rng.integers(0, self.max_d + 1) if self.max_d > 0 else 0
        sh = self.rng.integers(0, self.max_h + 1) if self.max_h > 0 else 0
        sw = self.rng.integers(0, self.max_w + 1) if self.max_w > 0 else 0
        s = self.subvolume_size

        sub = self.volume[sd:sd+s, sh:sh+s, sw:sw+s].copy()
        sub_t = torch.from_numpy(sub).float()

        if self.cube_symmetry is not None:
            sym_id = self.rng.integers(0, 48)
            sub_t = self.cube_symmetry.apply(sub_t, sym_id)

        # Scalar porosity = mean of the pore indicator (1 - solid).
        # `sub` stores 1 = solid, 0 = pore, so porosity = mean(1 - sub).
        porosity = float((1.0 - sub_t).mean())

        # Classifier-free condition dropout.
        if self.rng.random() < self.cond_dropout_p:
            porosity = self.NULL_POROSITY

        return {
            "x": sub_t.unsqueeze(0),               # (1, s, s, s)
            "y": {"porosity": torch.tensor(porosity).float()},
        }


# ---------------------------------------------------------------------------
# LR schedule — linear warmup over `warmup_steps`, then constant.
# ---------------------------------------------------------------------------

class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            return [lr * (step + 1) / self.warmup_steps for lr in self.base_lrs]
        return self.base_lrs


# ---------------------------------------------------------------------------
# Model construction — the TODO block Danilo is expected to verify.
# ---------------------------------------------------------------------------

def build_flow_model(cfg: dict) -> torch.nn.Module:
    """Construct the PUNetG denoiser for the base scalar-conditioned LDM.

    TODO (Danilo): reconcile this block with how the Naiff2026 repo
    built the denoiser that produced <rock>_pcond.ckpt. The defaults
    below match the article's description (PUNetG, 29.6 M params, GN,
    no self-attention, F=8 VAE compression, 4 latent channels) but the
    exact PUNetG channel multipliers / block counts may differ.
    """
    import diffsci2.nets.punetg as punetg
    import diffsci2.nets.punetg_config as punetg_config
    import diffsci2.nets.embedder as embedder

    # Scalar porosity embedder: a single scalar in [0,1], with a null
    # sentinel. CompositeEmbedder handles the null-token embedding when
    # the input is the sentinel.
    porosity_embedder = embedder.ScalarEmbedder(
        embedding_dim=128,
        null_value=VolumeSubvolumeScalarPorosityDataset.NULL_POROSITY,
    )
    cond_embedder = embedder.CompositeEmbedder({"porosity": porosity_embedder})

    net_config = punetg_config.PUNetGConfig(
        in_channels=4,                 # VAE latent channels
        out_channels=4,
        # TODO verify these three against Naiff2026's actual config:
        model_channels=64,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
    )
    return punetg.PUNetGCond(config=net_config, conditioning_embedder=cond_embedder)


def load_autoencoder(ckpt_path: Path) -> torch.nn.Module:
    """Load the pretrained VAE that maps binary volumes to 4-channel latents.

    TODO (Danilo): this delegates to the VAEModule loader pattern used
    by scripts/legacy/0003c (`load_autoencoder` from model_loaders). If
    your VAE checkpoint was saved with a different Lightning module
    class, adjust accordingly.
    """
    from diffsci2.models.vae.vaemodule import VAEModule
    module = VAEModule.load_from_checkpoint(ckpt_path, map_location="cpu")
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip())
    p.add_argument("--config", required=True, help="Rock YAML config.")
    p.add_argument("--devices", default="0",
                   help="Comma-separated GPU ids, e.g. '0,1,2,3'.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--checkpoint-dir", default=None,
                   help="Override output directory. Default: "
                        "savedmodels/experimental/<YYYYMMDD>-<rock>-base-scalar/")
    p.add_argument("--profile", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    rock = cfg["rock"]
    bt = cfg["base_training"]

    devices = parse_devices(args.devices)
    strategy = "ddp" if len(devices) > 1 else "auto"

    volume_path = raw_volume_path(cfg["data"]["volume_filename"])
    print(f"Rock:          {rock}")
    print(f"Volume:        {volume_path}")
    print(f"Devices:       {devices}")
    print()

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (
        model_root() / "experimental" /
        f"{datetime.now().strftime('%Y%m%d')}-{rock.lower()}-base-scalar"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir:    {ckpt_dir}")

    # Load volume.
    volume_data = np.fromfile(volume_path, dtype=np.uint8).reshape(cfg["data"]["volume_shape"])
    print(f"Loaded volume shape={volume_data.shape}")

    # Train/val split along last axis.
    split = bt["train_split_along_last_axis"]
    volume_train = volume_data[:, :, :split]
    volume_val = volume_data[:, :, split:]
    print(f"Train: {volume_train.shape}  Val: {volume_val.shape}")

    # Dataset + loaders.
    cube = diffsci2.data.CubeSymmetry()
    train_ds = VolumeSubvolumeScalarPorosityDataset(
        volume=volume_train,
        dataset_size=bt["samples_per_epoch"],
        subvolume_size=bt["subvolume_size"],
        cube_symmetry=cube,
        cond_dropout_p=bt["cond_dropout"],
    )
    val_ds = VolumeSubvolumeScalarPorosityDataset(
        volume=volume_val,
        dataset_size=bt["samples_per_epoch"] // 8,
        subvolume_size=bt["subvolume_size"],
        cube_symmetry=None,
        cond_dropout_p=0.0,
    )
    train_loader = DataLoader(
        train_ds, batch_size=bt["batch_size"], shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bt["batch_size"], shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Models.
    vae_ckpt = published_checkpoints(rock).vae
    print(f"VAE checkpoint: {vae_ckpt}")
    autoencoder = load_autoencoder(vae_ckpt)
    flow_model = build_flow_model(cfg)

    # Flow module (EDM / SI formulation, σ-space same as main.tex).
    flow_module_config = diffsci2.models.SIModuleConfig.from_edm_sigma_space(
        sigma_min=bt["sigma_min"],
        sigma_max=bt["sigma_max"],
        sigma_data=bt["sigma_data"],
        initial_norm=20.0,
        loss_formulation="denoiser",
    )
    flow_module = diffsci2.models.SIModule(
        config=flow_module_config,
        model=flow_model,
        autoencoder=autoencoder,
    )

    optimizer = torch.optim.AdamW(
        flow_module.model.parameters(),
        lr=bt["lr"],
        betas=(0.9, 0.999),
        weight_decay=bt["weight_decay"],
        eps=1e-8,
    )
    scheduler = WarmupScheduler(optimizer, warmup_steps=bt["warmup_steps"])
    flow_module.set_optimizer_and_scheduler(
        optimizer=optimizer, scheduler=scheduler, scheduler_interval="step",
    )

    # Callbacks.
    callbacks: list = [
        diffsci2.models.NanToZeroGradCallback(),
        pl_callbacks.ModelCheckpoint(
            dirpath=str(ckpt_dir / "checkpoints"),
            filename="base-scalar-{epoch:03d}-{val_loss:.6f}",
            save_top_k=3, monitor="val_loss", mode="min", save_last=True,
        ),
        pl_callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    tb = lightning.pytorch.loggers.TensorBoardLogger(
        save_dir=str(ckpt_dir), name="logs", default_hp_metric=False,
    )
    profiler = None
    if args.profile:
        profiler = lightning.pytorch.profilers.PyTorchProfiler(
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(ckpt_dir / "profiler")
            ),
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            profile_memory=True, with_stack=True,
        )

    trainer = lightning.Trainer(
        max_epochs=bt["epochs"],
        default_root_dir=str(ckpt_dir),
        gradient_clip_val=bt["grad_clip"],
        callbacks=callbacks,
        devices=devices, strategy=strategy,
        logger=tb, log_every_n_steps=10,
        profiler=profiler, precision="16-mixed",
        accumulate_grad_batches=bt["accumulate_grad_batches"],
        check_val_every_n_epoch=1,
    )

    print("\nStarting training.")
    trainer.fit(flow_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("\nDone. Best:", trainer.checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()
