#!/usr/bin/env python
"""Stage 3 — field-conditioned fine-tune of the base LDM.

Implements §3.2 Algorithm 2 of main.tex. Starts from the scalar-conditioned
base checkpoint and continues training on paired (subvolume, porosity-field)
inputs, with the porosity field average-pooled to latent resolution (F=8).

Legacy counterpart: scripts/legacy/0003c-porosity-field-training-vol-correction.py

This is a thin port of 0003c that:
  - reads paths from YAML config + env vars instead of hardcoded constants;
  - accepts --variant fc257 / fc129 to pick the right averaging window;
  - writes to the research-log location savedmodels/experimental/<YYYYMMDD>-…
    (so nothing under savedmodels/pore/ is ever overwritten by a training
    run — the published view there is symlinks only).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import lightning
import lightning.pytorch.callbacks as pl_callbacks
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import diffsci2.data  # noqa: E402
import diffsci2.models  # noqa: E402
import diffsci2.nets  # noqa: E402

from pipelines._common import (  # noqa: E402
    load_config, model_root, raw_volume_path, field_dir,
    parse_devices, published_checkpoints,
)


# ---------------------------------------------------------------------------
# Dataset — identical semantics to legacy 0003c.
# ---------------------------------------------------------------------------

class VolumeSubvolumeWithPorosityDataset(Dataset):
    def __init__(
        self,
        volume: np.ndarray,
        porosity_volume: np.ndarray,
        dataset_size: int,
        subvolume_size: int = 256,
        downsample_factor: int = 8,
        cube_symmetry: diffsci2.data.CubeSymmetry | None = None,
    ):
        assert volume.shape == porosity_volume.shape, (
            f"Volume {volume.shape} != porosity {porosity_volume.shape}"
        )
        self.volume = volume
        self.porosity_volume = porosity_volume
        self.dataset_size = dataset_size
        self.subvolume_size = subvolume_size
        self.downsample_factor = downsample_factor
        self.cube_symmetry = cube_symmetry
        self.rng = np.random.default_rng()

        self.max_d = volume.shape[0] - subvolume_size
        self.max_h = volume.shape[1] - subvolume_size
        self.max_w = volume.shape[2] - subvolume_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        sd = self.rng.integers(0, self.max_d + 1) if self.max_d > 0 else 0
        sh = self.rng.integers(0, self.max_h + 1) if self.max_h > 0 else 0
        sw = self.rng.integers(0, self.max_w + 1) if self.max_w > 0 else 0
        s = self.subvolume_size

        sub = self.volume[sd:sd+s, sh:sh+s, sw:sw+s].copy()
        por = self.porosity_volume[sd:sd+s, sh:sh+s, sw:sw+s].copy()
        sub_t = torch.from_numpy(sub).float()
        por_t = torch.from_numpy(por).float()

        if self.cube_symmetry is not None:
            sym = self.rng.integers(0, 48)
            sub_t = self.cube_symmetry.apply(sub_t, sym)
            por_t = self.cube_symmetry.apply(por_t, sym)

        # Average-pool the porosity field to latent resolution.
        p_in = por_t.unsqueeze(0).unsqueeze(0)
        p_down = F.avg_pool3d(p_in, kernel_size=self.downsample_factor,
                              stride=self.downsample_factor).squeeze(0).squeeze(0)

        return {"x": sub_t.unsqueeze(0), "y": {"porosity": p_down}}


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup over `warmup_steps` steps, then cosine decay to 0."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            return [lr * (step + 1) / self.warmup_steps for lr in self.base_lrs]
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return [lr * cos for lr in self.base_lrs]


class EMACallback(pl_callbacks.Callback):
    def __init__(self, decay: float = 0.99):
        super().__init__()
        self.decay = decay
        self.ema = {}

    def on_train_start(self, trainer, pl_module):
        for n, p in pl_module.model.named_parameters():
            self.ema[n] = p.data.clone()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        with torch.no_grad():
            for n, p in pl_module.model.named_parameters():
                if n in self.ema:
                    self.ema[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def on_validation_start(self, trainer, pl_module):
        self._swap(pl_module)

    def on_validation_end(self, trainer, pl_module):
        self._swap(pl_module)

    def _swap(self, pl_module):
        with torch.no_grad():
            for n, p in pl_module.model.named_parameters():
                if n in self.ema:
                    tmp = p.data.clone()
                    p.data.copy_(self.ema[n])
                    self.ema[n].copy_(tmp)


# ---------------------------------------------------------------------------
# Model loading — starts from the base scalar checkpoint.
# ---------------------------------------------------------------------------

def load_base_checkpoint(rock: str) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Load the scalar-conditioned base denoiser + the shared VAE.

    TODO (Danilo): verify against the Naiff2026 repo that the base
    `<rock>_pcond.ckpt` can be loaded with SIModule directly, or whether
    a state-dict bridge is needed.
    """
    paths = published_checkpoints(rock)
    from diffsci2.models.vae.vaemodule import VAEModule
    vae = VAEModule.load_from_checkpoint(paths.vae, map_location="cpu")
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # Load the base LDM module as the initialization point.
    module = diffsci2.models.SIModule.load_from_checkpoint(paths.base, map_location="cpu")
    return module.model, vae


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip())
    p.add_argument("--config", required=True)
    p.add_argument("--variant", choices=["fc257", "fc129"], required=True)
    p.add_argument("--devices", default="0")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--profile", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    rock = cfg["rock"]
    ft = cfg["field_training"]
    radius = cfg["field"][args.variant]["radius"]
    kernel_size = 2 * radius + 1

    devices = parse_devices(args.devices)
    strategy = "ddp" if len(devices) > 1 else "auto"

    volume_path = raw_volume_path(cfg["data"]["volume_filename"])
    fdir = field_dir(radius, rock)
    porosity_path = fdir / f"{rock.lower()}_porosity_field.npy"

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (
        model_root() / "experimental" /
        f"{datetime.now().strftime('%Y%m%d')}-{rock.lower()}-gpdata4-{kernel_size}-porosity-field"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Rock:          {rock}")
    print(f"Variant:       {args.variant}  (r={radius}, kernel={kernel_size})")
    print(f"Volume:        {volume_path}")
    print(f"Porosity:      {porosity_path}")
    print(f"Checkpoint:    {ckpt_dir}")

    # Load volumes.
    volume = np.fromfile(volume_path, dtype=np.uint8).reshape(cfg["data"]["volume_shape"])
    porosity = np.load(porosity_path)
    print(f"Volume  shape: {volume.shape}")
    print(f"Porosity shape: {porosity.shape} (expected {volume.shape[0] - 2*radius} per axis)")

    # Inward crop of the binary volume to match the valid-convolution porosity field.
    r = radius
    volume = volume[r:-r, r:-r, r:-r]
    assert volume.shape == porosity.shape

    # Train/val split.
    split = ft["train_split_along_last_axis"]
    vol_tr, vol_va = volume[:, :, :split], volume[:, :, split:]
    por_tr, por_va = porosity[:, :, :split], porosity[:, :, split:]

    cube = diffsci2.data.CubeSymmetry()
    tr_ds = VolumeSubvolumeWithPorosityDataset(
        vol_tr, por_tr,
        dataset_size=ft["samples_per_epoch"],
        subvolume_size=ft["subvolume_size"],
        downsample_factor=ft["downsample_factor"],
        cube_symmetry=cube,
    )
    va_ds = VolumeSubvolumeWithPorosityDataset(
        vol_va, por_va,
        dataset_size=ft["samples_per_epoch"] // 8,
        subvolume_size=ft["subvolume_size"],
        downsample_factor=ft["downsample_factor"],
        cube_symmetry=None,
    )
    tr_loader = DataLoader(
        tr_ds, batch_size=ft["batch_size"], shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    va_loader = DataLoader(
        va_ds, batch_size=ft["batch_size"], shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Load base LDM and VAE.
    flow_model, autoencoder = load_base_checkpoint(rock)

    flow_module_config = diffsci2.models.SIModuleConfig.from_edm_sigma_space(
        sigma_min=cfg["base_training"]["sigma_min"],
        sigma_max=cfg["base_training"]["sigma_max"],
        sigma_data=cfg["base_training"]["sigma_data"],
        initial_norm=20.0,
        loss_formulation="denoiser",
    )
    flow_module = diffsci2.models.SIModule(
        config=flow_module_config,
        model=flow_model,
        autoencoder=autoencoder,
    )

    # Optimizer + schedule.
    batches_per_epoch = len(tr_loader)
    batches_per_gpu = batches_per_epoch // max(1, len(devices))
    opt_steps_per_epoch = batches_per_gpu // ft["accumulate_grad_batches"]
    total_opt_steps = opt_steps_per_epoch * ft["epochs"]
    warmup_steps = ft["warmup_epochs"] * opt_steps_per_epoch

    optimizer = torch.optim.AdamW(
        flow_module.model.parameters(),
        lr=ft["lr"],
        betas=(0.9, 0.999),
        weight_decay=ft["weight_decay"],
        eps=1e-8,
    )
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=warmup_steps, total_steps=total_opt_steps,
    )
    flow_module.set_optimizer_and_scheduler(
        optimizer=optimizer, scheduler=scheduler, scheduler_interval="step",
    )

    # Callbacks.
    callbacks: list = [
        diffsci2.models.NanToZeroGradCallback(),
        pl_callbacks.ModelCheckpoint(
            dirpath=str(ckpt_dir / "checkpoints"),
            filename="porosity-field-{epoch:03d}-{val_loss:.6f}",
            save_top_k=3, monitor="val_loss", mode="min", save_last=True,
        ),
        pl_callbacks.LearningRateMonitor(logging_interval="step"),
    ]
    if ft["ema_decay"] > 0:
        callbacks.append(EMACallback(decay=ft["ema_decay"]))

    tb = lightning.pytorch.loggers.TensorBoardLogger(
        save_dir=str(ckpt_dir), name="logs", default_hp_metric=False,
    )
    trainer = lightning.Trainer(
        max_epochs=ft["epochs"],
        default_root_dir=str(ckpt_dir),
        gradient_clip_val=ft["grad_clip"],
        callbacks=callbacks,
        devices=devices, strategy=strategy,
        logger=tb, log_every_n_steps=10,
        precision="16-mixed",
        accumulate_grad_batches=ft["accumulate_grad_batches"],
        check_val_every_n_epoch=1,
    )
    print("\nStarting fine-tune.")
    trainer.fit(flow_module, train_dataloaders=tr_loader, val_dataloaders=va_loader)
    print("\nDone. Best:", trainer.checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()
