from typing import Literal, Callable, Any
from jaxtyping import Float, Bool
from torch import Tensor

import warnings

import torch
import torch.nn as nn
import numpy as np
import lightning

from diffsci2.torchutils import broadcast_from_below, dict_unsqueeze, dict_to
from diffsci2.models.aux_scripts import DimensionAgnosticBatchNorm, ConstantBatchNorm, IdentityBatchNorm


SampleType = Float[Tensor, "batch *shape"]
ConditionType = Float[Tensor, "batch *yshape"]
TimeType = Float[Tensor, "batch"]


class SIScheduler(object):
    """
    Stochastic Interpolant Scheduler.

    Defines the interpolation between data (t=0) and noise (t=1) via:
        x_t = alpha(t) * x_0 + sigma(t) * epsilon

    Can operate in two modes:
    - Normalized mode: t in [0, 1], with sigma_fn mapping to actual noise levels
    - Sigma-space mode: t IS sigma directly, t in [sigma_min, sigma_max]
    """

    def __init__(
        self,
        alpha_fn: Callable[[float], float],
        sigma_fn: Callable[[float], float],
        alpha_fn_dot: Callable[[float], float],
        sigma_fn_dot: Callable[[float], float],
        sigma_fn_inv: Callable[[float], float],
        time_domain: Literal['normalized', 'sigma'] = 'normalized',
        sigma_min: float | None = None,
        sigma_max: float | None = None
    ):
        self.alpha_fn = alpha_fn
        self.sigma_fn = sigma_fn
        self.alpha_fn_dot = alpha_fn_dot
        self.sigma_fn_dot = sigma_fn_dot
        self.sigma_fn_inv = sigma_fn_inv
        self.time_domain = time_domain
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    @classmethod
    def linear(cls):
        return cls(
            alpha_fn=lambda t: 1 - t,
            sigma_fn=lambda t: t,
            alpha_fn_dot=lambda t: -1 * torch.ones_like(t),
            sigma_fn_dot=lambda t: torch.ones_like(t),
            sigma_fn_inv=lambda s: s,
        )

    @classmethod
    def cosine(cls):
        return cls(
            alpha_fn=lambda t: torch.cos(t * np.pi / 2),
            sigma_fn=lambda t: torch.sin(t * np.pi / 2),
            alpha_fn_dot=lambda t: -1 * torch.pi / 2 * torch.sin(t * np.pi / 2),
            sigma_fn_dot=lambda t: torch.pi / 2 * torch.cos(t * np.pi / 2),
            sigma_fn_inv=lambda s: (2 / np.pi) * torch.arcsin(s),
        )

    @classmethod
    def finterpolation(
        cls,
        f: Callable[[float], float],
        finv: Callable[[float], float],
        fdot: Callable[[float], float],
        sigma_min: float,
        sigma_max: float
    ):
        def sigma_fn(t):
            interpolated_finv_sigma = (1 - t) * finv(sigma_min) + t * finv(sigma_max)
            return f(interpolated_finv_sigma)

        def sigma_fn_inv(s):
            return (finv(s) - finv(sigma_min)) / (finv(sigma_max) - finv(sigma_min))

        def sigma_fn_dot(t):
            interpolated_finv_sigma = (1 - t) * finv(sigma_min) + t * finv(sigma_max)
            return fdot(interpolated_finv_sigma) * (finv(sigma_max) - finv(sigma_min))

        return cls(
            alpha_fn=lambda t: 0.0 * t + 1.0,
            sigma_fn=sigma_fn,
            alpha_fn_dot=lambda t: 0.0 * t,
            sigma_fn_dot=sigma_fn_dot,
            sigma_fn_inv=sigma_fn_inv,
        )

    @classmethod
    def edm(
        cls,
        expoent: float = 7.0,
        sigma_min: float = 0.02,
        sigma_max: float = 80.0
    ):
        f = lambda x: x**expoent  # noqa: E731
        finv = lambda x: x**(1 / expoent)  # noqa: E731
        fdot = lambda x: expoent * x**(expoent - 1)  # noqa: E731
        return cls.finterpolation(f, finv, fdot, sigma_min, sigma_max)

    @classmethod
    def sigma_space(
        cls,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0
    ):
        """
        Sigma-space scheduler where time variable IS sigma directly.

        In this mode:
        - t is in [sigma_min, sigma_max], not [0, 1]
        - sigma_fn(t) = t (identity)
        - alpha_fn(t) = 1 (constant, VE-style diffusion)

        This replicates the EDM paper's native parameterization
        where the model operates directly in sigma space.

        Args:
            sigma_min: Minimum noise level (typically 0.002)
            sigma_max: Maximum noise level (typically 80.0)
        """
        return cls(
            alpha_fn=lambda sigma: 1.0 + 0.0 * sigma,
            sigma_fn=lambda sigma: sigma,
            alpha_fn_dot=lambda sigma: 0.0 * sigma,
            sigma_fn_dot=lambda sigma: 1.0 + 0.0 * sigma,
            sigma_fn_inv=lambda s: s,
            time_domain='sigma',
            sigma_min=sigma_min,
            sigma_max=sigma_max
        )

    @classmethod
    def get_interpolator(cls, name, *args, **kwargs):
        if name not in cls.named_interpolators():
            raise ValueError(f"Invalid interpolator: {name}")
        if name == 'linear':
            return cls.linear(*args, **kwargs)
        elif name == 'cosine':
            return cls.cosine(*args, **kwargs)
        elif name == 'edm':
            return cls.edm(*args, **kwargs)
        elif name == 'finterpolation':
            return cls.finterpolation(*args, **kwargs)
        elif name == 'sigma_space':
            return cls.sigma_space(*args, **kwargs)

    @classmethod
    def named_interpolators(cls):
        return ['linear', 'cosine', 'edm', 'finterpolation', 'sigma_space']

    @property
    def is_sigma_space(self) -> bool:
        """Check if operating in sigma-space mode."""
        return self.time_domain == 'sigma'

    def get_time_bounds(self) -> tuple[float, float]:
        """
        Get the bounds of the time variable.

        Returns:
            (t_min, t_max) - In normalized mode: (0, 1), in sigma mode: (sigma_min, sigma_max)
        """
        if self.is_sigma_space:
            return (self.sigma_min, self.sigma_max)
        else:
            return (0.0, 1.0)


class Preconditioner(object):
    def __init__(
        self,
        scheduler: SIScheduler,
        precondition_fn: Literal['identity', 'edm', 'edm_denoiser'] | Callable | None = 'identity',
        is_autonomous: bool = False,
        **kwargs
    ):
        self.scheduler = scheduler
        self.precondition_fn = precondition_fn
        self.is_autonomous = is_autonomous
        self.kwargs = kwargs

    def __call__(self, model, x, t=None, y=None):
        return self.get_flow_field(model, x, t, y)

    def _get_sigma(self, t):
        """Get sigma from t, handling both normalized and sigma-space modes."""
        if self.scheduler.is_sigma_space:
            return t  # t IS sigma in sigma-space mode
        else:
            return self.scheduler.sigma_fn(t)

    def _get_sigma_dot(self, t):
        """Get d(sigma)/dt, handling both modes."""
        if self.scheduler.is_sigma_space:
            return 1.0 + 0.0 * t  # d(sigma)/d(sigma) = 1
        else:
            return self.scheduler.sigma_fn_dot(t)

    def get_flow_field(self, model, x, t=None, y=None):
        if self.precondition_fn is None:
            v = self.identity(model, x, t, y)
        elif isinstance(self.precondition_fn, str):
            if self.precondition_fn == 'identity':
                v = self.identity(model, x, t, y)
            elif self.precondition_fn == 'edm':
                v = self.edm(model, x, t, y)
            elif self.precondition_fn == 'edm_denoiser':
                v = self.edm_denoiser(model, x, t, y)
            else:
                raise ValueError(f"Invalid condition function: {self.precondition_fn}")
        else:
            if self.is_autonomous:
                v = self.precondition_fn(model, x, y=y)
            else:
                v = self.precondition_fn(model, x, t, y=y)
        return v

    def identity(self, model, x, t=None, y=None):
        if self.is_autonomous:
            return model(x, y=y)
        else:
            return model(x, t, y=y)

    def edm(self, model, x, t=None, y=None):
        """
        EDM preconditioner that outputs flow field.

        The model predicts a denoised image, which is converted to flow field:
            flow_field = (d_sigma/dt) / sigma * (x - D(x))

        In sigma-space mode, d_sigma/dt = 1, so:
            flow_field = (x - D(x)) / sigma
        """
        sigma_data = self.kwargs.get("sigma_data", 0.5)
        sigma = self._get_sigma(t)
        sigma_dot = self._get_sigma_dot(t)
        sigma = broadcast_from_below(sigma, x)
        sigma_dot = broadcast_from_below(sigma_dot, x)
        cin = 1 / torch.sqrt(sigma_data**2 + sigma**2)
        cout = sigma * sigma_data / torch.sqrt(sigma_data**2 + sigma**2)
        cskip = sigma_data ** 2 / (sigma_data**2 + sigma**2)
        if self.is_autonomous:
            flow_field = cskip * x + cout * model(x / cin, y=y)
        else:
            cnoise = 0.5 * torch.log(sigma)
            # Flatten cnoise for the model (it expects [batch] not [batch, 1, 1, ...])
            if cnoise.dim() > 1:
                cnoise = cnoise.view(cnoise.shape[0])
            denoiser = cskip * x + cout * model(cin * x, cnoise, y=y)

            flow_field = sigma_dot / sigma * (x - denoiser)
        return flow_field

    def edm_denoiser(self, model, x, t=None, y=None):
        """
        EDM preconditioner that outputs the denoised image directly.

        This is useful for computing score or when you need D(x) rather than flow field.
        """
        sigma_data = self.kwargs.get("sigma_data", 0.5)
        sigma = self._get_sigma(t)
        sigma = broadcast_from_below(sigma, x)
        cin = 1 / torch.sqrt(sigma_data**2 + sigma**2)
        cout = sigma * sigma_data / torch.sqrt(sigma_data**2 + sigma**2)
        cskip = sigma_data ** 2 / (sigma_data**2 + sigma**2)

        cnoise = 0.5 * torch.log(sigma)
        if cnoise.dim() > 1:
            cnoise = cnoise.view(cnoise.shape[0])
        denoiser = cskip * x + cout * model(cin * x, cnoise, y=y)
        return denoiser


class LossWeighting(object):
    def __init__(
        self,
        scheduler: SIScheduler,
        weighting_class: Literal['edm', 'uniform', 'edm_sigma'] | dict[str, Any] = 'uniform',
        **kwargs
    ):
        self.scheduler = scheduler
        self.kwargs = kwargs
        self.weighting_class = weighting_class

        if not isinstance(weighting_class, str):
            assert 'weighting_function' in weighting_class
            assert 'weighting_sampler' in weighting_class

    def weighting_function(self, t):
        if isinstance(self.weighting_class, str):
            if self.weighting_class == 'edm':
                return self.edm_weighting_function(t)
            elif self.weighting_class == 'edm_sigma':
                return self.edm_sigma_weighting_function(t)
            elif self.weighting_class == 'uniform':
                return self.uniform_weighting_function(t)
            else:
                raise ValueError(f"Invalid weighting class: {self.weighting_class}")
        else:
            return self.weighting_class['weighting_function'](t)

    def weighting_sampler(self, nsamples):
        if isinstance(self.weighting_class, str):
            if self.weighting_class == 'edm':
                return self.edm_weighting_sampler(nsamples)
            elif self.weighting_class == 'edm_sigma':
                return self.edm_sigma_weighting_sampler(nsamples)
            elif self.weighting_class == 'uniform':
                return self.uniform_weighting_sampler(nsamples)
            else:
                raise ValueError(f"Invalid weighting class: {self.weighting_class}")
        else:
            return self.weighting_class['weighting_sampler'](nsamples)

    def uniform_weighting_function(self, t):
        return 1.0 + 0.0 * t

    def uniform_weighting_sampler(self, nsamples):
        """Sample uniform in [0, 1] for normalized mode, or [sigma_min, sigma_max] for sigma mode."""
        if self.scheduler.is_sigma_space:
            # Uniform in log-sigma space
            log_min = np.log(self.scheduler.sigma_min)
            log_max = np.log(self.scheduler.sigma_max)
            log_sigma = torch.rand(nsamples) * (log_max - log_min) + log_min
            return torch.exp(log_sigma)
        else:
            return torch.rand(nsamples)

    def edm_weighting_function(self, t):
        return self.uniform_weighting_function(t)
        # sigma = self.scheduler.sigma_fn(t)
        # sigma_dot = self.scheduler.sigma_fn_dot(t)
        # sigma_data = self.kwargs.get("sigma_data", 1.0)
        # lambd = (sigma_data**2 + sigma**2) / ((sigma * sigma_data)**2)
        # weight = lambd * sigma_dot**2 / sigma**2
        # return weight

    def edm_weighting_sampler(self, nsamples):
        """EDM sampler for normalized time mode - samples sigma then converts to t."""
        pmean = self.kwargs.get("pmean", -1.2)
        pstd = self.kwargs.get("pstd", 1.2)
        logsigma = pstd * torch.randn(nsamples) + pmean
        sigma = torch.exp(logsigma)
        t = self.scheduler.sigma_fn_inv(sigma)
        return t

    def edm_sigma_weighting_function(self, sigma):
        """
        EDM loss weighting function for sigma-space mode.

        lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2

        This is the weighting from the EDM paper (Karras et al. 2022).
        """
        sigma_data = self.kwargs.get("sigma_data", 0.5)
        return (sigma**2 + sigma_data**2) / ((sigma * sigma_data)**2)

    def edm_sigma_weighting_sampler(self, nsamples):
        """
        EDM sampler for sigma-space mode - samples sigma directly.

        Samples from log-normal: log(sigma) ~ N(pmean, pstd^2)
        This is the training distribution from the EDM paper.
        """
        pmean = self.kwargs.get("pmean", -1.2)
        pstd = self.kwargs.get("pstd", 1.2)
        logsigma = pstd * torch.randn(nsamples) + pmean
        sigma = torch.exp(logsigma)
        return sigma  # Return sigma directly, not converted to t


class SIModuleConfig(torch.nn.Module):
    """
    Configuration for Stochastic Interpolant Module.

    Supports two time parameterizations:
    - Normalized mode (default): t in [0, 1], scheduler maps to noise levels
    - Sigma-space mode: t IS sigma directly, t in [sigma_min, sigma_max]

    Use the factory methods for common configurations:
    - SIModuleConfig.from_edm_sigma_space() for EDM paper-style behavior
    """

    def __init__(self,
                 scheduler: SIScheduler | str = 'linear',
                 scheduler_args: dict[str, Any] = {},
                 num_channels: int | None = None,
                 initial_norm: bool | float = False,
                 autonomous_flow: bool = False,
                 precondition_fn: Callable | str | None = None,
                 preconditioner_kwargs: dict[str, Any] = {},
                 loss_weighting: Literal['edm', 'uniform', 'edm_sigma'] | dict[str, Any] = 'uniform',
                 loss_weighting_kwargs: dict[str, Any] = {},
                 loss_metric: Literal['mse', 'huber'] = 'huber',
                 autoencoder_is_conditional: bool = False,
                 encode_condition: bool = False):
        super().__init__()
        if isinstance(scheduler, str):
            scheduler = SIScheduler.get_interpolator(scheduler, **scheduler_args)
        else:
            scheduler = scheduler
        self.scheduler = scheduler
        self.num_channels = num_channels
        self.initial_norm = initial_norm
        self.autonomous_flow = autonomous_flow
        self._loss_weighting_config = loss_weighting
        self._loss_weighting_kwargs = loss_weighting_kwargs
        self.loss_metric = loss_metric
        self.precondition_fn = precondition_fn
        self.preconditioner_kwargs = preconditioner_kwargs
        self.autoencoder_is_conditional = autoencoder_is_conditional
        self.encode_condition = encode_condition
        self.set_scheduling_functions()
        self.set_loss_metric_module()
        self.set_preconditioner()
        self.set_loss_weighting()

    @classmethod
    def from_edm_sigma_space(
        cls,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        pmean: float = -1.2,
        pstd: float = 1.2,
        num_channels: int | None = None,
        initial_norm: bool | float = False,
        loss_metric: Literal['mse', 'huber'] = 'huber',
        autoencoder_is_conditional: bool = False,
        encode_condition: bool = False
    ):
        """
        Create an EDM configuration that operates directly in sigma-space.

        This replicates the EDM paper behavior where:
        - Time variable t IS sigma directly (not normalized to [0, 1])
        - Model receives 0.5 * log(sigma) as noise conditioning
        - Training samples sigma from log-normal distribution
        - Integration happens in sigma space

        Args:
            sigma_min: Minimum noise level (default 0.002)
            sigma_max: Maximum noise level (default 80.0)
            sigma_data: Data standard deviation for preconditioning (default 0.5)
            pmean: Log-normal prior mean for training (default -1.2)
            pstd: Log-normal prior std for training (default 1.2)
            num_channels: Number of channels for batch norm
            initial_norm: Whether to use initial normalization
            loss_metric: Loss function ('mse' or 'huber')
            autoencoder_is_conditional: Whether autoencoder uses conditioning
            encode_condition: Whether to encode the condition

        Returns:
            SIModuleConfig configured for EDM sigma-space operation
        """
        scheduler = SIScheduler.sigma_space(sigma_min=sigma_min, sigma_max=sigma_max)

        return cls(
            scheduler=scheduler,
            num_channels=num_channels,
            initial_norm=initial_norm,
            autonomous_flow=False,
            precondition_fn='edm',
            preconditioner_kwargs={'sigma_data': sigma_data},
            loss_weighting='edm_sigma',
            loss_weighting_kwargs={'sigma_data': sigma_data, 'pmean': pmean, 'pstd': pstd},
            loss_metric=loss_metric,
            autoencoder_is_conditional=autoencoder_is_conditional,
            encode_condition=encode_condition
        )

    @property
    def is_sigma_space(self) -> bool:
        """Check if operating in sigma-space mode."""
        return self.scheduler.is_sigma_space

    def get_time_bounds(self) -> tuple[float, float]:
        """Get the bounds of the time variable."""
        return self.scheduler.get_time_bounds()

    def set_scheduling_functions(self):
        self.alpha_fn = self.scheduler.alpha_fn
        self.sigma_fn = self.scheduler.sigma_fn
        self.alpha_fn_dot = self.scheduler.alpha_fn_dot
        self.sigma_fn_dot = self.scheduler.sigma_fn_dot
        self.sigma_fn_inv = self.scheduler.sigma_fn_inv

    def set_loss_metric_module(self):
        if self.loss_metric == 'mse':
            self.loss_metric_module = torch.nn.MSELoss(reduction="none")
        elif self.loss_metric == 'huber':
            self.loss_metric_module = torch.nn.HuberLoss(reduction="none")
        else:
            raise ValueError(f"Invalid loss metric: {self.loss_metric}")

    def set_preconditioner(self):
        self.preconditioner = Preconditioner(
            self.scheduler,
            self.precondition_fn,
            self.autonomous_flow,
            **self.preconditioner_kwargs
        )

    def set_loss_weighting(self):
        if isinstance(self._loss_weighting_config, str):
            self.loss_weighting = LossWeighting(
                self.scheduler,
                self._loss_weighting_config,
                **self._loss_weighting_kwargs
            )
        else:
            self.loss_weighting = LossWeighting(self.scheduler, **self._loss_weighting_config)


class SIModule(lightning.LightningModule):
    def __init__(
        self,
        config: SIModuleConfig,
        model: nn.Module,
        autoencoder: nn.Module | None = None
    ):
        super().__init__()
        self.config = config
        self.model = model
        self.set_initial_norm()
        self.autoencoder = autoencoder
        if self.autoencoder:
            self.freeze_autoencoder()

    def freeze_autoencoder(self):
        """
        Freezes the autoencoder to prevent its weights from being updated
        during training.
        """
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def encode(self, x, y=None):
        if not self.autoencoder:
            return x, y
        if not self.config.autoencoder_is_conditional and not self.config.encode_condition:
            x = self.autoencoder.encode(x)
        elif self.config.autoencoder_is_conditional and not self.config.encode_condition:
            x = self.autoencoder.encode(x, y)
        elif not self.config.autoencoder_is_conditional and self.config.encode_condition:
            raise ValueError("Cannot encode condition if autoencoder is not conditional")
        else:
            x, y = self.autoencoder.encode(x, y)
        if isinstance(x, dict):  # Handle the case where the autoencoder returns a dict
            x = x['zsample']
        return x, y

    def decode(self, x, y=None):
        if not self.autoencoder:
            return x, y
        if not self.config.autoencoder_is_conditional:
            x = self.autoencoder.decode(x)
        else:
            x = self.autoencoder.decode(x, y)
        return x, y

    def set_initial_norm(self):
        if isinstance(self.config.initial_norm, bool):
            if self.config.initial_norm:
                self.initial_norm = DimensionAgnosticBatchNorm(self.config.num_channels)
            else:
                self.initial_norm = IdentityBatchNorm()
        elif isinstance(self.config.initial_norm, float) or isinstance(self.config.initial_norm, int):
            self.initial_norm = ConstantBatchNorm(self.config.initial_norm)
        else:
            raise ValueError(f"Invalid initial norm: {self.config.initial_norm}")

    def loss_fn(self,
                x: Float[Tensor, "batch *shape"],  # noqa: F821, typing
                t: Float[Tensor, "batch"],  # noqa: F821, typing
                y: None | Float[Tensor, "batch *yshape"] = None,  # noqa: F821, typing
                mask: None | Float[Tensor, "batch *shape"] = None  # noqa: F821, typing
                ) -> Float[Tensor, ""]:  # noqa: F821, F722
        x, y = self.encode(x, y)
        x = self.initial_norm(x)
        noise = torch.randn_like(x)
        t_broadcasted = broadcast_from_below(t, x)
        alpha, sigma = self.config.alpha_fn(t_broadcasted), self.config.sigma_fn(t_broadcasted)
        x_noised = alpha * x + sigma * noise
        flow_field = self.get_flow_field(x_noised, t, y=y, guidance=1.0)

        alpha_dot, sigma_dot = self.config.alpha_fn_dot(t_broadcasted), self.config.sigma_fn_dot(t_broadcasted)
        target = (alpha_dot * x + sigma_dot * noise)

        loss = self.config.loss_metric_module(flow_field, target)
        loss_weighting = self.config.loss_weighting.weighting_function(t_broadcasted)
        loss = loss * loss_weighting

        if mask is not None:
            # Apply the mask if it is provided
            # We assume that the mask is 1 where the data is absent
            mask = mask.expand_as(loss)
            loss = loss * (1 - mask)
        loss = loss.mean()
        return loss

    def sample_timestep(self, nsamples):
        # Sample from uniform 0 and 1
        t = self.config.loss_weighting.weighting_sampler(nsamples)
        return t

    def training_step(self, batch, batch_idx):
        x = batch['x']
        y = batch.get('y', None)
        mask = batch.get('mask', None)
        t = self.sample_timestep(x.shape[0]).to(x)
        loss = self.loss_fn(x, t, y, mask)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['x']
        y = batch.get('y', None)
        mask = batch.get('mask', None)
        t = self.sample_timestep(x.shape[0]).to(x)
        loss = self.loss_fn(x, t, y, mask)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def set_optimizer_and_scheduler(self,
                                    optimizer=None,
                                    scheduler=None,
                                    scheduler_interval="step"):
        """
        Parameters
        ----------
        optimizer : None | torch.optim.Optimizer
            if None, use the default optimizer AdamW,
            with learning rate 1e-3, betas=(0.9, 0.999),
            and weight decay 1e-4
        scheduler : None | torch.optim.lr_scheduler._LRScheduler
            if None, use the default scheduler CosineAnnealingWarmRestarts,
            with T_0=10.
        scheduler_interval : str
            "epoch" or "step", whether the scheduler should be called at the
            end of each epoch or each step.
        """
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                               lr=1e-4,
                                               betas=(0.9, 0.999),
                                               weight_decay=1e-4)
        if scheduler is not None:
            self.lr_scheduler = scheduler
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: 1.0 + 0 * step  # noqa
            )  # Neutral scheduler
        self.lr_scheduler_interval = scheduler_interval

    def configure_optimizers(self):
        lr_scheduler_config = {"scheduler": self.lr_scheduler,
                               "interval": self.lr_scheduler_interval}
        # self.hp_manager.add_runtime_optimizer_info(self.optimizer, self.lr_scheduler)
        # self.hp_manager.log_to_wandb()

        return [self.optimizer], [lr_scheduler_config]

    def get_flow_field(
            self,
            x_noised: Float[Tensor, "batch *shape"],  # noqa: F821, typing
            t: Float[Tensor, "batch"],  # noqa: F821, typing
            guidance: float = 1.0,
            y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821, typing,
            integrate_on_sigma: bool = False
    ) -> Float[Tensor, "batch *shape"]:  # noqa: F821, typing
        if guidance == 1.0 or y is None:  # Implictly no guidance
            flow_field = self.config.preconditioner(self.model, x_noised, t, y=y)
        else:
            flow_field = self.config.preconditioner(self.model, x_noised, t, y=y)
            unconditioned_flow_field = self.config.preconditioner(self.model, x_noised, t, y=None)
            flow_field = guidance * flow_field + (1 - guidance) * unconditioned_flow_field
        if integrate_on_sigma:
            sigma_dot = self.config.sigma_fn_dot(t)
            flow_field = flow_field / sigma_dot
        return flow_field

    def get_score_field(
        self,
        x_noised: Float[Tensor, "batch *shape"],  # noqa: F821, typing
        t: Float[Tensor, "batch"],  # noqa: F821, typing
        y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821, typing
        guidance: float = 1.0,
        integrate_on_sigma: bool = False
    ) -> Float[Tensor, "batch *shape"]:  # noqa: F821, typing
        flow_field = self.get_flow_field(x_noised, t, y=y, guidance=guidance, integrate_on_sigma=integrate_on_sigma)
        (alpha, sigma, alpha_dot, sigma_dot) = (
            self.config.alpha_fn(t),
            self.config.sigma_fn(t),
            self.config.alpha_fn_dot(t),
            self.config.sigma_fn_dot(t)
        )
        alpha = broadcast_from_below(alpha, x_noised)
        sigma = broadcast_from_below(sigma, x_noised)
        alpha_dot = broadcast_from_below(alpha_dot, x_noised)
        sigma_dot = broadcast_from_below(sigma_dot, x_noised)
        score_field = ((alpha * flow_field - alpha_dot * x_noised) /
                       (sigma * (alpha_dot * sigma - alpha * sigma_dot)))
        return score_field

    def get_score_field_from_flow_field(
        self,
        flow_field: Float[Tensor, "batch *shape"],  # noqa: F821, typing
        x_noised: Float[Tensor, "batch *shape"],  # noqa: F821, typing
        t: Float[Tensor, "batch"],  # noqa: F821, typing
    ) -> Float[Tensor, "batch *shape"]:  # noqa: F821, typing
        (alpha, sigma, alpha_dot, sigma_dot) = (
            self.config.alpha_fn(t),
            self.config.sigma_fn(t),
            self.config.alpha_fn_dot(t),
            self.config.sigma_fn_dot(t)
        )
        alpha = broadcast_from_below(alpha, flow_field)
        sigma = broadcast_from_below(sigma, flow_field)
        alpha_dot = broadcast_from_below(alpha_dot, flow_field)
        sigma_dot = broadcast_from_below(sigma_dot, flow_field)
        score_field = ((alpha * flow_field - alpha_dot * x_noised) /
                       (sigma * (alpha_dot * sigma - alpha * sigma_dot)))
        return score_field

    def create_time_schedule(
        self,
        nsteps: int,
        rho: float = 7.0
    ) -> Float[Tensor, "nsteps"]:  # noqa: F821
        """
        Create a time schedule for sampling.

        In normalized mode: returns t from 1 to 0 (linear spacing)
        In sigma-space mode: returns sigma from sigma_max to sigma_min (power-law spacing)

        Args:
            nsteps: Number of steps
            rho: Exponent for power-law spacing in sigma-space mode (default 7.0, from EDM)

        Returns:
            Time schedule tensor of shape [nsteps]
        """
        if self.config.is_sigma_space:
            # EDM-style power-law spacing in sigma space
            sigma_min = self.config.scheduler.sigma_min
            sigma_max = self.config.scheduler.sigma_max
            step_indices = torch.arange(nsteps)
            # sigma_i = (sigma_max^(1/rho) + i/(N-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
            sigma_min_inv_rho = sigma_min ** (1 / rho)
            sigma_max_inv_rho = sigma_max ** (1 / rho)
            sigmas = (sigma_max_inv_rho + step_indices / (nsteps - 1) * (sigma_min_inv_rho - sigma_max_inv_rho)) ** rho
            return sigmas
        else:
            # Linear spacing in normalized time
            return torch.linspace(1, 0, nsteps)

    def sample(self,
               nsamples: int,
               shape: list[int],
               y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821, typing
               guidance: float = 1.0,
               nsteps: int = 30,
               is_latent_shape: bool = False,
               integrate_on_sigma: bool = False,
               noise_injection: bool = False,
               return_latents: bool = False,
               orig_noise: Float[Tensor, "batch *shape"] | None = None,  # noqa: F821, typing
               rho: float = 7.0  # Exponent for sigma-space step schedule
               ) -> Float[Tensor, "batch *shape"]:  # noqa: F821, F722
        if torch.inference_mode():
            with torch.no_grad():
                if orig_noise is None:
                    x = torch.randn(nsamples, *shape).to(self.device)
                else:
                    assert orig_noise.shape[0] == nsamples, "Number of samples must match"
                    assert list(orig_noise.shape[1:]) == list(shape), "Shape of noise must match"
                    x = orig_noise.to(self.device)
                if y is not None:
                    warnings.warn("Moving y to device: {}".format(self.device))
                    y = dict_to(y, self.device)
                if not is_latent_shape and self.autoencoder:
                    # Need to do a stupid hack for getting correct shape
                    x, _ = self.encode(x, y)
                    x = torch.randn_like(x)
                if y is not None:
                    y = dict_unsqueeze(y, 0)
                time_schedule = self.create_time_schedule(nsteps, rho=rho).to(x)
                sigma_init = self.config.sigma_fn(time_schedule[0])
                x = x * sigma_init
                x = self.integrate_flow_field(
                    x,
                    time_schedule,
                    y,
                    guidance,
                    integrate_on_sigma=integrate_on_sigma,
                    noise_injection=noise_injection)
                if not return_latents:
                    x, _ = self.decode(x, y)
        return x  # noqa: F821, F722

    def inpaint(
        self,
        x_orig: Float[Tensor, "*shape"],  # noqa: F821, typing,
        mask: Float[Tensor, "*shape"],  # noqa: F821, typing,
        nsamples: int = 1,
        y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821, typing
        guidance: float = 1.0,
        nsteps: int = 30,
        integrate_on_sigma: bool = False,
        noise_injection: bool = False,
        orig_noise: Float[Tensor, "batch *shape"] | None = None,  # noqa: F821, typing
        # New parameters for improved inpainting:
        mask_falloff: int = 0,  # Soft mask gradient width (0 = hard mask)
        resample_steps: int = 0,  # Number of resample iterations (RePaint-style)
        jump_length: int = 1,  # Steps to jump back when resampling
        rho: float = 7.0  # Exponent for sigma-space step schedule
    ) -> Float[Tensor, "batch *shape"]:  # noqa F821, typing
        # mask: 1 for where data is present, 0 for where data is absent
        warnings.warn("We are assuming we are in latent space for inpainting")
        with torch.inference_mode():
            with torch.no_grad():
                if y is not None:
                    warnings.warn("Moving y to device: {}".format(self.device))
                    y = dict_to(y, self.device)
                x_orig = x_orig.to(self.device)
                mask = mask.to(self.device)
                shape = x_orig.shape

                # Create soft mask if falloff is specified
                if mask_falloff > 0:
                    soft_mask = self._create_soft_mask(mask, mask_falloff)
                else:
                    soft_mask = mask

                x_orig = x_orig.unsqueeze(0)
                x_orig = self.initial_norm(x_orig)
                if orig_noise is None:
                    x = torch.randn(nsamples, *shape).to(self.device)
                else:
                    assert orig_noise.shape[0] == nsamples, "Number of samples must match"
                    assert orig_noise.shape[1:] == shape, "Shape of noise must match"
                    x = orig_noise.to(self.device)
                time_schedule = self.create_time_schedule(nsteps, rho=rho).to(x)
                sigma_init = self.config.sigma_fn(time_schedule[0])
                x = x * sigma_init

                for i in range(len(time_schedule) - 1):
                    t_curr = time_schedule[i] * torch.ones(x.shape[0]).to(x)
                    t_next = time_schedule[i + 1] * torch.ones(x.shape[0]).to(x)

                    # Resample loop (RePaint-style jump back)
                    for r in range(resample_steps + 1):
                        x = self.integration_step(
                            x,
                            t_curr,
                            t_next,
                            y,
                            guidance,
                            method='euler_maruyama',
                            integrate_on_sigma=integrate_on_sigma,
                            noise_injection=True
                        )
                        sigma = broadcast_from_below(self.config.sigma_fn(t_next), x_orig)
                        alpha = broadcast_from_below(self.config.alpha_fn(t_next), x_orig)

                        x_patch = alpha * x_orig + sigma * torch.randn_like(x_orig)
                        x = (1 - soft_mask) * x + soft_mask * x_patch

                        # Jump back if not last resample iteration and not at final timestep
                        if r < resample_steps and i + jump_length < len(time_schedule) - 1:
                            # Jump back by adding noise
                            t_jump = time_schedule[i]  # Jump back to current timestep
                            sigma_jump = broadcast_from_below(
                                self.config.sigma_fn(t_jump), x
                            )
                            alpha_jump = broadcast_from_below(
                                self.config.alpha_fn(t_jump), x
                            )
                            # Re-noise the sample
                            x = alpha_jump * x + sigma_jump * torch.randn_like(x)
                            # Also update the patch for the jumped state
                            x_patch_jump = alpha_jump * x_orig + sigma_jump * torch.randn_like(x_orig)
                            x = (1 - soft_mask) * x + soft_mask * x_patch_jump

                x = self.initial_norm.unnorm(x)
                return x

    def _create_soft_mask(
        self,
        mask: Float[Tensor, "*shape"],  # noqa: F821
        falloff: int
    ) -> Float[Tensor, "*shape"]:  # noqa: F821
        """
        Create a soft mask with cosine falloff at the boundary.

        Args:
            mask: Binary mask (1 = known, 0 = unknown)
            falloff: Width of the gradient transition zone in voxels

        Returns:
            Soft mask with smooth transition at boundaries
        """
        if falloff <= 0:
            return mask

        # Use average pooling to create distance-like field, then rescale
        # This is a simple approximation that works for 3D data
        import torch.nn.functional as F

        # Determine spatial dimensions (assume first dim is channels)
        ndim = mask.dim() - 1  # Number of spatial dimensions

        # Expand mask for pooling (need batch dim)
        m = mask.unsqueeze(0).float()  # [1, C, ...]

        # Apply average pooling to get approximate distance field
        kernel_size = 2 * falloff + 1
        padding = falloff

        if ndim == 3:
            # 3D case
            m_dilated = F.avg_pool3d(
                m, kernel_size=kernel_size, stride=1, padding=padding
            )
            m_eroded = F.avg_pool3d(
                1 - m, kernel_size=kernel_size, stride=1, padding=padding
            )
        elif ndim == 2:
            # 2D case
            m_dilated = F.avg_pool2d(
                m, kernel_size=kernel_size, stride=1, padding=padding
            )
            m_eroded = F.avg_pool2d(
                1 - m, kernel_size=kernel_size, stride=1, padding=padding
            )
        else:
            # Fallback: no soft mask for other dimensions
            return mask

        # Create soft mask: 1 in known region, 0 in unknown, gradient at boundary
        # Use the pooled values to create smooth transition
        soft_mask = m_dilated / (m_dilated + m_eroded + 1e-8)

        # Apply cosine smoothing for nicer transition
        soft_mask = (1 - torch.cos(soft_mask * np.pi)) / 2

        return soft_mask.squeeze(0)  # Remove batch dim

    def integrate_flow_field(
        self,
        x: Float[Tensor, "batch *shape"],  # noqa: F821, typing
        time_schedule: Float[Tensor, "nsteps"],  # noqa: F821, typing
        y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821, typing
        guidance: float = 1.0,
        return_history: bool = False,
        integrate_on_sigma: bool = False,
        noise_injection: bool = False
    ) -> Float[Tensor, "batch *shape"]:  # noqa: F821, typing
        # Integrate the flow field x' = v(x, t) using the Heun method
        self.model.eval()
        if return_history:
            history = [(time_schedule[0], x)]

        for i in range(len(time_schedule) - 1):
            t_curr = time_schedule[i] * torch.ones(x.shape[0]).to(x)
            t_next = time_schedule[i + 1] * torch.ones(x.shape[0]).to(x)

            if noise_injection:
                method = 'euler_maruyama'
            else:
                method = 'euler' if i == len(time_schedule) - 2 else 'heun'

            x = self.integration_step(
                x,
                t_curr,
                t_next,
                y,
                guidance,
                method=method,
                integrate_on_sigma=integrate_on_sigma,
                noise_injection=noise_injection
            )

            if return_history:
                history.append((time_schedule[i + 1], x))

        if not return_history:
            x = self.initial_norm.unnorm(x)
            return x
        else:
            history = list(map(lambda tx: (tx[0], self.initial_norm.unnorm(tx[1])), history))
            return history

    def integration_step(
        self,
        x: Float[Tensor, "batch *shape"],  # noqa: F821, typing
        t_curr: Float[Tensor, "batch"],  # noqa: F821, typing
        t_next: Float[Tensor, "batch"],  # noqa: F821, typing
        y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821, typing
        guidance: float = 1.0,
        method: Literal['euler', 'heun', 'euler_maruyama'] = 'euler',
        integrate_on_sigma: bool = False,
        noise_injection: bool = False
    ) -> Float[Tensor, "batch *shape"]:  # noqa: F821, typing

        if not integrate_on_sigma:
            dt = t_next - t_curr
        else:
            dt = self.config.sigma_fn(t_next) - self.config.sigma_fn(t_curr)
        dt = broadcast_from_below(dt, x)

        # Euler method
        if method in ['euler', 'heun']:
            assert not noise_injection, "Noise injection is not supported for Euler and Heun methods"

        if method == 'euler':
            v = self.get_flow_field(x, t_curr, y=y, guidance=guidance, integrate_on_sigma=integrate_on_sigma)
            return x + dt * v
        elif method == 'heun':
            # Heun method
            # First step - Euler
            v1 = self.get_flow_field(x, t_curr, y=y, guidance=guidance, integrate_on_sigma=integrate_on_sigma)
            x_euler = x + dt * v1

            # Second step - correction
            v2 = self.get_flow_field(x_euler, t_next, y=y, guidance=guidance, integrate_on_sigma=integrate_on_sigma)
            return x + dt * (v1 + v2) / 2
        elif method == 'euler_maruyama':
            if not noise_injection:
                raise ValueError("Noise injection is required for Euler-Maruyama method")
            v = self.get_flow_field(x, t_curr, y=y, guidance=guidance, integrate_on_sigma=integrate_on_sigma)
            score_field = self.get_score_field_from_flow_field(v, x, t_curr)
            omega = self.config.sigma_fn(t_curr)  # TODO: Allow for more complex integration methods
            omega = broadcast_from_below(omega, x)
            x = x + dt * (v - 0.5 * omega * score_field)
            noise = torch.sqrt(omega * torch.abs(dt)) * torch.randn_like(x)
            x = x + noise
            return x
        else:
            raise ValueError(f"Invalid integration method: {method}")
