import warnings

import torch.optim as optim


class InverseLR(optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: optim.Optimizer,
        inv_gamma: float = 1.0,
        power: float = 1.0,
        warmup: float = 0.0,
        final_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.inv_gamma = inv_gamma
        self.power = power
        if not 0.0 <= warmup < 1:
            raise ValueError("Invalid value for warmup")
        self.warmup = warmup
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Return per-group LRs; delegates to :meth:`_get_closed_form_lr`."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self) -> list[float]:
        """Compute LR analytically from the current epoch index."""
        warmup = 1 - self.warmup ** (self.last_epoch + 1)
        lr_mult = (1 + self.last_epoch / self.inv_gamma) ** -self.power
        return [
            warmup * max(self.final_lr, base_lr * lr_mult) for base_lr in self.base_lrs
        ]
