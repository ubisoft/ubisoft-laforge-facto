from typing import Sequence, List
from itertools import pairwise

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

LRELU_SLOPE = 0.1


def padding(kernel_size: int) -> int:
    return (kernel_size - 1) // 2


def fold(x: torch.Tensor, n: int) -> torch.Tensor:
    pad = (n - (x.shape[-1] % n)) % n
    x = F.pad(x, (0, pad))
    return x.reshape(*x.shape[:2], -1, n)


class PeriodDiscriminator(nn.Module):
    def __init__(
        self,
        period: int,
        channels: Sequence[int] = (1, 16, 64, 256, 512, 512),
        kernel_size: int = 5,
        stride: int = 3,
    ) -> None:
        super().__init__()
        self.period = period
        self.layers = nn.ModuleList(
            weight_norm(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(kernel_size, 1),
                    stride=(stride, 1),
                    padding=(kernel_size // 2, 0),
                )
            )
            for in_channels, out_channels in pairwise(channels)
        )
        self.output = weight_norm(nn.Conv2d(channels[-1], 1, (3, 1), padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = fold(x, self.period)
        features: list[torch.Tensor] = []
        for layer in self.layers:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            features.append(x)

        x = self.output(x)
        score = torch.flatten(x, 1, -1)
        return score, features


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: Sequence[int] = (2, 3, 5, 7, 11, 13, 17, 23)) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            PeriodDiscriminator(period) for period in periods
        )

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        scores: list[torch.Tensor] = []
        features: list[torch.Tensor] = []
        for discriminator in self.discriminators:
            s, f = discriminator(x)
            scores.append(s)
            features.extend(f)
        return scores, features


class SpectralDiscriminator(nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        channels: Sequence[int] = (1, 32, 64, 128, 256, 512),
        kernel_sizes: Sequence[int] = (5, 5, 5, 5, 3),
        strides: Sequence[int] = (2, 2, 2, 2, 1),
    ) -> None:
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.pad = (self.win_length - self.hop_length) // 2

        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

        self.layers = nn.ModuleList()
        for (in_channel, out_channel), kernel_size, stride in zip(
            pairwise(channels), kernel_sizes, strides
        ):
            self.layers.append(
                weight_norm(
                    nn.Conv2d(
                        in_channel,
                        out_channel,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding(kernel_size),
                    )
                )
            )
        self.output = weight_norm(
            nn.Conv2d(channels[-1], 1, kernel_size=3, padding=padding(3))
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = F.pad(x, (self.pad, self.pad), mode="reflect")
        x = torch.stft(
            x.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            normalized=True,
            center=False,
            return_complex=True,
        )
        # Power-compressed magnitude: |X|^0.5, with small epsilon for stability
        x = torch.pow(x.abs() + 1e-6, 0.5).unsqueeze(1)
        features: list[torch.Tensor] = []
        for layer in self.layers:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            features.append(x)
        x = self.output(x)
        return x, features


class MultiResolutionDiscriminator(nn.Module):
    def __init__(
        self,
        n_ffts: Sequence[int] = (206, 334, 542, 876, 1418, 2296),
        hop_lengths: Sequence[int] = (103, 167, 271, 438, 709, 1148),
        win_lengths: Sequence[int] = (206, 334, 542, 876, 1418, 2296),
    ) -> None:
        super().__init__()

        self.discriminators = nn.ModuleList(
            SpectralDiscriminator(n_fft, hop_length, win_length)
            for n_fft, hop_length, win_length in zip(n_ffts, hop_lengths, win_lengths)
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        scores: list[torch.Tensor] = []
        features: list[list[torch.Tensor]] = []
        for discriminator in self.discriminators:
            s, f = discriminator(x)
            scores.append(s)
            features.append(f)
        return scores, features


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.period_discriminator = MultiPeriodDiscriminator()
        self.resolution_discriminator = MultiResolutionDiscriminator()

    def forward(self, x: torch.Tensor) -> tuple[
        tuple[list[torch.Tensor], list[torch.Tensor]],
        tuple[list[torch.Tensor], list[list[torch.Tensor]]],
    ]:
        pscores, pfeatures = self.period_discriminator(x)
        rscores, rfeatures = self.resolution_discriminator(x)
        return (pscores, rscores), (pfeatures, rfeatures)


def hinge_loss(
    scores_real: List[torch.Tensor], scores_fake: List[torch.Tensor]
) -> torch.Tensor:
    return sum(
        F.relu(1 - real).mean() + F.relu(1 + fake).mean()
        for real, fake in zip(scores_real, scores_fake)
    )


def hinge_loss_adversarial(scores_fake: List[torch.Tensor]) -> torch.Tensor:
    return sum(-fake.mean() for fake in scores_fake)


def l1_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(x.detach(), y)


def feature_matching_loss(
    features_real: List[List[torch.Tensor]],
    features_fake: List[List[torch.Tensor]],
) -> torch.Tensor:
    matching_loss: torch.Tensor = 0
    for real, fake in zip(features_real, features_fake):
        matching_loss += sum(map(l1_loss, real, fake))
    return matching_loss
