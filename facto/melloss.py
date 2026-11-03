from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchaudio.transforms import MelSpectrogram


class MelSpectralLoss(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 160,
        n_mels: int = 100,
    ) -> None:
        super().__init__()
        self.melspectrogram = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=1.0,
            center=False,
            norm="slaney",
            mel_scale="slaney",
        )
        self.pad = (win_length - hop_length) // 2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, self.pad), "reflect")
        x = self.melspectrogram(x)
        x = torch.log(torch.clamp(x, min=1e-5))

        y = F.pad(y, (self.pad, self.pad), "reflect")
        y = self.melspectrogram(y)
        y = torch.log(torch.clamp(y, min=1e-5))

        loss = F.l1_loss(x, y)
        return loss


class MultiResolutionMelSpectralLoss(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_ffts: Sequence[int] = (1024, 2048, 512),
        win_lengths: Sequence[int] = (1024, 2048, 512),
        hop_lengths: Sequence[int] = (160, 320, 80),
        n_mels: Sequence[int] = (100, 128, 64),
    ) -> None:
        super().__init__()

        self.losses = nn.ModuleList(
            [
                MelSpectralLoss(
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    win_length=win_length,
                    hop_length=hop_length,
                    n_mels=n_mel,
                )
                for n_fft, win_length, hop_length, n_mel in zip(
                    n_ffts, win_lengths, hop_lengths, n_mels
                )
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = sum(criterion(x, y) for criterion in self.losses)
        return loss / len(self.losses)
