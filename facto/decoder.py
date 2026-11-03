from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

LRELU_SLOPE = 0.1


def get_padding(k: int, d: int) -> int:
    return d * (k - 1) // 2


class ConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.LeakyReLU(LRELU_SLOPE),
            weight_norm(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    get_padding(kernel_size, dilation),
                    dilation,
                )
            ),
            nn.LeakyReLU(LRELU_SLOPE),
            weight_norm(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    get_padding(kernel_size, 1),
                    1,
                )
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.convs(x)


class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 3, 5),
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[ConvBlock(channels, kernel_size, dilation) for dilation in dilations]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class FiLM(nn.Module):
    def __init__(self, channels: int, speaker_channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(speaker_channels, 2 * channels)

        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, speaker: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(speaker).unsqueeze(-1).chunk(2, dim=1)
        x = (1 + gamma) * x + beta
        return x


class FiLMConvBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        speaker_channels: int,
    ) -> None:
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                1,
                get_padding(kernel_size, dilation),
                dilation,
            )
        )
        self.film1 = FiLM(channels, speaker_channels)

        self.conv2 = weight_norm(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                1,
                get_padding(kernel_size, 1),
                1,
            )
        )
        self.film2 = FiLM(channels, speaker_channels)

    def forward(self, x: torch.Tensor, speaker: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.film1(h, speaker)
        h = F.leaky_relu(h, LRELU_SLOPE)

        h = self.conv2(h)
        h = self.film2(h, speaker)
        h = F.leaky_relu(h, LRELU_SLOPE)

        return x + h


class FiLMResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        speaker_channels: int,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 3, 5),
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            FiLMConvBlock(channels, kernel_size, dilation, speaker_channels)
            for dilation in dilations
        )

    def forward(self, x: torch.Tensor, speaker: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, speaker)
        return x


class MRFBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        speaker_channels: int,
        upsample_kernel_size: int,
        upsample_stride: int,
        resblock_kernel_sizes: Sequence[int],
        resblock_dilations: Sequence[Sequence[int]],
    ) -> None:
        super().__init__()
        self.upsample = weight_norm(
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size=upsample_kernel_size,
                stride=upsample_stride,
                padding=(upsample_kernel_size - upsample_stride) // 2,
            )
        )

        self.resblocks = nn.ModuleList(
            FiLMResBlock(out_channels, speaker_channels, kernel_size, dilation)
            for kernel_size, dilation in zip(resblock_kernel_sizes, resblock_dilations)
        )

    def forward(self, x: torch.Tensor, speaker: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.upsample(x)
        out = sum(block(x, speaker) for block in self.resblocks)
        return out / len(self.resblocks)


class SpeakerEmbedder(nn.Module):
    def __init__(
        self, in_channels: int, channels: int, out_channels: int, num_blocks: int = 3
    ) -> None:
        super().__init__()
        self.conv_pre = weight_norm(nn.Conv1d(in_channels, channels, 7, 1, padding=3))
        self.resblocks = nn.ModuleList(
            ResBlock(channels, 3, (1, 3, 5)) for _ in range(num_blocks)
        )
        self.proj = nn.Linear(channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = sum(block(x) for block in self.resblocks)
        x = x / len(self.resblocks)
        x = torch.mean(x, dim=-1)
        x = self.proj(x)
        x = F.normalize(x, dim=-1)
        return x


class Decoder(torch.nn.Module):
    """HiFiGAN decoder with a speaker embedder and FiLM conditioning.

    The speaker embedder maps the content-normalized residual to a global
    speaker embedding, which conditions every upsampling block.

    - ``embed_speaker`` computes the speaker embedding from the residual.
    - ``decode`` synthesizes audio from content and a speaker embedding.
    - ``forward`` does both, taking the speaker from the residual.

    Call ``remove_weight_norm`` to fold the weight-norm parametrizations into the
    weights for inference.
    """

    def __init__(
        self,
        content_channels: int = 19,
        residual_channels: int = 1024,
        speaker_channels: int = 512,
        resblock_dilations: Sequence[Sequence[int]] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        ),
        resblock_kernel_sizes: Sequence[int] = (3, 7, 11),
        upsample_initial_channels: int = 512,
        upsample_factors: Sequence[int] = (10, 8, 8, 2),
        upsample_kernel_sizes: Sequence[int] = (20, 16, 16, 4),
    ) -> None:
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_factors)

        self.speaker_embedder = SpeakerEmbedder(
            residual_channels, speaker_channels, speaker_channels
        )

        self.conv_pre = weight_norm(
            nn.Conv1d(content_channels, upsample_initial_channels, 7, 1, padding=3)
        )
        self.film = FiLM(upsample_initial_channels, speaker_channels)

        self.mrfblocks = nn.ModuleList()
        in_channels = upsample_initial_channels

        for upsample_kernel_size, upsample_stride in zip(
            upsample_kernel_sizes, upsample_factors
        ):
            out_channels = in_channels // 2
            self.mrfblocks.append(
                MRFBlock(
                    in_channels,
                    out_channels,
                    speaker_channels,
                    upsample_kernel_size,
                    upsample_stride,
                    resblock_kernel_sizes,
                    resblock_dilations,
                )
            )
            in_channels = out_channels

        self.conv_post = weight_norm(nn.Conv1d(in_channels, 1, 7, 1, padding=3))

    def remove_weight_norm(self) -> None:
        for module in self.modules():
            if nn.utils.parametrize.is_parametrized(module, "weight"):
                nn.utils.parametrize.remove_parametrizations(module, "weight")

    def embed_speaker(self, residual: torch.Tensor) -> torch.Tensor:
        """
        Args:
            residual: content-normalized features of shape
                ``(B, T, residual_channels)``.

        Returns:
            Unit-norm speaker embedding of shape ``(B, speaker_channels)``.
        """
        return self.speaker_embedder(residual.transpose(1, 2))

    def forward(self, content: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Args:
            content: content of shape ``(B, T, content_channels)``.
            residual: content-normalized features at the 50 Hz feature rate, of
                shape ``(B, T_features, residual_channels)``.

        Returns:
            16 kHz audio of shape ``(B, 1, samples)``.
        """
        return self.decode(content, self.embed_speaker(residual))

    def decode(self, content: torch.Tensor, speaker: torch.Tensor) -> torch.Tensor:
        """
        Args:
            content: content of shape ``(B, T, content_channels)``.
            speaker: speaker embedding of shape ``(B, speaker_channels)``.

        Returns:
            16 kHz audio of shape ``(B, 1, samples)``.
        """
        content = content.transpose(1, 2)

        x = self.conv_pre(content)
        x = self.film(x, speaker)
        for block in self.mrfblocks:
            x = block(x, speaker)

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return torch.tanh(x)
