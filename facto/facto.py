import torch
import torch.nn as nn

from .estimator import ContentEstimator, KMeans
from .decoder import Decoder
from .encoder import Encoder


class Facto(nn.Module):
    """Facto speech tokenizer.

    ``encode`` maps 16 kHz audio to content tokens and a speaker embedding, and
    ``decode`` maps tokens and a speaker embedding back to audio.
    """

    def __init__(
        self,
        encoder: Encoder,
        cestimator: ContentEstimator,
        kmeans: KMeans,
        decoder: Decoder,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.cestimator = cestimator
        self.kmeans = kmeans
        self.decoder = decoder

    def encode(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            wav: 16 kHz audio of shape ``(B, 1, samples)``.

        Returns:
            Tuple of:
            - **indices** — content tokens of shape ``(B, T)``.
            - **speaker** — speaker embedding of shape ``(B, speaker_channels)``.
        """
        content, residual = self.cestimator(self.encoder(wav))
        indices, _ = self.kmeans(content)
        speaker = self.decoder.embed_speaker(residual)
        return indices, speaker

    def decode(self, indices: torch.Tensor, speaker: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: content tokens of shape ``(B, T)``.
            speaker: speaker embedding of shape ``(B, speaker_channels)``.

        Returns:
            16 kHz audio of shape ``(B, 1, samples)``.
        """
        return self.decoder.decode(self.kmeans.codebook(indices), speaker)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav: 16 kHz audio of shape ``(B, 1, samples)``.

        Returns:
            Resynthesized 16 kHz audio of shape ``(B, 1, samples)``.
        """
        return self.decode(*self.encode(wav))
