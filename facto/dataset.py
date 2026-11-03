from pathlib import Path
import random

import torch
import torch.nn.functional as F
from torchcodec.decoders import AudioDecoder
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        train: bool,
        min_frames: int = 200,
        hop_length: int = 320,
        win_length: int = 400,
        sample_rate: int = 16000,
    ):
        self.min_samples: int = (min_frames - 1) * hop_length + win_length
        self.sample_rate = sample_rate
        self.train = train

        pattern = "train-*/**/*.flac" if train else "dev-*/**/*.flac"

        self.paths = [path for path in dataset_dir.rglob(pattern)]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.paths[index]
        decoder = AudioDecoder(path, sample_rate=self.sample_rate)
        samples = decoder.get_all_samples()

        wav: torch.Tensor = samples.data
        if wav.size(-1) < self.min_samples:
            pad = self.min_samples - wav.size(-1)
            wav = F.pad(wav, (0, int(pad)), value=0)

        max_offset = wav.size(-1) - self.min_samples
        offset = random.randint(0, max_offset) if self.train else max_offset // 2
        wav = wav[:, offset : offset + self.min_samples]
        return wav
