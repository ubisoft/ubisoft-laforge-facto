import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from torchcodec.decoders import AudioDecoder

from facto.encoder import Encoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def encode_dataset(args):
    logger.info(f"Loading encoder from {args.encoder}")
    encoder = Encoder()
    encoder.load_state_dict(torch.load(args.encoder, map_location="cpu"))
    encoder = encoder.cuda().eval()

    logger.info(f"Encoding dataset at {args.in_dir}")
    for in_path in tqdm(list(args.in_dir.rglob("*.flac"))):
        decoder = AudioDecoder(in_path, sample_rate=16000, num_channels=1)

        wav = decoder.get_all_samples()
        wav = wav.data.unsqueeze(0).cuda()

        with torch.inference_mode():
            x = encoder(wav)

        out_path = args.out_dir / in_path.relative_to(args.in_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Saved in float16 to halve disk usage; readers convert back to float32.
        np.save(
            out_path.with_suffix(".npy"),
            x.squeeze().cpu().numpy().astype(np.float16),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode an audio dataset.")
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        help="path to the dataset directory.",
        type=Path,
    )
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        help="path to the output directory.",
        type=Path,
    )
    parser.add_argument(
        "--encoder",
        required=True,
        help="path to the WavLM encoder checkpoint (encoder.ckpt).",
        type=Path,
    )
    args = parser.parse_args()
    encode_dataset(args)
