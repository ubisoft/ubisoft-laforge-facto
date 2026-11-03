import argparse
import logging
from pathlib import Path

import torch
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO = "ubisoft/ubisoft-laforge-facto"
SAMPLE_RATE = 16000


def load_audio(path: Path, device: str) -> torch.Tensor:
    decoder = AudioDecoder(path, sample_rate=SAMPLE_RATE, num_channels=1)
    return decoder.get_all_samples().data.unsqueeze(0).to(device)


@torch.inference_mode()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading Facto ({args.variant})")
    facto = torch.hub.load(REPO, "facto", variant=args.variant, trust_repo=True)
    facto = facto.to(device)

    tokens, _ = facto.encode(load_audio(args.source, device))
    _, speaker = facto.encode(load_audio(args.target, device))

    audio = facto.decode(tokens, speaker)
    AudioEncoder(audio.squeeze(0).cpu(), sample_rate=SAMPLE_RATE).to_file(args.out_path)

    logger.info(f"Saved {audio.shape[-1] / SAMPLE_RATE:.1f} secs of audio")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a source clip to a target voice."
    )
    parser.add_argument(
        "source",
        help="path to the source clip.",
        type=Path,
    )
    parser.add_argument(
        "target",
        help="path to the target clip.",
        type=Path,
    )
    parser.add_argument(
        "out_path",
        metavar="out-path",
        help="path to save the converted audio (e.g. out.wav).",
        type=Path,
    )
    parser.add_argument(
        "--variant",
        default="12hz-8192",
        choices=["12hz-8192", "12hz-16384", "25hz-8192", "25hz-16384"],
        help="frame rate and codebook size (default: 12hz-8192).",
    )
    args = parser.parse_args()
    main(args)
