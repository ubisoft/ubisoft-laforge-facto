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

TEMPERATURE = 0.5
REPETITION_PENALTY = 1.1
MAX_SECONDS = 30  # longest generation
MIN_SECONDS = 1  # shortest generation


@torch.inference_mode()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frame_rate = {"12hz": 12.5, "25hz": 25.0}[args.variant.split("-")[0]]

    logger.info(f"Loading Facto and the TTS model ({args.variant})")
    facto = torch.hub.load(REPO, "facto", variant=args.variant, trust_repo=True)
    lm, tokenizer = torch.hub.load(REPO, "tts", variant=args.variant, trust_repo=True)
    facto, lm = facto.to(device), lm.to(device)

    # Tokenize the reference clip.
    decoder = AudioDecoder(args.reference, sample_rate=SAMPLE_RATE, num_channels=1)
    wav = decoder.get_all_samples().data.unsqueeze(0).to(device)
    reference_tokens, speaker = facto.encode(wav)

    input_ids, attention_mask = tokenizer.encode(
        [(args.reference_text, args.text, reference_tokens[0])]
    )
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

    output_ids = lm.generate(
        input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        temperature=TEMPERATURE,
        top_k=0,
        repetition_penalty=REPETITION_PENALTY,
        max_new_tokens=int(MAX_SECONDS * frame_rate),
        min_new_tokens=int(MIN_SECONDS * frame_rate),
        cache_implementation="dynamic",
    )
    tokens = tokenizer.decode(output_ids, input_ids.shape[1])[0]

    if len(tokens) == 0:
        raise SystemExit(
            "The model generated no speech tokens. Please double check that inputs."
        )

    audio = facto.decode(tokens.unsqueeze(0), speaker)
    AudioEncoder(audio.squeeze(0).cpu(), sample_rate=SAMPLE_RATE).to_file(args.out_path)

    logger.info(f"Saved {audio.shape[-1] / SAMPLE_RATE:.1f} secs of audio")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthesize speech in the voice of a reference clip."
    )
    parser.add_argument(
        "reference",
        help="path to the reference audio clip.",
        type=Path,
    )
    parser.add_argument(
        "reference_text",
        metavar="reference-text",
        help="transcript of the reference clip.",
    )
    parser.add_argument(
        "text",
        help="text to synthesize.",
    )
    parser.add_argument(
        "out_path",
        metavar="out-path",
        help="path to save the generated audio (e.g. out.wav).",
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
