import argparse
import logging
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from facto.factorize import align, factorize
from facto.estimator import estimate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOP_RATE = 50  # frame rate of the WavLM features (Hz)
MINUTES = 5  # minutes of speech per speaker (T = MINUTES * 60 * HOP_RATE)
RANK = 12  # rank of the content space
SEED = 0  # seed for the anchor frames and randomized SVD
NUM_WORKERS = os.cpu_count()  # processes used to load the features


def load_speaker(speaker_dir: Path, num_frames: int) -> np.ndarray | None:
    """Load the first ``num_frames`` frames of features for a speaker."""
    xs, length = [], 0
    for path in sorted(speaker_dir.rglob("*.npy")):
        x = np.load(path).astype(np.float32)  # features are stored in float16
        xs.append(x)
        length += x.shape[0]
        if length >= num_frames:
            return np.concatenate(xs, axis=0)[:num_frames]
    return None


def main(args):
    num_frames = int(MINUTES * 60 * HOP_RATE)
    speaker_dirs = sorted(path for path in args.in_dir.iterdir() if path.is_dir())

    logger.info(
        f"Loading {MINUTES} min ({num_frames} frames) from each of "
        f"{len(speaker_dirs)} speakers in {args.in_dir}"
    )
    with Pool(NUM_WORKERS) as pool:
        loaded = list(
            tqdm(
                pool.imap(partial(load_speaker, num_frames=num_frames), speaker_dirs),
                total=len(speaker_dirs),
            )
        )

    skipped = [d.name for d, x in zip(speaker_dirs, loaded) if x is None]
    if skipped:
        logger.warning(
            f"Skipping {len(skipped)} speakers with less than {MINUTES} min of "
            f"speech: {', '.join(skipped)}"
        )
    loaded = [x for x in loaded if x is not None]

    X = np.stack(loaded, axis=1)
    del loaded
    T, K, D = X.shape
    logger.info(f"Loaded features of shape (K={K}, T={T}, D={D})")

    # Build the anchor set by choosing each frame from a random speaker.
    rng = np.random.default_rng(SEED)
    ks = rng.integers(0, K, size=T)
    anchor = X[np.arange(T), ks, :]

    # Align every speaker to the anchor.
    logger.info("Aligning speakers to the anchor set")
    for k in tqdm(range(K)):
        X[:, k, :] = align(anchor, X[:, k, :])

    logger.info(f"Factorizing with rank {RANK}")
    C, M, _ = factorize(X, RANK, random_state=SEED)

    logger.info("Fitting the least-squares content estimator")
    # estimate expects speaker-major (K, T, D) features.
    X = X.transpose(1, 0, 2)
    X = np.ascontiguousarray(X)
    W, b = estimate(X, C)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "W": torch.from_numpy(W.T).contiguous(),
            "b": torch.from_numpy(b),
            "M": torch.from_numpy(M),
        },
        args.out_dir / "content-proj.ckpt",
    )
    logger.info(f"Saved content projection to {args.out_dir / 'content-proj.ckpt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Factorize WavLM features into content and speaker components."
    )
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        help="path to the extracted features (one subdirectory per speaker).",
        type=Path,
    )
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        help="path to the output directory.",
        type=Path,
    )
    args = parser.parse_args()
    main(args)
