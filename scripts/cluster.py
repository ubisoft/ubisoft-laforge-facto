import argparse
import logging
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import faiss
import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOP_RATE = 50  # frame rate of the WavLM features (Hz)
VARIANCE = 0.95  # fraction of variance retained by the PCA
NITER = 50  # k-means iterations
MAX_POINTS_PER_CENTROID = None  # subsample patches per cluster for k-means
SEED = 0  # k-means seed
NUM_WORKERS = os.cpu_count()  # processes used to load the features


def load_patches(
    path: Path, W: np.ndarray, b: np.ndarray, patch_size: int
) -> np.ndarray:
    """Project one utterance into the content space and stack it into patches."""
    features = np.load(path).astype(np.float32)  # features are stored in float16
    content = features @ W.T + b
    T, D = content.shape
    T_trim = (T // patch_size) * patch_size
    return content[:T_trim].reshape(T_trim // patch_size, patch_size * D)


def fit_pca(x: np.ndarray, variance: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit a PCA keeping the fewest components that explain ``variance``.

    Returns ``(A, b)`` such that the reduced features are ``x @ A.T + b``.
    """
    dim_in = x.shape[1]
    pca = faiss.PCAMatrix(dim_in, dim_in)
    pca.train(x)
    eigenvalues = faiss.vector_to_array(pca.eigenvalues)
    explained = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    dim_out = int(np.searchsorted(explained, variance) + 1)
    logger.info(
        f"Keeping {dim_out} of {dim_in} components ({explained[dim_out - 1]:.1%} of variance)"
    )
    A = faiss.vector_to_array(pca.A).reshape(dim_in, dim_in)
    b = faiss.vector_to_array(pca.b)
    return A[:dim_out], b[:dim_out]


def main(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    W, b = checkpoint["W"].numpy(), checkpoint["b"].numpy()

    paths = sorted(args.in_dir.rglob("*.npy"))
    logger.info(
        f"Projecting {len(paths)} files from {args.in_dir} into the content space "
        f"(patch size {args.patch_size})"
    )

    load = partial(load_patches, W=W, b=b, patch_size=args.patch_size)
    with Pool(NUM_WORKERS) as pool:
        x = list(tqdm(pool.imap(load, paths, chunksize=64), total=len(paths)))
    x = np.concatenate(x, axis=0)
    x = np.ascontiguousarray(x)

    logger.info(f"Loaded {x.shape[0]} patches of dimension {x.shape[1]}")

    logger.info(f"Fitting PCA to retain {VARIANCE:.0%} of the variance")
    A, b = fit_pca(x, VARIANCE)
    x = np.ascontiguousarray(x @ A.T + b)

    num_points = x.shape[0]
    if MAX_POINTS_PER_CENTROID is not None:
        num_points = min(num_points, MAX_POINTS_PER_CENTROID * args.num_clusters)
    logger.info(
        f"Clustering into {args.num_clusters} clusters using {num_points} patches"
    )
    kmeans = faiss.Kmeans(
        x.shape[1],
        args.num_clusters,
        niter=NITER,
        seed=SEED,
        max_points_per_centroid=MAX_POINTS_PER_CENTROID or x.shape[0],
        verbose=True,
    )
    kmeans.train(x)

    rate = f"{int(HOP_RATE / args.patch_size)}hz"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pca_path = args.out_dir / f"patch-proj-{rate}.ckpt"
    kmeans_path = args.out_dir / f"kmeans-{rate}-{args.num_clusters}.ckpt"

    # Checkpoints in the formats of PatchProjection and KMeans.
    torch.save({"A": torch.from_numpy(A), "b": torch.from_numpy(b)}, pca_path)
    torch.save({"codebook.weight": torch.from_numpy(kmeans.centroids)}, kmeans_path)

    logger.info(f"Saved {pca_path} and {kmeans_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit the PCA and k-means codebook on the content space."
    )
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        help="path to the extracted WavLM features.",
        type=Path,
    )
    parser.add_argument(
        "checkpoint",
        help="path to the content projection checkpoint (content-proj.ckpt).",
        type=Path,
    )
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        help="path to the output directory.",
        type=Path,
    )
    parser.add_argument(
        "--patch-size",
        default=4,
        help="number of frames per patch: 4 for 12.5 Hz, 2 for 25 Hz (default: 4).",
        type=int,
    )
    parser.add_argument(
        "--num-clusters",
        default=8192,
        help="codebook size (default: 8192).",
        type=int,
    )
    args = parser.parse_args()
    main(args)
