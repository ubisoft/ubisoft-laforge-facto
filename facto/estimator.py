import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentProjection(nn.Module):
    """Projects self-supervised features into the factorized content space.

    Uses the least-squares estimator ``C = X W + b``. Also returns the
    content-normalized residual ``X - C M``, where ``M`` is the average speaker
    transformation.
    """

    def __init__(self, hidden_size: int, content_size: int) -> None:
        super().__init__()
        self.W = nn.Parameter(
            torch.zeros(content_size, hidden_size), requires_grad=False
        )
        self.b = nn.Parameter(torch.zeros(content_size), requires_grad=False)
        self.M = nn.Parameter(
            torch.zeros(content_size, hidden_size), requires_grad=False
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: features of shape ``(B, T, hidden_size)``.

        Returns:
            Tuple of the content of shape ``(B, T, content_size)`` and the
            residual of shape ``(B, T, hidden_size)``.
        """
        content = F.linear(hidden_states, self.W, self.b)
        residual = hidden_states - content @ self.M
        return content, residual


def patch(content: torch.Tensor, patch_size: int) -> torch.Tensor:
    B, T, D = content.shape
    T_trim = (T // patch_size) * patch_size
    return content[:, :T_trim].reshape(B, T_trim // patch_size, patch_size * D)


class PatchProjection(nn.Module):
    """Stacks ``patch_size`` neighboring content frames into patches and applies PCA."""

    def __init__(self, content_size: int, patch_size: int, output_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.content_size = content_size
        self.A = nn.Parameter(
            torch.zeros(output_size, content_size * patch_size), requires_grad=False
        )
        self.b = nn.Parameter(torch.zeros(output_size), requires_grad=False)

    def forward(self, content: torch.Tensor) -> torch.Tensor:
        """
        Args:
            content: content features of shape ``(B, T, content_size)``.

        Returns:
            Projected patches of shape ``(B, T // patch_size, output_size)``.
        """
        return F.linear(patch(content, self.patch_size), self.A, self.b)


class KMeans(nn.Module):
    """Quantizes content patches to their nearest codebook vectors."""

    def __init__(self, n_clusters: int, content_size: int) -> None:
        super().__init__()
        self.codebook = nn.Embedding(n_clusters, content_size)
        self.codebook.weight.requires_grad_(False)

    def forward(self, content: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            content: features of shape ``(B, T, content_size)``.

        Returns:
            Tuple of:
            - **indices** — nearest cluster indices of shape ``(B, T)``.
            - **quantized** — the corresponding codebook vectors of shape
              ``(B, T, content_size)``.
        """
        B, T, D = content.shape

        codebook = self.codebook.weight
        K, _ = codebook.shape

        content = content.view(-1, D)

        c_norm = torch.sum(content**2, dim=1, keepdim=True)
        k_norm = torch.sum(codebook**2, dim=1).view(1, K)

        # Compute: ||x - c||^2 = ||x||^2 + ||c||^2 - 2xc^T
        dists = torch.addmm(
            k_norm + c_norm,
            content,
            codebook.t(),
            alpha=-2.0,
            beta=1.0,
        )
        indices = torch.argmin(dists, dim=1)

        indices = indices.view(B, T)
        return indices, self.codebook(indices)


class ContentEstimator(nn.Module):
    """Maps self-supervised features to content patches.

    Applies the content projection, then patching and PCA.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        content_size: int = 12,
        patch_size: int = 4,
        output_size: int = 19,
    ) -> None:
        super().__init__()
        self.content_proj = ContentProjection(hidden_size, content_size)
        self.patch_proj = PatchProjection(content_size, patch_size, output_size)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: WavLM features of shape ``(B, T, hidden_size)``.

        Returns:
            Tuple of:
            - **content** — continuous content patches of shape
              ``(B, T // patch_size, output_size)``.
            - **residual** — content-normalized features ``X - C M`` of shape
              ``(B, T, hidden_size)``, used as input to the speaker embedder.
        """
        content, residual = self.content_proj(hidden_states)
        return self.patch_proj(content), residual


def estimate(X, C):
    r"""Fit the least-squares content estimator.

    Finds ``(W, b)`` minimizing ``\sum_{k=1}^K ||X_k W + b - C||_F^2``.

    Args:
        X: input features of shape ``(K, T, N)``, where K is the number of
            speakers, T is the number of frames, and N is the feature dimension.
        C: ground-truth content space of shape ``(T, P)``, where P is the
            content space rank.

    Returns:
        W: weight of shape (N, P)
        b: bias of shape (P,)
    """
    K, T, N = X.shape

    XMU = np.mean(X, axis=(0, 1))
    CMU = np.mean(C, axis=0)

    X = X - XMU
    C = C - CMU

    XTX = sum(X[k, :, :].T @ X[k, :, :] for k in range(K))
    XTC = sum(X[k, :, :].T @ C for k in range(K))

    W = np.linalg.solve(XTX, XTC)

    b = CMU - (XMU @ W)
    return W, b
