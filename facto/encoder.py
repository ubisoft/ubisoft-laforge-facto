from typing import Optional
import math
from itertools import pairwise

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class LayerNormConv(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int,
        stride: int,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
        )
        self.layer_norm = nn.LayerNorm(out_dim, elementwise_affine=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv(hidden_states)

        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)

        hidden_states = F.gelu(hidden_states)
        return hidden_states


class FeatureEncoder(nn.Module):
    def __init__(
        self,
        dims: tuple[int, ...],
        kernel_sizes: tuple[int, ...],
        strides: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            *[
                LayerNormConv(dim[0], dim[1], kernel_size, stride)
                for dim, kernel_size, stride in zip(
                    pairwise(dims), kernel_sizes, strides
                )
            ]
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        return self.conv_layers(input_values)


class FeatureProjection(nn.Module):
    def __init__(
        self,
        conv_dim: int,
        hidden_size: int,
        layer_norm_eps: float,
        feat_proj_dropout: float,
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(conv_dim, eps=layer_norm_eps)
        self.projection = nn.Linear(conv_dim, hidden_size)
        self.dropout = nn.Dropout(feat_proj_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class PositionalConvEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_conv_pos_embeddings: int,
        num_conv_pos_embedding_groups: int,
    ) -> None:
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=num_conv_pos_embeddings,
            padding=num_conv_pos_embeddings // 2,
            groups=num_conv_pos_embedding_groups,
        )
        self.conv = weight_norm(self.conv, name="weight", dim=2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.conv(hidden_states)
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        hidden_states = F.gelu(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation_dropout: float,
        hidden_dropout: float,
    ) -> None:
        super().__init__()
        self.intermediate_dropout = nn.Dropout(activation_dropout)

        self.intermediate_dense = nn.Linear(hidden_size, intermediate_size)
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_dropout = nn.Dropout(hidden_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        num_buckets: int = 320,
        max_distance: int = 800,
        has_relative_position_bias: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.num_buckets = num_buckets
        self.max_distance = max_distance

        self.gru_rel_pos_const = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))
        self.gru_rel_pos_linear = nn.Linear(self.head_dim, 8)

        if has_relative_position_bias:
            self.rel_attn_embed = nn.Embedding(self.num_buckets, self.num_heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, tgt_len, _ = hidden_states.size()

        # first pass of attention layer creates position bias
        if position_bias is None:
            position_bias = self.compute_bias(tgt_len, tgt_len)
            position_bias = position_bias.unsqueeze(0).repeat(bsz, 1, 1, 1)
            position_bias = position_bias.view(bsz * self.num_heads, tgt_len, tgt_len)

        gated_position_bias = self.compute_gated_bias(hidden_states, position_bias)

        attn_output = self.multi_head_self_attention(
            hidden_states, attention_mask, gated_position_bias
        )

        return attn_output, position_bias

    def multi_head_self_attention(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        gated_position_bias: torch.Tensor,
    ) -> torch.Tensor:
        bsz, tgt_len, _ = hidden_states.shape
        head_dim = self.embed_dim // self.num_heads

        qkv = self.in_proj(hidden_states)
        qkv = qkv.view(bsz, tgt_len, 3, self.num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if attention_mask is not None:
            key_padding_mask = attention_mask.ne(1)
            key_padding_mask = key_padding_mask[:, None, None, :]
            gated_position_bias = gated_position_bias.masked_fill(
                key_padding_mask, float("-inf")
            )

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=gated_position_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, tgt_len, self.embed_dim)
        return self.out_proj(attn_output)

    def compute_gated_bias(
        self, hidden_states: torch.Tensor, position_bias: torch.Tensor
    ) -> torch.Tensor:
        bsz, tgt_len, _ = hidden_states.shape

        # Compute relative position bias:
        # 1) get reshape hidden_states
        gated_hidden_states = hidden_states.view(bsz, tgt_len, self.num_heads, -1)
        gated_hidden_states = gated_hidden_states.permute(0, 2, 1, 3)

        # 2) project hidden states
        relative_position_proj = self.gru_rel_pos_linear(gated_hidden_states)
        relative_position_proj = relative_position_proj.view(
            bsz, self.num_heads, tgt_len, 2, 4
        ).sum(-1)

        # 3) compute gate for position bias from projected hidden states
        gate_a, gate_b = torch.sigmoid(relative_position_proj).chunk(2, dim=-1)
        gate_output = gate_a * (gate_b * self.gru_rel_pos_const - 1.0) + 2.0

        # 4) apply gate to position bias to compute gated position_bias
        gated_position_bias = (
            gate_output.view(bsz * self.num_heads, -1, 1) * position_bias
        )
        # gated_position_bias = gated_position_bias.view((-1, tgt_len, tgt_len))
        gated_position_bias = gated_position_bias.view(
            bsz, self.num_heads, tgt_len, tgt_len
        )
        return gated_position_bias

    def compute_bias(self, query_length: int, key_length: int) -> torch.Tensor:
        context_position = torch.arange(
            query_length,
            dtype=torch.long,
            device=self.rel_attn_embed.weight.device,
        )
        memory_position = torch.arange(
            key_length,
            dtype=torch.long,
            device=self.rel_attn_embed.weight.device,
        )
        relative_position = memory_position[None, :] - context_position[:, None]
        relative_position_bucket = self._relative_positions_bucket(relative_position)
        values = self.rel_attn_embed(relative_position_bucket)
        values = values.permute(2, 0, 1)
        return values

    def _relative_positions_bucket(
        self, relative_positions: torch.Tensor
    ) -> torch.Tensor:
        num_buckets = self.num_buckets // 2

        relative_buckets = (relative_positions > 0).to(torch.long) * num_buckets
        relative_positions = torch.abs(relative_positions)

        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact

        relative_positions_if_large = torch.log(relative_positions.float() / max_exact)
        relative_positions_if_large = relative_positions_if_large / math.log(
            self.max_distance / max_exact
        )
        relative_positions_if_large = relative_positions_if_large * (
            num_buckets - max_exact
        )
        relative_position_if_large = (max_exact + relative_positions_if_large).to(
            torch.long
        )
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(
            is_small, relative_positions, relative_position_if_large
        )
        return relative_buckets


class TransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        attention_dropout: float,
        num_buckets: int,
        max_bucket_distance: int,
        hidden_dropout: float,
        activation_dropout: float,
        layer_norm_eps: float,
        has_relative_position_bias: bool = True,
    ) -> None:
        super().__init__()
        self.attention = SelfAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            num_buckets=num_buckets,
            max_distance=max_bucket_distance,
            has_relative_position_bias=has_relative_position_bias,
        )
        self.dropout = nn.Dropout(hidden_dropout)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.feed_forward = FeedForward(
            hidden_size, intermediate_size, activation_dropout, hidden_dropout
        )
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, position_bias = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = hidden_states + self.feed_forward(
            self.final_layer_norm(hidden_states)
        )

        outputs = (hidden_states, position_bias)

        return outputs


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        attention_dropout: float,
        num_buckets: int,
        max_bucket_distance: int,
        num_conv_pos_embeddings: int,
        num_conv_pos_embedding_groups: int,
        layer_norm_eps: float,
        hidden_dropout: float,
        activation_dropout: float,
    ) -> None:
        super().__init__()

        self.pos_conv_embed = PositionalConvEmbedding(
            hidden_size, num_conv_pos_embeddings, num_conv_pos_embedding_groups
        )
        self.dropout = nn.Dropout(hidden_dropout)
        self.layers = nn.ModuleList(
            TransformerLayer(
                hidden_size,
                intermediate_size,
                num_attention_heads,
                attention_dropout,
                num_buckets,
                max_bucket_distance,
                hidden_dropout,
                activation_dropout,
                layer_norm_eps,
                has_relative_position_bias=i == 0,
            )
            for i in range(num_hidden_layers)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            # make sure padded tokens are not attended to
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(
                1, 1, hidden_states.shape[2]
            )
            hidden_states[~expand_attention_mask] = 0

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)

        position_bias = None
        for layer in self.layers:
            hidden_states, position_bias = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias,
            )

        return hidden_states


class Encoder(nn.Module):
    """WavLM-Large truncated to layer six.

    Maps 16 kHz audio to self-supervised features at 50 Hz (one frame every 320
    samples). The default arguments match the pretrained checkpoint.
    """

    def __init__(
        self,
        conv_dims: tuple[int, ...] = (1, 512, 512, 512, 512, 512, 512, 512),
        conv_kernel_sizes: tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2),
        conv_strides: tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2),
        hidden_size: int = 1024,
        layer_norm_eps: float = 1e-5,
        feat_proj_dropout: float = 0,
        num_hidden_layers: int = 6,
        intermediate_size: int = 4096,
        num_attention_heads: int = 16,
        attention_dropout: float = 0,
        num_buckets: int = 320,
        max_bucket_distance: int = 800,
        num_conv_pos_embeddings: int = 128,
        num_conv_pos_embedding_groups: int = 16,
        hidden_dropout: float = 0,
        activation_dropout: float = 0,
    ) -> None:
        super().__init__()

        self.feature_extractor = FeatureEncoder(
            conv_dims, conv_kernel_sizes, conv_strides
        )
        self.feature_projection = FeatureProjection(
            conv_dims[-1], hidden_size, layer_norm_eps, feat_proj_dropout
        )

        self.encoder = TransformerEncoder(
            num_hidden_layers,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            attention_dropout,
            num_buckets,
            max_bucket_distance,
            num_conv_pos_embeddings,
            num_conv_pos_embedding_groups,
            layer_norm_eps,
            hidden_dropout,
            activation_dropout,
        )

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_values: audio of shape ``(B, 1, samples)``.
            attention_mask: optional boolean mask of shape ``(B, T)`` over output
                frames; ``False`` marks padding.

        Returns:
            Features of shape ``(B, T, hidden_size)``, where ``hidden_size`` is
            1024 by default.
        """
        hidden_states = self.feature_extractor(input_values)
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.feature_projection(hidden_states)

        hidden_states = self.encoder(hidden_states, attention_mask)
        return hidden_states
