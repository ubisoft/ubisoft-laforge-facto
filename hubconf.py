import torch

from facto.encoder import Encoder
from facto.decoder import Decoder
from facto.estimator import ContentEstimator, KMeans
from facto.facto import Facto

dependencies = ["torch"]

RELEASE_URL = (
    "https://github.com/ubisoft/ubisoft-laforge-facto/releases/download/v0.1.0"
)

_DECODER_CONFIGS = {
    "12hz": dict(
        content_channels=19,
        upsample_factors=(10, 8, 8, 2),
        upsample_kernel_sizes=(20, 16, 16, 4),
    ),
    "25hz": dict(
        content_channels=12,
        upsample_factors=(10, 8, 4, 2),
        upsample_kernel_sizes=(20, 16, 8, 4),
    ),
}

_CESTIMATOR_CONFIGS = {
    "12hz": dict(patch_size=4, output_size=19),
    "25hz": dict(patch_size=2, output_size=12),
}

_VARIANTS = ("12hz-8192", "12hz-16384", "25hz-8192", "25hz-16384")


_CHECKPOINTS = {
    "encoder": "encoder-2bc62958.ckpt",
    "content-proj": "content-proj-23e55fa2.ckpt",
    "patch-proj-12hz": "patch-proj-12hz-3fbaab5b.ckpt",
    "patch-proj-25hz": "patch-proj-25hz-8c90f35d.ckpt",
    "kmeans-12hz-8192": "kmeans-12hz-8192-716b5e75.ckpt",
    "kmeans-12hz-16384": "kmeans-12hz-16384-e6936ddf.ckpt",
    "kmeans-25hz-8192": "kmeans-25hz-8192-72d20d5a.ckpt",
    "kmeans-25hz-16384": "kmeans-25hz-16384-fdb2a98e.ckpt",
    "decoder-12hz-8192": "decoder-12hz-8192-9a2e4864.ckpt",
    "decoder-12hz-16384": "decoder-12hz-16384-a2bcf41e.ckpt",
    "decoder-25hz-8192": "decoder-25hz-8192-b96d4bf4.ckpt",
    "decoder-25hz-16384": "decoder-25hz-16384-2d7e3295.ckpt",
    "tts-12hz-8192": "tts-12hz-8192-a7763bd6.ckpt",
    "tts-12hz-16384": "tts-12hz-16384-7c4acd08.ckpt",
    "tts-25hz-8192": "tts-25hz-8192-4ca76648.ckpt",
    "tts-25hz-16384": "tts-25hz-16384-227585d8.ckpt",
}

# Llama configuration of the TTS language models.
_TTS_CONFIG = dict(
    hidden_size=256,
    intermediate_size=768,
    num_hidden_layers=64,
    num_attention_heads=4,
    num_key_value_heads=4,
    head_dim=64,
    hidden_act="silu",
    max_position_embeddings=2048,
    rms_norm_eps=1e-5,
    rope_theta=10000,
    rope_scaling=None,
    attention_bias=False,
    attention_dropout=0.0,
    mlp_bias=False,
    initializer_range=0.02,
    pretraining_tp=1,
    tie_word_embeddings=True,
    bos_token_id=1,
    eos_token_id=2,
    use_cache=True,
    offset=8192,
)
_TTS_VOCAB_SIZES = {
    "12hz-8192": 16448,
    "12hz-16384": 24640,
    "25hz-8192": 16448,
    "25hz-16384": 24640,
}


def _check_variant(variant: str) -> None:
    if variant not in _VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}, expected one of {_VARIANTS}")


def encoder(pretrained: bool = True, progress: bool = True) -> Encoder:
    """WavLM-Large truncated to layer six.

    Maps 16 kHz audio of shape ``(B, 1, samples)`` to features of shape
    ``(B, T, 1024)`` at 50 Hz.
    """
    model = Encoder()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            f"{RELEASE_URL}/{_CHECKPOINTS['encoder']}",
            map_location="cpu",
            progress=progress,
            check_hash=True,
        )
        model.load_state_dict(state_dict)
    model.eval()
    return model


def decoder(
    variant: str = "12hz-8192", pretrained: bool = True, progress: bool = True
) -> Decoder:
    """HiFiGAN decoder and speaker embedder.

    Args:
        variant: one of ``"12hz-8192"``, ``"12hz-16384"``, ``"25hz-8192"`` or
            ``"25hz-16384"``
            (frame rate and codebook size).
    """
    _check_variant(variant)
    frame_rate = variant.split("-")[0]
    model = Decoder(**_DECODER_CONFIGS[frame_rate])
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            f"{RELEASE_URL}/{_CHECKPOINTS[f'decoder-{variant}']}",
            map_location="cpu",
            progress=progress,
            check_hash=True,
        )
        model.load_state_dict(state_dict)
    model.eval()
    return model


def facto(
    variant: str = "12hz-8192", pretrained: bool = True, progress: bool = True
) -> Facto:
    """The Facto Tokenizer: audio -> (tokens, speaker) -> audio.

    Args:
        variant: one of ``"12hz-8192"``, ``"12hz-16384"``, ``"25hz-8192"`` or
            ``"25hz-16384"``
            (frame rate and codebook size).
    """
    _check_variant(variant)
    model_decoder = decoder(variant, pretrained, progress)
    frame_rate, n_clusters = variant.split("-")
    config = _CESTIMATOR_CONFIGS[frame_rate]
    cestimator = ContentEstimator(**config)
    kmeans = KMeans(int(n_clusters), config["output_size"])
    if pretrained:
        for module, name in [
            (cestimator.content_proj, "content-proj"),
            (cestimator.patch_proj, f"patch-proj-{frame_rate}"),
            (kmeans, f"kmeans-{variant}"),
        ]:
            state_dict = torch.hub.load_state_dict_from_url(
                f"{RELEASE_URL}/{_CHECKPOINTS[name]}",
                map_location="cpu",
                progress=progress,
                check_hash=True,
            )
            module.load_state_dict(state_dict)
    model = Facto(
        encoder(pretrained, progress),
        cestimator,
        kmeans,
        model_decoder,
    )
    model.eval()
    return model


def tts(variant: str = "12hz-8192", pretrained: bool = True, progress: bool = True):
    """Text-to-speech language model and based on the Facto tokenizer.

    Requires ``transformers``. The tokenizer downloads the ``PleIAs/Monad`` text
    tokenizer from the Hugging Face Hub.

    Args:
        variant: one of ``"12hz-8192"``, ``"12hz-16384"``, ``"25hz-8192"`` or
            ``"25hz-16384"``
            (frame rate and codebook size). Decode the generated tokens with the
            ``facto`` entry of the same variant.

    Returns:
        Tuple of ``(model, tokenizer)``: a ``LlamaForCausalTTS`` in bfloat16 and
        the matching ``TTSTokenizer``.
    """
    from transformers import GenerationConfig, LlamaConfig

    from facto.tts import LlamaForCausalTTS, TTSTokenizer

    _check_variant(variant)
    tokenizer = TTSTokenizer(n_codes=int(variant.split("-")[1]))
    config = LlamaConfig(**_TTS_CONFIG, vocab_size=_TTS_VOCAB_SIZES[variant])
    # Build directly in bfloat16 (rather than casting afterwards) so the rotary
    # embedding frequencies stay in float32, matching from_pretrained.
    model = LlamaForCausalTTS._from_config(config, dtype=torch.bfloat16)
    if pretrained:
        # lm_head is tied to the input embeddings, so it isn't stored.
        state_dict = torch.hub.load_state_dict_from_url(
            f"{RELEASE_URL}/{_CHECKPOINTS[f'tts-{variant}']}",
            map_location="cpu",
            progress=progress,
            check_hash=True,
        )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing != ["lm_head.weight"] or unexpected:
            raise RuntimeError(
                f"Unexpected TTS checkpoint keys: missing={missing}, "
                f"unexpected={unexpected}"
            )
    model.generation_config = GenerationConfig(
        bos_token_id=config.bos_token_id,
        eos_token_id=tokenizer.end_of_speech_id,
        pad_token_id=tokenizer.pad_id,
        cache_implementation="static",
    )
    model.eval()
    return model, tokenizer
