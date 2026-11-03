from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaModel, LlamaPreTrainedModel


class LlamaForCausalTTS(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.offset = config.offset
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        # Only compute logits for the positions we need.
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        hidden_states = outputs.last_hidden_state[:, slice_indices, :]
        logits = F.linear(hidden_states, self.lm_head.weight[self.offset :])
        logits = F.pad(logits, (self.offset, 0), value=float("-inf"))

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class TTSTokenizer:
    """Text and speech tokenizer for TTS.

    Token layout, where ``n_codes`` is the Facto codebook size:

    - ``[0, offset)``: text tokens from ``PleIAs/Monad`` (``offset = 8192``).
    - ``[offset, offset + n_codes)``: speech tokens.
    - ``offset + n_codes + {0, 1, 2}``: ``<|pad|>``, ``<|start_of_speech|>``,
      ``<|end_of_speech|>``.

    Prompt format::

        <|begin_of_text|> text <|end_of_text|> <|start_of_speech|> prompt speech

    The model continues with speech tokens and ends with ``<|end_of_speech|>``.

    ``encode`` builds a left-padded batch of prompts.
    ``decode`` extracts the generated speech tokens.
    """

    def __init__(self, n_codes: int, model_name: str = "PleIAs/Monad") -> None:
        self.text_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.n_codes = n_codes
        self.offset = len(self.text_tokenizer)
        self.pad_id = self.offset + n_codes
        self.start_of_speech_id = self.pad_id + 1
        self.end_of_speech_id = self.pad_id + 2

    def encode(
        self, prompts: list[tuple[str, str, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            prompts: ``(prompt_text, target_text, prompt_tokens)`` tuples, where
                ``prompt_text`` is the transcript of the reference audio,
                ``target_text`` is the text to synthesize, and ``prompt_tokens``
                are the reference audio's Facto tokens (1-D, in ``[0, n_codes)``).

        Returns:
            ``input_ids`` of shape ``(B, T)`` and a boolean ``attention_mask``.
        """
        sequences = []
        for prompt_text, target_text, prompt_tokens in prompts:
            # Tokenize both texts together so the boundary doesn't split a token.
            text_ids = self.text_tokenizer.encode(
                prompt_text + " " + target_text, add_special_tokens=False
            )
            special = [self.text_tokenizer.bos_token_id, *text_ids]
            special += [self.text_tokenizer.eos_token_id, self.start_of_speech_id]
            sequences.append(
                torch.cat([torch.tensor(special), prompt_tokens.cpu() + self.offset])
            )
        input_ids = nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=self.pad_id, padding_side="left"
        )
        return input_ids, input_ids.ne(self.pad_id)

    def decode(self, output_ids: torch.Tensor, prompt_len: int) -> list[torch.Tensor]:
        """
        Args:
            output_ids: ``generate()`` output of shape ``(B, prompt_len + new)``.
            prompt_len: length of the (padded) prompt, ``input_ids.shape[1]``.

        Returns:
            One 1-D tensor of tokens in ``[0, n_codes)`` per sequence.
        """
        tokens = output_ids[:, prompt_len:] - self.offset
        # Keep the speech tokens up to the first non-speech token, which is
        # normally <|end_of_speech|>.
        keep = ((tokens >= 0) & (tokens < self.n_codes)).cumprod(-1).bool()
        return [row[mask] for row, mask in zip(tokens, keep)]
