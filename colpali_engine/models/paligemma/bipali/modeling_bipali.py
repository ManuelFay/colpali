from typing import Optional

import torch
from torch import nn
from transformers.models.paligemma.configuration_paligemma import PaliGemmaConfig
from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration, PaliGemmaPreTrainedModel


class BiPali(PaliGemmaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"model\.lm_head\.weight"]
    """
    BiPali is an implementation from the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.
    Representations are average pooled to obtain a single vector representation.
    """

    _checkpoint_conversion_mapping = {
        "^model.language_model.model": "model.model.language_model",
        "^model.vision_tower": "model.model.vision_tower",
        "^model.multi_modal_projector": "model.model.multi_modal_projector",
        "^model.language_model.lm_head": "model.lm_head",
        r"^base_model\.model\.custom_text_proj": "custom_text_proj",
    }

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        key_mapping = kwargs.pop("key_mapping", None)
        if key_mapping is None:
            key_mapping = cls._checkpoint_conversion_mapping
        return super().from_pretrained(*args, **kwargs, key_mapping=key_mapping)

    def __init__(self, config: PaliGemmaConfig):
        super(BiPali, self).__init__(config=config)
        model: PaliGemmaForConditionalGeneration = PaliGemmaForConditionalGeneration(config)
        if model.model.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [
                f"model.model.language_model.{k}" for k in model.model.language_model._tied_weights_keys
            ]
        self.model: PaliGemmaForConditionalGeneration = model
        self.main_input_name = "doc_input_ids"
        self.post_init()

    def get_input_embeddings(self):
        return self.model.model.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.model.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.model.model.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.model.model.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.model.model.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.model.language_model.get_decoder()

    def tie_weights(self, *args, **kwargs):
        return self.model.model.language_model.tie_weights(*args, **kwargs)

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.model.model.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.model.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def forward(self, *args, **kwargs):
        # delete output_hidden_states from kwargs
        kwargs.pop("output_hidden_states", None)
        if "pixel_values" in kwargs:
            kwargs["pixel_values"] = kwargs["pixel_values"].to(dtype=self.dtype)

        outputs = self.model(*args, output_hidden_states=True, **kwargs)
        last_hidden_states = outputs.hidden_states[-1]  # (batch_size, sequence_length, hidden_size)
        # pooling -mean on attention mask==1
        proj = torch.sum(last_hidden_states * kwargs["attention_mask"].unsqueeze(-1), dim=1) / torch.sum(
            kwargs["attention_mask"], dim=1, keepdim=True
        )
        proj = proj / proj.norm(dim=-1, keepdim=True)
        return proj


class BiPaliProj(PaliGemmaPreTrainedModel):
    """
    BiPaliProj is a BiPali model with a projection layer for dimensionality reduction.
    """

    def __init__(self, config: PaliGemmaConfig):
        super(BiPaliProj, self).__init__(config=config)
        model: PaliGemmaForConditionalGeneration = PaliGemmaForConditionalGeneration(config)
        if model.model.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [
                f"model.model.language_model.{k}" for k in model.model.language_model._tied_weights_keys
            ]
        self.model: PaliGemmaForConditionalGeneration = model
        self.main_input_name = "doc_input_ids"
        self.dim = 1024
        self.custom_text_proj = nn.Linear(self.model.config.text_config.hidden_size, self.dim)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.model.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.model.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.model.model.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.model.model.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.model.model.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.model.language_model.get_decoder()

    def tie_weights(self, *args, **kwargs):
        return self.model.model.language_model.tie_weights(*args, **kwargs)

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.model.model.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.model.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def forward(self, *args, **kwargs):
        # delete output_hidden_states from kwargs
        kwargs.pop("output_hidden_states", None)
        if "pixel_values" in kwargs:
            kwargs["pixel_values"] = kwargs["pixel_values"].to(dtype=self.dtype)

        outputs = self.model(*args, output_hidden_states=True, **kwargs)
        last_hidden_states = outputs.hidden_states[-1]  # (batch_size, sequence_length, hidden_size)

        # pooling -mean on attention mask==1
        proj = torch.sum(last_hidden_states * kwargs["attention_mask"].unsqueeze(-1), dim=1) / torch.sum(
            kwargs["attention_mask"], dim=1, keepdim=True
        )
        proj = self.custom_text_proj(proj)
        proj = proj / proj.norm(dim=-1, keepdim=True)
        return proj
