from typing import ClassVar, Optional

import torch
from torch import nn
from transformers.models.paligemma.modeling_paligemma import (
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaPreTrainedModel,
)


class ColPali(PaliGemmaPreTrainedModel):
    """
    ColPali model implementation from the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.

    Args:
        config (PaliGemmaConfig): The model configuration.
        mask_non_image_embeddings (Optional[bool]): Whether to ignore all tokens embeddings
            except those of the image at inference.
            Defaults to False --> Do not mask any embeddings during forward pass.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related
    _keys_to_ignore_on_load_missing = [r"model\.lm_head\.weight"]
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

    def __init__(self, config: PaliGemmaConfig, mask_non_image_embeddings: bool = False):
        super().__init__(config=config)

        model = PaliGemmaForConditionalGeneration(config=config)
        if model.model.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [
                f"model.model.language_model.{k}" for k in model.model.language_model._tied_weights_keys
            ]
        self.model = model

        # TODO: Wait for ColPali2 to create a ColPaliConfig to allow specifying the embedding dimension.
        # We could do it now but it would break all the models trying to load the model from the checkpoint.
        self.dim = 128
        self.custom_text_proj = nn.Linear(self.model.config.text_config.hidden_size, self.dim)

        self.mask_non_image_embeddings = mask_non_image_embeddings

        self.post_init()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Delete output_hidden_states from kwargs
        kwargs.pop("output_hidden_states", None)
        if "pixel_values" in kwargs:
            kwargs["pixel_values"] = kwargs["pixel_values"].to(dtype=self.dtype)

        outputs = self.model(*args, output_hidden_states=True, **kwargs)  # (batch_size, sequence_length, hidden_size)
        last_hidden_states = outputs.hidden_states[-1]  # (batch_size, sequence_length, hidden_size)
        proj = self.custom_text_proj(last_hidden_states)  # (batch_size, sequence_length, dim)

        # L2 normalization
        proj = proj / proj.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)

        proj = proj * kwargs["attention_mask"].unsqueeze(-1)  # (batch_size, sequence_length, dim)

        if "pixel_values" in kwargs and self.mask_non_image_embeddings:
            # Pools only the image embeddings
            image_mask = (kwargs["input_ids"] == self.config.image_token_index).unsqueeze(-1)
            proj = proj * image_mask
        return proj

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

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of=None,
    ) -> nn.Embedding:
        model_embeds = self.model.model.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # Update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.model.vocab_size = model_embeds.num_embeddings

        return model_embeds

    @property
    def patch_size(self) -> int:
        return self.model.vision_tower.config.patch_size
