"""
ColGemma3 Model - Implementation for late interaction retrieval.

This module implements ColGemma3 for late interaction retrieval, following
the ColQwen2 architecture pattern.

Key features:
    - Direct inheritance from Gemma3Model for compatibility
    - Custom projection layer for multi-vector embeddings
    - MaxSim scoring support
"""

from typing import ClassVar

import torch
from torch import nn
from transformers.models.gemma3 import Gemma3Config, Gemma3Model


class ColGemma3(Gemma3Model):
    """
    ColGemma3 model for late interaction retrieval.

    This model extends Gemma3 to produce multi-vector embeddings suitable for
    efficient document retrieval. Each input (image or text) is encoded into a
    sequence of contextualized vectors, which can be compared using MaxSim scoring.

    Args:
        config (Gemma3Config): The model configuration.
        mask_non_image_embeddings (bool, optional): Whether to ignore all tokens embeddings
            except those of the image at inference. Defaults to False.

    Example:
        >>> model = ColGemma3.from_pretrained("google/gemma-3-4b-it")
        >>> embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
        >>> print(embeddings.shape)  # (batch_size, seq_len, 128)
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related
    _checkpoint_conversion_mapping = {
        r"^base_model\.model\.custom_text_proj": "custom_text_proj",
    }

    def __init__(
        self,
        config: Gemma3Config,
        mask_non_image_embeddings: bool = False,
    ):
        super().__init__(config=config)
        self.dim = 128
        self.custom_text_proj = nn.Linear(self.config.text_config.hidden_size, self.dim)
        self.padding_side = "left"
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.post_init()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        key_mapping = kwargs.pop("key_mapping", None)
        if key_mapping is None:
            key_mapping = dict(getattr(super(), "_checkpoint_conversion_mapping", {}))
            key_mapping.update(cls._checkpoint_conversion_mapping)
        return super().from_pretrained(*args, **kwargs, key_mapping=key_mapping)

    @property
    def device(self):
        """Get the device of the model."""
        return next(self.parameters()).device

    @property
    def dtype(self):
        """Get the dtype of the model."""
        return next(self.parameters()).dtype

    def forward(self, *args, **kwargs) -> torch.Tensor:
        kwargs.pop("return_dict", True)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)
        hidden_states = (
            super()
            .forward(*args, **kwargs, use_cache=False, output_hidden_states=True, return_dict=True)
            .last_hidden_state
        )  # (batch_size, sequence_length, hidden_size)

        proj = self.custom_text_proj(hidden_states)  # (batch_size, sequence_length, dim)

        # L2 normalization
        proj = proj / proj.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)  # (batch_size, sequence_length, dim)

        if "pixel_values" in kwargs and self.mask_non_image_embeddings:
            # Pools only the image embeddings
            image_mask = (kwargs["input_ids"] == self.config.image_token_id).unsqueeze(-1)
            proj = proj * image_mask
        return proj

    @property
    def patch_size(self) -> int:
        return self.visual.config.patch_size
