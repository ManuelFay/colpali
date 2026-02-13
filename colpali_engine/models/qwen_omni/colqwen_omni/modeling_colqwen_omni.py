from typing import ClassVar

import torch
from torch import nn
from transformers.conversion_mapping import get_checkpoint_conversion_mapping, register_checkpoint_conversion_mapping
from transformers.core_model_loading import WeightRenaming
from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration


class ColQwen2_5Omni(Qwen2_5OmniThinkerForConditionalGeneration):  # noqa: N801
    """
    ColQwen2.5 Omni model with custom text projection layer.
    This model is a modified version of the Qwen2.5 Omni model, which includes a custom text projection layer
    for better performance in visual-textual tasks.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    _checkpoint_conversion_mapping = {
        r"^base_model\.model\.custom_text_proj": "custom_text_proj",
    }

    def __init__(self, config: Qwen2_5OmniThinkerConfig, mask_non_image_embeddings: bool = False):
        super().__init__(config=config)
        self.dim = 128
        self.custom_text_proj = nn.Linear(self.model.config.hidden_size, self.dim)
        self.lm_head = nn.Identity()  # Disable the original lm_head
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

    def get_output_embeddings(self) -> None:  # -> None | Any:
        """
        Transformers >=4.54.0 fails during resize_token_embeddings() due to a new get_output_embeddings()
        impl. The latter used to return None unless overridden, but they made it try harder to return
        *something*. Of course, this was not flagged as a breaking change. Eventually I found the
        responsible PR, and it endorses this change: https://github.com/huggingface/transformers/pull/39339
        """
        return None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # # Handle the custom "pixel_values" input obtained with `ColQwen2Processor` through unpadding
        # if "pixel_values" in kwargs:
        #     offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]  # (batch_size,)
        #     kwargs["pixel_values"] = torch.cat(
        #         [pixel_sequence[:offset] for pixel_sequence, offset in zip(kwargs["pixel_values"], offsets)],
        #         dim=0,
        #     )
        # pop return dict and output hidden states
        kwargs.pop("return_dict", True)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)
        last_hidden_states = (
            super().forward(*args, **kwargs, use_cache=False, output_hidden_states=True, return_dict=True).logits
        )  # (batch_size, sequence_length, hidden_size)# (batch_size, sequence_length, hidden_size)
        proj = self.custom_text_proj(last_hidden_states)  # (batch_size, sequence_length, dim)

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

    @property
    def spatial_merge_size(self) -> int:
        return self.visual.config.spatial_merge_size

    @spatial_merge_size.setter
    def spatial_merge_size(self, value):
        # allow assignment
        self.visual.config.spatial_merge_size = value


if get_checkpoint_conversion_mapping("qwen2_5_omni_thinker") is None:
    register_checkpoint_conversion_mapping(
        "qwen2_5_omni_thinker",
        [
            WeightRenaming(source_patterns=k, target_patterns=v)
            for k, v in ColQwen2_5Omni._checkpoint_conversion_mapping.items()
        ],
    )
