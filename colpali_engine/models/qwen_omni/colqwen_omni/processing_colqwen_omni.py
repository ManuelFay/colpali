from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import BatchFeature
from transformers.models.qwen2_5_omni import Qwen2_5OmniProcessor

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class ColQwen2_5OmniProcessor(BaseVisualRetrieverProcessor, Qwen2_5OmniProcessor):  # noqa: N801
    """
    Processor for ColQwen2.5 Omni.

    Args:
        *args: Variable length argument list to be passed to the parent `Qwen2VLProcessor` class.
        max_num_visual_tokens: The maximum number of visual tokens that can be processed by the model.
        **kwargs: Arbitrary keyword arguments to be passed to the parent `Qwen2VLProcessor` class.
    """

    query_prefix: ClassVar[str] = "Query: "
    query_augmentation_token: ClassVar[str] = "<|endoftext|>"

    def __init__(
        self,
        image_processor=None,
        video_processor=None,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
    ):
        super().__init__(
            image_processor=image_processor,
            video_processor=video_processor,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )
        self.tokenizer.padding_side = "left"
        self.chat_template = self.tokenizer.chat_template

    @classmethod
    def from_pretrained(
        cls,
        *args,
        device_map: Optional[str] = None,
        **kwargs,
    ):
        instance = super().from_pretrained(
            *args,
            device_map=device_map,
            **kwargs,
        )

        # if "max_num_visual_tokens" in kwargs:
        #     instance.image_processor.max_pixels = kwargs["max_num_visual_tokens"] * 28 * 28
        #     instance.image_processor.size["longest_edge"] = instance.image_processor.max_pixels

        return instance

    def process_conversations(self, conversations: List[dict]) -> BatchFeature:
        batch_doc = super().apply_chat_template(
            conversations,
            # transformers is bugged and doesn't support standalone audio when this flag is True
            load_audio_from_video=False,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            # video_fps=1,
            padding=True,
            use_audio_in_video=False,
        )

        # if "pixel_values" in batch_doc:
        #     # # NOTE: The following adjustment ensures correct behavior with DDP on multiple GPUs.
        #     offsets = batch_doc["image_grid_thw"][:, 1] * batch_doc["image_grid_thw"][:, 2]  # (batch_size,)

        #     # Split the pixel_values tensor into a list of tensors, one per image
        #     pixel_values = list(
        #         torch.split(batch_doc["pixel_values"], offsets.tolist())
        #     )  # [(num_patches_image_0, pixel_values), ..., (num_patches_image_n, pixel_values)]

        #     # Pad the list of pixel_value tensors to the same length along the sequence dimension
        #     batch_doc["pixel_values"] = torch.nn.utils.rnn.pad_sequence(
        #         pixel_values, batch_first=True
        #     )  # (batch_size, max_num_patches, pixel_values)
        return batch_doc

    def process_images(self, images: List[Image.Image]) -> BatchFeature:
        """
        Process images for ColQwen2.5.

        Args:
            images: List of PIL images or paths/urls to images.
        """

        conversations = [
            [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable "
                            "of perceiving auditory and visual inputs, as well as generating text and speech.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image.convert("RGB")},
                        {"type": "text", "text": "Describe the content."},
                    ],
                },
            ]
            for image in images
        ]
        batch_doc = self.process_conversations(conversations)
        return batch_doc

    def process_audios(self, audios) -> BatchFeature:
        """
        Process audios for ColQwen2.5.

        Args:
            audios: List of Numpy array of WAV files (or paths/URLs to WAV).
        """

        conversations = [
            [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable "
                            "of perceiving auditory and visual inputs, as well as generating text and speech.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "audio", "path": audio}, {"type": "text", "text": "Describe the content."}],
                },
            ]
            for audio in audios
        ]
        batch_doc = self.process_conversations(conversations)
        return batch_doc

    def process_videos(self, videos) -> BatchFeature:
        """
        Process videos for ColQwen2.5.

        Args:
            videos: List of videos or paths/urls to videos. Each video can be a 4D NumPy array or PyTorch
            tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
        """

        conversations = [
            [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable "
                            "of perceiving auditory and visual inputs, as well as generating text and speech.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "video", "path": video}, {"type": "text", "text": "Describe the content."}],
                },
            ]
            for video in videos
        ]
        batch_doc = self.process_conversations(conversations)
        return batch_doc

    def process_texts(
        self,
        texts: List[str],
    ) -> BatchFeature:
        """
        Process texts for ColQwenOmni.
        """

        conversations = [
            [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable "
                            "of perceiving auditory and visual inputs, as well as generating text and speech.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": text}],
                },
            ]
            for text in texts
        ]
        batch_query = self.process_conversations(conversations)
        return batch_query

    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings.
        """
        return self.score_multi_vector(qs, ps, device=device, **kwargs)

    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        spatial_merge_size: int,
    ) -> Tuple[int, int]:
        """
        Get the number of patches (n_patches_x, n_patches_y) that will be used to process an image of
        size (height, width) with the given patch size.

        The `spatial_merge_size` is the number of patches that will be merged spatially. It is stored in
        as a `Qwen2VLForConditionalGeneration` attribute under `model.spatial_merge_size`.
        """
        raise NotImplementedError("ColQwen2.5 Omni does not support the `get_n_patches` method. ")
