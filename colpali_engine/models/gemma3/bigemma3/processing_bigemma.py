from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature
from transformers.models.gemma3 import Gemma3Processor

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class BiGemmaProcessor3(BaseVisualRetrieverProcessor, Gemma3Processor):  # noqa: N801
    """
    Processor for BiGemma.

    Args:
        *args: Variable length argument list to be passed to the parent `Gemma3Processor` class.
        max_num_visual_tokens: The maximum number of visual tokens that can be processed by the model.
        **kwargs: Arbitrary keyword arguments to be passed to the parent `Gemma3Processor` class.
    """

    query_augmentation_token: ClassVar[str] = "<eos>"

    def __init__(
        self,
        image_processor,
        tokenizer,
        chat_template=None,
        image_seq_length: int = 256,
        **kwargs,
    ):
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            image_seq_length=image_seq_length,
            **kwargs,
        )
        self.tokenizer.padding_side = "left"

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

        if "max_num_visual_tokens" in kwargs:
            instance.image_processor.max_pixels = kwargs["max_num_visual_tokens"] * 56 * 56
            instance.image_processor.size["longest_edge"] = instance.image_processor.max_pixels

        return instance

    def process_images(
        self,
        images: List[Image.Image],
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process images for BiGemma3.

        Args:
            images: List of PIL images.
        """
        images = [image.convert("RGB") for image in images]

        # Process each image using chat template
        batch_docs = []
        for image in images:
            # Create message in chat format
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Describe this image"},
                    ],
                }
            ]

            # Apply chat template to get formatted text
            formatted_text = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)

            # Process with formatted text
            batch_doc = self(
                text=[formatted_text],
                images=[image],
                padding="longest",
                return_tensors="pt",
            )
            batch_docs.append(batch_doc)

        if len(batch_docs) == 1:
            return batch_docs[0]

        # Concatenate results along batch dimension
        concatenated = {}
        for key in batch_docs[0].keys():
            if isinstance(batch_docs[0][key], torch.Tensor):
                concatenated[key] = torch.cat([doc[key] for doc in batch_docs], dim=0)
            else:
                # For non-tensors, take from first (assuming same)
                concatenated[key] = batch_docs[0][key]
        return BatchFeature(concatenated)

    def process_texts(self, texts: List[str]) -> Union[BatchFeature, BatchEncoding]:
        """
        Process texts for BiGemma3.

        Args:
            texts: List of input texts.

        Returns:
            Union[BatchFeature, BatchEncoding]: Processed texts.
        """
        # Format each text using chat template
        formatted_texts = []
        for text in texts:
            # Create message in chat format
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Query: {text}"},
                    ],
                }
            ]

            # Apply chat template to get formatted text
            formatted_text = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
            formatted_texts.append(formatted_text)

        return self(
            text=formatted_texts,
            return_tensors="pt",
            padding="longest",
        )

    def get_n_patches(
        self,
        image_size: Tuple[int, int],  # noqa: ARG002
        patch_size: int,  # noqa: ARG002
    ) -> Tuple[int, int]:
        """
        Get the number of patches (n_patches_x, n_patches_y) for dense embedding.

        For dense models like BiGemma, the entire image is embedded as a single vector,
        so we return (1, 1) to indicate a single "patch" representing the whole image.
        """
        return (1, 1)

    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,  # noqa: ARG002
    ) -> torch.Tensor:
        """
        Compute the cosine similarity for the given query and passage embeddings.
        """
        return self.score_single_vector(qs, ps, device=device)
