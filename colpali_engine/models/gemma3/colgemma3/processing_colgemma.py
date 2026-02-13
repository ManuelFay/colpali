"""
ColGemma3 Processor - Fixed implementation following the ground truth raw implementation.

This module implements the processor for ColGemma3, handling the preprocessing of
images and text queries following the validated patterns from the ground truth.

Key features:
    - Image processing with chat template formatting
    - Text query processing with chat template formatting
    - MaxSim scoring for multi-vector embeddings (ColBERT-style)
    - Batch processing support
"""

from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature
from transformers.models.gemma3 import Gemma3Processor

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class ColGemmaProcessor3(BaseVisualRetrieverProcessor, Gemma3Processor):
    """
    Processor for ColGemma3 model.

    This processor handles image and text preprocessing for the ColGemma3 model,
    which returns multi-vector embeddings for efficient retrieval using MaxSim scoring.

    The processor uses Gemma3's chat template format for consistency with the base model.

    Based on the ground truth implementation from modal_scripts/colgemma/colgemma_raw.py.

    Args:
        *args: Variable length argument list to be passed to the parent `Gemma3Processor` class.
        max_num_visual_tokens (int, optional): The maximum number of visual tokens that can be processed.
        **kwargs: Arbitrary keyword arguments to be passed to the parent `Gemma3Processor` class.

    Example:
        >>> processor = ColGemmaProcessor3.from_pretrained("google/gemma-3-4b-it")
        >>> images = [Image.open("doc.png")]
        >>> batch_images = processor.process_images(images)
        >>> queries = ["What is this document about?"]
        >>> batch_queries = processor.process_queries(queries)
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
        # Set padding side to left (important for decoder-only models)
        self.tokenizer.padding_side = "left"

    @classmethod
    def from_pretrained(
        cls,
        *args,
        device_map: Optional[str] = None,
        **kwargs,
    ):
        """
        Load processor from pretrained model.

        Args:
            *args: Arguments for Gemma3Processor
            device_map: Device map for model placement
            **kwargs: Additional keyword arguments

        Returns:
            ColGemmaProcessor3: Initialized processor instance
        """
        instance = super().from_pretrained(
            *args,
            device_map=device_map,
            **kwargs,
        )

        # Configure max visual tokens if specified
        if "max_num_visual_tokens" in kwargs:
            # Gemma3 uses 56x56 patches
            instance.image_processor.max_pixels = kwargs["max_num_visual_tokens"] * 56 * 56
            instance.image_processor.size["longest_edge"] = instance.image_processor.max_pixels

        return instance

    def process_images(
        self,
        images: List[Image.Image],
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process images for ColGemma3 model.

        Each image is formatted using Gemma3's chat template with a descriptive prompt.
        This ensures the model generates contextual embeddings for the visual content.

        Args:
            images (List[Image.Image]): List of PIL images.

        Returns:
            Union[BatchFeature, BatchEncoding]: Processed images ready for model input.
                Contains input_ids, attention_mask, pixel_values, etc.
        """
        # Convert all images to RGB (handle RGBA, grayscale, etc.)
        images = [image.convert("RGB") for image in images]

        # Process each image using chat template
        batch_docs = []
        for image in images:
            # Create message in chat format
            # The prompt "Describe this image" encourages rich contextual embeddings
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

        # Handle single image case
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

    def process_queries(
        self,
        queries: List[str],
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process text queries for ColGemma3 model.

        Alias for process_texts for consistency with other processors.

        Args:
            queries (List[str]): List of input text queries.

        Returns:
            Union[BatchFeature, BatchEncoding]: Processed queries ready for model input.
        """
        return self.process_texts(queries)

    def process_texts(
        self,
        texts: List[str],
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process text queries for ColGemma3 model.

        Each text is formatted using Gemma3's chat template to match the format
        used during training/fine-tuning.

        Args:
            texts (List[str]): List of input text queries.

        Returns:
            Union[BatchFeature, BatchEncoding]: Processed texts ready for model input.
                Contains input_ids, attention_mask, etc.
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

        # Process all formatted texts
        return self(
            text=formatted_texts,
            return_tensors="pt",
            padding="longest",
        )

    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute similarity scores for the given query and document embeddings.

        This delegates to the multi-vector MaxSim scorer from BaseVisualRetrieverProcessor.

        Args:
            qs: Query embeddings (list of tensors or single tensor)
            ps: Document embeddings (list of tensors or single tensor)
            device: Device to perform computation on
            **kwargs: Additional arguments (unused, for compatibility)

        Returns:
            torch.Tensor: Similarity scores of shape (num_queries, num_docs)
        """
        return self.score_multi_vector(qs, ps, device=device, **kwargs)

    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        patch_size: int,
    ) -> Tuple[int, int]:
        """
        Get the number of patches (n_patches_x, n_patches_y) for an image.

        Args:
            image_size (Tuple[int, int]): Image size (width, height).
            patch_size (int): Size of each patch.

        Returns:
            Tuple[int, int]: Number of patches in x and y directions.
        """
        if hasattr(self.image_processor, "size"):
            if isinstance(self.image_processor.size, dict):
                width = self.image_processor.size.get("width", image_size[0])
                height = self.image_processor.size.get("height", image_size[1])
            else:
                width = height = self.image_processor.size

            n_patches_x = width // patch_size
            n_patches_y = height // patch_size
            return n_patches_x, n_patches_y

        # Fallback: calculate from provided image_size
        return image_size[0] // patch_size, image_size[1] // patch_size
