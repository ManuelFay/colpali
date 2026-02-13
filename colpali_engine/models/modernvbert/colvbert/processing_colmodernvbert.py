import math
from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature, Idefics3Processor

from colpali_engine.utils.processing_utils import (
    BaseVisualRetrieverProcessor,
    Idefics3SplitImageInterpretabilityMixin,
)


class ColModernVBertProcessor(
    Idefics3SplitImageInterpretabilityMixin,
    BaseVisualRetrieverProcessor,
    Idefics3Processor,
):
    """
    Processor for ColModernVBert.
    """

    query_augmentation_token: ClassVar[str] = "<end_of_utterance>"
    image_token: ClassVar[str] = "<image>"
    visual_prompt_prefix: ClassVar[str] = (
        "<|begin_of_text|>User:<image>Describe the image.<end_of_utterance>\nAssistant:"
    )

    def __init__(
        self,
        image_processor,
        tokenizer=None,
        image_seq_len=64,
        chat_template=None,
        **kwargs,
    ):
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            image_seq_len=image_seq_len,
            chat_template=chat_template,
            **kwargs,
        )
        self.tokenizer.padding_side = "left"

    def process_images(
        self,
        images: List[Image.Image],
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process images for ColModernVBert.

        Args:
            images: List of PIL images.
        """
        images = [image.convert("RGB") for image in images]

        batch_doc = self(
            text=[self.visual_prompt_prefix] * len(images),
            images=images,
            padding="longest",
            return_tensors="pt",
        )
        return batch_doc

    def process_texts(self, texts: List[str]) -> Union[BatchFeature, BatchEncoding]:
        """
        Process texts for ColModernVBert.

        Args:
            texts: List of input texts.

        Returns:
            Union[BatchFeature, BatchEncoding]: Processed texts.
        """
        return self(
            text=texts,
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
        Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings.
        """
        return self.score_multi_vector(qs, ps, device=device, **kwargs)

    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        patch_size: Optional[int] = None,
        *args,
        **kwargs,
    ) -> Tuple[int, int]:
        """
        Get the number of patches (n_patches_x, n_patches_y) that will be used to process an image of
        size (height, width) with the given patch size.

        This method mirrors the Idefics3 image processing logic with image splitting:
        1. Resize the image so the longest edge equals the processor's longest_edge setting
        2. Split into sub-patches of max_image_size (512x512)
        3. Each sub-patch becomes image_seq_len tokens (8x8 grid)

        Note: The patch_size parameter is kept for API compatibility but is not used in the
        calculation. The actual patch dimensions are determined by the image splitting logic
        and image_seq_len.

        Args:
            image_size: Tuple of (height, width) in pixels.
            patch_size: The size of each square patch in pixels (unused, kept for API compatibility).

        Returns:
            Tuple of (n_patches_x, n_patches_y) representing the number of token patches
            along the width and height dimensions respectively (excluding global patch).
        """
        # Get the longest_edge from the image processor's size configuration
        longest_edge = self.image_processor.size.get("longest_edge", 2048)

        # Step 1: Calculate resized dimensions using the mixin helper method
        height_new, width_new = self._calculate_resized_dimensions(image_size, longest_edge)

        # Step 2: Calculate number of sub-patches (512x512 patches)
        # This mirrors the split_image logic from Idefics3ImageProcessor
        max_image_size = self.image_processor.max_image_size.get("longest_edge", 512)
        n_subpatches_x = math.ceil(width_new / max_image_size)
        n_subpatches_y = math.ceil(height_new / max_image_size)

        # Step 3: Calculate token grid dimensions
        # Each sub-patch becomes image_seq_len tokens (typically 64 = 8x8 grid)
        tokens_per_subpatch_side = int(math.sqrt(self.image_seq_len))
        n_patches_x = n_subpatches_x * tokens_per_subpatch_side
        n_patches_y = n_subpatches_y * tokens_per_subpatch_side

        return n_patches_x, n_patches_y
