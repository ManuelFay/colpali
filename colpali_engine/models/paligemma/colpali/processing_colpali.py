from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer, BatchEncoding, BatchFeature, PaliGemmaProcessor

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class ColPaliProcessor(BaseVisualRetrieverProcessor, PaliGemmaProcessor):
    """
    Processor for ColPali.
    """

    visual_prompt_prefix: ClassVar[str] = "<image><bos>Describe the image."

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(image_processor=image_processor, tokenizer=tokenizer, chat_template=chat_template, **kwargs)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Load the processor from a checkpoint.

        Transformers v5 enforces that `PaliGemmaProcessor` is initialized with both
        an image processor and tokenizer. Some older ColPali checkpoints only expose
        the underlying `preprocessor_config.json` + tokenizer files, so we fallback to
        loading each component explicitly when needed.
        """
        try:
            instance = super().from_pretrained(*args, **kwargs)
            if getattr(instance, "image_processor", None) is None or getattr(instance, "tokenizer", None) is None:
                raise ValueError("Loaded processor is missing image_processor or tokenizer.")
            return instance
        except ValueError as exc:
            # Trigger fallback only for known processor-loading incompatibilities.
            msg = str(exc)
            if "image_seq_length" not in msg and "missing image_processor" not in msg:
                raise

        load_kwargs = {
            key: kwargs[key]
            for key in ("cache_dir", "force_download", "local_files_only", "revision", "token", "trust_remote_code")
            if key in kwargs
        }
        image_processor = AutoImageProcessor.from_pretrained(*args, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(*args, **load_kwargs)
        return cls(image_processor=image_processor, tokenizer=tokenizer)

    @property
    def query_augmentation_token(self) -> str:
        """
        Return the query augmentation token.
        Query augmentation buffers are used as reasoning buffers during inference.
        """
        return self.tokenizer.pad_token

    def process_images(
        self,
        images: List[Image.Image],
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process images for ColPali.

        Args:
            images: List of PIL images.
        """
        images = [image.convert("RGB") for image in images]

        batch_doc = self(
            text=[self.visual_prompt_prefix] * len(images),
            images=images,
            return_tensors="pt",
            padding="longest",
        )
        return batch_doc

    def process_texts(self, texts: List[str]) -> Union[BatchFeature, BatchEncoding]:
        """
        Process texts for ColPali.

        Args:
            texts: List of input texts.

        Returns:
            Union[BatchFeature, BatchEncoding]: Processed texts.
        """
        return self.tokenizer(
            [self.tokenizer.bos_token + text for text in texts],
            text_pair=None,
            return_token_type_ids=False,
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
        patch_size: int,
    ) -> Tuple[int, int]:
        n_patches_x = self.image_processor.size["width"] // patch_size
        n_patches_y = self.image_processor.size["height"] // patch_size

        return n_patches_x, n_patches_y

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        return batch_images.input_ids == self.image_token_id
