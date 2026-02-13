from typing import Generator, cast
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from colpali_engine.models import ColPaliProcessor


@pytest.fixture(scope="module")
def model_name() -> str:
    return "vidore/colpali-v1.2"


@pytest.fixture(scope="module")
def processor_from_pretrained(model_name: str) -> Generator[ColPaliProcessor, None, None]:
    yield cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(model_name))


def test_load_processor_from_pretrained(processor_from_pretrained: ColPaliProcessor):
    assert isinstance(processor_from_pretrained, ColPaliProcessor)


def test_process_images(processor_from_pretrained: ColPaliProcessor):
    # Create a dummy image
    image_size = (16, 32)
    image = Image.new("RGB", image_size, color="black")
    images = [image]

    # Process the image
    batch_feature = processor_from_pretrained.process_images(images)

    # Assertions
    assert "pixel_values" in batch_feature
    assert batch_feature["pixel_values"].shape == torch.Size([1, 3, 448, 448])


def test_process_texts(processor_from_pretrained: ColPaliProcessor):
    queries = [
        "Is attention really all you need?",
        "Are Benjamin, Antoine, Merve, and Jo best friends?",
    ]

    # Process the queries
    batch_encoding = processor_from_pretrained.process_texts(queries)

    # Assertions
    assert "input_ids" in batch_encoding
    assert isinstance(batch_encoding["input_ids"], torch.Tensor)
    assert cast(torch.Tensor, batch_encoding["input_ids"]).shape[0] == len(queries)


def test_process_queries(processor_from_pretrained: ColPaliProcessor):
    queries = [
        "Is attention really all you need?",
        "Are Benjamin, Antoine, Merve, and Jo best friends?",
    ]

    # Process the queries
    batch_encoding = processor_from_pretrained.process_queries(queries)

    # Assertions
    assert "input_ids" in batch_encoding
    assert isinstance(batch_encoding["input_ids"], torch.Tensor)
    assert cast(torch.Tensor, batch_encoding["input_ids"]).shape[0] == len(queries)


def test_colpali_processor_has_modalities(processor_from_pretrained: ColPaliProcessor):
    """Regression test for Transformers v5 processor registration changes."""
    assert hasattr(processor_from_pretrained, "tokenizer")
    assert hasattr(processor_from_pretrained, "image_processor")


def test_load_processor_fallback_when_processor_bundle_is_incomplete():
    class DummyProcessor:
        pass

    with (
        patch(
            "transformers.PaliGemmaProcessor.from_pretrained",
            side_effect=ValueError("Image processor is missing an `image_seq_length` attribute."),
        ),
        patch(
            "colpali_engine.models.paligemma.colpali.processing_colpali.AutoImageProcessor.from_pretrained"
        ) as mock_image,
        patch(
            "colpali_engine.models.paligemma.colpali.processing_colpali.AutoTokenizer.from_pretrained"
        ) as mock_tokenizer,
        patch(
            "colpali_engine.models.paligemma.colpali.processing_colpali.ColPaliProcessor.__init__", return_value=None
        ) as mock_init,
    ):
        mock_image.return_value = DummyProcessor()
        mock_tokenizer.return_value = DummyProcessor()

        processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2", revision="main")

    assert isinstance(processor, ColPaliProcessor)
    mock_image.assert_called_once()
    mock_tokenizer.assert_called_once()
    mock_init.assert_called_once_with(image_processor=mock_image.return_value, tokenizer=mock_tokenizer.return_value)
