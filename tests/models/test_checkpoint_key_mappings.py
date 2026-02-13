import re

from transformers.conversion_mapping import get_checkpoint_conversion_mapping

from colpali_engine.models.gemma3.colgemma3.modeling_colgemma import ColGemma3
from colpali_engine.models.modernvbert.colvbert.modeling_colmodernvbert import ColModernVBert
from colpali_engine.models.paligemma.bipali.modeling_bipali import BiPali
from colpali_engine.models.paligemma.colpali.modeling_colpali import ColPali
from colpali_engine.models.qwen2.colqwen2.modeling_colqwen2 import ColQwen2
from colpali_engine.models.qwen2_5.colqwen2_5.modeling_colqwen2_5 import ColQwen2_5
from colpali_engine.models.qwen3.colqwen3.modeling_colqwen3 import ColQwen3
from colpali_engine.models.qwen_omni.colqwen_omni.modeling_colqwen_omni import ColQwen2_5Omni


def _apply_mapping(key: str, mapping: dict[str, str]) -> str:
    remapped = key
    for pattern, replacement in mapping.items():
        remapped = re.sub(pattern, replacement, remapped)
    return remapped


def test_qwen2_adapter_key_mapping_remaps_custom_text_proj_and_layers():
    assert (
        _apply_mapping(
            "base_model.model.custom_text_proj.lora_A.default.weight",
            ColQwen2._checkpoint_conversion_mapping,
        )
        == "custom_text_proj.lora_A.default.weight"
    )
    assert (
        _apply_mapping(
            "model.layers.17.self_attn.v_proj.lora_B.default.weight",
            ColQwen2._checkpoint_conversion_mapping,
        )
        == "language_model.layers.17.self_attn.v_proj.lora_B.default.weight"
    )


def test_qwen2_5_adapter_key_mapping_remaps_custom_text_proj_and_layers():
    assert (
        _apply_mapping(
            "base_model.model.custom_text_proj.lora_B.default.weight",
            ColQwen2_5._checkpoint_conversion_mapping,
        )
        == "custom_text_proj.lora_B.default.weight"
    )
    assert (
        _apply_mapping(
            "model.layers.3.mlp.down_proj.lora_A.default.weight",
            ColQwen2_5._checkpoint_conversion_mapping,
        )
        == "language_model.layers.3.mlp.down_proj.lora_A.default.weight"
    )


def test_colpali_adapter_key_mapping_remaps_custom_text_proj():
    assert (
        _apply_mapping(
            "base_model.model.custom_text_proj.lora_A.default.weight",
            ColPali._checkpoint_conversion_mapping,
        )
        == "custom_text_proj.lora_A.default.weight"
    )


def test_colgemma3_adapter_key_mapping_remaps_custom_text_proj():
    assert (
        _apply_mapping(
            "base_model.model.custom_text_proj.lora_A.default.weight",
            ColGemma3._checkpoint_conversion_mapping,
        )
        == "custom_text_proj.lora_A.default.weight"
    )


def test_pali_wrappers_ignore_expected_missing_lm_head_weight():
    assert r"model\.lm_head\.weight" in ColPali._keys_to_ignore_on_load_missing
    assert r"model\.lm_head\.weight" in BiPali._keys_to_ignore_on_load_missing


def test_colmodernvbert_adapter_key_mapping_remaps_custom_text_proj():
    assert (
        _apply_mapping(
            "base_model.model.custom_text_proj.lora_A.default.weight",
            ColModernVBert._checkpoint_conversion_mapping,
        )
        == "custom_text_proj.lora_A.default.weight"
    )


def test_colmodernvbert_conversion_mapping_is_registered_for_adapter_loading():
    mapping = get_checkpoint_conversion_mapping("modernvbert")
    assert mapping is not None

    key = "base_model.model.custom_text_proj.lora_B.default.weight"
    for renaming in mapping:
        if not hasattr(renaming, "source_patterns") or not hasattr(renaming, "target_patterns"):
            continue
        for pattern, replacement in zip(renaming.source_patterns, renaming.target_patterns):
            key = re.sub(pattern, replacement, key)

    assert key == "custom_text_proj.lora_B.default.weight"


def test_colqwen2_5_omni_adapter_key_mapping_remaps_custom_text_proj():
    assert (
        _apply_mapping(
            "base_model.model.custom_text_proj.lora_A.default.weight",
            ColQwen2_5Omni._checkpoint_conversion_mapping,
        )
        == "custom_text_proj.lora_A.default.weight"
    )


def test_colqwen2_5_omni_conversion_mapping_is_registered_for_adapter_loading():
    mapping = get_checkpoint_conversion_mapping("qwen2_5_omni_thinker")
    assert mapping is not None

    key = "base_model.model.custom_text_proj.lora_B.default.weight"
    for renaming in mapping:
        if not hasattr(renaming, "source_patterns") or not hasattr(renaming, "target_patterns"):
            continue
        for pattern, replacement in zip(renaming.source_patterns, renaming.target_patterns):
            key = re.sub(pattern, replacement, key)

    assert key == "custom_text_proj.lora_B.default.weight"


def test_colqwen3_adapter_key_mapping_remaps_custom_text_proj_and_layers():
    assert (
        _apply_mapping(
            "base_model.model.custom_text_proj.lora_A.default.weight",
            ColQwen3._checkpoint_conversion_mapping,
        )
        == "custom_text_proj.lora_A.default.weight"
    )


def test_colqwen3_conversion_mapping_is_registered_for_adapter_loading():
    mapping = get_checkpoint_conversion_mapping("qwen3_vl")
    assert mapping is not None

    key = "base_model.model.custom_text_proj.lora_B.default.weight"
    for renaming in mapping:
        if not hasattr(renaming, "source_patterns") or not hasattr(renaming, "target_patterns"):
            continue
        for pattern, replacement in zip(renaming.source_patterns, renaming.target_patterns):
            key = re.sub(pattern, replacement, key)

    assert key == "custom_text_proj.lora_B.default.weight"
