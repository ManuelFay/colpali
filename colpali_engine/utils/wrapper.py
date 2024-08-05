import importlib

from colpali_engine.models.clip_baselines import ColSigLIP, SigLIP
from colpali_engine.models.colbert_architectures import (BiBERT, BiXLMRoBERTa,
                                                         ColBERT, ColCamembert,
                                                         ColLlama,
                                                         ColXLMRoBERTa)
from colpali_engine.models.idefics_colbert_architecture import (BiIdefics,
                                                                ColIdefics)
from colpali_engine.models.paligemma_colbert_architecture import (BiNewSiglip,
                                                                  BiPaliLast,
                                                                  BiPaliMean,
                                                                  ColNewSiglip,
                                                                  ColPali)

if importlib.util.find_spec("transformers") is not None:
    from transformers import AutoProcessor, AutoTokenizer
    from transformers.tokenization_utils import PreTrainedTokenizer

    class AutoProcessorWrapper:
        def __new__(cls, *args, **kwargs):
            return AutoProcessor.from_pretrained(*args, **kwargs)

    class AutoTokenizerWrapper(PreTrainedTokenizer):
        def __new__(cls, *args, **kwargs):
            return AutoTokenizer.from_pretrained(*args, **kwargs)

    class AutoColModelWrapper:
        def __new__(cls, *args, **kwargs):
            pretrained_model_name_or_path = None
            if args:
                pretrained_model_name_or_path = args[0]
            elif kwargs:
                pretrained_model_name_or_path = kwargs["pretrained_model_name_or_path"]

            training_objective = kwargs.pop("training_objective", "colbertv1")

            if "camembert" in pretrained_model_name_or_path:
                return ColCamembert.from_pretrained(*args, **kwargs)
            elif "xlm-roberta" in pretrained_model_name_or_path:
                if training_objective == "biencoder":
                    return BiXLMRoBERTa.from_pretrained(*args, **kwargs)
                return ColXLMRoBERTa.from_pretrained(*args, **kwargs)
            elif (
                "llama" in pretrained_model_name_or_path.lower() or "croissant" in pretrained_model_name_or_path.lower()
            ):
                return ColLlama.from_pretrained(*args, **kwargs)
            elif "idefics2" in pretrained_model_name_or_path:
                if training_objective == "biencoder":
                    return BiIdefics.from_pretrained(*args, **kwargs)
                return ColIdefics.from_pretrained(*args, **kwargs)
            elif "siglip" in pretrained_model_name_or_path:
                if training_objective == "biencoder_mean":
                    return SigLIP.from_pretrained(*args, **kwargs)
                elif training_objective == "colbertv1":
                    return ColSigLIP.from_pretrained(*args, **kwargs)
                else:
                    raise ValueError(f"Training objective {training_objective} not recognized")
            elif "paligemma" in pretrained_model_name_or_path:
                if training_objective == "biencoder_mean":
                    return BiPaliMean.from_pretrained(*args, **kwargs)
                elif training_objective == "biencoder_last":
                    return BiPaliLast.from_pretrained(*args, **kwargs)
                elif training_objective == "biencoder_mean_vision":
                    return BiNewSiglip.from_pretrained(*args, **kwargs)
                elif training_objective == "colbertv1_vision":
                    return ColNewSiglip.from_pretrained(*args, **kwargs)
                elif training_objective == "colbertv1":
                    return ColPali.from_pretrained(*args, **kwargs)
                else:
                    raise ValueError(f"Training objective {training_objective} not recognized")
            else:
                if training_objective == "biencoder":
                    return BiBERT.from_pretrained(*args, **kwargs)
                return ColBERT.from_pretrained(*args, **kwargs)

else:
    raise ModuleNotFoundError("Transformers must be loaded")
