# ColPali: Efficient Document Retrieval with Vision Language Models


[[Blog]](https://huggingface.co/blog/manu/colpali)
[[Paper]](https://arxiv.org/abs/2407.01449)
[[ColPali Model card]](https://huggingface.co/vidore/colpali)
[[ViDoRe Benchmark]](https://huggingface.co/vidore)
<!---[[Colab example]]()-->
[[HuggingFace Demo]](https://huggingface.co/spaces/manu/ColPali-demo)


## Associated Paper

**ColPali: Efficient Document Retrieval with Vision Language Models**
Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, Pierre Colombo

This repository contains the code for training custom Colbert retriever models.
Notably, we train colbert with LLMs (decoders) as well as Image Language models !

## Installation

### From git
```bash
pip install git+https://github.com/illuin-tech/colpali
```

### From source
```bash
git clone https://github.com/illuin-tech/colpali
cd colpali
pip install -r requirements.txt
```

## Usage

Example usage of the model is shown in the `scripts` directory.

```bash
# hackable example script to adapt
python scripts/infer/run_inference_with_python.py
```


```python
import torch
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor
from PIL import Image

from colpali_engine.models.paligemma_colbert_architecture import ColPali
from colpali_engine.trainer.retrieval_evaluator import CustomEvaluator
from colpali_engine.utils.colpali_processing_utils import process_images, process_queries
from colpali_engine.utils.image_from_page_utils import load_from_dataset


def main() -> None:
    """Example script to run inference with ColPali"""
    # Load model
    model_name = "vidore/colpali"
    model = ColPali.from_pretrained("google/paligemma-3b-mix-448", torch_dtype=torch.bfloat16, device_map="cuda").eval()
    model.load_adapter(model_name)
    processor = AutoProcessor.from_pretrained(model_name)

    # select images -> load_from_pdf(<pdf_path>),  load_from_image_urls(["<url_1>"]), load_from_dataset(<path>)
    images = load_from_dataset("vidore/docvqa_test_subsampled")
    queries = ["From which university does James V. Fiorca come ?", "Who is the japanese prime minister?"]

    # run inference - docs
    dataloader = DataLoader(
        images,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: process_images(processor, x),
    )
    ds = []
    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

    # run inference - queries
    dataloader = DataLoader(
        queries,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: process_queries(processor, x, Image.new("RGB", (448, 448), (255, 255, 255))),
    )

    qs = []
    for batch_query in dataloader:
        with torch.no_grad():
            batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
            embeddings_query = model(**batch_query)
        qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

    # run evaluation
    retriever_evaluator = CustomEvaluator(is_multi_vector=True)
    scores = retriever_evaluator.evaluate(qs, ds)
    print(scores.argmax(axis=1))


if __name__ == "__main__":
    typer.run(main)
```

Detais are also given in the model card for the base Colpali model on HuggingFace: [ColPali Model card](https://huggingface.co/vidore/colpali).

## Training

```bash
USE_LOCAL_DATASET=0 python scripts/train/train_colbert.py scripts/configs/siglip/train_siglip_model_debug.yaml
```

or 

```bash
accelerate launch scripts/train/train_colbert.py scripts/configs/train_colidefics_model.yaml
```

### Configurations
All training arguments can be set through a configuration file.
The configuration file is a yaml file that contains all the arguments for training.

The construction is as follows:

```python
@dataclass
class ColModelTrainingConfig:
    model: PreTrainedModel
    tr_args: TrainingArguments = None
    output_dir: str = None
    max_length: int = 256
    run_eval: bool = True
    run_train: bool = True
    peft_config: Optional[LoraConfig] = None
    add_suffix: bool = False
    processor: Idefics2Processor = None
    tokenizer: PreTrainedTokenizer = None
    loss_func: Optional[Callable] = ColbertLoss()
    dataset_loading_func: Optional[Callable] = None
    eval_dataset_loader: Optional[Dict[str, Callable]] = None
    pretrained_peft_model_name_or_path: Optional[str] = None
```
### Example

An example configuration file is:

```yaml
config:
  (): colpali_engine.utils.train_colpali_engine_models.ColModelTrainingConfig
  output_dir: !path ../../../models/without_tabfquad/train_colpali-3b-mix-448
  processor:
    () : colpali_engine.utils.wrapper.AutoProcessorWrapper
    pretrained_model_name_or_path: "./models/paligemma-3b-mix-448"
    max_length: 50
  model:
    (): colpali_engine.utils.wrapper.AutoColModelWrapper
    pretrained_model_name_or_path: "./models/paligemma-3b-mix-448"
    training_objective: "colbertv1"
    # attn_implementation: "eager"
    torch_dtype:  !ext torch.bfloat16
#    device_map: "auto"
#    quantization_config:
#      (): transformers.BitsAndBytesConfig
#      load_in_4bit: true
#      bnb_4bit_quant_type: "nf4"
#      bnb_4bit_compute_dtype:  "bfloat16"
#      bnb_4bit_use_double_quant: true

  dataset_loading_func: !ext colpali_engine.utils.dataset_transformation.load_train_set
  eval_dataset_loader: !import ../data/test_data.yaml

  max_length: 50
  run_eval: true
  add_suffix: true
  loss_func:
    (): colpali_engine.loss.colbert_loss.ColbertPairwiseCELoss
  tr_args: !import ../tr_args/default_tr_args.yaml
  peft_config:
    (): peft.LoraConfig
    r: 32
    lora_alpha: 32
    lora_dropout: 0.1
    init_lora_weights: "gaussian"
    bias: "none"
    task_type: "FEATURE_EXTRACTION"
    target_modules: '(.*(language_model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)'
    # target_modules: '(.*(language_model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)'
```


#### Local training

```bash
USE_LOCAL_DATASET=0 python scripts/train/train_colbert.py scripts/configs/siglip/train_siglip_model_debug.yaml
```


#### SLURM

```bash
sbatch --nodes=1 --cpus-per-task=16 --mem-per-cpu=32GB --time=20:00:00 --gres=gpu:1  -p gpua100 --job-name=colidefics --output=colidefics.out --error=colidefics.err --wrap="accelerate launch scripts/train/train_colbert.py  scripts/configs/train_colidefics_model.yaml"

sbatch --nodes=1  --time=5:00:00 -A cad15443 --gres=gpu:8  --constraint=MI250 --job-name=colpali --wrap="python scripts/train/train_colbert.py scripts/configs/train_colpali_model.yaml"
```

## CITATION

```bibtex
@misc{faysse2024colpaliefficientdocumentretrieval,
      title={ColPali: Efficient Document Retrieval with Vision Language Models}, 
      author={Manuel Faysse and Hugues Sibille and Tony Wu and Bilel Omrani and Gautier Viaud and Céline Hudelot and Pierre Colombo},
      year={2024},
      eprint={2407.01449},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2407.01449}, 
}
```
