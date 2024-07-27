# HuggingFace trainer
import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
from datasets import concatenate_datasets
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, Idefics2Processor, PreTrainedModel, PreTrainedTokenizer, TrainingArguments

from colpali_engine.dataset.custom_collator import CustomCollator
from colpali_engine.dataset.hard_neg_collator import HardNegCollator
from colpali_engine.loss.colbert_loss import BiEncoderLoss, BiPairwiseCELoss, ColbertLoss, ColbertPairwiseCELoss
from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer, ContrastiveNegativeTrainer
from colpali_engine.trainer.retrieval_evaluator import CustomEvaluator
from colpali_engine.utils.gpu_stats import print_gpu_utilization, print_summary


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

    def __post_init__(self):
        if self.output_dir is None:
            sanitized_name = str(self.model.name_or_path).replace("/", "_")
            self.output_dir = f"./models/{sanitized_name}"

        if self.tr_args is None:
            self.tr_args = TrainingArguments(output_dir=self.output_dir)
        elif self.tr_args.output_dir is None:
            self.tr_args.output_dir = self.output_dir

        # cast if string
        if isinstance(self.tr_args.learning_rate, str):
            self.tr_args.learning_rate = float(self.tr_args.learning_rate)
        self.tr_args.remove_unused_columns = False

        if self.processor is None and self.tokenizer is None:
            print("Using textual model tokenization")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model.name_or_path)

        if self.pretrained_peft_model_name_or_path is not None:
            self.model.load_adapter(self.pretrained_peft_model_name_or_path)
            print(f"Loaded pretrained adapter from {self.pretrained_peft_model_name_or_path}")

        if self.peft_config is not None:
            print("Configurating PEFT model")
            if self.processor is None:
                # Might be deprecated - use the "else" branch
                self.model = prepare_model_for_kbit_training(self.model)  # use_gradient_checkpointing=True
                # self.model.enable_input_require_grads()
                self.model = get_peft_model(self.model, self.peft_config)
                self.model.print_trainable_parameters()
            else:
                # Ugly debugging hack
                # if self.model.model.config.text_config.vocab_size == 32000:
                #     print("DEBUG: Resizing token embeddings - This should not happen in a real scenario!")
                #     self.model.model.text_model.resize_token_embeddings(32003)
                #     self.model.model.vision_model.encoder.layers = self.model.model.vision_model.encoder.layers[0:2]
                # self.model.enable_input_require_grads()
                if self.pretrained_peft_model_name_or_path is None:
                    self.model.add_adapter(self.peft_config)
                    self.model.enable_adapters()
                else:
                    print(f"Adapter already loaded from {self.pretrained_peft_model_name_or_path}. Not overwriting.")

    print_gpu_utilization()


class ColModelTraining:
    def __init__(self, config: ColModelTrainingConfig) -> None:
        self.config = config
        self.model = self.config.model
        self.dataset = self.config.dataset_loading_func()
        if isinstance(self.dataset, Tuple):
            neg_dataset = self.dataset[1]
            self.dataset = self.dataset[0]
            self.collator = HardNegCollator(
                processor=self.config.processor,
                tokenizer=self.config.tokenizer,
                max_length=self.config.max_length,
                image_dataset=neg_dataset
            )
        else:
            self.collator = CustomCollator(
                processor=self.config.processor,
                tokenizer=self.config.tokenizer,
                max_length=self.config.max_length
            )
        self.current_git_hash = os.popen("git rev-parse HEAD").read().strip()
        self.retriever_evaluator = CustomEvaluator(
            is_multi_vector=(
                isinstance(self.config.loss_func, ColbertLoss)
                or isinstance(self.config.loss_func, ColbertPairwiseCELoss)
            )
        )

    def train(self) -> None:

        if isinstance(self.collator, HardNegCollator):
            print("Training with hard negatives")
            trainer = ContrastiveNegativeTrainer(
                model=self.model,
                train_dataset=self.dataset["train"],
                eval_dataset=self.dataset["test"],
                args=self.config.tr_args,
                data_collator=self.collator,
                loss_func=self.config.loss_func,
                is_vision_model=self.config.processor is not None,
            )
        else:
            print("Training with in-batch negatives")
            trainer = ContrastiveTrainer(
                model=self.model,
                train_dataset=self.dataset["train"],
                eval_dataset=self.dataset["test"],
                args=self.config.tr_args,
                data_collator=self.collator,
                loss_func=self.config.loss_func,
                is_vision_model=self.config.processor is not None,
            )
        trainer.args.remove_unused_columns = False

        result = trainer.train()
        print_summary(result)

    def eval_dataset(self, test_dataset):

        self.model.eval()

        # # debug
        # if len(test_dataset) > 200:
        #     test_dataset = test_dataset.select(range(0, 100))

        idx_with_query = [idx for idx, sample in enumerate(test_dataset["query"]) if sample is not None]
        idx_without_query = [idx for idx, sample in enumerate(test_dataset["query"]) if sample is None]

        dataloader_with_query = DataLoader(
            test_dataset.select(idx_with_query),
            batch_size=self.config.tr_args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.collator,
        )
        dataloader_without_query = DataLoader(
            test_dataset.select(idx_without_query),
            batch_size=self.config.tr_args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.collator,
        )

        # dataset is ordered so that non-null queries come first
        test_dataset = concatenate_datasets(
            [test_dataset.select(idx_with_query), test_dataset.select(idx_without_query)]
        )

        relevant_docs = {}
        docidx_2_docid = {}
        qsidx_2_query = []
        for idx, sample in enumerate(test_dataset):
            doc_id = sample["image_filename"] if "image_filename" in sample else str(hash(sample["doc"]))
            # query_id = sample["query_id"] if "query_id" in sample else str(hash(sample["query"]))
            if sample["query"] is not None:
                relevant_docs[str(idx)] = {doc_id: 1}
                qsidx_2_query.append(str(idx))
            docidx_2_docid[str(idx)] = doc_id

        qs = []
        ps = []

        device = self.model.device
        with (torch.no_grad()):
            for dataloader in [dataloader_with_query, dataloader_without_query]:
                for batch in tqdm(dataloader):
                    if "doc_pixel_values" not in batch:
                        doc = self.model(
                            input_ids=batch["doc_input_ids"].to(device),
                            attention_mask=batch["doc_attention_mask"].to(device),
                        )

                    else:
                        if "doc_pixel_attention_mask" in batch:
                            doc = self.model(
                                input_ids=batch["doc_input_ids"].to(device),
                                attention_mask=batch["doc_attention_mask"].to(device),
                                pixel_values=batch["doc_pixel_values"].to(device),
                                pixel_attention_mask=batch["doc_pixel_attention_mask"].to(device),
                            )
                        else:
                            doc = self.model(
                                input_ids=batch["doc_input_ids"].to(device),
                                attention_mask=batch["doc_attention_mask"].to(device),
                                pixel_values=batch["doc_pixel_values"].to(device),
                            )

                    ps.extend(list(torch.unbind(doc.to("cpu"))))

                    if "query_input_ids" in batch:
                        query = self.model(
                            input_ids=batch["query_input_ids"].to(device),
                            attention_mask=batch["query_attention_mask"].to(device),
                        )
                        # variable len
                        qs.extend(list(torch.unbind(query.to("cpu"))))

        print("Embeddings computed, evaluating")
        scores = self.retriever_evaluator.evaluate(qs, ps)
        # scores is 2d array of shape (n_queries, n_docs)
        # turn it into a dict
        results = {}
        assert scores.shape[0] == len(qsidx_2_query)
        for idx, scores_per_query in enumerate(scores):
            results[qsidx_2_query[idx]] = {
                docidx_2_docid[str(docidx)]: float(score) for docidx, score in enumerate(scores_per_query)
            }

        # evaluate
        metrics = self.retriever_evaluator.compute_metrics(relevant_docs, results)
        print(metrics)
        return metrics

    def eval(self) -> None:

        print("Evaluating on validation set")
        metrics = self.eval_dataset(self.dataset["test"])
        print(f"Metrics for validation set: {metrics}")
        all_metrics = {"validation_set": metrics}

        if self.config.eval_dataset_loader is not None:
            for test_name, test_dataset_loading_func in self.config.eval_dataset_loader.items():
                print(f"Evaluating {test_name}")
                test_ds = test_dataset_loading_func()
                metrics = self.eval_dataset(test_ds)
                all_metrics[test_name] = metrics
                print(f"Metrics for {test_name}: {metrics}")

                # checkpoint dumps
                with open(f"{self.config.output_dir}/results.json", "w") as f:
                    json.dump(all_metrics, f)

        # save results as json
        with open(f"{self.config.output_dir}/results.json", "w") as f:
            json.dump(all_metrics, f)

    def save(self, config_file):
        # save model
        self.model.save_pretrained(self.config.output_dir)
        if self.config.tokenizer is not None:
            self.config.tokenizer.save_pretrained(self.config.output_dir)
        if self.config.processor is not None:
            self.config.processor.save_pretrained(self.config.output_dir)  # save config

        # copy-paste the yml file with os
        os.system(f"cp {config_file} {self.config.output_dir}/training_config.yml")

        # save git hash of the commit at beginning of training
        with open(f"{self.config.output_dir}/git_hash.txt", "w") as f:
            f.write(self.current_git_hash)
