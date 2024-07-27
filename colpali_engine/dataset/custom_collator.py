from transformers import PreTrainedTokenizer, ProcessorMixin


class CustomCollator:
    def __init__(
        self,
        processor: ProcessorMixin = None,
        tokenizer: PreTrainedTokenizer = None,
        max_length: int = 2048,
        add_suffix: bool = False,
    ):
        self.processor = processor
        self.tokenizer = tokenizer
        self.image_token_id = None
        self.max_length = max_length
        self.suffix = ""
        if add_suffix:
            self.suffix = "\n" * 10

        if tokenizer is None and processor is None:
            raise ValueError("Either processor or tokenizer should be provided.")

        if self.processor is not None:
            if self.processor.__class__.__name__ != "SiglipProcessor":
                self.image_token_id = self.processor.tokenizer.additional_special_tokens_ids[
                    self.processor.tokenizer.additional_special_tokens.index("<image>")
                ]

            if self.tokenizer is not None:
                raise ValueError("Only one of processor or tokenizer should be provided.")

        if self.tokenizer and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.processor.__class__.__name__ == "PaliGemmaProcessor":
            if self.processor.tokenizer.padding_side != "right":
                print("Setting padding side to right")
                self.processor.tokenizer.padding_side = "right"

    def __call__(self, examples):
        if self.processor is None:
            return self.forward_text(examples)
        if self.processor.__class__.__name__ == "Idefics2Processor":
            return self.forward_vision_idefics(examples)
        if self.processor.__class__.__name__ == "PaliGemmaProcessor":
            return self.forward_vision_pali(examples)
        if self.processor.__class__.__name__ == "SiglipProcessor":
            return self.forward_vision_siglip(examples)
        raise ValueError("Processor not supported")

    def forward_text(self, examples):
        texts_doc = []
        texts_query = []
        for example in examples:
            text_query = example["query"] + self.suffix
            text_doc = example["doc"]

            texts_doc.append(text_doc.strip())
            texts_query.append(text_query.strip())

        batch_doc = self.tokenizer(
            texts_doc, max_length=self.max_length, padding="longest", truncation=True, return_tensors="pt"
        )
        batch_query = self.tokenizer(
            texts_query, max_length=self.max_length, padding="longest", truncation=True, return_tensors="pt"
        )

        # prefix each key with "doc_" or "query_" to avoid key conflicts
        batch_doc = {f"doc_{k}": v for k, v in batch_doc.items()}
        batch_query = {f"query_{k}": v for k, v in batch_query.items()}
        batch_doc.update(batch_query)

        return batch_doc

    def forward_vision_idefics(self, examples):
        texts_doc = []
        texts_query = []
        images = []
        for example in examples:
            image = example["image"]

            text_query = None
            if example["query"] is not None:
                query = example["query"]
                messages_query = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Question: {query}<end_of_utterance><end_of_utterance><end_of_utterance><end_of_utterance><end_of_utterance>",
                            },
                        ],
                    },
                ]
                text_query = self.processor.apply_chat_template(messages_query, add_generation_prompt=False).strip()

            messages_doc = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the image."},
                        {"type": "image"},
                    ],
                },
            ]

            text_doc = self.processor.apply_chat_template(messages_doc, add_generation_prompt=False)

            texts_doc.append(text_doc.strip())
            texts_query.append(text_query)
            images.append([image])

        batch_doc = self.processor(
            text=texts_doc, images=images, return_tensors="pt", padding="longest", max_length=self.max_length
        )

        batch_query = None
        if all([t is None for t in texts_query]):
            print("All queries are None. Returning None for all queries.")
        elif any([t is None for t in texts_query]):
            raise ValueError("Some queries are None. This collator does not support None queries yet.")
        else:
            batch_query = self.processor(
                text=texts_query, return_tensors="pt", padding="longest", max_length=self.max_length
            )

        # prefix each key with "doc_" or "query_" to avoid key conflicts
        batch_doc = {f"doc_{k}": v for k, v in batch_doc.items()}

        if batch_query is not None:
            batch_query = {f"query_{k}": v for k, v in batch_query.items()}
            batch_doc.update(batch_query)

        return batch_doc

    def forward_vision_pali(self, examples):
        texts_doc = []
        texts_query = []
        images = []
        for example in examples:

            if example["image"] is None:
                raise ValueError("Image is None - This collator does not support None images yet.")

            image = example["image"].convert("RGB")
            images.append(image)
            texts_doc.append("Describe the image.")

            if example["query"] is None:
                texts_query.append(None)
            else:
                query = example["query"]
                query = f"Question: {query}<unused0><unused0><unused0><unused0><unused0>"
                texts_query.append(query)

        batch_doc = self.processor(
            text=texts_doc,
            images=images,
            return_tensors="pt",
            padding="longest",
            max_length=self.max_length + self.processor.image_seq_length,
        )

        batch_query = None
        # check if some but not all queries are None
        if all([t is None for t in texts_query]):
            print("All queries are None. Returning None for all queries.")
        elif any([t is None for t in texts_query]):
            # if it's the first query that is not None but the rest are None, then it's hard negatives
            if texts_query[0] is not None and all([t is None for t in texts_query[1:]]):
                batch_query = self.processor(
                    images=images[:1],
                    # NOTE: the image is not used in batch_query but it is required for calling the processor
                    text=texts_query[:1],
                    return_tensors="pt",
                    padding="longest",
                    max_length=self.max_length + self.processor.image_seq_length,
                )
            else:
                raise ValueError("Some queries are None. This collator does not support None queries yet.")
        else:
            batch_query = self.processor(
                images=images,  # NOTE: the image is not used in batch_query but it is required for calling the processor
                text=texts_query,
                return_tensors="pt",
                padding="longest",
                max_length=self.max_length + self.processor.image_seq_length,
            )
            del batch_query["pixel_values"]
            batch_query["input_ids"] = batch_query["input_ids"][..., self.processor.image_seq_length :]
            batch_query["attention_mask"] = batch_query["attention_mask"][..., self.processor.image_seq_length :]

        # prefix each key with "doc_" or "query_" to avoid key conflicts
        batch_doc = {f"doc_{k}": v for k, v in batch_doc.items()}

        if batch_query is not None:
            batch_query = {f"query_{k}": v for k, v in batch_query.items()}
            batch_doc.update(batch_query)

        return batch_doc

    def forward_vision_siglip(self, examples):
        texts_doc = []
        texts_query = []
        images = []
        for example in examples:

            if example["image"] is None:
                raise ValueError("Image is None - This collator does not support None images yet.")

            image = example["image"].convert("RGB")
            images.append(image)
            texts_doc.append("Describe the image.")

            if example["query"] is None:
                texts_query.append(None)
            else:
                query = f"Question: {example['query']}"
                texts_query.append(query)

        batch_doc = self.processor(
            text=texts_doc,
            images=images,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        batch_query = None
        # check if some but not all queries are None
        if all([t is None for t in texts_query]):
            # print("All queries are None.")
            pass
        elif any([t is None for t in texts_query]):
            raise ValueError("Some queries are None. This collator does not support None queries yet.")
        else:
            batch_query = self.processor(
                images=images,
                text=texts_query,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            del batch_query["pixel_values"]

        # prefix each key with "doc_" or "query_" to avoid key conflicts
        batch_doc = {f"doc_{k}": v for k, v in batch_doc.items()}

        if batch_query is not None:
            batch_query = {f"query_{k}": v for k, v in batch_query.items()}
            batch_doc.update(batch_query)
            # add attention mask for queries
            batch_doc["query_attention_mask"] = batch_doc["query_input_ids"].ne(0).long()

        # add attention mask for docs
        batch_doc["doc_attention_mask"] = batch_doc["doc_input_ids"].ne(0).long()

        return batch_doc
