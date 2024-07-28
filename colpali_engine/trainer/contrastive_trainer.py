import torch
from transformers import Trainer


class ContrastiveTrainer(Trainer):
    def __init__(self, loss_func, is_vision_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = loss_func
        self.is_vision_model = is_vision_model

    def compute_loss(self, model, inputs, return_outputs=False):
        query_outputs = model(input_ids=inputs["query_input_ids"], attention_mask=inputs["query_attention_mask"])
        if self.is_vision_model:
            if "doc_pixel_attention_mask" not in inputs:
                doc_outputs = model(
                    input_ids=inputs["doc_input_ids"],
                    attention_mask=inputs["doc_attention_mask"],
                    pixel_values=inputs["doc_pixel_values"],
                )
            else:
                doc_outputs = model(
                    input_ids=inputs["doc_input_ids"],
                    attention_mask=inputs["doc_attention_mask"],
                    pixel_values=inputs["doc_pixel_values"],
                    pixel_attention_mask=inputs["doc_pixel_attention_mask"],
                )
        else:
            doc_outputs = model(input_ids=inputs["doc_input_ids"], attention_mask=inputs["doc_attention_mask"])

        loss = self.loss_func(query_outputs, doc_outputs)
        return (loss, (query_outputs, doc_outputs)) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=True):
        """This function is used to generate predictions and return the loss for the given inputs."""
        if not prediction_loss_only:
            raise ValueError("prediction_step is only called with prediction_loss_only=True")

        with torch.no_grad():
            if self.is_vision_model:
                if "doc_pixel_attention_mask" not in inputs:
                    doc_outputs = model(
                        input_ids=inputs["doc_input_ids"],
                        attention_mask=inputs["doc_attention_mask"],
                        pixel_values=inputs["doc_pixel_values"],
                    )
                else:
                    doc_outputs = model(
                        input_ids=inputs["doc_input_ids"],
                        attention_mask=inputs["doc_attention_mask"],
                        pixel_values=inputs["doc_pixel_values"],
                        pixel_attention_mask=inputs["doc_pixel_attention_mask"],
                    )
                query_outputs = model(
                    input_ids=inputs["query_input_ids"], attention_mask=inputs["query_attention_mask"]
                )
            else:

                query_outputs = model(
                    input_ids=inputs["query_input_ids"], attention_mask=inputs["query_attention_mask"]
                )
                doc_outputs = model(input_ids=inputs["doc_input_ids"], attention_mask=inputs["doc_attention_mask"])

            loss = self.loss_func(query_outputs, doc_outputs)
            return loss, None, None


class ContrastiveNegativeTrainer(Trainer):
    def __init__(self, loss_func, is_vision_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = loss_func
        self.is_vision_model = is_vision_model

    def compute_loss(self, model, inputs, return_outputs=False):
        query_outputs = model(input_ids=inputs["query_input_ids"], attention_mask=inputs["query_attention_mask"])

        # save matplotlib image visualization
        import matplotlib.pyplot as plt
        import random

        random_num = random.randint(0, 1000)
        arr = inputs["neg_doc_pixel_values"][0].to("cpu").numpy()
        plt.imsave(f"neg_doc_pixel_values_{random_num}.png", arr)
        arr = inputs["doc_pixel_values"][0].to("cpu").numpy()
        plt.imsave(f"doc_pixel_values_{random_num}.png", arr)


        if self.is_vision_model:
            if "doc_pixel_attention_mask" not in inputs:
                doc_outputs = model(
                    input_ids=inputs["doc_input_ids"],
                    attention_mask=inputs["doc_attention_mask"],
                    pixel_values=inputs["doc_pixel_values"],
                )
            else:
                doc_outputs = model(
                    input_ids=inputs["doc_input_ids"],
                    attention_mask=inputs["doc_attention_mask"],
                    pixel_values=inputs["doc_pixel_values"],
                    pixel_attention_mask=inputs["doc_pixel_attention_mask"],
                )

            if "neg_doc_pixel_attention_mask" not in inputs:
                neg_doc_outputs = model(
                    input_ids=inputs["neg_doc_input_ids"],
                    attention_mask=inputs["neg_doc_attention_mask"],
                    pixel_values=inputs["neg_doc_pixel_values"],
                )
            else:
                neg_doc_outputs = model(
                    input_ids=inputs["neg_doc_input_ids"],
                    attention_mask=inputs["neg_doc_attention_mask"],
                    pixel_values=inputs["neg_doc_pixel_values"],
                    pixel_attention_mask=inputs["neg_doc_pixel_attention_mask"],
                )
        else:
            raise NotImplementedError("Only vision models are supported for now")

        loss = self.loss_func(query_outputs, doc_outputs, neg_doc_outputs)
        return (loss, (query_outputs, doc_outputs, neg_doc_outputs)) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=True):
        """This function is used to generate predictions and return the loss for the given inputs."""
        if not prediction_loss_only:
            raise ValueError("prediction_step is only called with prediction_loss_only=True")

        with torch.no_grad():
            if self.is_vision_model:
                if "doc_pixel_attention_mask" not in inputs:
                    doc_outputs = model(
                        input_ids=inputs["doc_input_ids"],
                        attention_mask=inputs["doc_attention_mask"],
                        pixel_values=inputs["doc_pixel_values"],
                    )
                else:
                    doc_outputs = model(
                        input_ids=inputs["doc_input_ids"],
                        attention_mask=inputs["doc_attention_mask"],
                        pixel_values=inputs["doc_pixel_values"],
                        pixel_attention_mask=inputs["doc_pixel_attention_mask"],
                    )
                if "neg_doc_pixel_attention_mask" not in inputs:
                    neg_doc_outputs = model(
                        input_ids=inputs["neg_doc_input_ids"],
                        attention_mask=inputs["neg_doc_attention_mask"],
                        pixel_values=inputs["neg_doc_pixel_values"],
                    )
                else:
                    neg_doc_outputs = model(
                        input_ids=inputs["neg_doc_input_ids"],
                        attention_mask=inputs["neg_doc_attention_mask"],
                        pixel_values=inputs["neg_doc_pixel_values"],
                        pixel_attention_mask=inputs["neg_doc_pixel_attention_mask"],
                    )
                query_outputs = model(
                    input_ids=inputs["query_input_ids"], attention_mask=inputs["query_attention_mask"]
                )
            else:

                query_outputs = model(
                    input_ids=inputs["query_input_ids"], attention_mask=inputs["query_attention_mask"]
                )
                doc_outputs = model(input_ids=inputs["doc_input_ids"], attention_mask=inputs["doc_attention_mask"])

            loss = self.loss_func(query_outputs, doc_outputs, neg_doc_outputs)
            return loss, None, None
