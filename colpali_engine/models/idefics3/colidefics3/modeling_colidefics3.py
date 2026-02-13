from torch import nn
from transformers import Idefics3Model, Idefics3PreTrainedModel


class ColIdefics3(Idefics3PreTrainedModel):
    """
    Initializes the ColIdefics3 model.

    Args:
        config : The model configuration.
        mask_non_image_embeddings (Optional[bool]): Whether to ignore all tokens embeddings
        except those of the image at inference.
        Defaults to False --> Do not mask any embeddings during forward pass.
    """

    def __init__(self, config, mask_non_image_embeddings: bool = False):
        super(ColIdefics3, self).__init__(config=config)
        self.model: Idefics3Model = Idefics3Model(config)
        self.dim = 128
        self.linear = nn.Linear(self.model.config.text_config.hidden_size, self.dim)
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.main_input_name = "doc_input_ids"
        self.post_init()

    def forward(self, *args, **kwargs):
        """
        Forward pass through Llama and the linear layer for dimensionality reduction

        Args:
        - input_ids (torch.LongTensor): The input tokens tensor.
        - attention_mask (torch.LongTensor): The attention mask tensor.

        Returns:
        - torch.Tensor: Embeddings of shape (batch_size, num_tokens, dim)
        """
        outputs = self.model(*args, **kwargs)
        last_hidden_states = outputs[0]  # (batch_size, sequence_length, hidden_size)
        proj = self.linear(last_hidden_states)
        # normalize l2 norm
        proj = proj / proj.norm(dim=-1, keepdim=True)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)

        if "pixel_values" in kwargs and self.mask_non_image_embeddings:
            # Pools only the image embeddings
            image_mask = (kwargs["input_ids"] == self.config.image_token_id).unsqueeze(-1)
            proj = proj * image_mask
        return proj
