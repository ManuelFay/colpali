from torch import nn
from transformers import (BertModel, BertPreTrainedModel, CamembertModel,
                          CamembertPreTrainedModel, LlamaModel,
                          LlamaPreTrainedModel, XLMRobertaModel,
                          XLMRobertaPreTrainedModel)


class ColCamembert(CamembertPreTrainedModel):
    def __init__(self, config):
        super(ColCamembert, self).__init__(config=config)
        self.roberta: CamembertPreTrainedModel = CamembertModel(config)
        self.dim = 128
        self.linear = nn.Linear(self.roberta.config.hidden_size, self.dim)
        self.main_input_name = "doc_input_ids"

    def forward(self, *args, **kwargs):
        """
        Forward pass through Camenbert and the linear layer for dimensionality reduction

        Args:
        - input_ids (torch.LongTensor): The input tokens tensor.
        - attention_mask (torch.LongTensor): The attention mask tensor.

        Returns:
        - torch.Tensor: Embeddings of shape (batch_size, num_tokens, dim)
        """
        outputs = self.roberta(*args, **kwargs)
        last_hidden_states = outputs[0]  # (batch_size, sequence_length, hidden_size)
        proj = self.linear(last_hidden_states)
        # normalize l2 norm
        proj = proj / proj.norm(dim=-1, keepdim=True)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)
        return proj


class ColXLMRoBERTa(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super(ColXLMRoBERTa, self).__init__(config=config)
        self.roberta: XLMRobertaPreTrainedModel = XLMRobertaModel(config)
        self.dim = 128
        self.linear = nn.Linear(self.roberta.config.hidden_size, self.dim)
        self.main_input_name = "doc_input_ids"

    def forward(self, *args, **kwargs):
        """
        Forward pass through Roberta and the linear layer for dimensionality reduction

        Args:
        - input_ids (torch.LongTensor): The input tokens tensor.
        - attention_mask (torch.LongTensor): The attention mask tensor.

        Returns:
        - torch.Tensor: Embeddings of shape (batch_size, num_tokens, dim)
        """
        outputs = self.roberta(*args, **kwargs)
        last_hidden_states = outputs[0]  # (batch_size, sequence_length, hidden_size)
        proj = self.linear(last_hidden_states)
        # normalize l2 norm
        proj = proj / proj.norm(dim=-1, keepdim=True)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)
        return proj


class BiXLMRoBERTa(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super(BiXLMRoBERTa, self).__init__(config=config)
        self.roberta: XLMRobertaPreTrainedModel = XLMRobertaModel(config)
        self.main_input_name = "doc_input_ids"

    def forward(self, *args, **kwargs):
        """
        Forward pass through Roberta and the linear layer for dimensionality reduction

        Args:
        - input_ids (torch.LongTensor): The input tokens tensor.
        - attention_mask (torch.LongTensor): The attention mask tensor.

        Returns:
        - torch.Tensor: Embeddings of shape (batch_size, num_tokens, dim)
        """
        outputs = self.roberta(*args, **kwargs)
        last_hidden_states = outputs[0]  # (batch_size, sequence_length, hidden_size)
        # pooling - mean tokens that have attention mask == 1
        proj = last_hidden_states * kwargs["attention_mask"].unsqueeze(-1)
        proj = proj.sum(dim=1) / kwargs["attention_mask"].sum(dim=1, keepdim=True)
        # normalize l2 norm
        proj = proj / proj.norm(dim=-1, keepdim=True)
        return proj


class ColBERT(BertPreTrainedModel):
    def __init__(self, config):
        super(ColBERT, self).__init__(config=config)
        self.bert: BertModel = BertModel(config)
        self.dim = 128
        self.linear = nn.Linear(self.bert.config.hidden_size, self.dim)
        self.main_input_name = "doc_input_ids"

    def forward(self, *args, **kwargs):
        """
        Forward pass through BERT and the linear layer for dimensionality reduction

        Args:
        - input_ids (torch.LongTensor): The input tokens tensor.
        - attention_mask (torch.LongTensor): The attention mask tensor.

        Returns:
        - torch.Tensor: Embeddings of shape (batch_size, num_tokens, dim)
        """
        outputs = self.bert(*args, **kwargs)
        last_hidden_states = outputs[0]  # (batch_size, sequence_length, hidden_size)
        proj = self.linear(last_hidden_states)
        # normalize l2 norm
        proj = proj / proj.norm(dim=-1, keepdim=True)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)
        return proj


class BiBERT(BertPreTrainedModel):
    def __init__(self, config):
        super(BiBERT, self).__init__(config=config)
        self.bert: BertModel = BertModel(config)
        self.main_input_name = "doc_input_ids"

    def forward(self, *args, **kwargs):
        """
        Forward pass through BERT and the linear layer for dimensionality reduction

        Args:
        - input_ids (torch.LongTensor): The input tokens tensor.
        - attention_mask (torch.LongTensor): The attention mask tensor.

        Returns:
        - torch.Tensor: Embeddings of shape (batch_size, num_tokens, dim)
        """
        outputs = self.bert(*args, **kwargs)
        last_hidden_states = outputs[0]  # (batch_size, sequence_length, hidden_size)
        # pooling - mean tokens that have attention mask == 1
        proj = last_hidden_states * kwargs["attention_mask"].unsqueeze(-1)
        proj = proj.sum(dim=1) / kwargs["attention_mask"].sum(dim=1, keepdim=True)
        # normalize l2 norm
        proj = proj / proj.norm(dim=-1, keepdim=True)
        return proj


class ColLlama(LlamaPreTrainedModel):
    def __init__(self, config):
        super(ColLlama, self).__init__(config=config)
        self.model: LlamaModel = LlamaModel(config)
        self.dim = 128
        self.linear = nn.Linear(self.model.config.hidden_size, self.dim)
        self.main_input_name = "doc_input_ids"

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
        return proj
