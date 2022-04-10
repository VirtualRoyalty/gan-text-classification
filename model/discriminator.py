from base import *
from transformers import AutoModel


class Discriminator(BaseModel):
    """Discriminator model class with transformer backbone"""

    def __init__(self, backbone: AutoModel,
                 input_size: int = 512,
                 hidden_size: int = 512,
                 hidden_layers: int = 1,
                 num_labels: int = 10,
                 dropout_rate: float = 0.1):
        super(Discriminator, self).__init__()
        # define model layers
        self.backbone = backbone
        self.input_dropout = nn.Dropout(p=dropout_rate)
        hidden_sizes = [input_size] + [hidden_size] * hidden_layers
        layers = []
        for i in range(len(hidden_sizes) - 1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                           nn.LeakyReLU(0.2, inplace=True),
                           nn.Dropout(dropout_rate)])
        self.layers = nn.Sequential(*layers)
        self.to_logits = nn.Linear(hidden_sizes[-1], num_labels + 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids: torch.Tensor = None,
                input_mask: torch.Tensor = None,
                external_states: torch.Tensor = None):
        # simple check
        if input_ids is None and external_states is None:
            raise AssertionError('Empty input: input_ids and external states are empty')

        # get last hidden state of transformer backbone
        trf_output = self.backbone(input_ids, attention_mask=input_mask)
        # get [CLS] token embedding as sentence embedding
        hidden_states = trf_output.last_hidden_state[:, 0, :]

        # add generator input to hidden states
        if external_states is not None:
            hidden_states = torch.cat([hidden_states, external_states], dim=0)
        hidden_states = self.input_dropout(hidden_states)
        last_hidden_states = self.layers(hidden_states)
        logits = self.to_logits(last_hidden_states)
        probs = self.softmax(logits)
        return last_hidden_states, logits, probs
