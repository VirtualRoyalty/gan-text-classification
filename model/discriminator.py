import base
from base import *
from transformers import AutoModel


class Discriminator(BaseModel):
    """Discriminator model class with transformer backbone"""

    def __init__(self, backbone: AutoModel,
                 input_size: int = 512,
                 hidden_size: int = 512,
                 hidden_layers: int = 1,
                 num_labels: int = 10,
                 dropout_rate: float = 0.1,
                 model_name: str = None,
                 **kwargs):
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
        self.to_logits = nn.Linear(hidden_sizes[-1], num_labels)
        self.softmax = nn.Softmax(dim=-1)
        self.model_name = model_name

    def forward(self, input_ids: torch.Tensor = None,
                input_mask: torch.Tensor = None,
                external_states: torch.Tensor = None):
        # simple check
        if input_ids is None and external_states is None:
            raise AssertionError('Empty input: input_ids and external states are empty')

        if input_ids is not None:
            # get last hidden state of transformer backbone
            trf_output = self.backbone(input_ids, attention_mask=input_mask)
            # get [CLS] token embedding as sentence embedding
            trf_states = trf_output.last_hidden_state[:, 0]
            # add generator input to hidden states
            if external_states is not None:
                hidden_states = torch.cat([trf_states, external_states], dim=0)
            else:
                hidden_states = trf_states
        else:
            hidden_states = external_states
            trf_states = external_states

        hidden_states = self.input_dropout(hidden_states)
        last_hidden_states = self.layers(hidden_states)
        logits = self.to_logits(last_hidden_states)
        probs = self.softmax(logits)
        return last_hidden_states, logits, probs, trf_states

    def predict(self, loader: torch.utils.data.DataLoader, device: torch.device, gan=True) -> tuple:
        predict = []
        ground_true = []
        self.eval()
        for inputs, masks, labels, label_mask in loader:
            inputs = inputs.to(device)
            masks = masks.to(device)
            _, logits, probs, _ = self(input_ids=inputs, input_mask=masks)
            if gan:
                logits = logits[:, :-1]
            result = np.argmax(logits.cpu().detach().numpy(), axis=1).tolist()
            predict.extend(result)
            ground_true.extend(labels.detach().numpy().tolist())
        return ground_true, predict

    def freeze_backbone(self) -> None:
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad = False