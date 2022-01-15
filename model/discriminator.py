from base import *


class Discriminator(BaseModel):
    def __init__(self, backbone, input_size: int = 512, hidden_size: int = 512, hidden_layers: int = 1,
                 num_labels: int = 2, dropout_rate: float = 0.1):
        super(Discriminator, self).__init__()
        self.backbone = backbone
        self.input_dropout = nn.Dropout(p=dropout_rate)
        layers = []
        hidden_sizes = [input_size] + [hidden_size] * hidden_layers
        for i in range(len(hidden_sizes) - 1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                           nn.LeakyReLU(0.2, inplace=True),
                           nn.Dropout(dropout_rate)])
        self.layers = nn.Sequential(*layers)
        self.logit = nn.Linear(hidden_sizes[-1], num_labels + 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, input_mask, external_states=None):
        model_output = self.backbone(input_ids, attention_mask=input_mask)
        # hidden_states = model_output[-1]
        hidden_states = model_output.last_hidden_state[:, 0, :]
        if external_states is not None:
            hidden_states = torch.cat([hidden_states, external_states], dim=0)
        hidden_states = self.input_dropout(hidden_states)
        last_states = self.layers(hidden_states)
        logits = self.logit(last_states)
        probs = self.softmax(logits)
        return last_states, logits, probs

    # def __train__(self):
    #     self.
    #     super().train()

# class MnistModel(BaseModel):
#     def __init__(self, num_classes=10):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, num_classes)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)
