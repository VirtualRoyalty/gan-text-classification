from base import *


class Generator(BaseModel):
    """Generator model class"""

    def __init__(self, noise_size: int = 100,
                 output_size: int = 512,
                 hidden_size: int = 512,
                 hidden_layers: int = 1,
                 dropout_rate: float = 0.1):
        super(Generator, self).__init__()
        # define model layers
        layers = []
        hidden_sizes = [noise_size] + [hidden_size] * hidden_layers
        for i in range(len(hidden_sizes) - 1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                           nn.LeakyReLU(0.2, inplace=True),
                           nn.Dropout(dropout_rate),
                           nn.BatchNorm1d(hidden_sizes[i])])
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, noise: torch.Tensor):
        output_rep = self.layers(noise)
        return output_rep
