import torch

from base import *
from .generator import Generator


class Autoencoder(BaseModel):
    """Autoencoder model class for Generator pre-training"""

    def __init__(self, decoder: Generator, input_size: int = 728, **kwargs):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(True),
            nn.Dropout(0.2),
            nn.Linear(256, 100),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(100))
        self.decoder = decoder

    def forward(self, x, labels=None):
        x = self.encoder(x)
        if labels is None:
            x = self.decoder(x)
        else:
            x = self.decoder(x, labels)
        return x

    def get_encoder(self, x):
        x = self.encoder(x)
        return x
