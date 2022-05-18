import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel
from typing import Dict, Tuple, List

from model import Autoencoder


class AETrainer:
    def __init__(self, config: Dict,
                 autoencoder: Autoencoder,
                 dataloader: torch.utils.data.DataLoader,
                 device: torch.device = None,
                 *args, **kwargs):
        self.config = config
        self.autoencoder = autoencoder
        self.dataloader = dataloader
        self.training_stats = []
        self.device = device
        pass

    def train_epoch(self, feature_extractor: AutoModel):
        self.config['pretrained_generator'] = True
        self.autoencoder.train()
        criterion = nn.MSELoss()
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)

        feature_extractor.eval()
        for epoch in range(self.config['pretrain_gen_num_epochs']):
            for batch in self.data_loader:
                input_ids = batch[0].to(self.device)
                input_mask = batch[1].to(self.device)
                trf_output = feature_extractor(input_ids, attention_mask=input_mask)
                hidden_states = trf_output.last_hidden_state[:, 0, :]
                if self.config['conditional_generator']:
                    labels = batch[2].to(self.device)
                    output = self.autoencoder(hidden_states, labels)
                else:
                    output = self.autoencoder(hidden_states)
                loss = criterion(output, hidden_states)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('epoch [{}/{}], loss: {:.4f}'.format(epoch + 1, self.config['pretrain_gen_num_epochs'], loss.item()))
        return

    @torch.no_grad()
    def validation(self):
        self.autoencoder.eval()
        return
