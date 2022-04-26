import torch
import numpy as np
import torch.nn.functional as F
# import time
# from base import BaseTrainer
# from utils import inf_loop, MetricTracker
from transformers import AutoModel
from model import Generator, Discriminator
from typing import Dict, Tuple


class Trainer:
    def __init__(self, config: Dict, backbone: AutoModel,
                 discriminator: Discriminator,
                 train_dataloader, valid_dataloader,
                 discriminator_optimizer, scheduler_d=None,
                 device=None):

        self.config = config
        self.backbone = backbone
        self.discriminator = discriminator
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.discriminator_optimizer = discriminator_optimizer
        self.scheduler_d = scheduler_d
        self.device = device
        self.training_stats = []
        pass

    def train_epoch(self, log_env=None) -> float:
        tr_d_loss = 0
        self.backbone.train()
        self.discriminator.train()

        for step, batch in enumerate(self.train_dataloader):
            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            b_label_mask = batch[3].to(self.device)

            # Generate the output of the Discriminator for real and fake data.
            hidden_states, logits, probs = self.discriminator(input_ids=b_input_ids,
                                                              input_mask=b_input_mask)
            # Disciminator's loss estimation
            log_probs = F.log_softmax(logits, dim=-1)
            label2one_hot = torch.nn.functional.one_hot(b_labels, self.config['num_labels'])
            per_example_loss = -torch.sum(label2one_hot * log_probs, dim=-1)
            per_example_loss = torch.masked_select(per_example_loss, b_label_mask.to(self.device))
            labeled_example_count = per_example_loss.type(torch.float32).numel()

            discriminator_loss = torch.div(torch.sum(per_example_loss.to(self.device)), labeled_example_count)
            if log_env:
                log_env['train/discriminator_loss'].log(discriminator_loss.item())

            # Avoid gradient accumulation
            self.discriminator_optimizer.zero_grad()

            # Calculate weights updates
            discriminator_loss.backward()
            # Apply modifications
            self.discriminator_optimizer.step()
            # Save the losses to print them later
            tr_d_loss += discriminator_loss.item()
            # Update the learning rate with the scheduler
            if self.config['apply_scheduler']:
                self.scheduler_d.step()
        return tr_d_loss

    @torch.no_grad()
    def validation(self, tr_d_loss, epoch_i, vervose=True, *args, **kwargs):

        # Calculate the average loss over all of the batches.
        discriminator_loss = tr_d_loss / len(self.train_dataloader)
        print(f"\tAverage training loss discriminator: {discriminator_loss:.3f}")

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        self.backbone.eval()
        self.discriminator.eval()

        # Tracking variables
        total_test_accuracy = 0
        total_test_loss = 0
        nb_test_steps = 0
        all_preds = []
        all_labels_ids = []
        # loss
        nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

        # Evaluate data for one epoch
        for batch in self.valid_dataloader:
            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                # model_outputs = self.backbone(b_input_ids, attention_mask=b_input_mask)
                _, logits, probs = self.discriminator(input_ids=b_input_ids, input_mask=b_input_mask)
                # Accumulate the test loss.
                total_test_loss += nll_loss(logits, b_labels)

            # Accumulate the predictions and the input labels
            _, preds = torch.max(logits, 1)
            all_preds += preds.detach().cpu()
            all_labels_ids += b_labels.detach().cpu()

        # Report the final accuracy for this validation run.
        all_preds = torch.stack(all_preds).numpy()
        all_labels_ids = torch.stack(all_labels_ids).numpy()
        test_accuracy = np.sum(all_preds == all_labels_ids) / len(all_preds)
        print("  Accuracy: {0:.3f}".format(test_accuracy))

        # Calculate the average loss over all of the batches.
        avg_test_loss = total_test_loss / len(self.valid_dataloader)
        avg_test_loss = avg_test_loss.item()
        # Record all statistics from this epoch.
        info_dct = {
            'epoch': epoch_i + 1,
            'Training Loss discriminator': discriminator_loss,
            'discriminator_loss': avg_test_loss,
            'discriminator_accuracy': test_accuracy,
        }
        self.training_stats.append(info_dct)
        if verbose:
            print(f"\tAverage training loss discriminator: {discriminator_loss:.3f}")
            # print("  Training epcoh took: {:}".format(training_time))
            print("  Accuracy: {0:.3f}".format(test_accuracy))
        return info_dct
