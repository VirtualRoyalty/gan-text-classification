import torch
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report
from typing import Dict, Tuple
import torch.nn.functional as F
from transformers import AutoModel, get_constant_schedule_with_warmup

from model import Discriminator


class Trainer:
    def __init__(self, config: Dict,
                 discriminator: Discriminator,
                 train_dataloader: torch.utils.data.DataLoader,
                 valid_dataloader: torch.utils.data.DataLoader,
                 device=None):

        self.D = discriminator
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.device = device
        self.valid_nll = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.training_stats = []

        # define trainable parameters and optimizer
        if config['frozen_backbone']:
            self.D.freeze_backbone()
        d_vars = [p for p in self.D.parameters() if p.requires_grad]
        print(f'Trainable layers {len(d_vars)}')
        self.optimizer = torch.optim.AdamW(d_vars, lr=config['learning_rate_discriminator'])

        # define scheduler
        if config['apply_scheduler']:
            config['num_train_steps'] = config['num_train_examples'] / config['batch_size'] * config['num_train_epochs']
            config['num_train_steps'] = int(config['num_train_steps'])
            config['num_warmup_steps_d'] = int(config['num_train_steps'] * config['warmup_proportion_d'])
            self.scheduler = get_constant_schedule_with_warmup(self.optimizer,
                                                               num_warmup_steps=config['num_warmup_steps_d'])
        # log config
        self.config = config

    def train_epoch(self, log_env=None) -> float:
        tr_d_loss = 0
        self.D.train()

        for step, batch in enumerate(self.train_dataloader):
            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            b_label_mask = batch[3].to(self.device)

            # Get the output of the Discriminator
            hidden_states, logits, probs, _ = self.D(input_ids=b_input_ids,
                                                     input_mask=b_input_mask)
            # Discriminator loss estimation
            log_probs = F.log_softmax(logits, dim=-1)
            label2one_hot = torch.nn.functional.one_hot(b_labels, self.config['num_labels'])
            per_example_loss = -torch.sum(label2one_hot * log_probs, dim=-1)
            per_example_loss = torch.masked_select(per_example_loss, b_label_mask.to(self.device))
            labeled_example_count = per_example_loss.type(torch.float32).numel()

            D_loss = torch.div(torch.sum(per_example_loss.to(self.device)), labeled_example_count)
            if log_env:
                log_env['train/discriminator_loss'].log(D_loss.item())

            # Avoid gradient accumulation
            self.optimizer.zero_grad()
            # Calculate weights updates
            D_loss.backward()
            self.optimizer.step()
            # Save the losses to print them later
            tr_d_loss += D_loss.item()
            # Update the learning rate with the scheduler
            if self.config['apply_scheduler']:
                self.scheduler.step()
        return tr_d_loss

    @torch.no_grad()
    def validation(self, tr_d_loss, epoch_i, verbose=True, log_env=None, *args, **kwargs):

        # Calculate the average loss over all of the batches.
        discriminator_loss = tr_d_loss / len(self.train_dataloader)

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        self.D.eval()

        # Tracking variables
        total_test_loss = 0
        all_preds = []
        all_labels_ids = []

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
                _, logits, probs, _ = self.D(input_ids=b_input_ids, input_mask=b_input_mask)
                # Accumulate the test loss.
                total_test_loss += self.valid_nll(logits, b_labels)

            # Accumulate the predictions and the input labels
            _, preds = torch.max(logits, 1)
            all_preds += preds.detach().cpu()
            all_labels_ids += b_labels.detach().cpu()

        # Report the final accuracy for this validation run.
        all_preds = torch.stack(all_preds).numpy()
        all_labels_ids = torch.stack(all_labels_ids).numpy()
        valid_accuracy = np.sum(all_preds == all_labels_ids) / len(all_preds)
        f1_macro = f1_score(all_labels_ids, all_preds, average='macro')
        f1_micro = f1_score(all_labels_ids, all_preds, average='micro')

        # Calculate the average loss over all of the batches.
        avg_test_loss = total_test_loss / len(self.valid_dataloader)
        avg_test_loss = avg_test_loss.item()
        # Record all statistics from this epoch.
        info_dct = {
            'epoch': epoch_i + 1,
            'Training Loss discriminator': discriminator_loss,
            'discriminator_loss': avg_test_loss,
            'discriminator_accuracy': valid_accuracy,
        }

        if log_env:
            log_env['valid/discriminator_loss'].log(avg_test_loss)
            log_env['valid/discriminator_accuracy'].log(valid_accuracy)
            log_env['valid/f1_macro'].log(f1_macro)
            log_env['valid/f1_micro'].log(f1_micro)
            fig, _ = self.get_error_matrix(all_labels_ids, all_preds)
            log_env['valid/cmatrix'].log(fig)
            fig, _ = self.get_error_matrix(all_labels_ids, all_preds, normalize="true")
            log_env['valid/cmatrix_norm'].log(fig)
            fig, _ = self.get_error_matrix(all_labels_ids, all_preds, normalize="true", values_format=".2g")
            log_env['valid/cmatrix_norm_annot'].log(fig)

        self.training_stats.append(info_dct)
        if verbose:
            print(f"\tAverage training loss discriminator: {discriminator_loss:.3f}")
            print("  Accuracy: {0:.3f}".format(valid_accuracy))
        return info_dct

    def get_error_matrix(self, labels, pred, normalize=None, include_values=False, values_format=None):
        cm = confusion_matrix(labels, pred, normalize=normalize)
        cmf = ConfusionMatrixDisplay(cm)
        fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
        cmf.plot(ax=ax, cmap='Blues', include_values=include_values, values_format=values_format)
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, cm
