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
                 train_dataloader,valid_dataloader,
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

    def train_epoch(self) -> Tuple[float, float]:
        tr_g_loss = 0
        tr_d_loss = 0
        self.backbone.train()
        self.generator.train()
        self.discriminator.train()

        for step, batch in enumerate(self.train_dataloader):
            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            b_label_mask = batch[3].to(self.device)

            # Encode real data in the Transformer
            # model_outputs = self.backbone(b_input_ids, attention_mask=b_input_mask)
            # hidden_states = model_outputs[-1]

            # Generate fake data that should have the same distribution of the ones
            # noise = torch.zeros(b_input_ids.shape[0], self.config['noise_size'], device=self.device).uniform_(0, 1)
            # gen_rep = self.generator(noise)

            # Generate the output of the Discriminator for real and fake data.
            # disciminator_input = torch.cat([hidden_states, gen_rep], dim=0)
            # features, logits, probs = self.discriminator(disciminator_input)
            features, logits, probs = self.discriminator(input_ids=b_input_ids,
                                                         input_mask=b_input_mask,
                                                         # external_states=gen_rep
                                                         )

            # features_list = torch.split(features, self.config['batch_size'])
            D_real_features = features
            # D_fake_features = features_list[1]

            # logits_list = torch.split(logits, self.config['batch_size'])
            # D_real_logits, D_fake_logits = logits_list[0], logits_list[1]
            D_real_logits = logits

            # probs_list = torch.split(probs, self.config['batch_size'])
            # D_real_probs, D_fake_probs = probs_list[0], probs_list[1]
            D_real_probs = probs

            # Generator's LOSS estimation
            # g_loss_d = -1 * torch.mean(torch.log(1 - D_fake_probs[:, -1] + self.config['epsilon']))
            # g_feat_reg = torch.mean(
            #     torch.pow(torch.mean(D_real_features, dim=0) - torch.mean(D_fake_features, dim=0), 2))
            # g_loss = g_loss_d + g_feat_reg

            # Disciminator's LOSS estimation
            logits = D_real_logits[:, 0:-1]
            log_probs = F.log_softmax(logits, dim=-1)
            # The discriminator provides an output for labeled and unlabeled real data
            # so the loss evaluated for unlabeled data is ignored (masked)
            label2one_hot = torch.nn.functional.one_hot(b_labels, self.config['num_labels'])
            per_example_loss = -torch.sum(label2one_hot * log_probs, dim=-1)
            per_example_loss = torch.masked_select(per_example_loss, b_label_mask.to(self.device))
            labeled_example_count = per_example_loss.type(torch.float32).numel()

            # It may be the case that a batch does not contain labeled examples,
            # so the "supervised loss" in this case is not evaluated
            if labeled_example_count == 0:
                D_L_Supervised = 0
            else:
                D_L_Supervised = torch.div(torch.sum(per_example_loss.to(self.device)), labeled_example_count)

            # D_L_unsupervised1U = -1 * torch.mean(torch.log(1 - D_real_probs[:, -1] + self.config['epsilon']))
            # D_L_unsupervised2U = -1 * torch.mean(torch.log(D_fake_probs[:, -1] + self.config['epsilon']))
            d_loss = D_L_Supervised # + D_L_unsupervised1U + D_L_unsupervised2U

            # Avoid gradient accumulation
            # self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()

            # Calculate weigth updates
            # retain_graph=True is required since the underlying graph will be deleted after backward
            # g_loss.backward(retain_graph=True)
            d_loss.backward()
            # Apply modifications
            # self.generator_optimizer.step()
            self.discriminator_optimizer.step()
            # Save the losses to print them later
            # tr_g_loss += g_loss.item()
            tr_d_loss += d_loss.item()
            # Update the learning rate with the scheduler
            if self.config['apply_scheduler']:
                self.scheduler_d.step()
                # self.scheduler_g.step()
        return None, tr_d_loss

    @torch.no_grad()
    def validation(self, tr_d_loss, epoch_i, *args, **kwargs):

        # Calculate the average loss over all of the batches.
        # avg_train_loss_g = tr_g_loss / len(self.train_dataloader)
        avg_train_loss_d = tr_d_loss / len(self.train_dataloader)
        # print(f"\tAverage training loss generetor: {avg_train_loss_g:.3f}")
        print(f"\tAverage training loss discriminator: {avg_train_loss_d:.3f}")
        # print("  Training epcoh took: {:}".format(training_time))

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        self.backbone.eval()
        self.discriminator.eval()
        # self.generator.eval()

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
                # hidden_states = model_outputs[-1]
                _, logits, probs = self.discriminator(input_ids=b_input_ids, input_mask=b_input_mask)
                # log_probs = F.log_softmax(probs[:,1:], dim=-1)
                filtered_logits = logits[:, 0:-1]
                # Accumulate the test loss.
                total_test_loss += nll_loss(filtered_logits, b_labels)

            # Accumulate the predictions and the input labels
            _, preds = torch.max(filtered_logits, 1)
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
        self.training_stats.append(
            {
                'epoch': epoch_i + 1,
                # 'Training Loss generator': avg_train_loss_g,
                'Training Loss discriminator': avg_train_loss_d,
                'Valid. Loss': avg_test_loss,
                'Valid. Accur.': test_accuracy,
            }
        )
        return
