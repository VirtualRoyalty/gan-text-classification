import numpy as np
import torch
import torch.nn.functional as F
# import time
# from base import BaseTrainer
# from utils import inf_loop, MetricTracker
from model import *
from typing import *


class GANTrainer:
    def __init__(self, config: Dict, backbone,
                 generator: Generator, discriminator: Discriminator,
                 train_dataloader, valid_dataloader,
                 generator_optimizer, discriminator_optimizer,
                 device):
        self.config = config
        self.backbone = backbone
        self.generator = generator
        self.discriminator = discriminator
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.device = device
        self.training_stats = []
        pass

    def train_epoch(self):
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
            model_outputs = self.backbone(b_input_ids, attention_mask=b_input_mask)
            hidden_states = model_outputs[-1]

            # Generate fake data that should have the same distribution of the ones
            noise = torch.zeros(b_input_ids.shape[0], self.config['NOISE_SIZE'], device=self.device).uniform_(0, 1)
            gen_rep = self.generator(noise)

            # Generate the output of the Discriminator for real and fake data.
            disciminator_input = torch.cat([hidden_states, gen_rep], dim=0)
            features, logits, probs = self.discriminator(disciminator_input)

            features_list = torch.split(features, self.config['BATCH_SIZE'])
            D_real_features = features_list[0]
            D_fake_features = features_list[1]

            logits_list = torch.split(logits, self.config['BATCH_SIZE'])
            D_real_logits, D_fake_logits = logits_list[0], logits_list[1]

            probs_list = torch.split(probs, self.config['BATCH_SIZE'])
            D_real_probs, D_fake_probs = probs_list[0], probs_list[1]

            # Generator's LOSS estimation
            g_loss_d = -1 * torch.mean(torch.log(1 - D_fake_probs[:, -1] + self.config['EPSILON']))
            g_feat_reg = torch.mean(
                torch.pow(torch.mean(D_real_features, dim=0) - torch.mean(D_fake_features, dim=0), 2))
            g_loss = g_loss_d + g_feat_reg

            # Disciminator's LOSS estimation
            logits = D_real_logits[:, 0:-1]
            log_probs = F.log_softmax(logits, dim=-1)
            # The discriminator provides an output for labeled and unlabeled real data
            # so the loss evaluated for unlabeled data is ignored (masked)
            label2one_hot = torch.nn.functional.one_hot(b_labels, self.config['NUM_LABELS'])
            per_example_loss = -torch.sum(label2one_hot * log_probs, dim=-1)
            per_example_loss = torch.masked_select(per_example_loss, b_label_mask.to(self.device))
            labeled_example_count = per_example_loss.type(torch.float32).numel()

            # It may be the case that a batch does not contain labeled examples,
            # so the "supervised loss" in this case is not evaluated
            if labeled_example_count == 0:
                D_L_Supervised = 0
            else:
                D_L_Supervised = torch.div(torch.sum(per_example_loss.to(self.device)), labeled_example_count)

            D_L_unsupervised1U = -1 * torch.mean(torch.log(1 - D_real_probs[:, -1] + self.config['EPSILON']))
            D_L_unsupervised2U = -1 * torch.mean(torch.log(D_fake_probs[:, -1] + self.config['EPSILON']))
            d_loss = D_L_Supervised + D_L_unsupervised1U + D_L_unsupervised2U

            # Avoid gradient accumulation
            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()

            # Calculate weigth updates
            # retain_graph=True is required since the underlying graph will be deleted after backward
            g_loss.backward(retain_graph=True)
            d_loss.backward()
            # Apply modifications
            self.generator_optimizer.step()
            self.discriminator_optimizer.step()
            # Save the losses to print them later
            tr_g_loss += g_loss.item()
            tr_d_loss += d_loss.item()
            # Update the learning rate with the scheduler
            if self.config['APPLY_SCHEDULER']:
                self.scheduler_d.step()
                self.scheduler_g.step()
        return

    @torch.no_grad()
    def validation(self, tr_g_loss, tr_d_loss, epoch_i):

        # Calculate the average loss over all of the batches.
        avg_train_loss_g = tr_g_loss / len(self.train_dataloader)
        avg_train_loss_d = tr_d_loss / len(self.train_dataloader)
        print(f"\tAverage training loss generetor: {avg_train_loss_g:.3f}")
        print(f"\tAverage training loss discriminator: {avg_train_loss_d:.3f}")
        # print("  Training epcoh took: {:}".format(training_time))

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        self.backbone.eval()
        self.discriminator.eval()
        self.generator.eval()

        # Tracking variables
        total_test_accuracy = 0
        total_test_loss = 0
        nb_test_steps = 0
        all_preds = []
        all_labels_ids = []
        # loss
        nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

        # Evaluate data for one epoch
        for batch in self.test_dataloader:
            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                model_outputs = self.backbone(b_input_ids, attention_mask=b_input_mask)
                hidden_states = model_outputs[-1]
                _, logits, probs = self.discriminator(hidden_states)
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
        avg_test_loss = total_test_loss / len(self.test_dataloader)
        avg_test_loss = avg_test_loss.item()
        # Record all statistics from this epoch.
        self.training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss generator': avg_train_loss_g,
                'Training Loss discriminator': avg_train_loss_d,
                'Valid. Loss': avg_test_loss,
                'Valid. Accur.': test_accuracy,
            }
        )
        return


# class Trainer():
#     def __init__(self, writer, config):
#         if "test" not in config["name"] and writer is None:
#             print(f"Error: init writer for train section!")
#             raise ValueError
#         self.writer = writer
#         self.config = config
#         self.step = 0
#         self.train_metrics = {
#             "train_losses": [], "train_accs": [], "train_FAs": [], "train_FRs": [],
#         }
#         self.val_metrics = {
#             "val_losses": [], "val_accs": [], "val_FAs": [], "val_FRs": [], "val_au_fa_fr": [], 'val_time_inference': []
#         }
#         self.last_val_metric = 0
#
#     def get_mean_val_au_fa_fr(self):
#         return self.last_val_metric
#
#     def log_train(self, logits, loss, labels):
#         probs = F.softmax(logits, dim=-1)
#         argmax_probs = torch.argmax(probs, dim=-1)
#         FA, FR = count_FA_FR(argmax_probs, labels)
#         acc = torch.sum(argmax_probs == labels).item() / torch.numel(argmax_probs)
#
#         self.train_metrics["train_losses"].append(loss)
#         self.train_metrics["train_accs"].append(acc)
#         self.train_metrics["train_FAs"].append(FA)
#         self.train_metrics["train_FRs"].append(FR)
#
#         if self.step % self.config["log_step"] == 0:
#             if self.writer is not None:
#                 self.writer.set_step(self.step)
#                 self.writer.add_scalars("train", {'loss': np.mean(self.train_metrics["train_losses"]),
#                                                   'acc': np.mean(self.train_metrics["train_accs"]),
#                                                   'FA': np.mean(self.train_metrics["train_FAs"]),
#                                                   'FR': np.mean(self.train_metrics["train_FRs"])})
#             self.train_metrics = {
#                 "train_losses": [], "train_accs": [], "train_FAs": [], "train_FRs": [],
#             }
#
#     def log_after_train_epoch(self, config_writer):
#         if self.writer is not None:
#             self.writer.add_scalar(f"epoch", config_writer["epoch"])
#         else:
#             print({'loss': np.mean(self.train_metrics["train_losses"]),
#                    'acc': np.mean(self.train_metrics["train_accs"]),
#                    'FA': np.mean(self.train_metrics["train_FAs"]),
#                    'FR': np.mean(self.train_metrics["train_FRs"])})
#         print(f"Epoch end, acc {np.mean(self.train_metrics['train_accs'])}")
#
#     def train_kd_mimic_logits(self, teacher, student, opt, loader, log_melspec, device, config_writer):
#         teacher.eval()
#         student.train()
#         for i, (batch, labels) in tqdm(enumerate(loader), desc="train", total=len(loader)):
#             self.step += 1
#             batch, labels = batch.to(device), labels.to(device)
#             batch = log_melspec(batch)
#
#             opt.zero_grad()
#
#             logits_st = student(batch)
#             with torch.no_grad():
#                 logits_teach = teacher(batch)
#
#             loss = F.mse_loss(logits_st, logits_teach)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(student.parameters(), 5)
#
#             opt.step()
#
#             # logging
#             self.log_train(logits_st, loss.item(), labels)
#
#         self.log_after_train_epoch(config_writer)
#
#     def train_kd_soft_labels(self, teacher, student, opt, loader, log_melspec, device, config_writer):
#         def softXEnt(input_, target_):
#             logprobs = F.log_softmax(input_, dim=-1)
#             return -(target_ * logprobs).sum() / input_.shape[0]
#
#         teacher.eval()
#         student.train()
#         T = config_writer['T']
#         for i, (batch, labels) in tqdm(enumerate(loader), desc="train", total=len(loader)):
#             self.step += 1
#             batch, labels = batch.to(device), labels.to(device)
#             batch_prepared = log_melspec(batch)
#
#             logits_st = student(batch_prepared)
#             with torch.no_grad():
#                 logits_teach = teacher(batch_prepared)  # .detach()
#
#             hard_predictions = F.softmax(logits_st, dim=-1)
#             soft_predictions = F.softmax(logits_st / T, dim=-1)
#             soft_labels = F.softmax(logits_teach / T, dim=-1)
#             distillation_loss = softXEnt(soft_predictions, soft_labels) / (T ** 2)
#             student_loss = F.cross_entropy(hard_predictions, labels)
#
#             loss = config_writer['lambda'] * distillation_loss + (1.0 - config_writer['lambda']) * student_loss
#             loss.backward()
#
#             torch.nn.utils.clip_grad_norm_(student.parameters(), 5)
#             opt.step()
#
#             opt.zero_grad()
#
#             # logging
#             self.log_train(logits_st, loss.item(), labels)
#
#         self.log_after_train_epoch(config_writer)
#
#     def train_epoch(self, model, opt, loader, log_melspec, device, config_writer):
#         model.train()
#         acc = torch.tensor([0.0])
#
#         for i, (batch, labels) in tqdm(enumerate(loader), desc="train", total=len(loader)):
#             self.step += 1
#
#             batch, labels = batch.to(device), labels.to(device)
#             batch = log_melspec(batch)
#
#             opt.zero_grad()
#             logits = model(batch)
#             loss = F.cross_entropy(logits, labels)
#
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
#             opt.step()
#
#             # logging
#             self.log_train(logits, loss.item(), labels)
#
#         self.log_after_train_epoch(config_writer)
#
#     @torch.no_grad()
#     def validation(self, model, loader, log_melspec, device, config_writer):
#         model.eval()
#         self.val_metrics = {
#             "val_losses": [], "val_accs": [], "val_FAs": [], "val_FRs": [], "val_au_fa_fr": [], 'val_time_inference': []
#         }
#         all_probs, all_labels = [], []
#         for i, (batch, labels) in tqdm(enumerate(loader), desc="val", total=len(loader)):
#             batch, labels = batch.to(device), labels.to(device)
#             batch = log_melspec(batch)
#
#             start = datetime.now()
#
#             output = model(batch)
#             probs = F.softmax(output, dim=-1)  # we need probabilities so we use softmax & CE separately
#
#             if config_writer["type"] == "train":
#                 loss = F.cross_entropy(output, labels)
#
#             # logging
#             argmax_probs = torch.argmax(probs, dim=-1)
#             time_infer = (datetime.now() - start).total_seconds()
#             all_probs.append(probs[:, 1].cpu())
#             all_labels.append(labels.cpu())
#             acc = torch.sum(argmax_probs == labels).item() / torch.numel(argmax_probs)
#             FA, FR = count_FA_FR(argmax_probs, labels)
#
#             if config_writer["type"] == "train":
#                 self.val_metrics["val_losses"].append(loss.item())
#             else:
#                 self.val_metrics["val_losses"].append(0)
#
#             self.val_metrics["val_accs"].append(acc)
#             self.val_metrics["val_FAs"].append(FA)
#             self.val_metrics["val_FRs"].append(FR)
#             self.val_metrics["val_time_inference"].append(time_infer)
#         # area under FA/FR curve for whole loader
#         au_fa_fr = get_au_fa_fr(torch.cat(all_probs, dim=0).cpu(), all_labels)
#
#         print({'mean_val_loss': round(np.mean(self.val_metrics["val_losses"]), 5),
#                'mean_val_acc': round(np.mean(self.val_metrics["val_accs"]), 5),
#                'mean_val_FA': round(np.mean(self.val_metrics["val_FAs"]), 5),
#                'mean_val_FR': round(np.mean(self.val_metrics["val_FRs"]), 5),
#                'val_time_inference': round(np.mean(self.val_metrics["val_time_inference"]), 5),
#                'au_fa_fr': round(au_fa_fr, 5)})
#         self.last_val_metric = au_fa_fr
#         self.step += 1
#         if config_writer["type"] == "train" and self.writer is not None:
#             self.writer.set_step(self.step, "valid")
#             self.writer.add_scalars("val", {'mean_loss': np.mean(self.val_metrics["val_losses"]),
#                                             'mean_acc': np.mean(self.val_metrics["val_accs"]),
#                                             'mean_FA': np.mean(self.val_metrics["val_FAs"]),
#                                             'mean_FR': np.mean(self.val_metrics["val_FRs"]),
#                                             'au_fa_fr': au_fa_fr})
#
#         return np.mean(self.val_metrics["val_losses"])
