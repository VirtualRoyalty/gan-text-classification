import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoModel
from typing import Dict, Tuple, List

from model import Generator, Discriminator


class GANTrainer:
    def __init__(self, config: Dict,
                 discriminator: Discriminator,
                 generator: Generator,
                 train_dataloader: torch.utils.data.DataLoader,
                 valid_dataloader: torch.utils.data.DataLoader,
                 generator_optimizer, discriminator_optimizer,
                 scheduler_d: torch.optim.lr_scheduler.LambdaLR = None,
                 scheduler_g: torch.optim.lr_scheduler.LambdaLR = None,
                 device: torch.device = None,
                 *args, **kwargs):
        self.config = config
        self.generator = generator
        self.discriminator = discriminator
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.scheduler_d = scheduler_d
        self.scheduler_g = scheduler_g
        self.noise_type = config.get('noise_type', None)
        self.label2stat = config.get('label2stat', None)
        self.device = device
        self.training_stats = []
        pass

    def train_epoch(self, log_env=None) -> Tuple[float, float]:
        total_g_loss = 0
        total_d_loss = 0
        self.generator.train()
        self.discriminator.train()

        for step, batch in enumerate(self.train_dataloader):
            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            b_label_mask = batch[3].to(self.device)

            # Generate the output of the Discriminator for real data
            real_states, real_logits, real_probs, enc_states = self.discriminator(input_ids=b_input_ids,
                                                                                  input_mask=b_input_mask)

            # Generate fake data that should have the same distribution of the ones
            noise = torch.zeros(b_input_ids.shape[0], self.config['noise_size'], device=self.device)

            if self.noise_type == 'uniform':
                noise = noise.uniform_(*self.config['noise_range'])
            elif self.noise_type == 'normal':
                noise = noise.normal_(*self.config['noise_range'])
            elif self.noise_type == 'conditional':
                noises = []
                K = 8
                for i in range(np.ceil(b_input_ids.shape[0] / K)):
                    if b_input_ids.shape[0] - (i * K) < K:
                        k = b_input_ids.shape[0] - (i * K)
                    else:
                        k = K
                    _label = np.random.choice(list(self.label2stat.keys()))
                    _noise = np.random.multivariate_normal(
                            self.label2stat[_label]['mean'],
                            self.label2stat[_label]['cov'], size=k)
                    noises.append(_noise)
                    if k != K:
                        break
                noise = np.vstack(noises).astype('float32')
                noise = torch.from_numpy(noise).to(self.device)
            else:
                noise = noise.uniform_(*self.config['noise_range'])

            if self.config['conditional_generator']:
                rand_labels = np.random.randint(0, self.config['num_labels'], b_input_ids.shape[0], dtype='int')
                rand_labels = torch.from_numpy(rand_labels).to(self.device)
                generator_states = self.generator(noise, rand_labels)
            else:
                generator_states = self.generator(noise)

            if self.config['NDA'] and not self.config['conditional_generator']:
                if self.config['nda_alpha'] is None:
                    self.config['nda_alpha'] = 0.9
                alpha = min(np.random.normal(self.config['nda_alpha'], 0.1), 0.95)
                generator_states = alpha * generator_states + (1 - alpha) * enc_states

            # Generate the output of the Discriminator for fake data
            fake_states, fake_logits, fake_probs, _ = self.discriminator(external_states=generator_states)

            # generator loss estimation
            cheat_rate_loss = -1 * torch.mean(torch.log(1 - fake_probs[:, -1] + self.config['epsilon']))
            feature_sim_loss = torch.mean(torch.pow(torch.mean(real_states, dim=0) - torch.mean(fake_states, dim=0), 2))
            generator_loss = self.config['cheat_rate_weight'] * cheat_rate_loss + \
                             self.config['feature_sim_weight'] * feature_sim_loss
            generator_loss *= self.config['generator_weight']

            # discriminator loss estimation
            logits = real_logits[:, 0:-1]
            log_probs = F.log_softmax(logits, dim=-1)
            # The discriminator provides an output for labeled and unlabeled real data
            label2one_hot = torch.nn.functional.one_hot(b_labels, self.config['num_labels'])
            per_example_loss = -torch.sum(label2one_hot * log_probs, dim=-1)
            per_example_loss = torch.masked_select(per_example_loss, b_label_mask.to(self.device))
            labeled_example_count = per_example_loss.type(torch.float32).numel()
            # It may be the case that a batch does not contain labeled examples,
            # so the "supervised loss" in this case is not evaluated
            if labeled_example_count == 0:
                supervised_loss = 0
            else:
                supervised_loss = torch.div(torch.sum(per_example_loss.to(self.device)), labeled_example_count)
            unsupervised_real_loss = -1 * torch.mean(torch.log(1 - real_probs[:, -1] + self.config['epsilon']))
            unsupervised_fake_loss = -1 * torch.mean(torch.log(fake_probs[:, -1] + self.config['epsilon']))
            discriminator_loss = self.config['supervised_weight'] * supervised_loss + \
                                 self.config['unsupervised_weight'] * (unsupervised_real_loss + unsupervised_fake_loss)
            discriminator_loss *= self.config['discriminator_weight']

            # Avoid gradient accumulation
            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            # Calculate weights updates
            # retain_graph=True is required since the underlying graph will be deleted after backward
            generator_loss.backward(retain_graph=True)
            discriminator_loss.backward()
            # Apply modifications
            self.generator_optimizer.step()
            self.discriminator_optimizer.step()

            # Save the losses to print them later
            total_g_loss += generator_loss.item()
            total_d_loss += discriminator_loss.item()
            # Update the learning rate with the scheduler
            if self.config['apply_scheduler']:
                self.scheduler_d.step()
                self.scheduler_g.step()
            if log_env:
                # if total_d_loss != 0:
                log_env['train/generator_loss'].log(generator_loss.item())
                log_env['train/discriminator_loss'].log(discriminator_loss.item())

        # Calculate the average loss over all of the batches.
        avg_loss_g = total_g_loss / len(self.train_dataloader)
        avg_loss_d = total_d_loss / len(self.train_dataloader)
        return avg_loss_g, avg_loss_d

    @torch.no_grad()
    def validation(self, generator_loss, discriminator_loss, epoch_i, verbose=True):
        # Put the model in evaluation mode
        # the dropout layers behave differently during evaluation.
        self.discriminator.eval()
        self.generator.eval()

        # Tracking variables
        total_test_loss = 0
        all_preds = []
        all_labels_ids = []
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
                # Accumulate the test loss.
                _, logits, probs, _ = self.discriminator(input_ids=b_input_ids, input_mask=b_input_mask)
                filtered_logits = logits[:, 0:-1]
                total_test_loss += nll_loss(filtered_logits, b_labels)
            # Accumulate the predictions and the input labels
            _, preds = torch.max(filtered_logits, 1)
            all_preds += preds.detach().cpu()
            all_labels_ids += b_labels.detach().cpu()

        # Report the final accuracy for this validation run.
        all_preds = torch.stack(all_preds).numpy()
        all_labels_ids = torch.stack(all_labels_ids).numpy()
        test_accuracy = np.sum(all_preds == all_labels_ids) / len(all_preds)
        # Calculate the average loss over all of the batches.
        avg_test_loss = total_test_loss / len(self.valid_dataloader)

        # Record all statistics from this epoch.
        info_dct = {
            'epoch': epoch_i + 1,
            'Training Loss generator': generator_loss,
            'Training Loss discriminator': discriminator_loss,
            'discriminator_loss': avg_test_loss.item(),
            'discriminator_accuracy': test_accuracy,
        }
        self.training_stats.append(info_dct)
        if verbose:
            print(f"\tAverage training loss generetor: {generator_loss:.3f}")
            print(f"\tAverage training loss discriminator: {discriminator_loss:.3f}")
            # print("  Training epcoh took: {:}".format(training_time))
            print("  Accuracy: {0:.3f}".format(test_accuracy))
        return info_dct
