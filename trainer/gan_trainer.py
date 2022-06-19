import torch
import numpy as np
from typing import Dict, Tuple, List
import torch.nn.functional as F
from transformers import AutoModel, get_constant_schedule_with_warmup

from model import Generator, Discriminator


class GANTrainer:
    def __init__(self, config: Dict,
                 discriminator: Discriminator,
                 generator: Generator,
                 train_dataloader: torch.utils.data.DataLoader,
                 valid_dataloader: torch.utils.data.DataLoader,
                 device: torch.device = None,
                 *args, **kwargs):
        self.G = generator
        self.D = discriminator
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.noise_type = config.get('noise_type', None)
        self.label2stat = config.get('label2stat', None)
        self.MF = config.get('MF', 1e-3)
        self.valid_nll = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.mse = torch.nn.MSELoss().to(device)
        self.device = device
        self.training_stats = []

        # define trainable parameters and optimizer
        if config['frozen_backbone']:
            self.D.freeze_backbone()
        d_vars = [p for p in self.D.parameters() if p.requires_grad]
        g_vars = [v for v in self.G.parameters()]
        print(f'Trainable layers {len(d_vars)}')
        self.optimizer = torch.optim.AdamW(d_vars, lr=config['learning_rate_discriminator'])
        self.discriminator_optimizer = torch.optim.AdamW(d_vars, lr=config['learning_rate_discriminator'])
        self.generator_optimizer = torch.optim.AdamW(g_vars, lr=config['learning_rate_generator'])

        # define schedulers
        if config['apply_scheduler']:
            config['num_train_steps'] = config['num_train_examples'] / config['batch_size'] * config['num_train_epochs']
            config['num_train_steps'] = int(config['num_train_steps'])
            config['num_warmup_steps_d'] = int(config['num_train_steps'] * config['warmup_proportion_d'])
            config['num_warmup_steps_g'] = int(config['num_train_steps'] * config['warmup_proportion_g'])
            self.scheduler_D = get_constant_schedule_with_warmup(self.discriminator_optimizer,
                                                                 num_warmup_steps=config['num_warmup_steps_d'])
            self.scheduler_G = get_constant_schedule_with_warmup(self.generator_optimizer,
                                                                 num_warmup_steps=config['num_warmup_steps_g'])
        # log config
        self.config = config

    def train_epoch(self, log_env=None) -> Tuple[float, float]:
        total_g_loss = 0
        total_d_loss = 0
        self.D.train()
        self.G.train()

        for step, batch in enumerate(self.train_dataloader):
            # Unpack this training batch from our dataloader.
            input_ids = batch[0].to(self.device)
            input_mask = batch[1].to(self.device)
            labels = batch[2].to(self.device)
            label_mask = batch[3].to(self.device)
            batch_size = input_ids.shape[0]

            # Generate the output of the Discriminator for real data
            real_states, real_logits, real_probs, enc_states = self.D(input_ids=input_ids,
                                                                      input_mask=input_mask)
            # Generate latent noise vector
            noise = self._gen_noise(batch_size)

            # Generated perturbed latent noise vector if manifold regularization
            if self.config['manifold']:
                perturbs = torch.randn(batch_size, self.config['noise_size']).to(self.device)
                noise_perturbed = noise + (perturbs * 1e-5)

            # Get Generator states from latent noise vectors
            #   if conditional generator or if vanilla generator used
            if self.config['conditional_generator']:
                rand_labels = np.random.randint(0, self.config['num_labels'], batch_size, dtype='int')
                rand_labels = torch.from_numpy(rand_labels).to(self.device)
                generator_states = self.G(noise, rand_labels)
                if self.config['manifold']:
                    generator_states_prtd = self.G(noise_perturbed, rand_labels)
            else:
                generator_states = self.G(noise)
                if self.config['manifold']:
                    generator_states_prtd = self.G(noise_perturbed)

            # Generate negative sample if NDA
            if self.config['NDA']:
                if self.config['nda_alpha'] is None:
                    self.config['nda_alpha'] = 0.9
                alpha = min(np.random.normal(self.config['nda_alpha'], 5e-3), 1.0)
                generator_states = alpha * generator_states + (1 - alpha) * enc_states

            # Get the output of Discriminator for the fake data
            fake_states, fake_logits, fake_probs, _ = self.D(external_states=generator_states)
            
            # Get the output of Discriminator for the perturbed fake data
            #   if manifold regularization
            if self.config['manifold']:
                fake_states_prtd, fake_logits_prtd, fake_probs_prtd, _ = self.D(external_states=generator_states_prtd)

            # Generator loss estimation
            cheat_rate_loss = -1 * torch.mean(torch.log(1 - fake_probs[:, -1] + self.config['epsilon']))
            feature_sim_loss = torch.mean(torch.pow(torch.mean(real_states, dim=0) - torch.mean(fake_states, dim=0), 2))
            G_loss = self.config['cheat_rate_weight'] * cheat_rate_loss
            G_loss += self.config['feature_sim_weight'] * feature_sim_loss

            # Discriminator loss estimation
            logits = real_logits[:, 0:-1]
            log_probs = F.log_softmax(logits, dim=-1)
            label2one_hot = torch.nn.functional.one_hot(labels, self.config['num_labels'])
            per_example_loss = -torch.sum(label2one_hot * log_probs, dim=-1)
            per_example_loss = torch.masked_select(per_example_loss, label_mask.to(self.device))
            labeled_example_count = per_example_loss.type(torch.float32).numel()
            # It may be the case that a batch does not contain labeled examples,
            if labeled_example_count == 0:
                supervised_loss = 0
            else:
                supervised_loss = torch.div(torch.sum(per_example_loss.to(self.device)), labeled_example_count)
            unsupervised_real_loss = -1 * torch.mean(torch.log(1 - real_probs[:, -1] + self.config['epsilon']))
            unsupervised_fake_loss = -1 * torch.mean(torch.log(fake_probs[:, -1] + self.config['epsilon']))

            # Calculate manifold regularization loss if needed
            if self.config['manifold']:
                self.config['unsupervised_weight'] = 0.5
                D_manifold_loss = self.MF * self.mse(fake_probs, fake_probs_prtd)
            else:
                D_manifold_loss = 0
            D_loss = self.config['supervised_weight'] * supervised_loss
            D_loss += self.config['unsupervised_weight'] * (unsupervised_real_loss + unsupervised_fake_loss)
            D_loss += D_manifold_loss
            
            # Avoid gradient accumulation
            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            # Calculate weights updates
            G_loss.backward(retain_graph=True)
            D_loss.backward()
            self.generator_optimizer.step()
            self.discriminator_optimizer.step()

            # Save the losses to print them later
            total_g_loss += G_loss.item()
            total_d_loss += D_loss.item()
            # Update the learning rate with the scheduler
            if self.config['apply_scheduler']:
                self.scheduler_D.step()
                self.scheduler_G.step()
            if log_env:
                # if total_d_loss != 0:
                log_env['train/generator_loss'].log(G_loss.item())
                log_env['train/discriminator_loss'].log(D_loss.item())

        # Calculate the average loss over all of the batches.
        avg_loss_g = total_g_loss / len(self.train_dataloader)
        avg_loss_d = total_d_loss / len(self.train_dataloader)
        return avg_loss_g, avg_loss_d

    def _gen_noise(self, batch_size):
        noise = torch.zeros(batch_size, self.config['noise_size'], device=self.device)
        if self.noise_type == 'uniform':
            noise = noise.uniform_(*self.config['noise_range'])
        elif self.noise_type == 'normal':
            noise = noise.normal_(*self.config['noise_range'])
        else:
            noise = noise.uniform_(*self.config['noise_range'])
        return noise

    @torch.no_grad()
    def validation(self, generator_loss, discriminator_loss, epoch_i, verbose=True):
        # Put the model in evaluation mode
        self.D.eval()
        self.G.eval()

        # Tracking variables
        total_test_loss = 0
        all_preds = []
        all_labels_ids = []
        
        # Evaluate data for one epoch
        for batch in self.valid_dataloader:
            # Unpack this training batch from our dataloader.
            input_ids = batch[0].to(self.device)
            input_mask = batch[1].to(self.device)
            labels = batch[2].to(self.device)

            # Tell pytorch not to bother with constructing the compute graph during
            with torch.no_grad():
                # Accumulate the test loss.
                _, logits, probs, _ = self.D(input_ids=input_ids, input_mask=input_mask)
                filtered_logits = logits[:, 0:-1]
                total_test_loss += self.valid_nll(filtered_logits, labels)
            # Accumulate the predictions and the input labels
            _, preds = torch.max(filtered_logits, 1)
            all_preds += preds.detach().cpu()
            all_labels_ids += labels.detach().cpu()

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
            print("  Accuracy: {0:.3f}".format(test_accuracy))
        return info_dct
