import torch
import numpy as np
import torch.nn.functional as F
from typing import Dict, Tuple, List
from transformers import AutoModel, get_constant_schedule_with_warmup
from tqdm import tqdm_notebook

from trainer import Trainer
from model import Generator, Discriminator
from data_loader import generate_data_loader


class GANDistilTrainer:
    def __init__(self, config: Dict,
                 discriminator: Discriminator,
                 generator: Generator,
                 train_dataloader: torch.utils.data.DataLoader,
                 valid_dataloader: torch.utils.data.DataLoader,
                 train_tensor: torch.Tensor,
                 device: torch.device = None,
                 *args, **kwargs):
        self.generator = generator
        self.discriminator = discriminator
        self.backbone = discriminator.backbone
        self.train_dataloader = train_dataloader
        self.train_tensor = train_tensor
        self.valid_dataloader = valid_dataloader
        self.noise_type = config.get('noise_type', None)
        self.label2stat = config.get('label2stat', None)
        self.device = device
        self.training_stats = []
        self.triplet_loss_fn = torch.nn.TripletMarginLoss(margin=1.0)

        # define trainable parameters and optimizer
        if config['frozen_backbone']:
            self.discriminator.freeze_backbone()
        d_vars = [p for p in self.discriminator.parameters() if p.requires_grad]
        g_vars = [v for v in self.generator.parameters()]
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
            self.scheduler_d = get_constant_schedule_with_warmup(self.discriminator_optimizer,
                                                                 num_warmup_steps=config['num_warmup_steps_d'])
            self.scheduler_g = get_constant_schedule_with_warmup(self.generator_optimizer,
                                                                 num_warmup_steps=config['num_warmup_steps_g'])
        # log config
        self.config = config

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
            del real_states

            # Generate fake data that should have the same distribution of the ones
            noise = torch.zeros(b_input_ids.shape[0], self.config['noise_size'], device=self.device)
            if self.noise_type == 'uniform':
                noise = noise.uniform_(*self.config['noise_range'])
            elif self.noise_type == 'normal':
                noise = noise.normal_(*self.config['noise_range'])
            else:
                noise = noise.uniform_(*self.config['noise_range'])

            rand_labels = np.random.randint(1, self.config['num_labels'], b_input_ids.shape[0], dtype='int')
            rand_labels = torch.from_numpy(rand_labels).to(self.device)
            generator_states = self.generator(noise, rand_labels)

            if self.config['NDA'] and not self.config['conditional_generator']:
                if self.config['nda_alpha'] is None:
                    self.config['nda_alpha'] = 0.9
                alpha = min(np.random.normal(self.config['nda_alpha'], 0.01), 0.95)
                generator_states = alpha * generator_states + (1 - alpha) * enc_states
            del enc_states

            # Generate the output of the Discriminator for fake data
            fake_states, fake_logits, fake_probs, _ = self.discriminator(external_states=generator_states)
            del fake_states

            # get easy end hard samples
            easy_ids, hard_ids = self.get_hard_easy_ids(rand_labels)
            easy_samples, hard_samples = self.train_tensor[easy_ids].to(self.device), self.train_tensor[hard_ids].to(
                self.device)
            easy_states = self.backbone(easy_samples, attention_mask=b_input_mask).last_hidden_state[:, 0]
            del easy_samples

            hard_states = self.backbone(hard_samples, attention_mask=b_input_mask).last_hidden_state[:, 0]
            del hard_samples

            # generator loss estimation
            dist_loss = self.triplet_loss_fn(generator_states, hard_states, easy_states)
            del easy_states
            del hard_states
            del generator_states
            torch.cuda.empty_cache()

            cheat_rate_loss = -1 * torch.mean(torch.log(1 - fake_probs[:, -1] + self.config['epsilon']))
            # feature_sim_loss = torch.mean(torch.pow(torch.mean(real_states, dim=0) - torch.mean(fake_states, dim=0), 2))
            generator_loss = self.config['cheat_rate_weight'] * cheat_rate_loss + dist_loss
            # self.config['feature_sim_weight'] * feature_sim_loss
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

    def get_hard_easy_ids(self, b_labels):
        easy_ids = []
        hard_ids = []
        for b_label in b_labels:
            easy_list = self.config['label2easy_id'].get(b_label)
            if easy_list is None:
                easy_ids.append(np.random.choice(self.config['easy_ids']))
            else:
                easy_ids.append(np.random.choice(easy_list))
            hard_list = self.config['label2hard_id'].get(b_label)
            if hard_list is None:
                hard_ids.append(np.random.choice(self.config['hard_ids']))
            else:
                hard_ids.append(np.random.choice(hard_list))
        return easy_ids, hard_ids

    def hard_mining(self, labeled_dataloader: torch.utils.data.DataLoader,
                    local_dataloader: torch.utils.data.DataLoader,
                    hard_mine_epoch: int = 3):
        _CONFIG = self.config.copy()
        _CONFIG['num_labels'] = _CONFIG['num_labels'] + 1
        # Pretraining of Discriminator and hard/easy sample mining
        pretrainer = Trainer(config=_CONFIG,
                             discriminator=self.discriminator,
                             train_dataloader=labeled_dataloader,
                             valid_dataloader=None,
                             device=self.device)
        for epoch_i in range(0, hard_mine_epoch):
            print(f"======== Epoch {epoch_i + 1} / {hard_mine_epoch} ========")
            _ = pretrainer.train_epoch()

        # Hard/easy samples mining
        self.discriminator.eval()

        hard_easy_embs = []
        easy_embs = []
        easy_labels = []
        easy_ids = []
        hard_embs = []
        hard_labels = []
        hard_ids = []
        with torch.no_grad():
            for batch in tqdm_notebook(local_dataloader):
                _ids = batch[0].to(self.device)
                _input_ids = batch[1].to(self.device)
                input_mask = batch[2].to(self.device)
                _labels = batch[3].to(self.device)
                _, logits, probs, _embs = self.discriminator(_input_ids, input_mask=input_mask)
                embs = _embs.cpu().detach().numpy()
                ids = _ids.cpu().detach().numpy()
                labels = _labels.cpu().detach().numpy()
                probs = probs.cpu().detach().numpy()
                predicted_labels = probs.argmax(axis=1)

                indexes = np.arange(_input_ids.shape[0])
                hard_indexes = indexes[labels != predicted_labels].tolist()
                hard_indexes += indexes[probs.max(axis=1) < 0.5].tolist()
                easy_indexes = indexes[labels == predicted_labels].tolist()

                # easy
                easy_embs.append(embs[easy_indexes])
                easy_labels.extend(labels[easy_indexes].tolist())
                easy_ids.extend(ids[easy_indexes].tolist())

                # hard
                hard_embs.append(embs[hard_indexes])
                hard_labels.extend(labels[hard_indexes].tolist())
                hard_ids.extend(ids[hard_indexes].tolist())
                hard_easy_embs.append(embs)

                del _input_ids
                del _labels
                del _embs
                torch.cuda.empty_cache()

        easy_embs = np.vstack(easy_embs)
        hard_embs = np.vstack(hard_embs)

        label2easy_id = {}
        for i, label in enumerate(easy_labels):
            if label not in label2easy_id:
                label2easy_id[label] = [easy_ids[i]]
            else:
                label2easy_id[label].append(easy_ids[i])

        label2hard_id = {}
        for i, label in enumerate(hard_labels):
            if label not in label2hard_id:
                label2hard_id[label] = [hard_ids[i]]
            else:
                label2hard_id[label].append(hard_ids[i])

        self.config['label2hard_id'] = label2hard_id
        self.config['label2easy_id'] = label2easy_id
        self.config['easy_ids'] = easy_ids
        self.config['hard_ids'] = hard_ids
        # return hard_embs, easy_embs

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
