import os
import gc
import sys
import json
import torch
import numpy as np
import pandas as pd
import importlib as imp
import neptune.new as neptune
from tqdm import tqdm, tqdm_notebook
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import warnings

from model import Discriminator, Generator, ConditionalGenerator, Autoencoder
from data_loader import generate_data_loader
from trainer.trainer import Trainer
from trainer.gan_trainer import GANTrainer
from trainer.gan_distil_trainer import GANDistilTrainer
from trainer.autoencoder_trainer import AETrainer

warnings.simplefilter('ignore')
sys.path.append('..')


class Experiment:

    def __init__(self, config, train_df, test_df, secret, device, tags=None):
        self.config = config
        self.train_df = train_df
        self.test_df = test_df
        self.secret = secret
        self.device = device
        self.tags = tags
        return

    def split_data(self, seed=42):
        UNKNOWN_LABEL_NAME = "UNK"
        PROPORTION = self.config['labeled_proportion']
        UNLABEL_PROPORTION = self.config['unlabeled_proportion']
        self.labeled_df, self.unlabeled_df, _, _ = train_test_split(self.train_df, self.train_df,
                                                                    test_size=(1 - PROPORTION),
                                                                    random_state=seed, stratify=self.train_df.label)
        self.unlabeled_df = self.unlabeled_df.sample(frac=UNLABEL_PROPORTION)
        self.unlabeled_df['label'] = UNKNOWN_LABEL_NAME
        print(f"Labeled: {len(self.labeled_df)} Unlabeled: {len(self.unlabeled_df)}")

        self.label_list = [UNKNOWN_LABEL_NAME] + self.train_df.label.unique().tolist()
        self.label2id = {label: _id for _id, label in enumerate(self.label_list)}
        self._label2id = {label: _id for _id, label in enumerate(self.train_df.label.unique().tolist())}

        self.config['labeled_df'] = len(self.labeled_df)
        self.config['unlabeled_df'] = len(self.unlabeled_df)
        self.config['num_labels'] = len(self.label_list)
        print(f"Label count: {len(self._label2id)}")

    def create_dataloaders(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        train_examples = self.labeled_df.values.tolist()
        # CONFIG['num_train_examples'] = len(train_examples)
        unlabeled_examples = self.unlabeled_df.values.tolist()
        test_examples = self.test_df.values.tolist()

        train_label_masks = np.ones(len(self.labeled_df), dtype=bool)
        self._train_dataloader = generate_data_loader(train_examples,
                                                      train_label_masks,
                                                      label_map=self._label2id,
                                                      batch_size=self.config['batch_size'],
                                                      tokenizer=tokenizer,
                                                      max_seq_length=self.config['max_seq_length'],
                                                      do_shuffle=True,
                                                      balance_label_examples=self.config['apply_balance'])

        test_label_masks = np.ones(len(test_examples), dtype=bool)
        self._test_dataloader = generate_data_loader(test_examples,
                                                     test_label_masks,
                                                     label_map=self._label2id,
                                                     batch_size=self.config['batch_size'],
                                                     tokenizer=tokenizer,
                                                     max_seq_length=self.config['max_seq_length'],
                                                     do_shuffle=False, balance_label_examples=False)
        if len(unlabeled_examples) > 0:
            labeled_examples = train_examples.copy()
            labeled_masks = train_label_masks.copy()
            train_examples = train_examples + unlabeled_examples
            tmp_masks = np.zeros(len(unlabeled_examples), dtype=bool)
            train_label_masks = np.concatenate([train_label_masks, tmp_masks])

        self.labeled_dataloader = generate_data_loader(labeled_examples,
                                                       labeled_masks,
                                                       label_map=self.label2id,
                                                       batch_size=self.config['batch_size'],
                                                       tokenizer=tokenizer,
                                                       max_seq_length=self.config['max_seq_length'],
                                                       do_shuffle=True,
                                                       balance_label_examples=self.config['apply_balance'])

        self.train_dataloader = generate_data_loader(train_examples,
                                                     train_label_masks,
                                                     label_map=self.label2id,
                                                     batch_size=self.config['batch_size'],
                                                     tokenizer=tokenizer,
                                                     max_seq_length=self.config['max_seq_length'],
                                                     do_shuffle=True,
                                                     balance_label_examples=self.config['apply_balance'])

        self.test_dataloader = generate_data_loader(test_examples,
                                                    test_label_masks,
                                                    label_map=self.label2id,
                                                    batch_size=self.config['batch_size'],
                                                    tokenizer=tokenizer,
                                                    max_seq_length=self.config['max_seq_length'],
                                                    do_shuffle=False, balance_label_examples=False)

    def train_only_classifier(self):
        gc.collect()
        torch.cuda.empty_cache()
        config = AutoConfig.from_pretrained(self.config['model_name'])
        self.config['hidden_size'] = int(config.hidden_size)
        # Define the number and width of hidden layers
        transformer = AutoModel.from_pretrained(self.config['model_name'])
        discriminator = Discriminator(backbone=transformer, input_size=self.config['hidden_size'],
                                      hidden_size=self.config['hidden_size'],
                                      num_labels=self.train_df.label.nunique(),
                                      dropout_rate=self.config['out_dropout_rate'],
                                      model_name=self.config['model_name'])
        transformer.to(self.device)
        discriminator.to(self.device)
        if self.config['multi_gpu']:
            transformer = torch.nn.DataParallel(transformer)

        self.config['num_labels'] = self.train_df.label.nunique()
        self.config['num_train_examples'] = len(self.labeled_dataloader.dataset)
        dtrainer = Trainer(config=self.config,
                           discriminator=discriminator,
                           train_dataloader=self._train_dataloader,
                           valid_dataloader=self._test_dataloader,
                           device=self.device)
        run = neptune.init(project=self.secret['project'], api_token=self.secret['token'], tags=self.tags)
        self.config['GAN'] = False
        run['config'] = dtrainer.config
        for epoch_i in range(0, self.config['num_train_epochs']):
            print(f"======== Epoch {epoch_i + 1} / {self.config['num_train_epochs']} ========")
            tr_d_loss = dtrainer.train_epoch(log_env=run)
            result = dtrainer.validation(tr_d_loss=tr_d_loss, epoch_i=epoch_i)
            run['valid/discriminator_loss'].log(result['discriminator_loss'])
            run['valid/discriminator_accuracy'].log(result['discriminator_accuracy'])

        _labels, _predicts = discriminator.predict(self._test_dataloader, device=self.device, gan=False)
        f1_macro = f1_score(_labels, _predicts, average='macro')
        f1_micro = f1_score(_labels, _predicts, average='micro')
        run['valid/f1_macro'].log(f1_macro)
        run['valid/f1_micro'].log(f1_micro)
        print('Only classifier')
        print(f'f1_macro {f1_macro:.3f}')
        print(f'f1_micro {f1_micro:.3f}')
        run.stop()
        del transformer
        del discriminator
        with torch.no_grad():
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        gc.collect()

    def train_gan(self):
        torch.cuda.empty_cache()
        gc.collect()
        config = AutoConfig.from_pretrained(self.config['model_name'])
        self.config['hidden_size'] = int(config.hidden_size)
        self.config['num_train_examples'] = len(self.train_dataloader.dataset)
        self.config['num_labels'] = len(self.label_list)

        transformer = AutoModel.from_pretrained(self.config['model_name'])
        discriminator = Discriminator(backbone=transformer,
                                      input_size=self.config['hidden_size'], hidden_size=self.config['hidden_size'],
                                      hidden_layers=self.config['num_hidden_layers_g'],
                                      num_labels=self.config['num_labels'] + 1,
                                      dropout_rate=self.config['out_dropout_rate'],
                                      model_name=self.config['model_name'])

        if self.config['conditional_generator']:
            generator = ConditionalGenerator(noise_size=self.config['noise_size'], num_labels=self.config['num_labels'],
                                             output_size=self.config['hidden_size'],
                                             hidden_size=self.config['hidden_size'],
                                             hidden_layers=self.config['num_hidden_layers_g'],
                                             dropout_rate=self.config['out_dropout_rate'])
        else:
            generator = Generator(noise_size=self.config['noise_size'], output_size=self.config['hidden_size'],
                                  hidden_size=self.config['hidden_size'],
                                  hidden_layers=self.config['num_hidden_layers_g'],
                                  dropout_rate=self.config['out_dropout_rate'])

        # Put everything in the GPU if available
        generator.to(self.device)
        transformer.to(self.device)
        discriminator.to(self.device)

        if self.config['pretrained_generator']:
            print('Start AE pre-training...')
            autoencoder = Autoencoder(generator, input_size=self.config['hidden_size'])
            autoencoder.to(self.device)
            ae_loader = (self.labeled_dataloader if self.config['conditional_generator'] else self.train_dataloader)
            ae_trainer = AETrainer(self.config, autoencoder=autoencoder,
                                   dataloader=ae_loader,
                                   device=self.device)
            transformer.eval()
            ae_trainer.train_epoch(feature_extractor=transformer)
            print('Generator is pretrained')

        if self.config['distil_gan']:
            aversarial_trainer = GANDistilTrainer(config=self.config,
                                                  generator=generator,
                                                  discriminator=discriminator,
                                                  train_dataloader=self.train_dataloader,
                                                  train_tensor=self.labeled_dataloader.dataset.tensors[0],
                                                  valid_dataloader=self.test_dataloader,
                                                  device=self.device)

            local_data_loader = generate_data_loader(self.labeled_examples,
                                                     self.labeled_masks,
                                                     label_map=self.label2id,
                                                     batch_size=100,
                                                     tokenizer=self.tokenizer,
                                                     max_seq_length=self.config['max_seq_length'],
                                                     do_shuffle=False,
                                                     balance_label_examples=self.config['apply_balance'],
                                                     return_ids=True)
            print('Hard/easy mining started...')
            aversarial_trainer.hard_mining(labeled_dataloader=self.labeled_dataloader,
                                           local_dataloader=local_data_loader)
            print('Hard/easy mining completed!')
        else:
            aversarial_trainer = GANTrainer(config=self.config,
                                            generator=generator,
                                            discriminator=discriminator,
                                            train_dataloader=self.train_dataloader,
                                            valid_dataloader=self.test_dataloader,
                                            device=self.device)
        run = neptune.init(project=self.secret['project'], api_token=self.secret['token'], tags=self.tags)
        run['config'] = aversarial_trainer.config
        for epoch_i in range(0, self.config['num_train_epochs']):
            print(f"======== Epoch {epoch_i + 1} / {self.config['num_train_epochs']} ========")
            tr_g_loss, tr_d_loss = aversarial_trainer.train_epoch(log_env=run)
            result = aversarial_trainer.validation(tr_d_loss, tr_g_loss, epoch_i=epoch_i)
            run['valid/discriminator_loss'].log(result['discriminator_loss'])
            run['valid/discriminator_accuracy'].log(result['discriminator_accuracy'])

        _labels, _predicts = discriminator.predict(self.test_dataloader, device=self.device)
        f1_macro = f1_score(_labels, _predicts, average='macro')
        f1_micro = f1_score(_labels, _predicts, average='micro')
        run['valid/f1_macro'].log(f1_macro)
        run['valid/f1_micro'].log(f1_micro)
        print('GAN')
        print(f'f1_macro {f1_macro:.3f}')
        print(f'f1_micro {f1_micro:.3f}')
        run.stop()
        del transformer
        del discriminator
        del generator
        with torch.no_grad():
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        gc.collect()

    def run(self):
        return
