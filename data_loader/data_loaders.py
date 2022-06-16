import math
import datetime
import numpy as np
from typing import Dict, Tuple, List

import torch
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from base import *


def generate_data_loader(input_examples,
                         label_masks,
                         label_map: Dict,
                         batch_size: int,
                         tokenizer: AutoTokenizer,
                         max_seq_length: int,
                         do_shuffle: bool = False,
                         balance_label_examples: bool = False,
                         return_ids: bool = False, **kwargs) -> torch.utils.data.DataLoader:
    """
    Generate a Dataloader given the input examples, eventually masked if they are
    to be considered NOT labeled.
    """
    examples = []
    # Count the percentage of labeled examples
    num_labeled_examples = 0
    for label_mask in label_masks:
        if label_mask:
            num_labeled_examples += 1
    label_mask_rate = num_labeled_examples / len(input_examples)

    # if required it applies the balance
    for index, ex in enumerate(input_examples):
        if label_mask_rate == 1 or not balance_label_examples:
            examples.append((ex, label_masks[index]))
        else:
            # it simulates a labeled example
            if label_masks[index]:
                balance = int(1 / label_mask_rate)
                balance = int(math.log(balance, 2))
                if balance < 1:
                    balance = 1
                for b in range(0, int(balance)):
                    examples.append((ex, label_masks[index]))
            else:
                examples.append((ex, label_masks[index]))

    # Generate input examples to the Transformer
    input_ids = []
    input_mask_array = []
    label_mask_array = []
    label_id_array = []

    # tokenization
    for (text, label_mask) in examples:
        encoded_sent = tokenizer.encode(text[0], add_special_tokens=True,
                                        max_length=max_seq_length, padding="max_length",
                                        truncation=True)
        input_ids.append(encoded_sent)
        label_id_array.append(label_map[text[1]])
        label_mask_array.append(label_mask)

    # Attention to token (to ignore padded input wordpieces)
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        input_mask_array.append(att_mask)

    # Convertion to Tensor
    input_ids = torch.tensor(input_ids)
    input_mask_array = torch.tensor(input_mask_array)
    label_id_array = torch.tensor(label_id_array, dtype=torch.long)
    label_mask_array = torch.tensor(label_mask_array)

    # Building the TensorDataset
    if return_ids:
        ids = torch.tensor(np.arange(input_ids.shape[0]))
        dataset = TensorDataset(ids, input_ids, input_mask_array, label_id_array, label_mask_array)
    else:
        dataset = TensorDataset(input_ids, input_mask_array, label_id_array, label_mask_array)
    if do_shuffle:
        sampler = RandomSampler
    else:
        sampler = SequentialSampler

    return DataLoader(dataset, sampler=sampler(dataset), batch_size=batch_size)