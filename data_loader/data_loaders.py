import math
import datetime
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from base import *

label_list = ["UNK_UNK", "ABBR_abb", "ABBR_exp", "DESC_def", "DESC_desc",
              "DESC_manner", "DESC_reason", "ENTY_animal", "ENTY_body",
              "ENTY_color", "ENTY_cremat", "ENTY_currency", "ENTY_dismed",
              "ENTY_event", "ENTY_food", "ENTY_instru", "ENTY_lang",
              "ENTY_letter", "ENTY_other", "ENTY_plant", "ENTY_product",
              "ENTY_religion", "ENTY_sport", "ENTY_substance", "ENTY_symbol",
              "ENTY_techmeth", "ENTY_termeq", "ENTY_veh", "ENTY_word", "HUM_desc",
              "HUM_gr", "HUM_ind", "HUM_title", "LOC_city", "LOC_country",
              "LOC_mount", "LOC_other", "LOC_state", "NUM_code", "NUM_count",
              "NUM_date", "NUM_dist", "NUM_money", "NUM_ord", "NUM_other",
              "NUM_perc", "NUM_period", "NUM_speed", "NUM_temp", "NUM_volsize",
              "NUM_weight"]


def get_qc_examples(input_file):
    """Creates examples for the training and dev sets."""
    examples = []

    with open(input_file, 'r') as f:
        contents = f.read()
        file_as_list = contents.splitlines()
        for line in file_as_list[1:]:
            split = line.split(" ")
            question = ' '.join(split[1:])

            text_a = question
            inn_split = split[0].split(":")
            label = inn_split[0] + "_" + inn_split[1]
            examples.append((text_a, label))
        f.close()

    return examples


def generate_data_loader(input_examples, label_masks, label_map, batch_size, tokenizer,
                         max_seq_length, do_shuffle=False, balance_label_examples=False):
    '''
    Generate a Dataloader given the input examples, eventually masked if they are
    to be considered NOT labeled.
    '''
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
            # IT SIMULATE A LABELED EXAMPLE
            if label_masks[index]:
                balance = int(1 / label_mask_rate)
                balance = int(math.log(balance, 2))
                if balance < 1:
                    balance = 1
                for b in range(0, int(balance)):
                    examples.append((ex, label_masks[index]))
            else:
                examples.append((ex, label_masks[index]))

    # -----------------------------------------------
    # Generate input examples to the Transformer
    # -----------------------------------------------
    input_ids = []
    input_mask_array = []
    label_mask_array = []
    label_id_array = []

    # Tokenization
    for (text, label_mask) in examples:
        encoded_sent = tokenizer.encode(text[0], add_special_tokens=True, max_length=max_seq_length,
                                        padding="max_length", truncation=True)
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
    dataset = TensorDataset(input_ids, input_mask_array, label_id_array, label_mask_array)

    if do_shuffle:
        sampler = RandomSampler
    else:
        sampler = SequentialSampler

    return DataLoader(dataset, sampler=sampler(dataset), batch_size=batch_size)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# class MnistDataLoader(BaseDataLoader):
#     """
#     MNIST data loading demo using BaseDataLoader
#     """
#     def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
#         trsfm = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])
#         self.data_dir = data_dir
#         self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
#         super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
