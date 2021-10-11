from tokenizers import Tokenizer

import pandas as pd
import numpy as np
import pickle as pkl

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler


def generate_examples(source, target):

    source = np.expand_dims(source, 0)
    target = np.expand_dims(target, 0)

    examples = np.append(source.T, target.T, axis=1)

    return examples


class build_dataset(Dataset):
    def __init__(self, source, target):

        self.examples = generate_examples(source, target)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def batchify_feature(batch, tokenizer):

    source = np.array(batch)[:, 0]
    target = np.array(batch)[:, 1]

    encoding_source = tokenizer.encode_batch(source)
    encoding_target = tokenizer.encode_batch(target)

    source_ids = [encoding.ids for encoding in encoding_source]
    target_ids = [encoding.ids for encoding in encoding_target]

    source_true_seq_length = [
        np.sum(encoding.attention_mask) for encoding in encoding_source
    ]
    target_true_seq_length = [
        np.sum(encoding.attention_mask) for encoding in encoding_target
    ]

    src_attention_mask = [encoding.attention_mask for encoding in encoding_source]
    trg_attention_mask = [encoding.attention_mask for encoding in encoding_target]

    source_ids = torch.LongTensor(source_ids)
    target_ids = torch.LongTensor(target_ids)

    source_true_seq_length = torch.LongTensor(source_true_seq_length)
    target_true_seq_length = torch.LongTensor(target_true_seq_length)

    src_attention_mask = torch.LongTensor(src_attention_mask)
    trg_attention_mask = torch.LongTensor(trg_attention_mask)

    return (
        source_ids,
        src_attention_mask,
        target_ids,
        trg_attention_mask,
        source_true_seq_length,
        target_true_seq_length,
    )


def generate_data_loader(batch_size, train_path, dev_path, test_path, tokenizer):

    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

    train_df = pkl.load(open(train_path, "rb"))
    dev_df = pkl.load(open(dev_path, "rb"))
    test_df = pkl.load(open(test_path, "rb"))

    train_df = train_df.head(512)
    dev_df = dev_df.head(64)
    test_df = test_df.head(64)

    train_dataset = build_dataset(
        train_df["abstract"].to_numpy(), train_df["title"].to_numpy()
    )
    train_sampler = RandomSampler(train_dataset)
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=lambda x: batchify_feature(x, tokenizer),
    )

    dev_dataset = build_dataset(
        dev_df["abstract"].to_numpy(), dev_df["title"].to_numpy()
    )
    dev_sampler = RandomSampler(dev_dataset)
    dev_data_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        sampler=dev_sampler,
        collate_fn=lambda x: batchify_feature(x, tokenizer),
    )

    test_dataset = build_dataset(
        test_df["abstract"].to_numpy(), test_df["title"].to_numpy()
    )
    test_sampler = SequentialSampler(test_dataset)
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        collate_fn=lambda x: batchify_feature(x, tokenizer),
    )

    return (
        train_dataset,
        train_data_loader,
        dev_dataset,
        dev_data_loader,
        test_dataset,
        test_data_loader,
    )
