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

    source = source.tolist()
    target = target.tolist()
    len_vect = np.vectorize(len)
    source_encoding = tokenizer(source)["input_ids"]
    target_encoding = tokenizer(target)["input_ids"]

    source_max_len = np.max([len(k) for k in source_encoding])
    target_max_len = np.max([len(k) for k in target_encoding])

    if source_max_len > 1024:
        source_max_len = 1024

    if target_max_len > 1024:
        target_max_len = 1024

    source_encoding = tokenizer(
        source, truncation=True, padding="max_length", max_length=source_max_len
    )
    target_encoding = tokenizer(
        target, truncation=True, padding="max_length", max_length=target_max_len
    )

    source_ids = source_encoding["input_ids"]
    target_ids = target_encoding["input_ids"]
    source_att_mask = source_encoding["attention_mask"]
    target_att_mask = target_encoding["attention_mask"]

    source_ids = torch.LongTensor(source_ids)
    target_ids = torch.LongTensor(target_ids)
    source_att_mask = torch.LongTensor(source_att_mask)
    target_att_mask = torch.LongTensor(target_att_mask)

    return source_ids, target_ids, source_att_mask, target_att_mask


def generate_data_loader(batch_size, train_path, dev_path, test_path, tokenizer):

    train_df = pkl.load(open(train_path, "rb"))
    dev_df = pkl.load(open(dev_path, "rb"))
    test_df = pkl.load(open(test_path, "rb"))

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
