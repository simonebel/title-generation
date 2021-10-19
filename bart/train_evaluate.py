import math

import torch
from torch.nn.utils import clip_grad_norm_

from seq2seq import compute_rouge_score_on_minibatch


def train(
    model,
    device,
    batch_size,
    accumulation_steps,
    train_dataset,
    train_data_loader,
    optimizer,
):
    clip_norm = 0.1
    epoch_loss, nb_tr_steps = 0, 0

    model.train()
    tot_iteration = math.ceil(train_dataset.__len__() / batch_size)

    for step, (source_ids, target_ids, source_att_mask, target_att_mask) in enumerate(
        train_data_loader
    ):
        source_ids, target_ids, source_att_mask = (
            source_ids.to(device),
            target_ids.to(device),
            source_att_mask.to(device),
        )

        _, current_loss = model(source_ids, target_ids, source_att_mask)
        epoch_loss += current_loss.item() / accumulation_steps
        current_loss.backward()

        if (step + 1) % accumulation_steps == 0:

            optimizer.zero_grad()
            clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        nb_tr_steps += 1

    return (epoch_loss * accumulation_steps) / nb_tr_steps


def evaluate(model, device, dev_data_loader, tokenizer):
    epoch_loss, epoch_rouge, nb_dev_steps = 0, 0, 0
    beam_size = 3

    model.eval()
    with torch.no_grad():
        for source_ids, target_ids, source_att_mask, target_att_mask in dev_data_loader:
            source_ids, target_ids, source_att_mask = (
                source_ids.to(device),
                target_ids.to(device),
                source_att_mask.to(device),
            )

            _, loss = model(source_ids, target_ids, source_att_mask)

            detokenize_candidates = model.evaluate(
                source_ids, tokenizer, beam_size, 100
            )
            ground_truth_target = tokenizer.decode(target_ids)
            rouge_score = compute_rouge_score_on_minibatch(
                detokenize_candidates, ground_truth_target, beam_size
            )

            epoch_loss += loss.item()
            epoch_rouge += rouge_score
            nb_dev_steps += 1

    return epoch_loss / nb_dev_steps, epoch_rouge / nb_dev_steps


def test(model, device, test_data_loader, tokenizer, beam_size=3):
    epoch_rouge, nb_dev_steps = 0, 0
    model.eval()
    with torch.no_grad():
        for (
            source_ids,
            target_ids,
            source_att_mask,
            target_att_mask,
        ) in test_data_loader:
            source_ids, target_ids, source_att_mask = (
                source_ids.to(device),
                target_ids.to(device),
                source_att_mask.to(device),
            )

            detokenize_candidates = model.evaluate(source_ids, beam_size, 100)
            ground_truth_target = tokenizer.decode(target_ids)
            rouge_score = compute_rouge_score_on_minibatch(
                detokenize_candidates, ground_truth_target, beam_size
            )

            epoch_rouge += rouge_score
            nb_dev_steps += 1

    return epoch_rouge / nb_dev_steps
