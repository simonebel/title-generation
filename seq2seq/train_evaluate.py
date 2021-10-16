import numpy as np
import math

import plotly.graph_objects as go

import torch
from torch.nn.utils import clip_grad_norm_
from rouge_score import rouge_scorer


def train(
    model, device, train_data_loader, optimizer, config, loss_fn, clip, scheduler=None
):
    epoch_loss, nb_tr_steps = 0, 0
    model.train()
    for (
        source_ids,
        src_attention_mask,
        target_ids,
        trg_attention_mask,
        source_true_seq_length,
        target_true_seq_length,
    ) in train_data_loader:

        source_ids, src_attention_mask, target_ids, trg_attention_mask = (
            source_ids.to(device),
            src_attention_mask.to(device),
            target_ids.to(device),
            trg_attention_mask.to(device),
        )
        source_true_seq_length, target_true_seq_length = (
            source_true_seq_length.cpu().detach(),
            target_true_seq_length.cpu().detach(),
        )

        # Prevent the gradient to be accumulated over epochs
        optimizer.zero_grad()

        predictions = model(
            source_ids,
            src_attention_mask,
            target_ids,
            trg_attention_mask,
            source_true_seq_length,
        )

        target_ids = target_ids[:, 1:].reshape(-1)
        predictions = predictions[:, 1:].reshape(-1, config.vocab_size)

        loss = loss_fn(predictions, target_ids)

        loss.backward()

        clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
        nb_tr_steps += 1

        del loss, predictions, target_ids

    return epoch_loss / nb_tr_steps


def greedy_search_decoder():
    pass


def left_to_right_beam_search_decoder(
    predictions, beam_size, seq_length, batch_size, config
):

    neg_log_predictions = -torch.log(predictions)

    last_conditioned_proba = neg_log_predictions[:, 0, :]
    topk_last_conditioned_proba, last_token_indices = torch.topk(
        last_conditioned_proba, beam_size, largest=False
    )
    topk_last_conditioned_proba = topk_last_conditioned_proba.unsqueeze(-1)

    candidate_ids = torch.empty(
        *(batch_size, beam_size, 1), dtype=torch.int64, device=config.device
    )
    candidate_ids[:, :, 0] = config.start_idx

    candidate_ids = torch.cat((candidate_ids, last_token_indices.unsqueeze(2)), 2)

    for index in range(seq_length - 1):

        next_log_proba = (
            neg_log_predictions[:, index + 1, :].unsqueeze(1).repeat(1, beam_size, 1)
        )
        conditioned_proba = topk_last_conditioned_proba + next_log_proba

        topk_conditioned_proba, token_indices = conditioned_proba.view(
            batch_size, -1
        ).topk(beam_size, largest=False)
        token_indices = token_indices % (config.vocab_size - 1)

        candidate_ids = torch.cat((candidate_ids, token_indices.unsqueeze(2)), 2)

        topk_last_conditioned_proba = topk_conditioned_proba.unsqueeze(-1)

    return candidate_ids


def detokenize_candidate_target(tokenizer, batch_size, candidates_ids, target_ids):

    detokenize_candidates = []
    for index in range(batch_size):

        beam_detokenize_candidates = tokenizer.decode_batch(
            candidates_ids[index, :, :].cpu().numpy()
        )
        detokenize_candidates.append(beam_detokenize_candidates)

    ground_truth_target = tokenizer.decode_batch(target_ids.cpu().numpy())

    return detokenize_candidates, ground_truth_target


def evaluate(
    model, device, tokenizer, dev_data_loader, config, loss_fn, beam_size, batch_size
):

    epoch_loss, epoch_rouge, nb_dev_steps = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for (
            source_ids,
            src_attention_mask,
            target_ids,
            trg_attention_mask,
            source_true_seq_length,
            target_true_seq_length,
        ) in dev_data_loader:
            source_ids, src_attention_mask, target_ids, trg_attention_mask = (
                source_ids.to(device),
                src_attention_mask.to(device),
                target_ids.to(device),
                trg_attention_mask.to(device),
            )
            source_true_seq_length, target_true_seq_length = (
                source_true_seq_length.cpu().detach(),
                target_true_seq_length.cpu().detach(),
            )

            predictions = model(
                source_ids,
                src_attention_mask,
                target_ids,
                trg_attention_mask,
                source_true_seq_length,
            )

            target_ids = target_ids[:, 1:]
            predictions = predictions[:, 1:]

            flat_target_ids = target_ids.reshape(-1)
            flat_predictions = predictions.reshape(-1, config.vocab_size)

            loss = loss_fn(flat_predictions, flat_target_ids)

            candidates_ids = left_to_right_beam_search_decoder(
                predictions,
                beam_size,
                predictions.size()[1],
                predictions.size()[0],
                config,
            )
            detokenize_candidates, ground_truth_target = detokenize_candidate_target(
                tokenizer, predictions.size()[0], candidates_ids, target_ids
            )
            print(detokenize_candidates)

            rouge_score = compute_rouge_score_on_minibatch(
                detokenize_candidates, ground_truth_target, beam_size
            )
            epoch_loss += loss.item()
            epoch_rouge += rouge_score
            nb_dev_steps += 1

    return epoch_loss / nb_dev_steps, epoch_rouge / nb_dev_steps


def compute_rouge_score_on_minibatch(hyps, refs, beam_size):

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"])
    rouge_score = []
    for idc, ref in enumerate(refs):
        bean_hyp = hyps[idc]
        partial_rougeL = 0
        for hyp in bean_hyp:
            rouges = scorer.score(ref, hyp)
            rougeL = rouges["rougeL"].fmeasure
            partial_rougeL += rougeL

        rouge_score.append(partial_rougeL / len(bean_hyp))

    return np.mean(rouge_score)


def plot_loss(train_loss_set, dev_loss_set, dev_bleu_set, epochs):

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[i for i in range(1, len(train_loss_set) + 1)],
            y=train_loss_set,
            mode="lines+markers",
            name="Train Loss",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[i for i in range(1, len(dev_loss_set) + 1)],
            y=dev_loss_set,
            mode="lines+markers",
            name="Validation Loss",
        )
    )
    fig.update_layout(
        xaxis_title="Epochs",
        yaxis_title="Loss Function",
        title=f"Evolution of the Loss Function during {epochs} epochs",
        title_x=0.5,
    )

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=[i for i in range(1 + len(dev_bleu_set) + 1)],
            y=dev_bleu_set,
            mode="lines+markers",
            name="Developpement ROUGE Score",
        )
    )
    fig2.update_layout(
        xaxis_title="Epochs",
        yaxis_title="BLEU Score",
        title=f"Evolution of the ROUGE Score during {epochs} epochs",
        title_x=0.5,
    )
    fig.show()
    fig2.show()


def set_step_size(train_dataset_len, batch_size, epochs_factor=2):
    epoch_iterations = math.ceil(train_dataset_len / batch_size)

    step_size = epochs_factor * epoch_iterations

    return step_size


def plot_lr_vs_accuracy(lr_train_set, dev_bleu_set):

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=lr_train_set, y=dev_bleu_set, mode="lines+markers", name="Learning Rate"
        )
    )
    fig.update_layout(
        xaxis_title="Learning Rate",
        yaxis_title="BLEU Score",
        title="BLEU Score as a function of increasing learning rate for 8 epochs (LR range test)",
        title_x=0.5,
    )
    fig.update_xaxes(type="log")

    fig.show()
