import numpy as np
import json

import torch
from torch import nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Seq2seqConfig:
    def __init__(
        self,
        vocab_file,
        device,
        embedding_size=620,
        hidden_size=1000,
        maxout_hidden_size=500,
        padding_idx=0,
        start_idx=1,
        vocab_size=30522,
        layer_norm_eps=1e-12,
        embedding_p_dropout=0.1,
    ):
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.maxout_hidden_size = maxout_hidden_size
        self.padding_idx = padding_idx
        self.start_idx = start_idx
        self.vocab_file = vocab_file
        if self.vocab_file is not None:
            self.vocab_size = self.get_vocab_size(self.vocab_file)
        else:
            self.vocab_size = vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.embedding_p_dropout = embedding_p_dropout
        self.device = device

    @staticmethod
    def get_vocab_size(vocab_file):
        vocab = json.load(open(vocab_file, "r"))
        return len(vocab["model"]["vocab"])


class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.word_embedding = nn.Embedding(
            config.vocab_size, config.embedding_size, padding_idx=config.padding_idx
        )
        self.LayerNorm = nn.LayerNorm(config.embedding_size, config.layer_norm_eps)
        self.Dropout = nn.Dropout(config.embedding_p_dropout)

    def forward(self, input_ids):
        embeddings = self.word_embedding(input_ids)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.Dropout(embeddings)

        return embeddings


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.rnn = nn.LSTM(
            config.embedding_size,
            config.hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.embedding_p_dropout)
        self._init_weights()

    def _init_weights(
        self,
    ):
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            if "bias" in name:
                nn.init.zeros_(param)

        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()

    def forward(self, source_embedding, source_true_seq_length):

        pack_source = pack_padded_sequence(
            source_embedding,
            source_true_seq_length,
            batch_first=True,
            enforce_sorted=False,
        )
        outputs, (last_hidden_encoder, last_cell_state_encoder) = self.rnn(pack_source)

        # outputs = [batch size, seq len, hid dim * 2]
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        # last_backward_hidden = [batch size, hid dim]
        # last_backward_cell_state = [batch size, hid dim]
        last_backward_hidden = torch.tanh(self.fc(last_hidden_encoder[-1, :, :]))
        last_backward_cell_state = last_cell_state_encoder[-1, :, :]

        return outputs, last_backward_hidden, last_backward_cell_state


class AdditiveAttention(nn.Module):
    def __init__(self, config):
        super(AdditiveAttention, self).__init__()

        self.v = nn.parameter.Parameter(torch.zeros(1, config.hidden_size))
        self.U_a = nn.Linear(config.hidden_size * 2, config.hidden_size)  # Value
        self.W_a = nn.Linear(config.hidden_size, config.hidden_size)  # Query
        self.softmax = nn.Softmax(dim=2)
        self._init_weights()

    def _init_weights(self):
        self.U_a.weight.data.normal_(mean=0.0, std=0.001)
        self.W_a.weight.data.normal_(mean=0.0, std=0.001)
        self.U_a.bias.data.zero_()
        self.W_a.bias.data.zero_()

    def forward(
        self, encoder_output, last_decoder_output, src_attention_mask
    ):  # value, query

        Q = self.W_a(last_decoder_output)
        V = self.U_a(encoder_output)

        energy = torch.matmul(self.v, torch.tanh(Q + V).transpose(1, 2))
        energy.masked_fill_(src_attention_mask.unsqueeze(1) == 0, -1e10)
        probs = self.softmax(energy)

        # weigted = [batch size, 1, hid dim * 2]
        weigted = torch.sum((probs.transpose(1, 2) * encoder_output), dim=1)
        weigted = weigted.unsqueeze(1)
        return weigted


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.rnnUnit = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True)
        self.fc_word = nn.Linear(config.embedding_size, config.hidden_size)
        self.fc_context = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self._init_weights()

    def _init_weights(self):
        self.fc_word.weight.data.normal_(mean=0.0, std=0.01)
        self.fc_word.bias.data.zero_()
        self.fc_context.weight.data.normal_(mean=0.0, std=0.01)
        self.fc_context.bias.data.zero_()

        for name, param in self.rnnUnit.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            if "bias" in name:
                nn.init.zeros_(param)

    def forward(self, w_embedding, context_vector, last_hidden, last_cell_state):

        rnn_input = self.fc_word(w_embedding) + self.fc_context(context_vector)

        # output = [batch size, 1, hid dim]
        # last_hidden = [1, 1, hid dim]
        # last_cell_state = [1, 1, hid dim]
        output, (last_hidden, last_cell_state) = self.rnnUnit(
            rnn_input, (last_hidden, last_cell_state)
        )

        return output, last_hidden, last_cell_state


class Seq2seq(nn.Module):
    def __init__(self, config, device):
        super(Seq2seq, self).__init__()

        self.word_embeddings = Embedding(config)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.attention = AdditiveAttention(config)

        self.fc_dec_hid = nn.Linear(config.hidden_size, config.maxout_hidden_size * 2)
        self.fc_w_emb = nn.Linear(config.embedding_size, config.maxout_hidden_size * 2)
        self.fc_context = nn.Linear(
            config.hidden_size * 2, config.maxout_hidden_size * 2
        )
        self.fc_maxout = nn.Linear(config.maxout_hidden_size, config.vocab_size)

        self.dropout = nn.Dropout(config.embedding_p_dropout)
        self.LayerNorm = nn.LayerNorm(config.embedding_size, config.layer_norm_eps)
        self.soft = nn.Softmax(dim=2)
        self.config = config
        self.device = device
        self._init_weight()

    def _init_weight(self):
        self.fc_dec_hid.weight.data.normal_(mean=0.0, std=0.01)
        self.fc_dec_hid.bias.data.zero_()
        self.fc_w_emb.weight.data.normal_(mean=0.0, std=0.01)
        self.fc_w_emb.bias.data.zero_()
        self.fc_context.weight.data.normal_(mean=0.0, std=0.01)
        self.fc_context.bias.data.zero_()
        self.fc_maxout.weight.data.normal_(mean=0.0, std=0.01)
        self.fc_maxout.bias.data.zero_()

    def forward(
        self,
        source_ids,
        src_attention_mask,
        target_ids,
        trg_attention_mask,
        source_true_seq_length,
        forcing_teacher_rate=0.5,
    ):

        source_embedding = self.word_embeddings(source_ids)
        target_embedding = self.word_embeddings(target_ids)

        encoder_outputs, last_hidden, last_cell_state = self.encoder(
            source_embedding, source_true_seq_length
        )

        seq_length = target_embedding.size()[1]
        batch_size = target_embedding.size()[0]

        predictions = torch.zeros(
            [batch_size, seq_length, self.config.vocab_size], dtype=torch.float64
        ).to(self.device)

        decoder_input = target_embedding[:, 0].unsqueeze(1)
        decoder_output = last_hidden.unsqueeze(1)
        last_hidden = last_hidden.unsqueeze(0)
        last_cell_state = last_cell_state.unsqueeze(0)

        for idx in range(1, seq_length):
            context = self.attention(
                encoder_outputs, decoder_output, src_attention_mask
            )

            decoder_output, last_hidden, last_cell_state = self.decoder(
                decoder_input, context, last_hidden, last_cell_state
            )
            deep_output = (
                self.fc_dec_hid(decoder_output)
                + self.fc_w_emb(decoder_input)
                + self.fc_context(context)
            )
            deep_output = deep_output.reshape(
                batch_size, 1, self.config.maxout_hidden_size, 2
            )
            maxout = deep_output.max(dim=3).values

            prediction = self.fc_maxout(maxout)
            prediction_ids = prediction.argmax(2).squeeze(0)
            predictions[:, idx] = prediction.squeeze(1)

            if np.random.random_sample() > forcing_teacher_rate:
                decoder_input = target_embedding[:, idx].unsqueeze(1)

            else:
                decoder_input = self.word_embeddings(prediction_ids)

        return predictions


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
