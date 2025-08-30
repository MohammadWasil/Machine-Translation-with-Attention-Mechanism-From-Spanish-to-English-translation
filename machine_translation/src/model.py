from __future__ import annotations

import random

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class Encoder(nn.Module):
    def __init__(self, embedding_size_spanish, embedding_dim, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.embedding_size_english = embedding_size_spanish
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim

        self.embedding = nn.Embedding(self.embedding_size_english, self.embedding_dim)

        self.rnn = nn.GRU(self.embedding_dim, self.encoder_hidden_dim, bidirectional=True)

        self.linear = nn.Linear(self.encoder_hidden_dim * 2, self.decoder_hidden_dim)

        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        # WRITE CODE HERE
        # input's shape : [sequence_len, batch_size]
        embeds = self.dropout(self.embedding(inputs))
        # https://stackoverflow.com/questions/49466894/how-to-correctly-give-inputs-to-embedding-lstm-and-linear-layers-in-pytorch
        # output of the embedding is [seq_len, batch_size, embedding_size]

        # input: [seq_len, batch_size, embedding_size] # noqa: ERA001
        rnn_enc, hidden = self.rnn(embeds)
        # rnn out shape : [sequence_len, batch_size, encoder_hidden_dim*2]
        # hidden shape   : [2, batch_size, encoder_hidden_dim]

        # concatenate both forward and backward hidden vectors
        hidden_f_b = torch.cat((hidden[0, :, :], hidden[1, :, :]), dim=1)
        # output shape: [batch_size, encoder_hidden_dim*2]

        # input: [batch_size, encoder_hidden_dim*2] # noqa: ERA001
        hidden_enc = self.linear(hidden_f_b)
        # output: [batch_size, decoder_hidden_dim] # noqa: ERA001

        hidden_enc = torch.tanh(hidden_enc)

        return rnn_enc, hidden_enc


class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear((encoder_hidden_dim * 2) + decoder_hidden_dim, decoder_hidden_dim)
        self.fc2 = nn.Linear(decoder_hidden_dim, 1)

    def forward(self, decoder_hidden, encoder_output):
        # WRITE CODE HERE

        # encoder_output shape : [sequence_len, batch_size, encoder_hidden_dim*2]
        # decoder_hidden shape : [batch_size, decoder_hidden_dim]

        # repeat decoder hidden state sequence_len Time's.
        # input shape : [batch_size, decoder_hidden_dim]
        decoder_hidden = torch.unsqueeze(decoder_hidden, 1)
        # output shape: [batch_size, 1, decoder_hidden_dim]

        decoder_hidden = decoder_hidden.repeat(1, encoder_output.shape[0], 1)
        # [batch_size, sequence_len, decoder_hidden_dim] # noqa: ERA001

        encoder_output = encoder_output.permute(1, 0, 2)
        # encoder_output- shape : [batch_size, sequence_len, encoder_hidden_dim*2]

        # concatenate encoder's output and decoder's hidden state
        # and feed into a neural network layer
        concat = torch.cat((encoder_output, decoder_hidden), dim=2)
        # shape- [batch_size, sequence_len, (encoder_hidden_dim*2) + decoder_hidden_dim], [batch_size, sequence_len, 3000] # noqa: ERA001

        # input shape: [batch_size, sequence_len, (encoder_hidden_dim*2) + decoder_hidden_dim]
        fc1 = self.fc1(concat)
        # output: [batch_size, sequence_len, decoder_hidden_dim] # noqa: ERA001

        fc1 = torch.tanh(fc1)
        # output: [batch_size, sequence_len, decoder_hidden_dim] # noqa: ERA001

        fc2 = self.fc2(fc1)
        # output: [batch_size, sequence_len, 1] # noqa: ERA001

        alpha = F.softmax(fc2, dim=1)
        # output: [batch_size, sequence_len, 1] # noqa: ERA001

        # attention vector to take the weighted sum of the encoder hidden state.
        alpha = alpha.permute(0, 2, 1)
        # alpha shape:            [batch_size, 1, sequence_len]
        # encoder_output- shape : [batch_size, sequence_len, encoder_hidden_dim*2]
        # https://pytorch.org/docs/stable/generated/torch.bmm.html

        return alpha @ encoder_output  # multiplying all the words in each sequence, from 1..N, wher N=sequence len.
        # [batch_size, 1, sequence_len] * [batch_size, sequence_len, encoder_hidden_dim*2] # noqa: ERA001
        # attention- shape : [batch_size, 1, encoder_hidden_dim*2]


class Decoder(nn.Module):
    def __init__(self, embedding_size_english, embedding_dim, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.embedding_size_english = embedding_size_english
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim

        self.attention = Attention(self.decoder_hidden_dim, self.encoder_hidden_dim)

        self.embedding = nn.Embedding(self.embedding_size_english, self.embedding_dim)

        self.rnn = nn.GRU(
            (self.encoder_hidden_dim * 2) + self.embedding_dim, self.decoder_hidden_dim, bidirectional=False
        )

        self.linear = nn.Linear(
            self.embedding_dim + (self.encoder_hidden_dim * 2) + self.decoder_hidden_dim, self.embedding_size_english
        )

        self.dropout = nn.Dropout(0.5)

    def forward(self, target_lang, decoder_hidden, encoder_output):
        # WRITE CODE HERE
        # input shape: [BATCH_SIZE]
        embeds = self.dropout(self.embedding(target_lang))
        # embeds shape: [BATCH_SIZE, embedding_dim]

        # attention layer - get the context vector from the attention layer.
        # encoder_output: [sequence_len, batch_size, encoder_hidden_dim*2] # noqa: ERA001
        # decoder_hidden: [batch_size, decoder_hidden_dim] # noqa: ERA001
        a = self.attention(decoder_hidden, encoder_output)
        # attention shape: [BATCH_SIZE, 1, encoder_hidden_dim*2]

        ### concatenate convext vector, "embeds" and attention output, "a".

        # expand 1 dim for embeds
        # input shape: [BATCH_SIZE, embedding_dim]
        embeds = embeds.unsqueeze(0)
        # output shape: [1, BATCH_SIZE, embedding_dim]

        # change the shape of context vector.
        # input shape:  [BATCH_SIZE, 1, encoder_hidden_dim*2]
        a = a.permute(1, 0, 2)
        # output shape: [1, BATCH_SIZE, encoder_hidden_dim*2]

        x = torch.cat((embeds, a), dim=2)
        # output shape: [1, BATCH_SIZE, encoder_hidden_dim*2+embedding_dim] i.e. [1, 16, 2030]

        # input shape: [1, BATCH_SIZE, encoder_hidden_dim*2+embedding_dim]
        rnn_dec, decoder_hidden = self.rnn(x, decoder_hidden.unsqueeze(0))
        # rnn_dec shape: [1, BATCH_SIZE, decoder_hidden_dim], *1 since it is not bidirectional.
        # hidden.shape: [1, BATCH_SIZE, decoder_hidden_dim], hidden[0] = 1, since it is not bidirectional.

        # prediction
        # input to the nn: embeds, a, rnn_dec
        # embeds output shape:    [1, BATCH_SIZE, embedding_dim]
        # attention output shape: [1, BATCH_SIZE, encoder_hidden_dim*2]
        # rnn_dec shape:          [1, BATCH_SIZE, decoder_hidden_dim]

        c1 = torch.cat((rnn_dec, a, embeds), dim=2)
        # c1 shape: [1, BATCH_SIZE, embedding_dim+(encoder_hidden_dim*2)+decoder_hidden_dim]

        pred = self.linear(c1).squeeze(0)
        # pred's shape: [BATCH_SIZE, embedding_size_spanish]

        return pred, decoder_hidden.squeeze(0)


class Model(nn.Module):
    def __init__(
        self,
        embedding_size_spanish,
        embedding_size_english,
        embedding_dim,
        encoder_hidden_dim,
        decoder_hidden_dim,
    ):
        super().__init__()
        self.encoder = Encoder(embedding_size_spanish, embedding_dim, encoder_hidden_dim, decoder_hidden_dim)
        self.decoder = Decoder(embedding_size_english, embedding_dim, encoder_hidden_dim, decoder_hidden_dim)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # batch_size = src.shape[1] # noqa: ERA001
        trg_len = trg.shape[0]
        # trg_vocab_size = embedding_size_english # noqa: ERA001

        outputs = []

        encoder_outputs, encoder_hidden = self.encoder(src)
        # the encoder' hidden state becomes decoder's hidden state at time step = 0 (i.e. when starting the decoding phase)
        decoder_hidden = encoder_hidden

        input_text = trg[0, :]

        for t in range(trg_len):
            output, decoder_hidden = self.decoder(input_text, decoder_hidden, encoder_outputs)
            # output shape:         [BATCH_SIZE, embedding_size_spanish]
            # decoder_hidden shape: [BATCH_SIZE, decoder_hidden_dim]

            if t == 0:
                output = torch.zero_(output)
            outputs.append(output)

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio  # noqa: S311

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            input_text = trg[t] if teacher_force else top1

        # converts list to torch tensor
        # outputs shape: [sequence_length, BATCH_SIZE, embedding_size_spanish]
        return torch.stack(outputs)
