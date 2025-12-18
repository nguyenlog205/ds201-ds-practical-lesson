import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        hidden = hidden.view(self.n_layers, 2, -1, self.hid_dim)
        cell = cell.view(self.n_layers, 2, -1, self.hid_dim)
        
        hidden = torch.sum(hidden, dim=1)
        cell = torch.sum(cell, dim=1)
        
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.W = nn.Linear(hid_dim, hid_dim * 2)

    def forward(self, decoder_hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        hidden = decoder_hidden.unsqueeze(1)
        project_hidden = self.W(hidden)
        encoder_outputs = encoder_outputs.permute(1, 2, 0)
        attention_scores = torch.bmm(project_hidden, encoder_outputs)
        return F.softmax(attention_scores, dim=2)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.fc_concat = nn.Linear(hid_dim * 3, hid_dim)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(0)
        
        embedded = self.dropout(self.embedding(input))

        rnn_output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        a = self.attention(rnn_output.squeeze(0), encoder_outputs) 

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        weighted = torch.bmm(a, encoder_outputs)
        
        rnn_output = rnn_output.squeeze(0)
        weighted = weighted.squeeze(1)
        
        concat_input = torch.cat((rnn_output, weighted), dim=1)

        concat_output = torch.tanh(self.fc_concat(concat_input)) 

        prediction = self.fc_out(concat_output) 
        
        return prediction, hidden, cell

class Model03(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src)

        input = trg[0,:]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            
            input = trg[t] if teacher_force else top1
            
        return outputs