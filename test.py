import os
import time
import torch
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', required=True)
# parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--eval_neg_sample', default=1000, type=int)
parser.add_argument('--backbone', default='mamba', type=str)
parser.add_argument('--name', default='mamba', type=str)
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--maxlen', default=20, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=501, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--state_dict_path', default=None, type=str)
args = parser.parse_args()


class GRU4Rec(torch.nn.Module):
    def __init__(self, itemnum,args):
        super(GRU4Rec,  self).__init__()
        gru_layers = 1
        d_model = args.hidden_units
        self.max_length = args.maxlen
        self.token_emb = torch.nn.Embedding(itemnum, d_model)
        self.dropout_prob = args.dropout_rate
        self.token_emb_dropout = torch.nn.Dropout(self.dropout_prob)
        self.gru = torch.nn.GRU(
            input_size=args.hidden_units,
            hidden_size=args.hidden_units,
            num_layers=gru_layers,
            batch_first=True,
            bias=False
        )
        self.fc = torch.nn.Linear(d_model, itemnum)



    def forward(self, seq_token, seq_pos):
        # Supervised Head
        seq_embeddings = self.token_emb(seq_token)
        seq_pos = seq_pos.to('cuda')
        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(seq_embeddings, seq_pos, batch_first=True,
                                                             enforce_sorted=False)
        emb_packed, hidden = self.gru(emb_packed)
        output = hidden[-1].view(-1, hidden[-1].shape[1])
        # output = hidden.view(-1, hidden.shape[2])
        # output = self.fc(hidden)
        return output



    def predict(self, seq_token, seq_pos):
        # Supervised Head
        seq_embeddings = self.token_emb(seq_token)
        seq_pos = seq_pos.to('cuda')
        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(seq_embeddings, seq_pos, batch_first=True,
                                                             enforce_sorted=False)
        emb_packed, hidden = self.gru(emb_packed)
        output = hidden[-1].view(-1, hidden[-1].shape[1])
        # output = hidden.view(-1, hidden.shape[2])
        # output = self.fc(hidden)
        return output
    



gru = GRU4Rec(5,args)


a = torch.tensor([[1,2,3,4,5]])
print(gru(a,5))