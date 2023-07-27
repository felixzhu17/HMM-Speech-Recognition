import math
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:,x.size(1),:]
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, encoding_dim = 32, n_heads = 4, dropout = 0.2):
        super().__init__()
        self.positional_encoding = PositionalEncoding(input_dim, dropout)
        self.fc1 = nn.Linear(input_dim, encoding_dim)
        self.attn_1 = nn.MultiheadAttention(encoding_dim, n_heads, dropout, batch_first=True)
        self.fc2 = nn.Linear(encoding_dim, encoding_dim)
        self.attn_2 = nn.MultiheadAttention(encoding_dim, n_heads, dropout, batch_first=True)
        self.ln = nn.LayerNorm(encoding_dim)
        self.fc3 = nn.Linear(encoding_dim, output_dim)
        self.fc4 = nn.Linear(output_dim, output_dim)

    def forward(self, x, mask = None):
        x = self.positional_encoding(x)
        x = F.relu(self.fc1(x))
        x, _ = self.attn_1(x, x, x, key_padding_mask = mask == False if mask is not None else None)
        x = F.relu(self.ln(self.fc2(x)))
        x, _ = self.attn_2(x, x, x, key_padding_mask = mask == False if mask is not None else None)
        x = x.mean(axis = 1)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x