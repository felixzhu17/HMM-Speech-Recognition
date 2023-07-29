import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:, x.size(1), :]
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, encoding_dim=64, n_heads=4, dropout=0.2):
        super().__init__()
        self.positional_encoding = PositionalEncoding(input_dim, dropout)
        self.transformation_1 = nn.Sequential(
            nn.Linear(input_dim, encoding_dim), nn.ReLU(), nn.LayerNorm(encoding_dim)
        )
        self.attn_1 = nn.MultiheadAttention(
            encoding_dim, n_heads, dropout, batch_first=True
        )
        self.transformation_2 = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim), nn.ReLU(), nn.LayerNorm(encoding_dim)
        )
        self.attn_2 = nn.MultiheadAttention(
            encoding_dim, n_heads, dropout, batch_first=True
        )
        self.transformation_3 = nn.Sequential(
            nn.Linear(encoding_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x, mask=None):
        key_padding_mask = mask == False if mask is not None is True else None

        x = self.positional_encoding(x)
        x = self.transformation_1(x)
        x, _ = self.attn_1(x, x, x, key_padding_mask=key_padding_mask)

        x = self.transformation_2(x)
        x, _ = self.attn_2(x, x, x, key_padding_mask=key_padding_mask)

        x = x.mean(axis=1)
        
        x = self.transformation_3(x)
        return x


def train_model(model, train_data, num_epochs=50, learning_rate=0.001, batch_size=32):
    """
    Trains a PyTorch model.

    Args:
        model (torch.nn.Module): The model to train.
        train_data (torch.utils.data.DataLoader): The data to train on.
        num_epochs (int, optional): The number of epochs to train for. Defaults to 50.
        learning_rate (float, optional): The learning rate for the Adam optimizer. Defaults to 0.001.
        batch_size (int, optional): The size of the batches for training. Defaults to 32.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Wrap your train_data in a DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        epoch_loss = 0.0  # initialize loss for this epoch
        num_batches = 0  # keep track of number of batches processed

        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()  # zero the parameter gradients

            outputs = model(x, mask)  # forward pass, get the output of the network

            loss = criterion(outputs, y)  # calculate the loss
            loss.backward()  # backward pass, compute gradient of the loss with respect to model parameters
            optimizer.step()  # update model parameters

            epoch_loss += loss.item()  # accumulate batch loss
            num_batches += 1  # increment batch count

        avg_epoch_loss = epoch_loss / num_batches  # calculate average epoch loss
        print(f"Epoch: {epoch+1}, Loss: {avg_epoch_loss:.4f}")
