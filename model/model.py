"""
This module defines an LSTM (Long Short-Term Memory) neural network model for sequence generation.

Classes:
    - LSTM(nn.Module): A custom LSTM class inheriting from PyTorch's nn.Module.
        Methods:
            - __init__(self, input_size, hidden_size, num_layers, output_size): Initializes the LSTM model.
            - forward(self, x, hidden, cell): Defines the forward pass of the model.
            - init_hidden(self, batch_size=1): Initializes the hidden and cell states of the model.

Usage:
    import torch
    import torch.nn as nn
    from lstm_model import LSTM

    # Example usage to create an LSTM model
    input_size = 50
    hidden_size = 128
    num_layers = 2
    output_size = 26  # Assuming an output size of 26 for an alphabet-based task

    lstm_model = LSTM(input_size, hidden_size, num_layers, output_size)

    # Example forward pass
    input_sequence = torch.randint(0, input_size, (1, 10))  # Example input sequence of length 10
    initial_hidden_state, initial_cell_state = lstm_model.init_hidden()
    output_sequence, final_states = lstm_model(input_sequence, initial_hidden_state, initial_cell_state)
"""
import torch
import torch.nn as nn

class LSTM(nn.Module):
    """
    Custom Long Short-Term Memory (LSTM) neural network model for sequence generation.

    Args:
        input_size (int): Size of the input vocabulary.
        hidden_size (int): Size of the hidden state.
        num_layers (int): Number of LSTM layers.
        output_size (int): Size of the output vocabulary.

    Attributes:
        hidden_size (int): Size of the hidden state.
        num_layers (int): Number of LSTM layers.
        device (str): Device used for training ('cuda' if available, 'cpu' otherwise).
        embedding (nn.Embedding): Embedding layer for input sequences.
        lstm (nn.LSTM): LSTM layer for sequence modeling.
        fc (nn.Linear): Fully connected layer for output prediction.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x, hidden, cell):
        """
        Forward pass of the LSTM model.

        Args:
            x (torch.Tensor): Input sequence tensor.
            hidden (torch.Tensor): Initial hidden state.
            cell (torch.Tensor): Initial cell state.

        Returns:
            torch.Tensor: Output tensor from the model.
            tuple: Tuple containing updated hidden and cell states.
        """
        output = self.embedding(x)
        output, (hidden, cell) = self.lstm(output.unsqueeze(1), (hidden, cell))
        output = self.fc(output.reshape(output.shape[0], -1))
        return output, (hidden, cell)


    def init_hidden(self, batch_size=1):
        """
        Initialize the hidden and cell states of the LSTM model.

        Args:
            batch_size (int, optional): Batch size for initialization. Defaults to 1.

        Returns:
            tuple: Tuple containing the initialized hidden and cell states.
        """
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return hidden, cell