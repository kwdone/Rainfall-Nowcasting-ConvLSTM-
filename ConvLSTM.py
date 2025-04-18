import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        # Calculate padding to maintain the same size after convolution
        self.padding = (kernel_size - 1) // 2

        # Convolutional layer to generate the gates (input, forget, cell, output)
        self.conv = nn.Conv2d(input_channels + hidden_channels,
                              4 * hidden_channels, kernel_size, padding=self.padding)

        # GroupNorm to normalize the gates
        self.group_norm = nn.GroupNorm(4 * hidden_channels // 32, 4 * hidden_channels)

    def forward(self, inputs, hidden_state=None):
        batch_size, input_channels, height, width = inputs.size()

        if hidden_state is None:
            h = torch.zeros(batch_size, self.hidden_channels, height, width).to(inputs.device)
            c = torch.zeros(batch_size, self.hidden_channels, height, width).to(inputs.device)
        else:
            h, c = hidden_state

        # Concatenate the input and the previous hidden state (hx)
        combined = torch.cat((inputs, h), 1)

        # Apply convolution to get the gates
        gates = self.conv(combined)
        gates = self.group_norm(gates)  # Normalize the gates

        # Split the gates into individual components (input, forget, cell, output)
        ingate, forgetgate, cellgate, outgate = torch.split(gates, self.hidden_channels, dim=1)

        # Apply the activations
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        # Update the cell and hidden states
        c_new = forgetgate * c + ingate * cellgate
        h_new = outgate * torch.tanh(c_new)

        return h_new, c_new


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [ConvLSTMCell(input_channels if i == 0 else hidden_channels,
                          hidden_channels, kernel_size) for i in range(num_layers)]
        )

    def forward(self, inputs):
        batch_size, seq_len, input_channels, height, width = inputs.size()

        # Initialize hidden state
        hidden_states = [(torch.zeros(batch_size, self.layers[0].hidden_channels, height, width).to(inputs.device),
                          torch.zeros(batch_size, self.layers[0].hidden_channels, height, width).to(inputs.device))
                         for _ in range(self.num_layers)]

        # Loop through the sequence
        for t in range(seq_len):
            x_t = inputs[:, t, :, :, :]
            for layer, (h, c) in zip(self.layers, hidden_states):
                h, c = layer(x_t, (h, c))
                x_t = h
        return x_t

