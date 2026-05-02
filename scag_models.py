import torch
import torch.nn as nn
import math


class ParametrizedMLP(nn.Module):
    def __init__(
            self,
            input_size: int = 64,
            num_classes: int = 8,
            layer_number: int = 2,
            layer_width: int = 128,
            dropout_prob: float = 0.5,
            use_batchnorm: bool = True
    ):
        """
        :param input_size: Size of the input vector (e.g., 16*4=64)
        :param num_classes: Number of classes
        :param hidden_layers: List with the number of neurons in each hidden layer
        :param dropout_prob: Dropout probability
        :param use_batchnorm: Whether to use BatchNorm after Linear layers
        """
        super().__init__()

        self.flatten = nn.Flatten()

        layers = []
        current_size = input_size
        hidden_layers = [layer_width // (2 ** i) for i in range(layer_number)]

        for h_size in hidden_layers:
            layers.append(nn.Linear(current_size, h_size))

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_size))

            layers.append(nn.ReLU(inplace=True))
            if layer_number > 1:
                layers.append(nn.Dropout(dropout_prob))
            current_size = h_size

        layers.append(nn.Linear(current_size, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class ParametrizedCNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 8,
        layer_number: int = 2,
        layer_width: int = 128,
        kernel_size: int = 3,
        stride: int = 2,
        use_batchnorm: bool = True,
        activation: str = 'ReLU',
        pool_type: str = 'max',
        pool_kernel_size: int = 2,
        dropout_prob: float = 0.5,
    ):
        """
        :param input_channels: Number of input channels (e.g., 3 for RGB)
        :param num_classes: Number of classes
        :param layer_number: Number of convolutional blocks
        :param layer_width: Base channel width (for the first layer)
        :param dropout_prob: Dropout probability
        :param use_batchnorm: Whether to use BatchNorm after Conv2d
        """
        super().__init__()

        layers = []
        current_channels = input_channels

        for i in range(layer_number):
            out_channels = layer_width // (2 ** i)

            layers.append(nn.Conv2d(
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2)
            )

            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))

            if activation == 'ReLU':
                layers.append(nn.ReLU(inplace=True))

            elif activation == 'LeakyReLU':
                layers.append(nn.LeakyReLU(inplace=True))

            elif activation == 'ELU':
                layers.append(nn.ELU(inplace=True))

            elif activation == 'PReLU':
                layers.append(nn.PReLU(out_channels))

            if pool_type == 'max':
                layers.append(nn.MaxPool2d(
                    kernel_size=pool_kernel_size,
                    stride=pool_kernel_size
                ))

            elif pool_type == 'avg':
                layers.append(nn.AvgPool2d(
                    kernel_size=pool_kernel_size,
                    stride=pool_kernel_size
                ))

            if dropout_prob > 0.0 and layer_number > 1:
                layers.append(nn.Dropout2d(dropout_prob))

            current_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(current_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ParametrizedRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 2,
        rnn_type: str = 'lstm',
        bidirectional: bool = True,
        dropout_prob: float = 0.3,
        use_batch_norm: bool = False,
        # use_layer_norm: bool = False,
        final_dropout_prob: float = 0.5,
    ):
        """
        :param input_size (int): Dimension of the input vector at each step (e.g., 18).
        :param hidden_size (int): Dimension of the hidden state in the RNN.
        :param num_classes (int): Number of classes for classification.
        :param num_layers (int): Number of recurrent layers.
        :param rnn_type (str): Type of cell ('lstm', 'gru').
        :param bidirectional (bool): Whether to use bidirectionality.
        :param dropout_prob (float): Dropout between RNN layers (if num_layers > 1).
        :param use_batchnorm (bool): Whether to apply BatchNorm to the input data.
        :param use_layer_norm (bool): Whether to apply LayerNorm inside LSTM/GRU.
        :param final_dropout_prob (float): Dropout before the classification layer.
        """
        super().__init__()

        self.use_batchnorm = use_batch_norm
        if use_batch_norm:
            self.batchnorm = nn.BatchNorm1d(input_size)

        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True, # Вход: [batch, seq_len, features]
                dropout=dropout_prob if num_layers > 1 else 0,
                bidirectional=bidirectional,
                proj_size=0, # Без проекции
                # layer_norm=use_layer_norm # Доступно в новых версиях PyTorch
            )
            self.is_lstm = True

        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_prob if num_layers > 1 else 0,
                bidirectional=bidirectional,
                # layer_norm=use_layer_norm
            )
            self.is_lstm = False

        else:
            raise ValueError("rnn_type can be 'lstm' or 'gru'")

        # 3. Классификационная головка
        # Размерность на входе в Linear зависит от двунаправленности и количества слоёв
        rnn_output_dim = hidden_size * self.num_directions

        self.final_dropout = nn.Dropout(final_dropout_prob)
        self.classifier = nn.Linear(rnn_output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_batchnorm:
            # BatchNorm1d: [B, F, T] -> [B*F, T]
            # batch_size, seq_len, features = x.size()
            x = x.permute(0, 2, 1).contiguous() # [B, F, T]
            x = self.batchnorm(x)
            x = x.permute(0, 2, 1).contiguous() # [B, T, F]

        if self.is_lstm:
            output, (hidden_n, cell_n) = self.rnn(x)
            # [num_directions, batch, hidden_size]
            last_hidden_state = hidden_n[-self.num_directions:]
            # -> [batch, hidden_size * num_directions]
            last_hidden_state = torch.cat([last_hidden_state[i] for i in range(self.num_directions)], dim=1)
        else:
            output, hidden_n = self.rnn(x)
            last_hidden_state = hidden_n[-self.num_directions:]
            last_hidden_state = torch.cat([last_hidden_state[i] for i in range(self.num_directions)], dim=1)

        x = self.final_dropout(last_hidden_state)
        logits = self.classifier(x)

        return logits


class ParametrizedTransformer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_classes: int,
            seq_len: int = 50,
            nhead: int = 4,
            num_encoder_layers: int = 3,
            dim_feedforward: int = 128,
            dropout_prob: float = 0.1,
            use_positional_encoding: bool = True,
            final_dropout_prob: float = 0.5,
    ):
        """
        :param input_dim (int): Dimension of the input vector at each step (e.g., 18).
        :param num_classes (int): Number of classes for classification.
        :param seq_len (int): Length of the input sequence (e.g., 50).
        :param nhead (int): Number of attention heads.
        :param num_encoder_layers (int): Number of encoder layers.
        :param dim_feedforward (int): Dimension of the inner feed-forward network layer.
        :param dropout_prob (float): Dropout inside Transformer layers.
        :param use_positional_encoding (bool): Whether to add positional encoding.
        :param final_dropout_prob (float): Dropout before the classifier.
        """
        super().__init__()

        d_model = dim_feedforward
        self.input_projection = nn.Linear(input_dim, d_model)

        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.positional_encoding = self._generate_positional_encoding(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_prob if num_encoder_layers > 1 else 0,
            batch_first=True  # Важно: вход [batch, seq, feature]
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.pooling = 'first_token'
        self.final_dropout = nn.Dropout(final_dropout_prob)
        self.classifier = nn.Linear(d_model, num_classes)

    def _generate_positional_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, seq_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)  # [B, S, D]

        if self.use_positional_encoding:
            pe = self.positional_encoding.to(x.device)
            x = x + pe

        x = self.transformer_encoder(x)  # [B, S, D]

        if self.pooling == 'first_token':
            x = x[:, 0, :]
        elif self.pooling == 'mean':
            x = x.mean(dim=1)

        x = self.final_dropout(x)
        logits = self.classifier(x)

        return logits
