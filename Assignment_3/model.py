import torch.nn as nn


class MLP(nn.Module):
    """
    Simple feedforward network for sentiment classification
    on top of document embeddings.
    """

    def __init__(self, input_dim, hidden_dims=None, num_classes=1, dropout=0.2):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        # Final layer: outputs logits for num_classes
        layers.append(nn.Linear(prev_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


