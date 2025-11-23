import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        # https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html
        self.net = nn.Sequential(
            # Insert layers here
        )

    def forward(self, x):
        return self.net(x)

