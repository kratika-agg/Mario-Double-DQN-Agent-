
import torch
import torch.nn as nn


class DQN(nn.Module):

    def __init__(self, input_channels=4, num_actions=2):
        super(DQN, self).__init__()

        self.network = nn.Sequential(

            # Convolutional Layer 1
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=32,
                kernel_size=8,
                stride=4
            ),
            nn.ReLU(),

            # Convolutional Layer 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2
            ),
            nn.ReLU(),

            # Convolutional Layer 3
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU(),

            # Flatten feature maps
            nn.Flatten(),

            # Fully connected layers
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),

            # Output Q-values
            nn.Linear(512, num_actions)
        )

    def forward(self, x):

        # Handle environment observations: (4, 1, 84, 84)
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1).unsqueeze(0)

        # Handle batches: (B, 4, 1, 84, 84)
        elif x.ndim == 5:
            x = x.squeeze(2)

        # Handle unbatched grayscale frames
        elif x.ndim == 3:
            x = x.unsqueeze(0)

        x = x.float()

        return self.network(x)
