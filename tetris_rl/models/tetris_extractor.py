import torch
import torch.nn as nn

from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TetrisCNNExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for Dict observations:
    - board -> CNN
    - piece -> MLP / identity
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        # Compute the actual final feature dimension below.
        super().__init__(observation_space, features_dim=1)

        board_shape = observation_space["board"].shape # (1, H, W)
        piece_dim = observation_space["piece"].shape[0]

        n_input_channels = board_shape[0]

        self.board_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_board = torch.as_tensor(
                observation_space["board"].sample()[None]
            ).float()
            board_flattened_dim = self.board_cnn(sample_board).shape[1]

        self.piece_net = nn.Sequential(
            nn.Linear(piece_dim, 32),
            nn.ReLU(),
        )

        self.combined_net = nn.Sequential(
            nn.Linear(board_flattened_dim + 32, features_dim),
            nn.ReLU(),
        )

        self._feature_dim = features_dim

    def forward(self, observations):
        board_features = self.board_cnn(observations["board"])
        piece_features = self.piece_net(observations["piece"])
        combined = torch.cat([board_features, piece_features], dim=1)
        return self.combined_net(combined)
