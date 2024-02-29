# from facebookresearch/mtrl
from typing import List
import torch
import torch.nn as nn

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
class ProbabilisticForwardModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        task_embedding_dim: int,
        action_shape: List[int],
        layer_width: int,
        max_sigma: float = 1e1,
        min_sigma: float = 1e-4,
    ):
        """Probabilistic model for predicting the transition dynamics.

        Args:
            encoder_feature_dim (int): size of the input feature.
            action_shape (List[int]): size of the action vector.
            layer_width (int): width for each layer.
            max_sigma (float, optional): maximum value of sigma (of the learned
                gaussian distribution). Larger values are clipped to this value.
                Defaults to 1e1.
            min_sigma (float, optional): minimum value of sigma (of the learned
                gaussian distribution). Smaller values are clipped to this value.
                Defaults to 1e-4.
        """
        super(ProbabilisticForwardModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + task_embedding_dim + action_shape[0], layer_width),
            # nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(layer_width, layer_width),
            nn.ReLU(),
        )
        #self.ln = nn.LayerNorm(layer_width)
        self.fc_mu = nn.Linear(layer_width, state_dim)
        self.fc_sigma = nn.Linear(layer_width, state_dim)

        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        assert self.max_sigma >= self.min_sigma

        self.apply(weight_init)

    def forward(self, x):
        x = self.fc(x)

        mu = self.fc_mu(x)
        sigma = torch.sigmoid(self.fc_sigma(x))  # range (0, 1.)
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma # scaled range (min_sigma, max_sigma)
        return mu, sigma

    def sample_prediction(self, x):
        mu, sigma = self(x)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    # def get_last_shared_layers(self) -> List[ModelType]:
    #     breakpoint()
    #     return [self.fc]