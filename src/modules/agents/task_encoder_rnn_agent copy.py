# code adapted from https://github.com/wendelinboehmer/dcg

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.nn_utils import weight_init, build_mlp

class TaskEncoderRNNAgent(nn.Module):
    def __init__(self, input_shape, args, obs_dim,):
        super(TaskEncoderRNNAgent, self).__init__()
        self.args = args

        self.obs_fc = nn.Linear(obs_dim, args.observation_embedding_dim)
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        rnn_dim = args.observation_embedding_dim + args.hidden_dim

        if self.args.use_rnn:
            self.rnn = nn.GRUCell(rnn_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(rnn_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, observation, inputs, hidden_state):
        obs_embedding = F.relu(self.obs_fc(observation))
        x = F.relu(self.fc1(inputs))
        #print(f"{x.shape=}, {observation.shape}, {obs_embedding.shape}") # [8, 64] [8, 128], [8, 64] for inference
        # print(f"{x.shape=}, {observation.shape}, {obs_embedding.shape}")
        # x.shape=torch.Size([256, 64]), torch.Size([32, 8, 128]), torch.Size([32, 8, 64])
        x = torch.cat([obs_embedding, x], dim=1)

        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)
        return q, h

    def observation_encode(self, observations):
        return self.obs_fc(observations)