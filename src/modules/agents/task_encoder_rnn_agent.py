# code adapted from https://github.com/wendelinboehmer/dcg

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.nn_utils import weight_init, build_mlp

def AvgL1Norm(x, eps=1e-8):
	return x/x.abs().mean(-1,keepdim=True).clamp(min=eps)

# class TaskEncoderRNNAgent(nn.Module):
#     def __init__(self, input_shape, args, obs_dim,):
#         super(TaskEncoderRNNAgent, self).__init__()
#         self.args = args

#         # suppose the observation includes the obs and agent_id
#         self.obs_fc = nn.Linear(obs_dim + args.n_agents, args.observation_embedding_dim)
#         # the input of self.fc1 includes the last action
#         self.fc1 = nn.Linear(input_shape, args.hidden_dim)
#         rnn_dim = args.observation_embedding_dim + args.hidden_dim

#         if self.args.use_rnn:
#             self.rnn = nn.GRUCell(rnn_dim, args.hidden_dim)
#         else:
#             self.rnn = nn.Linear(rnn_dim, args.hidden_dim)
#         self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)
#         if self.args.use_l1norm:
#             self.l1norm = AvgL1Norm
#         else:
#             self.l1norm = lambda x:x

#     def init_hidden(self):
#         # make hidden states on same device as model
#         return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

#     def forward(self, observations, inputs, hidden_state):
#         #obs_embedding = AvgL1Norm(self.obs_fc(observations))
#         obs_embedding = self.l1norm(self.obs_fc(observations))

#         x = F.relu(self.fc1(inputs))
#         #print(f"{x.shape=}, {observation.shape}, {obs_embedding.shape}") # [8, 64] [8, 128], [8, 64] for inference
#         # print(f"{x.shape=}, {observation.shape}, {obs_embedding.shape}")
#         # x.shape=torch.Size([256, 64]), torch.Size([32, 8, 128]), torch.Size([32, 8, 64])
#         x = torch.cat([x, obs_embedding], dim=1)

#         h_in = hidden_state.reshape(-1, self.args.hidden_dim)
#         if self.args.use_rnn:
#             h = self.rnn(x, h_in)
#         else:
#             h = F.relu(self.rnn(x))
#         q = self.fc2(h)
#         return q, h, None

#     def observation_encode(self, observations):
#         return self.obs_fc(observations)

class TaskEncoderRNNAgent(nn.Module):
    """as the Qmix is not stable on this architecture, we change it trying to obtain stability"""
    def __init__(self, input_shape, args, obs_dim,):
        super(TaskEncoderRNNAgent, self).__init__()
        self.args = args

        # suppose the observation includes the obs and agent_id
        self.obs_fc = nn.Linear(obs_dim + args.n_agents, args.observation_embedding_dim)
        # the input of self.fc1 includes the last action
        self.fc1 = nn.Linear(input_shape + args.observation_embedding_dim, args.hidden_dim)
        #rnn_dim = args.observation_embedding_dim + args.hidden_dim
        rnn_dim = args.hidden_dim
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(rnn_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(rnn_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

        if self.args.use_l1norm:
            self.l1norm = AvgL1Norm
        else:
            self.l1norm = lambda x:x

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, observations, inputs, hidden_state):
        #obs_embedding = AvgL1Norm(self.obs_fc(observations))
        obs_embedding = self.l1norm(self.obs_fc(observations))

        aug_inputs = torch.cat([inputs, obs_embedding], dim=1)
        x = F.relu(self.fc1(aug_inputs))
        #print(f"{x.shape=}, {observation.shape}, {obs_embedding.shape}") # [8, 64] [8, 128], [8, 64] for inference
        # print(f"{x.shape=}, {observation.shape}, {obs_embedding.shape}")
        # x.shape=torch.Size([256, 64]), torch.Size([32, 8, 128]), torch.Size([32, 8, 64])
        #x = torch.cat([x, obs_embedding], dim=1)

        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)
        return q, h, None

    def observation_encode(self, observations):
        return self.obs_fc(observations)

class ContrastiveRNNAgent(nn.Module):
    def __init__(self, input_shape, args, obs_dim,):
        super(ContrastiveRNNAgent, self).__init__()
        self.args = args

        if self.args.use_l1norm:
            self.l1norm = AvgL1Norm
        else:
            self.l1norm = lambda x:x

        # suppose the observation includes the obs and agent_id
        self.obs_fc = nn.Linear(obs_dim + args.n_agents, args.observation_embedding_dim)
        
        # self.auxiliary_obs_agent_id_fc = nn.Sequential(
        #     nn.Linear(obs_dim + args.n_agents, args.observation_embedding_dim),
        #     nn.ReLU(),
        #     nn.Linear(args.observation_embedding_dim, args.observation_embedding_dim),
        # )
        self.auxiliary_predicate_fc = nn.Sequential(
            nn.Linear(args.observation_embedding_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
        )

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

    def forward(self, observations, inputs, hidden_state):
        #obs_embedding = AvgL1Norm(self.obs_fc(observation))
        obs_embedding = self.l1norm(self.obs_fc(observations))
        aux_h = self.auxiliary_predicate_fc(obs_embedding)
        #bs = inputs.shape[0] // self.args.n_agents
        #agent_id = torch.eye(self.args.n_agents).to(obs_embedding.device) # [2, 2]
        #agent_id = agent_id.unsqueeze(0).expand(bs, -1, -1).reshape(bs*self.args.n_agents, -1)
        # print(f"{observation.shape=}, {agent_id.shape=}") observation.shape=torch.Size([2, 17]), agent_id.shape=torch.Size([4, 2])
        #auxiliary_input = torch.cat([observation, agent_id], dim=1)
        
        x = F.relu(self.fc1(inputs))
        x = torch.cat([x, obs_embedding], dim=1)

        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)
        return q, h, aux_h

    def observation_encode(self, observations):
        return self.obs_fc(observations)
    
    def aux_encode(self, observations):
        """
        Arguments:
            observations [bs*seq_len, n_agents, obs_dim]
        """

        bs = observations.shape[0]# // self.args.n_agents
        agent_id = torch.eye(self.args.n_agents).to(self.args.device)
        # print(f"{agent_id.shape=}") # [2, 2]
        agent_id = agent_id.unsqueeze(0).expand(bs, -1, -1)
        agent_id = agent_id.reshape(bs*self.args.n_agents, -1)
        #print(f"{observations.shape=}, {agent_id.shape=}") # [8608, 2, 17]), agent_id.shape=torch.Size([8608, 2, 2]
        
        observations = observations.reshape(bs*self.args.n_agents, -1)
        
        # print(f"{observations.shape=}, {agent_id.shape=}")# [17216, 17]), agent_id.shape=torch.Size([17216, 2])
        auxiliary_input = torch.cat([observations, agent_id], dim=1)
        #aux_h = self.auxiliary_obs_agent_id_fc(auxiliary_input)
        obs_embedding = AvgL1Norm(self.obs_fc(auxiliary_input))
        aux_h = self.auxiliary_predicate_fc(obs_embedding)

        return aux_h
    
    
class AuxiliaryTaskRNNAgent(nn.Module):
    def __init__(self, input_shape, args, obs_dim,):
        super(AuxiliaryTaskRNNAgent, self).__init__()
        self.args = args

        if self.args.use_l1norm:
            self.l1norm = AvgL1Norm
        else:
            self.l1norm = lambda x:x

        self.obs_fc = nn.Linear(obs_dim + args.n_agents, args.observation_embedding_dim)
        # self.auxiliary_obs_agent_id_fc = nn.Sequential(
        #     nn.Linear(obs_dim + args.n_agents, args.observation_embedding_dim),
        #     nn.ReLU(),
        #     nn.Linear(args.observation_embedding_dim, args.observation_embedding_dim),
        # )
        self.auxiliary_predicate_fc = nn.Sequential(
            nn.Linear(args.observation_embedding_dim, args.observation_embedding_dim),
            nn.ReLU(),
            nn.Linear(args.observation_embedding_dim, args.n_actions),
        )

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        #rnn_dim = args.observation_embedding_dim * 2 + args.hidden_dim
        rnn_dim = args.observation_embedding_dim + args.hidden_dim

        # self.auxiliary_predicate_fc = nn.Sequential(
        #     nn.Linear(rnn_dim, args.observation_embedding_dim),
        #     nn.ReLU(),
        #     nn.Linear(args.observation_embedding_dim, args.n_actions),
        # )

        if self.args.use_rnn:
            self.rnn = nn.GRUCell(rnn_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(rnn_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, observation, inputs, hidden_state):
        obs_embedding = self.l1norm(self.obs_fc(observation))
        avail_A_prediction = self.auxiliary_predicate_fc(obs_embedding)

        #bs = obs_embedding.shape[0] // self.args.n_agents
        #agent_id = torch.eye(self.args.n_agents).to(obs_embedding.device)
        # print(f"{agent_id.shape=}") # [2, 2]
        #agent_id = agent_id.unsqueeze(0).expand(bs, -1, -1).reshape(bs*self.args.n_agents, -1)
        # print(f"{observation.shape=}, {agent_id.shape=}") observation.shape=torch.Size([2, 17]), agent_id.shape=torch.Size([4, 2])
        #auxiliary_input = torch.cat([observation, agent_id], dim=1)
        #aux_h = self.auxiliary_obs_agent_id_fc(auxiliary_input)
        #aux_h = AvgL1Norm(aux_h)

        x = F.relu(self.fc1(inputs))
        #print(f"{x.shape=}, {observation.shape}, {obs_embedding.shape}") # [8, 64] [8, 128], [8, 64] for inference
        # print(f"{x.shape=}, {observation.shape}, {obs_embedding.shape}")
        # x.shape=torch.Size([256, 64]), torch.Size([32, 8, 128]), torch.Size([32, 8, 64])

        #x = torch.cat([x, obs_embedding, aux_h], dim=1)
        x = torch.cat([x, obs_embedding], dim=1)
        #avail_A_prediction = self.auxiliary_predicate_fc(x)

        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)
        return q, h, avail_A_prediction

    def observation_encode(self, observations):
        return self.obs_fc(observations)