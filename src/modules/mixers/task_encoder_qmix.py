import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TaskEncoderQMixer(nn.Module):
    def __init__(self, args):
        super(TaskEncoderQMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.state_encoder = nn.Linear(self.state_dim, args.state_embedding_dim)
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)

        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot

# class TaskEncoderQMixer(nn.Module):
#     def __init__(self, args):
#         super(TaskEncoderQMixer, self).__init__()

#         self.args = args
#         self.n_agents = args.n_agents
#         self.original_state_dim = int(np.prod(args.state_shape))
#         if self.args.state_available_action:
#             self.original_state_dim += int(args.n_actions*args.n_agents)
#         #self.state_dim = int(np.prod(args.state_shape))

#         self.embed_dim = args.mixing_embed_dim

#         if getattr(args, "hypernet_layers", 1) == 1:
#             self.hyper_w_1 = nn.Linear(args.state_embedding_dim, self.embed_dim * self.n_agents)
#             self.hyper_w_final = nn.Linear(args.state_embedding_dim, self.embed_dim)
#         elif getattr(args, "hypernet_layers", 1) == 2:
#             hypernet_embed = self.args.hypernet_embed
#             self.state_encoder = nn.Linear(self.original_state_dim, args.state_embedding_dim)
#             self.hyper_w_1 = nn.Sequential(nn.Linear(args.state_embedding_dim, hypernet_embed),
#                                            nn.ReLU(),
#                                            nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))

#             # self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
#             #                                nn.ReLU(),
#             #                                nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
#             self.hyper_w_final = nn.Sequential(nn.Linear(args.state_embedding_dim, hypernet_embed),
#                                            nn.ReLU(),
#                                            nn.Linear(hypernet_embed, self.embed_dim))
#         elif getattr(args, "hypernet_layers", 1) > 2:
#             raise Exception("Sorry >2 hypernet layers is not implemented!")
#         else:
#             raise Exception("Error setting number of hypernet layers.")

#         # State dependent bias for hidden layer
#         self.hyper_b_1 = nn.Linear(args.state_embedding_dim, self.embed_dim)

#         # V(s) instead of a bias for the last layers
#         self.V = nn.Sequential(nn.Linear(args.state_embedding_dim, self.embed_dim),
#                                nn.ReLU(),
#                                nn.Linear(self.embed_dim, 1))

#     def forward(self, agent_qs, states):
#         bs = agent_qs.size(0)
#         states = states.reshape(-1, self.original_state_dim)

#         #if self.args.use_task_encoder:
#         #    states = th.cat([states, task_embedding], dim=1)
#         agent_qs = agent_qs.view(-1, 1, self.n_agents)

#         # First layer
#         state_embedding = F.relu(self.state_encoder(states))
#         #print(f"{state_embedding.shape=}, {task_embedding.shape=}")
#         # state_embedding.shape=torch.Size([2080, 128]), task_embedding.shape=torch.Size([2080, 64])

#         w1 = th.abs(self.hyper_w_1(state_embedding))
#         # w1 = th.abs(self.hyper_w_1(states))
#         b1 = self.hyper_b_1(state_embedding)
#         w1 = w1.view(-1, self.n_agents, self.embed_dim)
#         b1 = b1.view(-1, 1, self.embed_dim)
#         hidden = F.elu(th.bmm(agent_qs, w1) + b1)

#         # Second layer
#         w_final = th.abs(self.hyper_w_final(state_embedding))
#         w_final = w_final.view(-1, self.embed_dim, 1)
#         # State-dependent bias
#         v = self.V(state_embedding).view(-1, 1, 1)
#         # Compute final output
#         y = th.bmm(hidden, w_final) + v
#         # Reshape and return
#         q_tot = y.view(bs, -1, 1)
#         return q_tot

#     def state_embedding(self, states):
#         states = states.reshape(-1, self.original_state_dim)
#         return F.relu(self.state_encoder(states))