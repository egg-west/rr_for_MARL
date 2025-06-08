import copy
import numpy as np

import ot
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from info_nce import InfoNCE, info_nce

from components.standarize_stream import RunningMeanStd
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.task_encoder_qmix import TaskEncoderQMixer
from modules.task_encoders.base_task_encoder import BaseTaskEncoder
from modules.dynamic_models.probabilistic_forward_model import ProbabilisticForwardModel

class NCEDistance(nn.Module):
    def __init__(self, embedding_size):
        super(NCEDistance, self).__init__()
        self.w = nn.Parameters

class AbstractQLearner:
    """Agent learner with state abstraction"""
    def __init__(self, mac, scheme, logger, args, wandb_logger):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.wandb_logger = wandb_logger

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                #self.mixer = QMixer(args)
                self.mixer = TaskEncoderQMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = Adam(params=self.params, lr=args.lr)

        # state_dim = int(np.prod(args.state_shape))
        # joint_action_dim = int(args.n_actions * args.n_agents)
        if args.use_agent_encoder:
            self.agent_forward_model = ProbabilisticForwardModel(
                args.observation_embedding_dim,
                args.agent_embedding_dim,
                [int(args.n_actions)],
                256
            ).to(args.device)
            self.AFM_optimiser = Adam(params=self.agent_forward_model.parameters(), lr=args.lr)
            self.agent_encoder = BaseTaskEncoder(self.args.n_agents + 3, self.args.agent_embedding_dim).to(args.device)
            self.AE_optimiser = Adam(params=self.agent_encoder.parameters(), lr=args.lr)

            self.agent_indices = torch.arange(args.n_agents).to(args.device)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.training_steps = 0
        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)
    
    def reset(self, new_mac):
        self.mac = new_mac
        self.params = list(new_mac.parameters())

        self.mixer = None
        if self.args.mixer is not None:
            if self.args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif self.args.mixer == "qmix":
                #self.mixer = QMixer(args)
                self.mixer = TaskEncoderQMixer(self.args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = Adam(params=self.params, lr=self.args.lr)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(new_mac)
        self.cuda()

    def train(
        self,
        batch: EpisodeBatch,
        t_env: int,
        episode_num: int,
        current_task_id: int,
        positive_samples=None,
        negative_samples=None,
        seq_id=None,
    ):
        bs = batch.batch_size
        seq_len = int(batch["task_indices_global"].shape[1] - 1)

        if self.args.use_agent_encoder:
            #print(f"for agent encoder: {task_embedding.shape=}")
            agent_indices = self.agent_indices.unsqueeze(0).unsqueeze(0).tile([bs, seq_len, 1])
            #print(f"{agent_indices.shape=}") # expect [32, seq_len, 8] # [32, 68, 8]
            agent_embedding = self.agent_encoder(agent_indices).reshape(-1, self.args.n_agents, self.args.agent_embedding_dim)
            #print(f"{agent_embedding.shape=}") # expect [32 * seq_len, 8, 64] # [2176, 8, 64]
            onehot_actions = F.one_hot(batch["actions"][:, :-1]).squeeze(-2).to(batch.device) # [32, 66, 8, 14]
            agent_action = onehot_actions.reshape((-1, self.args.n_agents, self.args.n_actions)) # [2176, 8, 14]
            h = self.mac.observation_encode(batch["obs"][:, :-1]).reshape(bs*seq_len, self.args.n_agents, self.args.observation_embedding_dim)
            #print(f"{h.shape=}") # [32*68, 8, 64]
            afm_inputs = torch.cat([agent_embedding.detach(), h, agent_action], dim=2)
            #print(f"{afm_inputs.shape=}") # 2176, 8, 206
            predicted_next_observations, sigma = self.agent_forward_model(afm_inputs.detach())
            # print(f"{predicted_next_observations.shape=}") # [2176, 8, 64]
            next_h = self.mac.observation_encode(batch["obs"][:, 1:]).reshape(bs*seq_len, self.args.n_agents, self.args.observation_embedding_dim)
            diff = (predicted_next_observations - next_h.detach()) / sigma
            afm_loss = torch.mean(0.5 * diff.pow(2) + torch.log(sigma))

            self.AFM_optimiser.zero_grad()
            afm_loss.backward()
            self.AFM_optimiser.step()
            self.wandb_logger.log({
                "SA/afm_loss":afm_loss.item(),
                "SA/afm_diff":torch.mean(diff.pow(2)).detach().item(),
                },
                t_env
            )

            #################################### update the agent embedding
            # TEST_SAMPLE_NUMBER = 10
            random_agent_indices_bias = torch.LongTensor(np.random.randint(low=1, high=self.args.n_agents, size=self.args.n_agents)).to(batch.device)
            # print(f"{self.agent_indices.shape=}, {random_agent_indices_bias.shape=}")# [8]
            random_agent_indices = (self.agent_indices + random_agent_indices_bias).unsqueeze(0).unsqueeze(0).tile([bs, seq_len, 1]) % self.args.n_agents
            #print(f"{random_agent_indices.shape=}") [32, 68, 8]
            random_agent_embedding = self.agent_encoder(random_agent_indices).reshape(-1, self.args.n_agents, self.args.agent_embedding_dim)

            #print(f"{random_agent_embedding.shape=}") # [2176, 8, 64]
            random_afm_inputs = torch.cat([random_agent_embedding, h.detach(), agent_action], dim=2)
            #print(f"{random_afm_inputs.shape=}") # [2176, 8, 206]
            random_agent_predicted_next_observation, sigma = self.agent_forward_model(random_afm_inputs)
            #print(f"{random_agent_predicted_next_observation.shape=}") # [2176, 8, 64]
            afm_inputs = torch.cat([agent_embedding, h.detach(), agent_action], dim=2)
            new_predicted_next_observations, sigma = self.agent_forward_model(afm_inputs)
            diff = (new_predicted_next_observations - next_h.detach()) / sigma
            afm_loss = torch.mean(0.5 * diff.pow(2) + torch.log(sigma))

            ae_diff = torch.norm(agent_embedding.reshape(-1, agent_embedding.shape[-1]) - random_agent_embedding.reshape(-1, random_agent_embedding.shape[-1]), dim=1)
            next_observation_diff = torch.norm(
                new_predicted_next_observations.reshape(-1, new_predicted_next_observations.shape[-1]) - random_agent_predicted_next_observation.reshape(-1, random_agent_predicted_next_observation.shape[-1]),
                dim=1
            )
            self.AE_optimiser.zero_grad()
            ae_loss = F.mse_loss(ae_diff, next_observation_diff.detach()) + afm_loss
            ae_loss.backward()
            self.AE_optimiser.step()
            self.wandb_logger.log({"SA/ae_loss":ae_loss.item()}, t_env)

        ####################### Update the agent

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        task_indices = batch["task_indices"]#[:, :-1] # include many tasks
        obs_dim = batch["obs"].shape[-1]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # Calculate estimated Q-Values
        mac_out = []
        correlation_value = 0.0
        corr_to_log = []

        self.mac.init_hidden(batch.batch_size)

        if self.args.use_agent_encoder:
            agent_embedding = self.agent_encoder(self.agent_indices).unsqueeze(0).tile([bs, 1, 1])
            # print(f"{agent_embedding.shape}")# [8, 64]
            #print(f"{agent_embedding.shape=}") # torch.Size([32, 8, 64])
            for t in range(batch.max_seq_length):
                agent_outs, _ = self.mac.forward(batch, t=t, agent_embedding=agent_embedding)
                mac_out.append(agent_outs)
        else:
            for t in range(batch.max_seq_length):
                agent_outs, h = self.mac.forward(batch, t=t)

                mac_out.append(agent_outs)

        mac_out = th.stack(mac_out, dim=1)  # Concat over time # [bs, seq_len, n_agent, n_action]

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        # chosen_action_qvals = [bs, seq_len, n_agent]

        # Calculate the Q-Values necessary for the target
        target_mac_out = []

        self.target_mac.init_hidden(batch.batch_size)

        h_var = None
        if self.args.use_agent_encoder:
            agent_embedding = self.agent_encoder(self.agent_indices).unsqueeze(0).tile([bs, 1, 1])
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t, agent_embedding=agent_embedding)
                target_mac_out.append(target_agent_outs)
        else:
            # for t in range(batch.max_seq_length):
            #     target_agent_outs, _ = self.target_mac.forward(batch, t=t)
            #     target_mac_out.append(target_agent_outs)
            h_list = []
            for t in range(batch.max_seq_length):
                target_agent_outs, h = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
                h_list.append(h.detach())
            h_var = torch.stack(h_list).var(dim=0).mean().item()
            # print(f"{h.shape=}") # [256, 64]

        # Calcullate dormant ratio
        batch_mean = torch.abs(h).mean(dim=0) # should be [h_dim]
        batch_mean /= batch_mean.sum() # h should be [batch, h_dim]

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        if self.args.standardise_returns:
            target_max_qvals = target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        # print(f"{td_error.shape=}") # [bs, seq_len, 1]

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (self.training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            mask_elems = mask.sum().item()
            train_stats = {}
            train_stats["inspect/h_variance"] = h_var
            train_stats["train/loss"] = loss.item()
            train_stats["train/grad_norm"] = grad_norm.item()
            train_stats["train/td_error_abs"] = masked_td_error.abs().sum().item() / mask_elems
            train_stats["train/q_taken_mean"] = (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents)
            train_stats["train/target_mean"] = (targets * mask).sum().item()/(mask_elems * self.args.n_agents)
            for dn_tau in [0.002, 0.005, 0.010, 0.020, 0.025, 0.1, 0.2, 0.3, 0.4, 0.5]:
                train_stats[f"monitor/dorm_neuro_tau={dn_tau}"] = (batch_mean <= dn_tau).sum() / np.prod(batch_mean.shape)
            self.wandb_logger.log(train_stats, t_env)

            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env
            return train_stats

    def get_agent_embedding(self):
        return self.agent_encoder(self.agent_indices)

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
