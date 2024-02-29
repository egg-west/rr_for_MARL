import numpy as np
import torch as th

from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY

# This multi-agent controller shares parameters between agents
class TaskEncoderMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.obs_dim = scheme["obs"]["vshape"]
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(
        self,
        ep_batch,
        t_ep,
        t_env,
        bs=slice(None),
        test_mode=False,
        agent_embedding=None,
    ):

        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode, agent_embedding=agent_embedding)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False, agent_embedding=None):
        agent_inputs = self._build_inputs(ep_batch, t, agent_embedding)
        observations = ep_batch["obs"][:, t]
        observations = observations.reshape((-1, observations.shape[-1]))
        avail_actions = ep_batch["avail_actions"][:, t]

        #print(f"in forward: {task_indices.shape}") # before reshape: [1, 5, 1], after [5]
        agent_outs, self.hidden_states = self.agent(observations, agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        #self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        self.agent = agent_REGISTRY["task_encoder_rnn"](
            input_shape,
            self.args,
            self.obs_dim,
        )

    def _build_inputs(self, batch, t, agent_embedding=None):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        # print(f'{batch["obs"].shape=}, {batch["obs"][:, t].shape=}')
        # batch["obs"].shape=torch.Size([1, 121, 5, 80]), batch["obs"][:, t].shape=torch.Size([1, 5, 80])
        #inputs.append(batch["obs"][:, t])  # [bs, 1, agent_num, observation_dim]
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            if self.args.use_agent_encoder:
                inputs.append(agent_embedding)
            else:
                inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        #print(f"{inputs.shape=}")# inputs.shape=torch.Size([8, 64])
        return inputs

    def _get_input_shape(self, scheme):
        #input_shape = scheme["obs"]["vshape"]
        input_shape = 0
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            if self.args.use_agent_encoder:
                input_shape += self.args.agent_embedding_dim
            else:
                input_shape += self.n_agents

        return input_shape

    def observation_encode(self, observations):
        return self.agent.observation_encode(observations)
