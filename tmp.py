
"""
generate mask to zerofy a random number of agent. The mask is independent on each sequence time step but
common across batch
"""
if self.args.mask:
    tmp_chosen_action_qvals = chosen_action_qvals.detach().clone() # [32, 50, 10]
    init_mask = torch.ones_like(tmp_chosen_action_qvals).to(tmp_chosen_action_qvals.device)

    xx = np.arange(self.args.n_agents)

    for i in range(chosen_action_qvals.shape[1]):
        init_mask[:, i, np.random.choice(xx, size=3)] = False
    #print(init_mask[0, :20, :])
    chosen_action_qvals *= init_mask
    chosen_action_qvals += tmp_chosen_action_qvals * (-(init_mask - 1))



if args.sub_batch_method != "":
    if args.sub_batch_method == "random":
        tot_id = list(range(args.batch_size))
        for _ in range(args.batch_size // args.sub_batch_size):
            sub_id = np.random.choice(tot_id, size=args.sub_batch_size)
            learner.train(episode_sample, sum(runner_step), episode, current_task_id, batch_indices=sub_id)
    elif args.sub_batch_method == "return":
        episode_returns = episode_sample["reward"][:, :-1].sum(dim=1).squeeze(-1)  # [batch_size]
        sorted_indices = torch.argsort(episode_returns, dim=0)
        #print(f"{sorted_indices.shape=}") # [32]
        sorted_indices = sorted_indices.cpu().numpy()
        #print(f"{sorted_indices.shape=}")
        #sorted_indices = sorted_indices.reshape((args.sub_batch_size, -1))
        sorted_indices = sorted_indices.reshape((-1, args.sub_batch_size))
        #print(f"{sorted_indices.shape=}")
        # for _ in range(args.batch_size // args.sub_batch_size):
        #     id_for_id = np.random.choice(np.arange(args.batch_size // args.sub_batch_size), args.sub_batch_size, )
        #     sub_id = [sorted_indices[i, id_for_id[i]] for i in range(args.sub_batch_size)]
        #     learner.train(episode_sample, sum(runner_step), episode, current_task_id, batch_indices=sub_id)
        for sub_step in range(args.batch_size // args.sub_batch_size):
            #id_for_id = list(range(sub_step * args.sub_batch_size, (sub_step + 1) * args.sub_batch_size))
            sub_id = sorted_indices[sub_step]#[id_for_id]
            learner.train(episode_sample, sum(runner_step), episode, current_task_id, batch_indices=sub_id)
        #print(final_id)
        # sort the indices acendingly
        # sample indices from the sourted indices list

        #raise NotImplementedError
    else:
        assert False, "args.sub_batch_method should be in `return` and `random`"
else:
    learner.train(episode_sample, sum(runner_step), episode, current_task_id)
    


elif self.args.availA_contrastive:
    aux_embedding_list = []
    pos_embedding_list = []
    neg_embedding_list = []
    for t in range(batch.max_seq_length):
        agent_outs, aux_embedding = self.mac.forward(batch, t=t)
        mac_out.append(agent_outs)

        aux_embedding_list.append(aux_embedding.reshape(-1, self.args.n_agents, self.args.observation_embedding_dim))
    aux_embedding_batch = th.stack(aux_embedding_list, dim=1)
    #print(f"{aux_embedding_batch.shape=}")# 32, 269, 2, 32

    with torch.no_grad():
        # positive_samples_th = th.FloatTensor(positive_samples).to(
        #     self.args.device).reshape(bs*(seq_len+1), self.args.n_agents, obs_dim)
        # negative_samples_th = th.FloatTensor(negative_samples).to(
        #     self.args.device).reshape(bs*(seq_len+1), self.args.n_agents, obs_dim)
        positive_samples_th = th.FloatTensor(positive_samples).to(
            self.args.device).reshape(bs*(self.args.contrastive_n_samples), self.args.n_agents, obs_dim)
        negative_samples_th = th.FloatTensor(negative_samples).to(
            self.args.device).reshape(bs*(self.args.contrastive_n_samples), self.args.n_agents, obs_dim)
        #print(f"{positive_samples_th.shape=}, {negative_samples_th.shape=}") # [8608, 2, 17]
        #pos_aux_embedding = self.mac.aux_encode(positive_samples_th)
        #neg_aux_embedding = self.mac.aux_encode(negative_samples_th)
        pos_aux_embedding = self.target_mac.aux_encode(positive_samples_th)
        neg_aux_embedding = self.target_mac.aux_encode(negative_samples_th)
        # print(f"{pos_aux_embedding.shape=}, {neg_aux_embedding.shape=}") # [17216, 32]

    try:
        avail_A_loss = self.contrastive_loss(
            #aux_embedding_batch.reshape((-1, self.args.observation_embedding_dim)),
            aux_embedding_batch[:, seq_id, :, :].reshape((-1, self.args.hidden_dim)),
            pos_aux_embedding,
            neg_aux_embedding.unsqueeze(1)
        )
    except:

        # aux_embedding_batch.shape= [32, 269, 2, 32]
        # pos_aux_embedding.shape=torch.Size([17216, 32]),
        # neg_aux_embedding.shape=torch.Size([17216, 32])
        print(f"{aux_embedding_batch.shape=}, {pos_aux_embedding.shape=}, {neg_aux_embedding.shape=}")
        assert False, "debug"
    self.wandb_logger.log({"train/contrastive_loss": avail_A_loss}, t_env)


# cor = torch.corrcoef(c.mean(dim=0)).mean()
# correlation_value += cor
# corr_to_log.append(cor.detach().cpu().numpy())

if args.availA_contrastive:
    self.contrastive_loss = InfoNCE(negative_mode='paired')
