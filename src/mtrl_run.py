import datetime
import os
import pprint
import time
import threading
import wandb
import pickle
import numpy as np
import torch as th
from copy import deepcopy
from tqdm import tqdm
from types import SimpleNamespace as SN
from utils.logging import Logger, Timer
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import torch

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer, ContrastiveBuffer
from components.transforms import OneHot


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    # unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    try:
        map_name = _config["env_args"]["map_name"]
    except:
        map_name = _config["env_args"]["key"]
    unique_token = f"{_config['name']}_seed{_config['seed']}_{map_name}_{datetime.datetime.now()}"

    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
            dirname(dirname(abspath(__file__))), "results", "tb_logs"
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def modify_env_config(args, config_name, target_value):
    _args = deepcopy(args)
    _args.env_args[config_name] = target_value
    return _args

def run_sequential(args, logger):
    #print(f"{args=}")
    group_name_prefix = f"{args.name}"

    #if args.state_available_action:
    #    group_name_prefix = f"{args.name}-availAction"
    if args.use_rnn:
        group_name_prefix += "-RNN-"
    if args.checkpoint_path != "":
        group_name_prefix += f"load{args.load_step}"

    wandb_run = wandb.init(
        #project=f"pyMARL_CONTRASTIVE_{args.learner}",
        project=f"pyMARL_CONTRASTIVE_q_learner",
        #group=f'{args.env_args["map_name"]}',
        #group=f'{group_name_prefix}-RR{args.replay_ratio}-Reset{args.use_reset}-{args.env_args["map_name"]}',
        group=f'{group_name_prefix}-RR{args.replay_ratio}-{args.env_args["map_name"]}',
        name=f'{args.seed}',
        #mode="offline",
        config=args
    )

    timer = Timer()
    #wandb.config.update(args)
    #wandb.config.update({"test": 0})

    # Init runner so we can get env info
    #runner = r_REGISTRY["mtrl"](args=args, logger=logger, wandb_logger=wandb_run)
    """test task uses index that is closest to the training set"""
    if args.is_debug:
        # TRAIN_SPEED = [1.4, 2.2]
        # INTERPOLATE_SPEED = [1.8]
        # INTERPOLATE_TASK_ID = [1]
        # EXTRAPOLATE_SPEED = [1]
        # EXTRAPOLATE_TASK_ID = [1]
        SHOOT_RANGE_LIST = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        TRAIN_SHOOT_RANGE = [6.0]
        INTERPOLATE_TASK_ID = [0]
        EXTRAPOLATE_TASK_ID = [0]
        INTERPOLATE_SHOOT_RANGE = [7.0]
        EXTRAPOLATE_SHOOT_RANGE = [5.0]
    else:
        # SPEED_LIST = [0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4]
        # TRAIN_SPEED = [1.0, 1.4, 2.2, 3.0]
        # INTERPOLATE_TASK_ID = [1, 2]
        # EXTRAPOLATE_TASK_ID = [0, 3]
        # INTERPOLATE_SPEED = [1.8, 2.6]
        # EXTRAPOLATE_SPEED = [0.6, 3.4]
        # SPEED_LIST = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        # TRAIN_SPEED = [2.0, 3.0, 5.0]
        # INTERPOLATE_TASK_ID = [1]
        # EXTRAPOLATE_TASK_ID = [0, 2]
        # INTERPOLATE_SPEED = [4.0]
        # EXTRAPOLATE_SPEED = [1.0, 6.0]
        SHOOT_RANGE_LIST = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        TRAIN_SHOOT_RANGE = [6.0]


    # train_runner_list = [r_REGISTRY["mtrl"](args=modify_env_config(args, "move_amount", TRAIN_SPEED[i]), logger=logger, wandb_logger=wandb_run, env_id=i) \
    #     for i in range(len(TRAIN_SPEED))]
    # if args.test_interpolate:
    #     interpolate_runner_list = [r_REGISTRY["mtrl"](args=modify_env_config(args, "move_amount", INTERPOLATE_SPEED[i]), logger=logger, wandb_logger=wandb_run, env_id=INTERPOLATE_TASK_ID[i]) \
    #         for i in range(len(INTERPOLATE_SPEED))]
    # if args.test_extrapolate:
    #     extrapolate_runner_list = [r_REGISTRY["mtrl"](args=modify_env_config(args, "move_amount", EXTRAPOLATE_SPEED[i]), logger=logger, wandb_logger=wandb_run, env_id=EXTRAPOLATE_TASK_ID[i]) \
    #         for i in range(len(EXTRAPOLATE_SPEED))]
    #train_runner_list = [r_REGISTRY["mtrl"](args=modify_env_config(args, "shoot_range", TRAIN_SHOOT_RANGE[i]), logger=logger, wandb_logger=wandb_run, env_id=i) \
    #    for i in range(len(TRAIN_SHOOT_RANGE))]

    if args.name not in ["mappo", "ippo", "ia2c", "maa2c"]: #["vdn", "qmix", "maddpg"]:
        train_runner_list = [r_REGISTRY["mtrl"](args=args, logger=logger, wandb_logger=wandb_run, env_id=i) \
            for i in range(len(TRAIN_SHOOT_RANGE))]
    else:
        train_runner_list = [r_REGISTRY["parallel"](args=args, logger=logger, wandb_logger=wandb_run) \
            for i in range(len(TRAIN_SHOOT_RANGE))]

    runner_step = [0 for _ in range(len(train_runner_list))]

    # Set up schemes and groups here
    env_info = train_runner_list[0].get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    #if args.state_available_action:
    #    state_shape = env_info["state_shape"] + env_info["n_agents"] * env_info["n_actions"]

    state_shape = env_info["state_shape"]
    scheme = {
        #"state": {"vshape": env_info["state_shape"]},
        "state": {"vshape": state_shape},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "reward": {"vshape": (1,)},
        "task_indices": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "task_indices_global": {"vshape": (1,), "dtype": th.long},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )
    # contrastive_replay = ContrastiveBuffer(args)

    # Setup multiagent controller here
    if args.mac in ["maddpg_mac", "basic_mac_cds"]:
        mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    else:
        mac = mac_REGISTRY["task_encoder_mac"](buffer.scheme, groups, args)

    # Give runner the scheme
    #runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    for rn in train_runner_list:
        rn.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    #learner = le_REGISTRY["abstract_q_learner"](mac, buffer.scheme, logger, args, wandb_run)
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args, wandb_run)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(args.checkpoint_path)
            )
            return
        model_path = os.path.join(args.checkpoint_path, "models")
        # Go through all files in args.checkpoint_path
        for name in os.listdir(model_path):
            full_name = os.path.join(model_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
                # choose the max timestep
                timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        if args.load_model:
            model_path = os.path.join(model_path, str(timestep_to_load))

            logger.console_logger.info(f"Loading model from {model_path}")
            learner.load_models(model_path)
        else:
            logger.console_logger.info("Warning!!! Model is not loaded")
        
        runner = train_runner_list[0]
        runner.t_env = timestep_to_load

        with open(os.path.join(args.checkpoint_path, "buffer", f"{timestep_to_load}.pkl"), "rb") as f:
            buffer = pickle.load(f)
            buffer.to("cuda")

        if args.evaluate or args.save_replay:
            runner.log_train_stats_t = runner.t_env
            evaluate_sequential(args, runner)
            logger.log_stat("episode", runner.t_env, runner.t_env)
            logger.print_recent_stats()
            logger.console_logger.info("Finished Evaluation")
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_save_T = -args.save_model_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))
    
    skip_interval = int(1 / args.replay_ratio)
    SKIP = False
    if skip_interval > 1:
        SKIP = True

    #while runner.t_env <= args.t_max:
    _, _ = timer.reset()
    current_task_id = 0
    n_updates = 0
    RESET_STEP_THRESHOLD = 10 # early updates should not reset the parameter
    pbar = tqdm(total=args.t_max)
    runner = train_runner_list[current_task_id]
    pbar.update(runner.t_env)
    runner_step[0] = runner.t_env # stores the number of episode for each task
    while sum(runner_step) <= args.t_max:
        runner = train_runner_list[current_task_id]

        agent_embedding = None
        if args.use_agent_encoder:
            agent_embedding = learner.get_agent_embedding().unsqueeze(0) # agent_embedding.shape=torch.Size([1, 8, 64])

        # Run for a whole episode at a time
        episode_batch, _ = runner.run(test_mode=False, agent_embedding=agent_embedding)
        #print(f'{episode_batch["obs"].shape=}') # [1, 301, 2, 17] Tensor
        #print(f'{episode_batch["avail_actions"].shape=}') # [1, 301, 2, 7]

        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):

            if SKIP:
                if n_updates % skip_interval == 0:
                    episode_sample = buffer.sample(args.batch_size)

                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)

                    learner.train(episode_sample, sum(runner_step), episode, current_task_id)

                n_updates +=1

                if args.use_reset and n_updates > RESET_STEP_THRESHOLD and n_updates % args.reset_freq == 0:
                    last_test_T = sum(runner_step)
                    #print(f"Before reset: {mac.agent.obs_fc.weight[:5, :5]=}" )
                    #print(f"Before reset: {mac.agent.agents[0].fc1.weight[:5, :5]=}" )
                    mac = mac_REGISTRY["task_encoder_mac"](buffer.scheme, groups, args)
                    learner.reset(mac)
                    #print("reset finished")
                    #print(f"After reset: {mac.agent.obs_fc.weight[:5, :5]=}" )
                    #print(f"After reset: {mac.agent.agents[0].fc1.weight[:5, :5]=}" )
                    wandb_run.log({"train/reset": 1.0}, last_test_T)

                    single_env_return_list = []
                    single_env_win_rate_list = []
                    for _ in range(n_test_runs):
                        _, statistics = train_runner_list[0].run(test_mode=True, agent_embedding=agent_embedding)
                        single_env_return_list.append(statistics["returns"])
                        single_env_win_rate_list.append(statistics.get("battle_won", 0))

                    prefix = f"single_env/speed_{TRAIN_SHOOT_RANGE[0]}_"
                    return_mean = np.mean(single_env_return_list)
                    win_rate_mean = np.mean(single_env_win_rate_list)
                    single_env_stat = {
                        f"{prefix}return_mean": return_mean,
                        f"{prefix}battle_won": win_rate_mean
                    }
                    wandb_run.log(single_env_stat, last_test_T)
                    print(f"reset finished, {last_test_T=}")
            else:
                for _ in range(args.replay_ratio):
                    # if args.data_augmentation:
                    #     episode_sample = deepcopy(episode_sample_init)
                    #     episode_sample["obs"] += \
                    #         torch.normal(mean=0.0,
                    #                     std=args.aug_std,
                    #                     size=episode_sample["obs"].shape
                    #         ).to(episode_sample["obs"].device)
                    # else:
                    #     #episode_sample = episode_sample_init
                    #     episode_sample = buffer.sample(args.batch_size)
                    #print(f"{episode_sample['obs'].shape=}") # [32, 181, 10, 176]
                    episode_sample = buffer.sample(args.batch_size)

                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)

                    learner.train(episode_sample, sum(runner_step), episode, current_task_id)
                    n_updates +=1

                    if args.use_reset and n_updates > RESET_STEP_THRESHOLD and n_updates % args.reset_freq == 0:
                        last_test_T = sum(runner_step)
                        #print(f"Before reset: {mac.agent.obs_fc.weight[:5, :5]=}" )
                        #print(f"Before reset: {mac.agent.agents[0].fc1.weight[:5, :5]=}" )
                        mac = mac_REGISTRY["task_encoder_mac"](buffer.scheme, groups, args)
                        learner.reset(mac)
                        #print("reset finished")
                        #print(f"After reset: {mac.agent.obs_fc.weight[:5, :5]=}" )
                        #print(f"After reset: {mac.agent.agents[0].fc1.weight[:5, :5]=}" )
                        wandb_run.log({"train/reset": 1.0}, last_test_T)

                        single_env_return_list = []
                        single_env_win_rate_list = []
                        for _ in range(n_test_runs):
                            _, statistics = train_runner_list[0].run(test_mode=True, agent_embedding=agent_embedding)
                            single_env_return_list.append(statistics["returns"])
                            single_env_win_rate_list.append(statistics.get("battle_won", 0))

                        prefix = f"single_env/speed_{TRAIN_SHOOT_RANGE[0]}_"
                        return_mean = np.mean(single_env_return_list)
                        win_rate_mean = np.mean(single_env_win_rate_list)
                        single_env_stat = {
                            f"{prefix}return_mean": return_mean,
                            f"{prefix}battle_won": win_rate_mean
                        }
                        wandb_run.log(single_env_stat, last_test_T)
                        print(f"reset finished, {last_test_T=}")

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)

        ######################################### TEST
        if (sum(runner_step) - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info(
                "t_env: {} / {}".format(sum(runner_step), args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, sum(runner_step), args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()

            elapsed_time, total_time = timer.reset()
            wandb_run.log({"train/ips": (sum(runner_step) - last_test_T) / elapsed_time}, sum(runner_step))
            last_test_T = sum(runner_step)
            multi_env_return_list = []
            multi_env_win_rate_list = []
            for i in range(len(train_runner_list)):
                test_runner = train_runner_list[i]

                single_env_return_list = []
                single_env_win_rate_list = []
                for _ in range(n_test_runs):

                    #_, statistics = test_runner.run(test_mode=True, agent_embedding=agent_embedding)
                    # dict_keys(['dead_allies', 'dead_enemies', 'battle_won', 'n_episodes', 'ep_length', 'epsilon', 'return_mean', 'return_std', 'returns'])
                    #single_env_return_list.append(statistics["returns"])
                    #if not "battle_won" in statistics:
                    #    print(f"{statistics=}") # statistics={'returns': 7.607843137254902}
                    #single_env_win_rate_list.append(statistics.get("battle_won", 0))
                    
                    _, statistics = test_runner.run(test_mode=True, agent_embedding=agent_embedding)
                    single_env_return_list.extend(statistics[0])
                    single_env_win_rate_list.extend(statistics[1])
                #prefix = f"single_env/speed_{TRAIN_SPEED[i]}_"
                prefix = f"single_env/speed_{TRAIN_SHOOT_RANGE[i]}_"
                return_mean = np.mean(single_env_return_list)
                win_rate_mean = np.mean(single_env_win_rate_list)
                single_env_stat = {
                    f"{prefix}return_mean": return_mean,
                    f"{prefix}battle_won": win_rate_mean
                }
                #wandb_run.log(single_env_stat, last_test_T)
                wandb_run.log(single_env_stat, statistics[-1])
                multi_env_return_list.append(return_mean)
                multi_env_win_rate_list.append(win_rate_mean)
            # stat = {
            #     "eval/return_train_set":np.mean(multi_env_return_list),
            #     "eval/return_std_train_set":np.std(multi_env_return_list),
            #     "eval/win_rate_train_set":np.mean(multi_env_win_rate_list),
            # }
            # wandb_run.log(stat, last_test_T)

        # if args.save_model and (
        #     runner.t_env - model_save_time >= args.save_model_interval
        #     or model_save_time == 0
        # ):
        if args.save_model and (sum(runner_step) - last_save_T) / args.save_model_interval >= 1.0:
            model_save_time = runner.t_env
            save_path = os.path.join(
                args.local_results_path, args.unique_token, "models", str(runner.t_env)
            )
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

            buffer_path = os.path.join(
                args.local_results_path, args.unique_token, "buffer", str(runner.t_env)
            )
            os.makedirs(buffer_path, exist_ok=True)
            buffer_name = f"{buffer_path}.pkl"
            buffer.to('cpu')
            with open(f"{buffer_name}", "wb") as f:
                pickle.dump(buffer, f)

            # with open(f"{buffer_path}/buffer.pkl", "rb") as f:
            #     buffer = pickle.load(f)

            buffer.to('cuda')
            logger.console_logger.info(f"Saving buffer to {buffer_name}")
            last_save_T = sum(runner_step)

        episode += args.batch_size_run

        # if (runner.t_env - last_log_T) >= args.log_interval:
        #     logger.log_stat("episode", episode, runner.t_env)
        #     logger.print_recent_stats()
        #     last_log_T = runner.t_env

        pbar.update(runner.t_env - runner_step[current_task_id])
        runner_step[current_task_id] = runner.t_env
        current_task_id += 1
        current_task_id = (current_task_id) % len(train_runner_list)

    for runner in train_runner_list:
        runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config
