from envs import REGISTRY as env_REGISTRY
from components import env_utils
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np

from components.featurize import *

# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run    # 并行环境数量

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        self.ps = [Process(target=env_worker,
                           args=(worker_conn, CloudpickleWrapper(env_fn), self.args.train_idx_list))
                   for worker_conn in self.worker_conns]

        for p in self.ps:
            p.daemon = True
            p.start()

        # self.parent_conns[0].send(("get_env_info", None))
        # self.env_info = self.parent_conns[0].recv()
        self.env_info = env_utils.get_env_info(self.args)
        self.episode_limit = self.env_info["episode_limit"]
        # 设置最大步长
        for parent_conn in self.parent_conns:
            parent_conn.send(("set_max_steps", args.max_step-1))

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    # def get_env_info(self):
    #     return self.env_info
    #
    # def save_replay(self):
    #     pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "board_state": [],
            "flat_state": [],
            "avail_actions": [],
            "board_obs": [],
            "flat_obs": []
        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["board_state"].append(data["board_state"])
            pre_transition_data["flat_state"].append(data["flat_state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["board_obs"].append(data["board_obs"])
            pre_transition_data["flat_obs"].append(data["flat_obs"])

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        self.reset()

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]   # 长度为并行环境的数量
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed] # 未结束的环境的编号
        # final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:
            # 将截至目前的 batch 发给所有 agent
            # 收到的是未结束的环境中，在这个 time_step 下 agents 的动作
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            cpu_actions = actions.to("cpu").numpy()
            # print("terminated:", terminated)
            # print('cpu_actions:', cpu_actions)
            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1)    # 变成列向量
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:
                    if not terminated[idx]:
                        parent_conn.send(("get_all_actions", None))

            action_idx = 0
            all_actions_list = []
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:
                    if not terminated[idx]:
                        all_actions = parent_conn.recv()
                        # print('all_actions_', idx, ':',all_actions)
                        for agent_idx, train_idx in enumerate(self.args.train_idx_list):
                            all_actions[train_idx] = int(cpu_actions[action_idx][agent_idx])   # 替换
                        all_actions_list.append(all_actions)
                    else:
                        all_actions_list.append(None)
                    action_idx += 1
            # print('all_actions:', all_actions_list)
            # print('================================')
            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # We produced actions for this env
                    if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", all_actions_list[action_idx]))
                    action_idx += 1 # actions is not a list over every env

            # Update envs_not_terminated, 全部环境都回合结束了才退出
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # 当前 timestep 的数据
            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # 下一 timestep 的数据
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "board_state": [],
                "flat_state": [],
                "avail_actions": [],
                "board_obs": [],
                "flat_obs": []
            }

            # 只收集未结束的环境的数据反馈
            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # 处理当前 timestep 的数据
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    # todo: 有问题
                    env_terminated = False
                    # if data["terminated"]:
                    #     final_env_infos.append(data["info"])
                    if data["terminated"]:# and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # 处理下一回合需要的数据
                    # Data for the next timestep needed to select an action
                    pre_transition_data["board_state"].append(data["board_state"])
                    pre_transition_data["flat_state"].append(data["flat_state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["board_obs"].append(data["board_obs"])
                    pre_transition_data["flat_obs"].append(data["flat_obs"])

            # 把当前 timestep 的数据添加到 episode batch 里
            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # timestep 加 1 之后，有关数据添加到 episode batch 里
            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run   # 加上当前 timestep 下未终止的环境跑的步数， t_env 是 total_timesteps

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats",None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] #+ final_env_infos
        # cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker(remote, env_fn, train_list):
    # Make environment
    env = env_fn.var()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            next_obs_list, reward_list, terminated, env_info = env.step(actions)

            state = env.get_state()
            board_state = to_board_state(state, train_list)
            flat_state = to_flat_state(state, train_list)

            obs_list = []     # 取训练的 obs
            for agent_idx in train_list:
                obs_list.append(next_obs_list[agent_idx])
            board_obs_list = to_board_obs(obs_list)
            flat_obs_list = to_flat_obs(obs_list)

            avail_actions = env_utils.get_avail_actions(obs_list, train_list)

            reward = reward_list[train_list[0]]  # 取其中一个训练智能体的奖励返回

            remote.send({
                # Data for the next timestep needed to pick an action
                "board_state": board_state,
                "flat_state": flat_state,
                "avail_actions": avail_actions,
                "board_obs": board_obs_list,
                "flat_obs": flat_obs_list,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            state = env.get_state()
            board_state = to_board_state(state, train_list)
            flat_state = to_flat_state(state, train_list)

            obs_list = env_utils.get_agent_obs(env, train_list)  # 包含了两个本方智能体 obs 的列表
            board_obs_list = to_board_obs(obs_list)
            flat_obs_list = to_flat_obs(obs_list)

            remote.send({
                "board_state": board_state,
                "flat_state": flat_state,
                "avail_actions": env_utils.get_avail_actions(obs_list, train_list),
                "board_obs": board_obs_list,
                "flat_obs": flat_obs_list
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        # elif cmd == "get_env_info":
        #     remote.send(env_utils.get_env_info(args=args))
        elif cmd == "get_stats":
            remote.send(env_utils.get_stats(env))
        elif cmd == "get_all_actions":
            all_obs = env_utils.get_all_obs(env)
            all_actions = env.act(all_obs)
            remote.send(all_actions)
        elif cmd == "set_max_steps":
            env._max_steps = data
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, var):
        self.var = var
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.var)
    def __setstate__(self, ob):
        import pickle
        self.var = pickle.loads(ob)

