from envs import REGISTRY as env_REGISTRY
from components import env_utils
from functools import partial
from components.episode_buffer import EpisodeBatch

from components.featurize import *


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env]()
        self.train_idx_list = self.args.train_idx_list  # train_idx_list 第一个是第一个智能体编号，不是0就是1
        self.episode_limit = self.args.max_step
        self.env._max_steps = self.args.max_step

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    # def get_env_info(self):
    #     return env_utils.get_env_info()

    # todo:决定要不要保留
    # def save_replay(self):
    #     self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            state = env_utils.get_state(self.env)
            board_state = to_board_state(state, self.train_idx_list)
            flat_state = to_flat_state(state)

            obs_list = env_utils.get_agent_obs(self.env, self.train_idx_list)   # 包含了两个本方智能体 obs 的列表
            board_obs_list = to_board_obs(obs_list, )
            flat_obs_list = to_flat_obs(obs_list)

            pre_transition_data = {
                "board_state": [board_state],
                "flat_state": [flat_state],
                "avail_actions": [env_utils.get_avail_actions(self.env)],
                "board_obs": [board_obs_list],
                "flat_obs": [flat_obs_list]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            agent_actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            # todo: action 加信息
            # 替换 all_actions 中我方智能体的动作
            all_obs = env_utils.get_all_obs(self.env)
            all_actions = env_utils.get_all_agent_actions(self.env, all_obs)
            for train_idx, agent_idx in zip(self.train_idx_list, range(len(self.train_idx_list))):
                one_of_action = agent_actions[0][agent_idx]     # 之所以第一个所以是零，是为了将形如[[1,2,3,4]]的输出剥出来
                all_actions[train_idx] = one_of_action

            next_obs_list, reward_list, terminated, env_info = self.env.step(all_actions)   # 传入动作列表

            # 处理 reward，取得我方智能体的 reward
            one_of_train_idx = self.train_idx_list[0]
            reward = reward_list[one_of_train_idx]
            episode_return += reward

            # 处理 terminated
            if next_obs_list[0]['step_count'] > self.episode_limit:
                terminated = True

            post_transition_data = {
                "actions": agent_actions,
                "reward": [(reward,)],
                "terminated": [(terminated,)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        # 最后一步的数据
        state = env_utils.get_state(self.env)
        board_state = to_board_state(state)
        flat_state = to_flat_state(state)
        obs_list = env_utils.get_agent_obs(self.env, self.train_idx_list)  # 包含了两个本方智能体 obs 的列表
        board_obs_list = to_board_obs(obs_list)
        flat_obs_list = to_flat_obs(obs_list)
        last_data = {
            "board_state": [board_state],
            "flat_state": [flat_state],
            "avail_actions": [env_utils.get_avail_actions(self.env)],
            "board_obs": [board_obs_list],
            "flat_obs": [flat_obs_list]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        agent_actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": agent_actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
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
