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
        self.env._max_steps = self.args.max_step-1

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

        """reward shaping 设置"""
        reward_dead = [5 for _ in range(4)]
        for i in self.train_idx_list:
            reward_dead[i] = -1
        reward_win = 0
        reward_lay_bomb = 1
        is_dead = [False for _ in range(4)]

        while not terminated:
            state = self.env.get_state()
            board_state = to_board_state(state, self.train_idx_list)
            flat_state = to_flat_state(state, self.train_idx_list)

            obs_list = env_utils.get_agent_obs(self.env, self.train_idx_list)   # 包含了两个本方智能体 obs 的列表
            board_obs_list = to_board_obs(obs_list)
            flat_obs_list = to_flat_obs(obs_list)

            avail_actions = env_utils.get_avail_actions(obs_list)

            pre_transition_data = {
                "board_state": [board_state],
                "flat_state": [flat_state],
                "avail_actions": [avail_actions],
                "board_obs": [board_obs_list],
                "flat_obs": [flat_obs_list]
            }
            # print('shape1:',board_state.shape)
            # print('shape2:', board_obs_list[0].shape)

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            agent_actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            # print(agent_actions)
            # todo: action 加信息
            # todo: 死了的话动作处理
            # 替换 all_actions 中我方智能体的动作
            all_obs = env_utils.get_all_obs(self.env)
            all_actions = env_utils.get_all_agent_actions(self.env, all_obs)
            for train_idx, agent_idx in zip(self.train_idx_list, range(len(self.train_idx_list))):
                one_of_action = agent_actions[0][agent_idx].item()    # 之所以第一个所以是零，是为了将形如[[1,2,3,4]]的输出剥出来
                all_actions[train_idx] = one_of_action

            # print(all_actions)
            next_obs_list, reward_list, terminated, env_info = self.env.step(all_actions)   # 传入动作列表

            # todo: 死了的话 reward 要处理
            # 处理 reward，取得我方智能体的 reward
            one_of_train_idx = self.train_idx_list[0]
            reward = reward_list[one_of_train_idx]

            """ 进行reward shaping """
            for i in range(4):
                if not is_dead[i] and not self.env._agents[i].is_alive:
                    if i in self.train_idx_list:
                        terminated = True
                    reward += reward_dead[i]
                    is_dead[i] = True
            for i, j in enumerate(self.train_idx_list):
                if obs_list[i]['ammo'] > 0 and all_actions[j] == 5:
                    reward += reward_lay_bomb

            episode_return += reward

            post_transition_data = {
                "actions": agent_actions,
                "reward": [(reward,)],
                "terminated": [(terminated,)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1
        # print('time_step:', self.t)
        # 最后一步的数据
        # print("next_obs_list[0]['step_count']", next_obs_list[0]['step_count'])
        # print(self.batch['flat_obs'].shape)
        state = self.env.get_state()
        board_state = to_board_state(state, self.train_idx_list)
        flat_state = to_flat_state(state, self.train_idx_list)
        # print('raw state:', state)
        obs_list = env_utils.get_agent_obs(self.env, self.train_idx_list)  # 包含了两个本方智能体 obs 的列表
        board_obs_list = to_board_obs(obs_list)
        flat_obs_list = to_flat_obs(obs_list)
        avail_actions = env_utils.get_avail_actions(obs_list)
        last_data = {
            "board_state": [board_state],
            "flat_state": [flat_state],
            "avail_actions": [avail_actions],
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
        # cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
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
