from components import featurize

episode_limit = 1000
n_agent = 2


def get_env_info():
    env_info = {"board_state_shape": get_board_state_size(),
                "flat_state_shape": get_flat_state_size(),
                "board_obs_shape": get_board_obs_size(),
                "flat_obs_shape": get_flat_obs_size(),
                "n_actions": get_total_actions(),
                "n_agents": n_agent,
                "episode_limit": episode_limit}
    return env_info


def get_state(env):
    return env.get_state()


# 得到训练的所有智能体的 obs
def get_agent_obs(env, train_idx_list):
    agent_obs = []
    all_obs = env.get_observations()
    for train_idx in train_idx_list:
        agent_obs.append(all_obs[train_idx])
    return agent_obs


# 包括敌方在内的所有 obs
def get_all_obs(env):
    return env.get_observations()


def get_board_state_size():
    return featurize.get_board_state_size()


def get_flat_state_size():
    return featurize.get_flat_state_size()


def get_board_obs_size():
    return featurize.get_board_obs_size()


def get_flat_obs_size():
    return featurize.get_flat_obs_size()


def get_avail_actions(env):
    raise NotImplementedError


def get_avail_agent_actions(env, agent_id):
    """ Returns the available actions for agent_id """
    raise NotImplementedError


def get_all_agent_actions(env, all_obs):
    return env.act(all_obs)


def get_total_actions():
    """ Returns the total number of actions an agent could ever take """
    # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
    raise NotImplementedError


def get_stats(env):
    stats = {
        "battles_won": self.battles_won,
        "battles_game": self.battles_game,
        "battles_draw": self.timeouts,
        "win_rate": self.battles_won / self.battles_game,
        "timeouts": self.timeouts,
        "restarts": self.force_restarts,
    }
    return stats

# todo: 决定要不要保留
# def seed(self):
#     raise NotImplementedError

# todo: 决定要不要保留
# def save_replay(self):
#     raise NotImplementedError


