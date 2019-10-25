episode_limit = 1000
n_agent = 2

def get_env_info():
    env_info = {"state_shape": get_state_size(),
                "obs_shape": get_obs_size(),
                "n_actions": get_total_actions(),
                "n_agents": n_agent,
                "episode_limit": episode_limit}
    return env_info


def get_obs(env, agent_id):
    """ Returns observation for agent_id """
    return env.get_observations()[agent_id]


def get_obs_size():
    """ Returns the shape of the observation """
    raise NotImplementedError


def get_state(env):
    return env.get_state()


def get_state_size():
    """ Returns the shape of the state"""
    raise NotImplementedError


def get_avail_actions(env):
    raise NotImplementedError


def get_avail_agent_actions(env, agent_id):
    """ Returns the available actions for agent_id """
    raise NotImplementedError


def get_total_actions():
    """ Returns the total number of actions an agent could ever take """
    # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
    raise NotImplementedError


def get_stats(env):
    pass
    # stats = {
    #     "battles_won": self.battles_won,
    #     "battles_game": self.battles_game,
    #     "battles_draw": self.timeouts,
    #     "win_rate": self.battles_won / self.battles_game,
    #     "timeouts": self.timeouts,
    #     "restarts": self.force_restarts,
    # }
    # return stats


# def reset():
#     """ Returns initial observations and states"""
#     raise NotImplementedError


# def render(self):
#     raise NotImplementedError


# def close(self):
#     raise NotImplementedError

# todo: 决定要不要保留
# def seed(self):
#     raise NotImplementedError

# todo: 决定要不要保留
# def save_replay(self):
#     raise NotImplementedError


