from components import featurize

n_agent = 2


def get_env_info(args):
    env_info = {"board_state_shape": get_board_state_size(),
                "flat_state_shape": get_flat_state_size(),
                "board_obs_shape": get_board_obs_size(),
                "flat_obs_shape": get_flat_obs_size(),
                "n_actions": get_total_actions(),
                "n_agents": n_agent,
                "episode_limit": args.max_step}
    return env_info


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


def get_avail_actions(obs_list):
    avail_actions = []
    for obs in obs_list:
        avail_actions.append(get_avail_agent_actions(obs))
    return avail_actions


# todo: board 物体、火焰、能不能踢等
def get_avail_agent_actions(obs):
    """ Returns the available actions for agent_id """
    avail_agent_actions = [1] * 6
    if obs['ammo'] == 0:
        avail_agent_actions[5] = 0      # 不能放炸弹
    for action in [1, 2, 3, 4]:         # 上下左右
        avail_agent_actions[action] = check_move(obs, action)
    return avail_agent_actions


def get_all_agent_actions(env, all_obs):
    return env.act(all_obs)


def get_total_actions():
    """ Returns the total number of actions an agent could ever take """
    return 6


def get_stats(env):
    stats = {
        # "battles_won": self.battles_won,
        # "battles_game": self.battles_game,
        # "battles_draw": self.timeouts,
        # "win_rate": self.battles_won / self.battles_game,
        # "timeouts": self.timeouts,
        # "restarts": self.force_restarts,
    }
    return stats


def check_move(obs, move):
    curr_pos = obs['position']
    if move == 1:
        next_pos_0 = curr_pos[0] - 1
        next_pos_1 = curr_pos[1]
    elif move == 2:
        next_pos_0 = curr_pos[0] + 1
        next_pos_1 = curr_pos[1]
    elif move == 3:
        next_pos_0 = curr_pos[0]
        next_pos_1 = curr_pos[1] - 1
    elif move == 4:
        next_pos_0 = curr_pos[0]
        next_pos_1 = curr_pos[1] + 1
    else:
        raise ValueError
    # 墙、木墙、边界
    if next_pos_1 < 0 or next_pos_1 > 10 or \
            next_pos_0 < 0 or next_pos_0 > 10 or \
            obs['board'][(next_pos_0, next_pos_1)] in [1, 2]:
        return 0
    else:
        return 1


