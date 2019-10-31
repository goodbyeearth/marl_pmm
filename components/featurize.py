import numpy as np


# TODO： 记得敌方跟己方的特征分开, 并且按顺序
def to_board_state(state, train_list):
    maps = []
    board = state['board']

    '''爆炸one-hot'''
    bomb_life = state['bomb_life']
    bomb_blast_strength = state['bomb_blast_strength']
    flame_life = state['flame_life']
    # 统一炸弹时间
    for x in range(11):
        for y in range(11):
            if bomb_blast_strength[(x, y)] > 0 :
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x + i, y)
                    if x + i > 10:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x,y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x - i, y)
                    if x - i < 0:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x,y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x, y + i)
                    if y + i > 10:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x,y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x, y - i)
                    if y - i < 0:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x,y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]

    bomb_life = np.where(bomb_life > 0, bomb_life + 3, bomb_life)
    flame_life = np.where(flame_life == 0, 15, flame_life)
    flame_life = np.where(flame_life == 1, 15, flame_life)
    bomb_life = np.where(flame_life != 15, flame_life, bomb_life)
    for i in range(2, 13):                        # 特征1, 11层
        maps.append(bomb_life == i)

    '''将bomb direction编码为one-hot'''
    bomb_moving_direction = state['bomb_moving_direction']
    for i in range(1, 5):
        maps.append(bomb_moving_direction == i)       # 特征2，4层

    """棋盘物体 one hot"""
    for i in range(9):  # [0, 1, ..., 8]
        maps.append(board == i)                        # 特征3，9层

    """四个智能体的位置"""
    assert train_list is not None
    for i in [10, 11, 12, 13]:
        if i not in train_list:
            # 敌人先后顺序无所谓
            maps.append(board == i)                    # 特征4，2层
    for i in [10, 11, 12, 13]:
        if i in train_list:                             # 特征5，2层
            # 这个先后顺序是有意义的，一个是自己一个是队友，故输入必须有编号才能分辨该特征
            maps.append(board == i)

    return np.stack(maps)


def get_board_state_size():
    return 28, 11, 11


# TODO： 记得敌方跟己方的特征分开, 并且按顺序
def to_flat_state(state, train_list):
    if train_list[0] == 0:
        enemy_list = [1, 3]
    else:
        enemy_list = [2, 4]

    # alive 特征
    ally_alive = [0, 0]        # 特征 1，长度2
    enemy_alive = [0, 0]       # 特征 2，长度2
    ally_idx = enemy_idx = 0
    for agent_idx in state['alive']:
        if agent_idx-10 in train_list:
            ally_alive[ally_idx] = 1
            ally_idx += 1
        else:
            enemy_alive[enemy_idx] = 1
            enemy_idx += 1

    # 曼哈顿距离: 和队友、敌人1、敌人2 的距离
    dist_list = []                  # 特征 3，长度 6
    for agent_idx in train_list:
        my_pos = state['agents_attrs'][agent_idx]['position']    # 我的位置
        tm_pos = state['agents_attrs'][(agent_idx+2) % 4]['position']  # 队友的位置
        e1_pos = state['agents_attrs'][enemy_list[0]]['position']  # 敌人1的位置
        e2_pos = state['agents_attrs'][enemy_list[1]]['position']  # 敌人2的位置
        tm_dist = abs(my_pos[0] - tm_pos[0]) + abs(my_pos[1] - tm_pos[1])  # 与队友的距离
        tm_dist = tm_dist / 20   # 归一化
        e1_dist = abs(my_pos[0] - e1_pos[0]) + abs(my_pos[1] - e1_pos[1])
        e1_dist = e1_dist / 20
        e2_dist = abs(my_pos[0] - e2_pos[0]) + abs(my_pos[1] - e2_pos[1])
        e2_dist = e2_dist / 20
        dist_list = dist_list + [tm_dist, e1_dist, e2_dist]

    # 己方炸弹爆炸范围、能否踢、弹药量
    ally_blast_strength = []
    ally_can_kick = []
    ally_ammo = []
    for agent_idx in train_list:
        attr = state['agents_attrs'][agent_idx]
        ally_blast_strength.append(attr['blast_strength'] / 5)
        can_kick = 1 if attr['can_kick'] else 0
        ally_can_kick.append(can_kick)
        ally_ammo.append(attr['ammo'] / 5)
    ally_attr = ally_blast_strength + ally_can_kick + ally_ammo   # 特征4，长度6

    # 敌方炸弹爆炸范围、能否踢、弹药量
    enemy_blast_strength = []
    enemy_can_kick = []
    enemy_ammo = []
    for agent_idx in enemy_list:
        attr = state['agents_attrs'][agent_idx]
        enemy_blast_strength.append(attr['blast_strength'] / 5)
        can_kick = 1 if attr['can_kick'] else 0
        enemy_can_kick.append(can_kick)
        enemy_ammo.append(attr['ammo'] / 5)
    enemy_attr = enemy_blast_strength + enemy_can_kick + enemy_ammo   # 特征5，长度6

    flat_state = np.concatenate((ally_alive, enemy_alive, dist_list, ally_attr, enemy_attr))
    return flat_state


def get_flat_state_size():
    return 22


def to_board_obs(obs_list):
    board_obs = []
    for obs in obs_list:
        board_obs.append(to_agent_board_obs(obs))
    return board_obs


def to_agent_board_obs(obs):
    maps = []
    board = obs['board']

    '''爆炸one-hot'''
    bomb_life = obs['bomb_life']
    bomb_blast_strength = obs['bomb_blast_strength']
    flame_life = obs['flame_life']
    # 统一炸弹时间
    for x in range(11):
        for y in range(11):
            if bomb_blast_strength[(x, y)] > 0:
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x + i, y)
                    if x + i > 10:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x - i, y)
                    if x - i < 0:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x, y + i)
                    if y + i > 10:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x, y - i)
                    if y - i < 0:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]

    bomb_life = np.where(bomb_life > 0, bomb_life + 3, bomb_life)
    flame_life = np.where(flame_life == 0, 15, flame_life)
    flame_life = np.where(flame_life == 1, 15, flame_life)
    bomb_life = np.where(flame_life != 15, flame_life, bomb_life)
    for i in range(2, 13):
        maps.append(bomb_life == i)             # 特征 1，11层

    '''将bomb direction编码为one-hot'''
    bomb_moving_direction = obs['bomb_moving_direction']
    for i in range(1, 5):
        maps.append(bomb_moving_direction == i)     # 特征2，4层

    """棋盘物体 one hot"""
    for i in range(9):  # [0, 1, ..., 8]
        maps.append(board == i)                     # 特征3，9层

    """一个队友的位置"""
    teammate_idx = obs['teammate'].value
    maps.append(board == teammate_idx)               # 特征4，1层
    """两个敌人的位置"""
    enemies_idx = []
    for e in obs['enemies']:
        if not e.value == 9:  # AgentDummy
            enemies_idx.append(e.value)
            maps.append(board == e.value)             # 特征5,2层
    """我的智能体的位置"""
    train_agent_idx = None
    for idx in [10, 11, 12, 13]:
        if idx not in enemies_idx + [teammate_idx]:
            train_agent_idx = idx
            break
    assert train_agent_idx is not None
    maps.append(board == train_agent_idx)              # 特征6，1层

    return np.stack(maps)


def get_board_obs_size():
    return 28, 11, 11


def to_flat_obs(obs_list):
    flat_obs_list = []
    for obs in obs_list:
        flat_obs_list.append(to_agent_flat_obs(obs))
    return flat_obs_list


def to_agent_flat_obs(obs):
    """确定编号"""
    teammate_idx = obs['teammate'].value
    enemies_idx = []
    for e in obs['enemies']:
        if not e.value == 9:  # AgentDummy
            enemies_idx.append(e.value)
    my_agent_idx = None
    for idx in [10, 11, 12, 13]:
        if idx not in enemies_idx + [teammate_idx]:
            my_agent_idx = idx
            break
    assert my_agent_idx is not None

    """alive 特征"""
    # 队友 alive
    tm_alive = [0]                 # 特征 1，长度1
    if teammate_idx in obs['alive']:
        tm_alive[0] = 1
    # 敌人 alive
    en_alive = [0, 0]                # 特征2，长度2
    for idx, i in zip(enemies_idx, range(2)):
        if idx in obs['alive']:
            en_alive[i] = 1
    # 自己 alive
    me_alive = [0]                    # 特征3，长度1
    if my_agent_idx in obs['alive']:
        me_alive[0] = 1

    # 炸弹爆炸范围
    blast_strength = [obs['blast_strength'] / 5]     # 特征4，长度1

    # 是否可踢
    can_kick = [1] if obs['can_kick'] else [0]          # 特征5，长度1

    # 剩余炸弹量
    ammo = [obs['ammo'] / 5]                        # 特征6，长度1

    # step count
    step_count = [obs['step_count'] / 500]           # 特征7，长度1

    # 我视野里敌人数量
    num_enemy_in_view = [0, 0, 0]                   # 特征8，长度3
    enemy_in_view = np.in1d(enemies_idx, obs['board'])  # 得到长度为2的bool数组
    if all(enemy_in_view):
        num_enemy_in_view[2] = 1        # 视野里敌人数量为2
    elif any(enemy_in_view):
        num_enemy_in_view[1] = 1        # 视野里敌人数量为1
    else:
        num_enemy_in_view[0] = 1        # 视野里敌人数量为0

    # flat_obs = tm_alive + en_alive + me_alive + blast_strength + can_kick + ammo + step_count + num_enemy_in_view

    flat_obs = np.concatenate((tm_alive, en_alive, me_alive, blast_strength,
                               can_kick, ammo, step_count, num_enemy_in_view))
    # print(flat_obs)
    return flat_obs


def get_flat_obs_size():
    return 11
