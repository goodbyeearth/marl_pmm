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
    for i in range(2, 13):
        maps.append(bomb_life == i)

    '''将bomb direction编码为one-hot'''
    bomb_moving_direction = state['bomb_moving_direction']
    for i in range(1, 5):
        maps.append(bomb_moving_direction == i)

    """棋盘物体 one hot"""
    for i in range(9):  # [0, 1, ..., 8]
        maps.append(board == i)

    """四个智能体的位置"""
    assert train_list is not None
    for i in [10, 11, 12, 13]:
        if i not in train_list:
            maps.append(board == i)     # 敌人先后顺序无所谓
    for i in [10, 11, 12, 13]:
        if i in train_list:
            maps.append(board == i)     # 这个先后顺序是有意义的，一个是自己一个是队友，故输入必须有编号才能分辨该特征

    return maps


def get_board_state_size():
    pass


# TODO： 记得敌方跟己方的特征分开, 并且按顺序
def to_flat_state(state, train_list):
    if train_list[0] == 0:
        enemy_list = [1, 3]
    else:
        enemy_list = [2, 4]

    # alive 特征
    ally_alive = [0, 0]    # 特征 1，长度2
    enemy_alive = [0, 0]   # 特征 2，长度2
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
    pass


def to_board_obs(obs_list):
    board_obs = []
    for obs in obs_list:
        pass


def get_board_obs_size():
    pass


def to_flat_obs(obs_list):
    # 我视野里敌人的数量
    pass


def get_flat_obs_size():
    pass