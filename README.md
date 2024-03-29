# run
```python main.py --config=parallel_coma --env-config=pmm```

# state
'agents_attrs': 
[{'position': (3, 1), 'blast_strength': 2, 'can_kick': False, 'teammate': <Item.Agent2: 12>, 'ammo': 0, 'enemies': [<Item.Agent1: 11>, <Item.Agent3: 13>, <Item.AgentDummy: 9>]},
 {'position': (9, 3), 'blast_strength': 2, 'can_kick': False, 'teammate': <Item.Agent3: 13>, 'ammo': 0, 'enemies': [<Item.Agent0: 10>, <Item.Agent2: 12>, <Item.AgentDummy: 9>]}, 
 {'position': (9, 7), 'blast_strength': 2, 'can_kick': False, 'teammate': <Item.Agent0: 10>, 'ammo': 0, 'enemies': [<Item.Agent1: 11>, <Item.Agent3: 13>, <Item.AgentDummy: 9>]}, 
 {'position': (1, 8), 'blast_strength': 2, 'can_kick': False, 'teammate': <Item.Agent1: 11>, 'ammo': 1, 'enemies': [<Item.Agent0: 10>, <Item.Agent2: 12>, <Item.AgentDummy: 9>]}]
```
# 全局 state:
state = {'alive': alive_agents}
state['board'] = curr_board

bomb_blast_strengths, bomb_life, bomb_moving_direction = make_bomb_maps(agents[0].position)
state['bomb_blast_strength'] = bomb_blast_strengths
state['bomb_life'] = bomb_life
state['bomb_moving_direction'] = bomb_moving_direction

flame_life = make_flame_map(agents[0].position)
state['flame_life'] = flame_life

agents_attrs = []
for agent in agents:
    attrs_dict = {}
    for attr in attrs:
        assert hasattr(agent, attr)
        attrs_dict[attr] = getattr(agent, attr)
    agents_attrs.append(attrs_dict)
state['agents_attrs'] = agents_attrs
```

# observation （其中一个智能体的某一帧的 observation）
### 较有可能用到的
  `活着的智能体编号`  
  'alive': [10, 11, 12]  
  
`棋盘编号`  
'board':   
array([[ 0,  1,  2,  1,  2,  1,  5,  5,  5,  5,  5],  
       [ 1,  0,  3,  0,  2,  2,  5,  5,  5,  5,  5],  
       [ 0,  0,  0,  1,  1,  2,  5,  5,  5,  5,  5],  
       [ 1, 10,  1,  0,  0,  0,  5,  5,  5,  5,  5],  
       [ 2,  2,  1,  0,  0,  2,  5,  5,  5,  5,  5],  
       [ 1,  2,  2,  0,  2,  0,  5,  5,  5,  5,  5],  
       [ 1,  2,  0,  1,  0,  2,  5,  5,  5,  5,  5],  
       [ 1,  0,  1,  1,  1,  2,  5,  5,  5,  5,  5],  
       [ 5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5],  
       [ 5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5],  
       [ 5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5]], dtype=uint8),   
       
`炸弹范围（视野外视为无炸弹，即为0）`  
'bomb_blast_strength':   
array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]),     

`炸弹生命值（视野外视为无炸弹，即为0）`  
'bomb_life':   
array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 3., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]),  
       
`炸弹运动方向（视野外视为无炸弹，即为0）`  
'bomb_moving_direction':   
array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]),   

`爆炸火花生命值（视野外视为无炸弹，即为0）`  
'flame_life':   
array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]),  

`本智能体位置 `  
'position': (3, 1),   

`本智能体炸弹范围`  
'blast_strength': 3,   

`本智能体是否可踢`  
'can_kick': False,   

`本智能体剩余炸弹量`  
 'ammo': 0,  

`队友传来的信息（队友死或刚开始为0，否则范围是[1, 8]）`  
 'message': (0, 0)}  
 
### 不太会用到的  
 
`走了几步`  
'step_count': 20,  
 
 `对局类型`  
 'game_type': 3,   

`对局环境`  
'game_env': 'pommerman.envs.v2:Pomme',   

`队友`  
'teammate': <Item.Agent2: 12\>,  

`敌人`  
 'enemies': [<Item.Agent1: 11\>, <Item.Agent3: 13\>, <Item.AgentDummy: 9\>],