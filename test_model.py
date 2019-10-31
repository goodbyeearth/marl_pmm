import pommerman
from pommerman import agents
from controllers.test_controller import TestMAC
import torch as th
from components.featurize import *

def main():
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)
    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        # agents.DockerAgent("d9fc50459a6d", port=33333),
        agents.SimpleAgent(),
        agents.RandomAgent(),
    ]
    env = pommerman.make('PommeRadioCompetition-v2', agent_list)
    env_info = {"board_state_shape": get_board_state_size(),
                "flat_state_shape": get_flat_state_size(),
                "board_obs_shape": get_board_obs_size(),
                "flat_obs_shape": get_flat_obs_size(),
                "n_actions": 6,
                "n_agents": 2,
                "episode_limit": 800}
    scheme = {
        "board_state": {"vshape": env_info["board_state_shape"]},
        "flat_state": {"vshape": env_info["flat_state_shape"]},
        "board_obs": {"vshape": env_info["board_obs_shape"], "group": "agents"},
        "flat_obs": {"vshape": env_info["flat_obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    rnn_hidden_dim = 256
    mac = TestMAC(scheme=scheme, agent_output_type=None, rnn_hidden_dim=rnn_hidden_dim,
                  model_load_path='/home/hiogdong/pymarl_pmm/results/models/coma_pmm__2019-10-30_20-02-55/356/agent.th')
    test_idx_list = [0, 2]


    n_episode = 400
    for i_episode in range(n_episode):
        obs = env.reset()
        mac.last_action = [th.zeros(6), th.zeros(6)]
        mac.init_hidden(1, rnn_hidden_dim)
        done = False
        frame = 0
        print('env max step:', env._max_steps)
        while not done:
            actions = env.act(obs)
            for idx, agent_idx in enumerate(test_idx_list):
                action_agent = mac.select_actions(obs[agent_idx], idx).item()
                temp = th.zeros(6)
                temp[action_agent] = 1
                mac.last_action[idx] = temp
                actions[agent_idx] = action_agent
            obs, reward, done, info = env.step(actions)
            env.render()




if __name__ == '__main__':
    main()