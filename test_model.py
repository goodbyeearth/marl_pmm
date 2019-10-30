from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from components import env_utils, featurize
import torch as th

class MyMAC:
    def __init__(self, scheme, args):
        self.args = args
        self.n_agents = args.n_agents
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None
        self.episode_start = True        # todo: 在每回合开始前设置
        self.last_action =

    def select_actions(self, obs, agent_idx):    # agent_idx: 0/1
        avail_actions = env_utils.get_avail_agent_actions(obs)
        agent_outputs = self.forward()

    def forward(self):
        pass

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, obs, agent_idx):
        board_obs = featurize.to_agent_board_obs(obs)  # todo: unsqueeze(0)
        flat_obs = featurize.to_agent_flat_obs(obs)
        inputs = {'board_inputs': board_obs, 'flat_inputs': [flat_obs]}

        if self.args.obs_last_action:
            if self.episode_start:
                self.episode_start = False
                inputs['flat_inputs'].append(th.zeros_like())

    def _get_input_shape(self, scheme):
        input_shape = {
            'board_shape': scheme['board_obs']['vshape'],
            'flat_shape': scheme["flat_obs"]["vshape"],
        }
        if self.args.obs_last_action:
            input_shape['flat_shape'] += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape['flat_shape'] += self.n_agents

        return input_shape