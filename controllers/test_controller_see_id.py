from modules.agents import REGISTRY as agent_REGISTRY
from components import env_utils, featurize
import torch as th


class TestSeeIdMAC:
    def __init__(self, scheme, agent_output_type, model_load_path, rnn_hidden_dim):
        self.n_agents = 2
        # input_shape = self._get_input_shape(scheme)
        # self._build_agents(input_shape)
        self.agent_output_type = agent_output_type
        self.input_shape = self._get_input_shape(scheme)
        self.agent = agent_REGISTRY['see_id_agent'](self.input_shape, rnn_hidden_dim)
        self.rnn_hidden_dim = rnn_hidden_dim
        self.load_models(path=model_load_path)
        self.hidden_states = None
        # self.episode_start = True
        self.last_action = None          # todo: 每一步都要设置

    def select_actions(self, obs, agent_idx):    # agent_idx: 0/1
        avail_actions = env_utils.get_avail_agent_actions(obs)
        avail_actions = th.Tensor(avail_actions).int()
        agent_output = self.forward(obs, agent_idx)
        chosen_action = select(agent_output, avail_actions)
        return chosen_action

    def forward(self, obs, agent_idx):
        agent_inputs = self._build_inputs(obs, agent_idx)
        agent_out, self.hidden_states = self.agent(agent_inputs, self.hidden_states, self.rnn_hidden_dim)
        print('agent_out:', agent_out)

        if self.agent_output_type == "pi_logits":
            # if getattr(self.args, "mask_before_softmax", True):
            #     agent_out[avail_actions == 0] = -1e10
            agent_out = th.nn.functional.softmax(agent_out)

        return agent_out

    def init_hidden(self, batch_size, rnn_hidden_dim):
        self.hidden_states = self.agent.init_hidden(rnn_hidden_dim).unsqueeze(0).expand(batch_size, 1, -1)

    def load_models(self, path):
        # self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        # self.agent.load_state_dict(th.load(path), map_location=lambda storage, loc: storage)
        self.agent.load_state_dict(th.load(path))

    def _build_inputs(self, obs, agent_idx):      # agent_idx: 0/1
        board_obs = featurize.to_agent_board_obs(obs)
        board_obs = th.Tensor(board_obs).unsqueeze(0)
        flat_obs = featurize.to_agent_flat_obs(obs)
        inputs = {'board_inputs': board_obs, 'flat_inputs': [flat_obs]}

        if agent_idx == 0 or agent_idx == 1:
            k = 0
        elif agent_idx == 2 or agent_idx == 3:
            k = 1
        else:
            raise ValueError
        inputs['flat_inputs'].append(self.last_action[k])

        id_inputs = th.Tensor([1, 0]) if agent_idx == 0 or agent_idx == 1 else th.Tensor([0, 1])
        inputs['flat_inputs'].append(id_inputs)

        inputs['flat_inputs'] = th.cat([th.Tensor(x) for x in inputs['flat_inputs']]).unsqueeze(0)

        inputs['id_inputs'] = id_inputs.unsqueeze(0)
        # print(inputs['board_inputs'].shape)
        # print(inputs['flat_inputs'].shape)

        return inputs

    def _get_input_shape(self, scheme):
        input_shape = {
            'board_shape': scheme['board_obs']['vshape'],
            'flat_shape': scheme["flat_obs"]["vshape"],
        }
        input_shape['flat_shape'] += 6
        input_shape['flat_shape'] += self.n_agents

        input_shape['id_shape'] = self.n_agents

        return input_shape


def select(agent_input, avail_action):
    masked_policies = agent_input.clone()
    avail_action = avail_action.unsqueeze(0)
    masked_policies[avail_action == 0.0] = 0.0

    # print(masked_policies)
    picked_actions = masked_policies.max(1)[1]
    # print(picked_actions)
    return picked_actions
