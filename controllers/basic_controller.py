from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]   # 当前回合的最后一步，即第 t_ep 步
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)  # 得到所有agent的动作
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        print('chosen_actions, ', chosen_actions)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, self.args.rnn_hidden_dim)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):     # todo: 设置该参数
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # 统计了动作数目
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    # 只统计可选动作数目
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden(self.args.rnn_hidden_dim).unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args.rnn_hidden_dim)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size

        board_obs = batch['board_obs'][:, t].reshape((bs*self.n_agents, ) + batch['board_obs'][:, t].shape[2:])
        # print('board_obs.shape', board_obs.shape)

        inputs = {'board_inputs': board_obs, 'flat_inputs': [batch['flat_obs'][:, t]]}

        if self.args.obs_last_action:
            if t == 0:
                inputs['flat_inputs'].append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs['flat_inputs'].append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            # 如果obs_agent_id为true，对智能体编号进行 onehot 后加入 input
            inputs['flat_inputs'].append(
                th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1)   # 得到 2 维数组
            )
        # print('flat_input[0]: ', inputs['flat_inputs'][0].shape)
        # print('flat_input[1]: ', inputs['flat_inputs'][1].shape)
        # print('flat_input[2]: ', inputs['flat_inputs'][2].shape)

        # 得到 2 维数组，第一维是样本编号，一个样本编号代表一个智能体一步的特征
        inputs['flat_inputs'] = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs['flat_inputs']], dim=1)
        # print('final shape: ', inputs['flat_inputs'].shape)
        # print(inputs['flat_inputs'].shape)
        return inputs

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
