from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class TwoAgentMAC:
    def __init__(self, scheme, groups, args):
        # self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states_1 = None
        self.hidden_states_2 = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        bs = ep_batch.batch_size
        avail_actions_1 = ep_batch["avail_actions"][:, t_ep][:, 0, :].unsqueeze(0)
        avail_actions_2 = ep_batch["avail_actions"][:, t_ep][:, 1, :].unsqueeze(0)
        agent_outputs_1, agent_outputs_2 = self.forward(ep_batch, t_ep, test_mode=test_mode)  # 得到所有agent的动作
        # print(avail_actions_1)
        # print(avail_actions_2)
        # print('agent_output_1:', agent_outputs_1)
        # print('agent_output_2:', agent_outputs_2)
        chosen_actions_1 = self.action_selector.select_action(agent_outputs_1, avail_actions_1, t_env,
                                                              test_mode=test_mode)
        chosen_actions_2 = self.action_selector.select_action(agent_outputs_2, avail_actions_2, t_env,
                                                              test_mode=test_mode)
        # print('chosen_actions_1:', chosen_actions_1)
        # print('chosen_actions_2:', chosen_actions_2)
        return chosen_actions_1, chosen_actions_2

    def forward(self, ep_batch, t, test_mode=False):
        bs = ep_batch.batch_size
        slice_batch_1 = slice(0, bs, 2)
        slice_batch_2 = slice(1, bs, 2)
        agent_inputs_1, agent_inputs_2 = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        avail_actions_1 = avail_actions[slice_batch_1]
        avail_actions_2 = avail_actions[slice_batch_2]
        agent_outs_1, self.hidden_states_1 = self.agent_1(agent_inputs_1, self.hidden_states_1,
                                                          self.args.rnn_hidden_dim)
        agent_outs_2, self.hidden_states_2 = self.agent_2(agent_inputs_2, self.hidden_states_2,
                                                          self.args.rnn_hidden_dim)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):     # todo: 设置该参数
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions_1 = avail_actions_1.reshape(ep_batch.batch_size, -1)
                reshaped_avail_actions_2 = avail_actions_2.reshape(ep_batch.batch_size, -1)
                agent_outs_1[reshaped_avail_actions_1 == 0] = -1e10
                agent_outs_2[reshaped_avail_actions_2 == 0] = -1e10

            agent_outs_1 = th.nn.functional.softmax(agent_outs_1, dim=-1)
            agent_outs_2 = th.nn.functional.softmax(agent_outs_2, dim=-1)

            if not test_mode:
                # 统计了动作数目
                # Epsilon floor
                epsilon_action_num_1 = agent_outs_1.size(-1)
                epsilon_action_num_2 = agent_outs_2.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    # 只统计可选动作数目
                    epsilon_action_num_1 = reshaped_avail_actions_1.sum(dim=1, keepdim=True).float()
                    epsilon_action_num_2 = reshaped_avail_actions_2.sum(dim=1, keepdim=True).float()

                agent_outs_1 = ((1 - self.action_selector.epsilon) * agent_outs_1
                               + th.ones_like(agent_outs_1) * self.action_selector.epsilon/epsilon_action_num_1)
                agent_outs_2 = ((1 - self.action_selector.epsilon) * agent_outs_2
                                + th.ones_like(agent_outs_2) * self.action_selector.epsilon / epsilon_action_num_2)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs_1[reshaped_avail_actions_1 == 0] = 0.0
                    agent_outs_2[reshaped_avail_actions_2 == 0] = 0.0

        # print(agent_outs_1.shape)
        # print(agent_outs_2.shape)
        return agent_outs_1.view(ep_batch.batch_size, 1, -1), agent_outs_2.view(ep_batch.batch_size, 1, -1)

    def init_hidden(self, batch_size):
        self.hidden_states_1 = self.agent_1.init_hidden(self.args.rnn_hidden_dim).unsqueeze(0).expand(batch_size, 1, -1)  # bav
        self.hidden_states_2 = self.agent_2.init_hidden(self.args.rnn_hidden_dim).unsqueeze(0).expand(batch_size, 1, -1)

    def parameters(self):
        return [self.agent_1.parameters(), self.agent_2.parameters()]

    def load_state(self, other_mac_1, other_mac_2):
        self.agent_1.load_state_dict(other_mac_1.agent.state_dict())
        self.agent_2.load_state_dict(other_mac_2.agent.state_dict())

    def cuda(self):
        self.agent_1.cuda()
        self.agent_2.cuda()

    def save_models(self, path):
        th.save(self.agent_1.state_dict(), "{}/agent_1.th".format(path))
        th.save(self.agent_2.state_dict(), "{}/agent_2.th".format(path))

    def load_models(self, path):
        self.agent_1.load_state_dict(th.load("{}/agent_1.th".format(path), map_location=lambda storage, loc: storage))
        self.agent_2.load_state_dict(th.load("{}/agent_2.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent_1 = agent_REGISTRY[self.args.agent](input_shape, self.args.rnn_hidden_dim)
        self.agent_2 = agent_REGISTRY[self.args.agent](input_shape, self.args.rnn_hidden_dim)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size

        # print('board shape:', batch['board_obs'].shape)
        board_obs = batch['board_obs'][:, t].reshape((bs*2, ) + batch['board_obs'][:, t].shape[2:])
        slice_batch_1 = slice(0, bs*2, 2)
        slice_batch_2 = slice(1, bs*2, 2)
        board_obs_1 = board_obs[slice_batch_1, :, :, :]
        board_obs_2 = board_obs[slice_batch_2, :, :, :]

        flat_inputs = [batch['flat_obs'][:, t]]
        if self.args.obs_last_action:
            if t == 0:
                flat_inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                flat_inputs.append(batch["actions_onehot"][:, t-1])

        # 得到 2 维数组，第一维是样本编号，一个样本编号代表一个智能体一步的特征
        flat_inputs = th.cat([x.reshape(bs*2, -1) for x in flat_inputs], dim=1)

        flat_inputs_1 = flat_inputs[slice_batch_1, :]
        flat_inputs_2 = flat_inputs[slice_batch_2, :]
        # print(board_obs.shape)
        # print(flat_inputs_1.shape)
        # print(flat_inputs_2.shape)
        # print('=========================')
        inputs_1 = {'board_inputs': board_obs_1, 'flat_inputs': flat_inputs_1}
        inputs_2 = {'board_inputs': board_obs_2, 'flat_inputs': flat_inputs_2}

        return inputs_1, inputs_2

    def _get_input_shape(self, scheme):
        input_shape = {
            'board_shape': scheme['board_obs']['vshape'],
            'flat_shape': scheme["flat_obs"]["vshape"],
        }
        if self.args.obs_last_action:
            input_shape['flat_shape'] += scheme["actions_onehot"]["vshape"][0]
        # if self.args.obs_agent_id:
        #     input_shape['flat_shape'] += self.n_agents

        return input_shape
