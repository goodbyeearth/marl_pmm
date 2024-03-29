import torch as th
import torch.nn as nn
import torch.nn.functional as F


class COMACritic(nn.Module):
    def __init__(self, scheme, args):
        super(COMACritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "q"

        # Set up network layers
        self.conv1 = nn.Conv2d(input_shape['board_state_shape'][0], 32, 3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        self.conv4 = nn.Conv2d(input_shape['board_obs_shape'][0], 32, 3, stride=1)
        self.conv5 = nn.Conv2d(32, 64, 3, stride=1)
        self.conv6 = nn.Conv2d(64, 64, 3, stride=1)

        self.fc1 = nn.Linear(64*5*5*2 + input_shape['flat_shape'], 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, self.n_actions)

    def forward(self, batch, t=None):
        inputs = self._build_inputs(batch, t=t)

        board_state = F.relu(self.conv1(inputs['board_state']))
        board_state = F.relu(self.conv2(board_state))
        board_state = F.relu(self.conv3(board_state))
        board_state = board_state.view(board_state.size(0), -1)

        board_obs = F.relu(self.conv4(inputs['board_obs']))
        board_obs = F.relu(self.conv5(board_obs))
        board_obs = F.relu(self.conv6(board_obs))
        board_obs = board_obs.view(board_obs.size(0), -1)

        feature_cat = th.cat([board_state, board_obs, inputs['flat_inputs']], dim=1)
        x = F.relu(self.fc1(feature_cat))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        q = q.reshape(batch.batch_size, -1, self.n_agents, self.n_actions)
        return q

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        # board_state
        board_state = batch["board_state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1, 1, 1)
        board_state = board_state.reshape((bs*max_t*self.n_agents, ) + board_state[:, ts].shape[3:])
        # print('board_state:', board_state.shape)

        # board_obs
        board_obs = batch["board_obs"][:, ts]
        board_obs = board_obs.reshape((bs*max_t*self.n_agents, ) + board_obs[:, ts].shape[3:])
        # print('board_obs:', board_obs.shape)

        inputs = {
            'board_state': board_state,
            'board_obs': board_obs,
            'flat_inputs': []
        }

        # flat_state
        inputs['flat_inputs'].append(batch["flat_state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # flat_obs
        inputs['flat_inputs'].append(batch["flat_obs"][:, ts])

        # actions
        actions = batch["actions_onehot"][:, ts].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        agent_mask = (1 - th.eye(self.n_agents, device=batch.device))
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
        inputs['flat_inputs'].append(actions * agent_mask.unsqueeze(0).unsqueeze(0))

        if t == 0:
            inputs['flat_inputs'].append(
                th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
            )
            # print('actions 1',
            #       th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1).shape)
        elif isinstance(t, int):
            inputs['flat_inputs'].append(
                batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
            )
            # print('actions 2', batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1).shape)
        else:
            last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
            last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
            inputs['flat_inputs'].append(last_actions)
            # print('actions 3', last_actions.shape)

        # agent idx
        inputs['flat_inputs'].append(
            th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1)
        )
        # print('agent idx: ', th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1).shape)

        inputs['flat_inputs'] = th.cat([x.reshape(bs*max_t*self.n_agents, -1) for x in inputs['flat_inputs']], dim=-1)
        # inputs['flat_inputs'] = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs['flat_inputs']], dim=-1)

        # print('board_input shape:', inputs['board_state'].shape)
        # print('flat_input shape:', inputs['flat_inputs'].shape)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = {
            'board_state_shape': scheme['board_state']['vshape'],
            'board_obs_shape': scheme['board_obs']['vshape'],
            'flat_shape': scheme["flat_obs"]["vshape"] + scheme['flat_state']['vshape'],
        }
        # actions and last actions
        input_shape['flat_shape'] += scheme["actions_onehot"]["vshape"][0] * self.n_agents * 2
        # print('actions and last actions: ', scheme["actions_onehot"]["vshape"][0] * self.n_agents * 2)
        # agent id
        input_shape['flat_shape'] += self.n_agents

        # print(input_shape)
        return input_shape
