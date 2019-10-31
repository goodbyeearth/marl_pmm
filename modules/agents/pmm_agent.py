import torch.nn as nn
import torch.nn.functional as F
import torch as th


class PmmAgent(nn.Module):
    def __init__(self, input_shape, rnn_hidden_dim):
        super(PmmAgent, self).__init__()
        # self.args = args
        self.conv1 = nn.Conv2d(input_shape['board_shape'][0], 32, 3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(64*5*5 + input_shape['flat_shape'], rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        self.fc2 = nn.Linear(rnn_hidden_dim, 6)

    def init_hidden(self, rnn_hidden_dim):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, rnn_hidden_dim):
        board_obs = F.relu(self.conv1(inputs['board_inputs']))
        board_obs = F.relu(self.conv2(board_obs))
        board_obs = F.relu(self.conv3(board_obs))
        board_obs = board_obs.view(board_obs.size(0), -1)    # 展开成 (x.size(0), x.size(1)*x.size(2)*x.size(3))

        obs = th.cat([board_obs, inputs['flat_inputs']], dim=1)
        # print(obs.shape)
        obs = F.relu(self.fc1(obs))
        # print('hidden_state.shape', hidden_state.shape)
        # print(type(rnn_hidden_dim))
        h_in = hidden_state.reshape(-1, rnn_hidden_dim)
        # print('h_in:', h_in.shape)
        # print('obs:', obs.shape)
        h = self.rnn(obs, h_in)
        q = self.fc2(h)

        return q, h
