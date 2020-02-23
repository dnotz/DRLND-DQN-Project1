import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, fc_layer_sizes, seed, dueling=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc_layer_sizes (list of int): Layer size of each FC layer
            seed (int): Random seed
            dueling (bool): Whether or not to use Dueling DQN Network architecture
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.dueling = dueling

        # define network fc layers 
        self.linears = nn.ModuleList([])
        if len(fc_layer_sizes) == 0:
            self.linears.append(nn.Linear(state_size, action_size))
        else:
            self.linears.append(nn.Linear(state_size, fc_layer_sizes[0]))
            for i in range(1, len(fc_layer_sizes)):
                self.linears.append(nn.Linear(fc_layer_sizes[i-1], fc_layer_sizes[i]))
            self.linears.append(nn.Linear(fc_layer_sizes[-1], action_size))

        # if Dueling DQN network: define last two fc layers for value function estimation
        if self.dueling:
            self.linears_val = nn.ModuleList([])
            if len(fc_layer_sizes) == 0:
                self.linears_val.append(nn.Linear(state_size, 1))
            elif len(fc_layer_sizes) == 1:
                self.linears_val.append(nn.Linear(state_size, fc_layer_sizes[-1]))
                self.linears_val.append(nn.Linear(fc_layer_sizes[-1], 1))
            else:
                self.linears_val.append(nn.Linear(fc_layer_sizes[-2], fc_layer_sizes[-1]))
                self.linears_val.append(nn.Linear(fc_layer_sizes[-1], 1))

    def forward(self, x):
        """Build a network that maps state -> action values."""
        if not self.dueling:
            for i in range(len(self.linears) - 1):
                x = F.relu(self.linears[i](x))
            x = self.linears[-1](x)
        else:
            for i in range(len(self.linears) - len(self.linears_val)):
                x = F.relu(self.linears[i](x))
            x_adv = x
            x_val = x
            for i in range(len(self.linears_val) - 1):
                x_adv = F.relu(self.linears[i + len(self.linears) - len(self.linears_val)](x_adv))
                x_val = F.relu(self.linears_val[i](x_val))
            x_adv = self.linears[-1](x_adv)
            x_val = self.linears_val[-1](x_val)

            # enforce zero average for advantage value, then add value and advantage
            x_adv -= x_adv.mean(1).unsqueeze(1)
            x = x_val + x_adv

        return x
