import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from tqdm import tqdm

from loss_functions import *

'''
A fully-connected neural network that parameterizes
the distribution induced by the driving forces

outputs a vector containing the probabilities of change
'''
class SimpleFCNet(nn.Module):

    def __init__(self, num_layers, in_dim, out_dim, hidden_dim, use_bn=False):
        super(SimpleFCNet, self).__init__()
        self.num_layers = num_layers
        self.sigm = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.fc_hidd = nn.Linear(hidden_dim, hidden_dim)

        layers = []
        for i in range(self.num_layers + 1):
            if i == 0:
                layers.append(self.fc_in)
                layers.append(self.relu)
            elif i == self.num_layers:
                layers.append(nn.Dropout(p=0.8))
                layers.append(self.fc_out)
                layers.append(self.sigm)
            else:
                layers.append(nn.Dropout(p=0.8))
                layers.append(self.fc_hidd)
                layers.append(self.relu)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# x = [parameters_to_fit, driving_forces]
def train(model, parameters_to_fit, driving_forces, y, timestep, all_opinion, writer=None):
    num_epochs = 10
    learning_rate = 0.01
    optimizer = torch.optim.SGD(params=
                                 [{'params': model.parameters()},
                                  {'params':parameters_to_fit}],
                                 lr=learning_rate,
                                 momentum=0.8)
    loss = nn.MSELoss()

    def opinion(x, p):
        res = (np.random.rand(1))[0]
        return 1 - x if res < p else x

    def opinion_no_return(x, p):
        if (x == 1): # shouldn't the condition be if has_changed[] == True?
            return 1
        else:
            return opinion(x, p)

    def get_opinion(current_opinion, probs, step):
        opinions = []
        for agent in range(len(probs)):
            new_state_no_return = opinion_no_return(current_opinion[agent], probs[agent])
            opinions.append(new_state_no_return)
        opinions = torch.tensor(opinions).float()
        opinions.requires_grad = True
        return opinions

    # training
    res = None
    for i in tqdm(range(num_epochs), desc="training model ..."):
        optimizer.zero_grad()
        inp = torch.cat((parameters_to_fit, driving_forces.detach())).float()
        res = model(inp)

        # torch cannot handle conditional updates
        x = res #get_opinion(all_opinion[timestep - 1], res, timestep)
        model_loss = loss(x, y) # TODO: create a meaningful loss function
        print(model_loss)
        model_loss.backward()
        optimizer.step()
        if writer is not None:
            writer.add_scalar('training loss time step {}'.format(timestep), model_loss, i)
    print("params:", parameters_to_fit)
    print("pgrad: ", parameters_to_fit.grad)

    return get_opinion(all_opinion[timestep - 1], res, timestep)


'''
parameters_to_fit: N * 4
driving_forces: N * num_agents 
all_opinions: N * num_agents 

y: real stat of opinions from the data
'''
def train_global(model, parameters_to_fit, driving_forces,  all_opinions, y, writer=None):
    num_epochs = 50
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(params=
                                 [{'params': model.parameters()},
                                  {'params':parameters_to_fit}],
                                 lr=learning_rate)
    loss = nn.MSELoss()
    dataset_length = len(driving_forces)
    print(dataset_length)

    # shuffle dataset
    shuffled_ids = torch.randperm(dataset_length)


    # training
    for i in tqdm(range(dataset_length), desc="training model ..."):
        optimizer.zero_grad()

        inp_1 = driving_forces[shuffled_ids[i]].detach()
        inp_2 = all_opinions[shuffled_ids[i]].detach()
        inp = torch.cat((parameters_to_fit, inp_1, inp_2)).float()
        x = model(inp) #opinions
        x = torch.round(x)
        print(x)

        model_loss = loss(x, y) # mean squared diff of opinions
        model_loss.backward()
        optimizer.step()
        if writer is not None:
            writer.add_scalar('training loss', model_loss, i)

    print("params:", parameters_to_fit)
    print("pgrad: ", parameters_to_fit.grad)
