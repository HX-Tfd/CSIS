import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

'''
A fully-connected neural network that parameterizes
the distribution induced by the driving forces

outputs a vector containing the probabilities of change
'''
#TODO: maybe use other non-linearities for the hidden layers (e.g. ReLU)
#TODO (optional): maybe use batchnorm
class SimpleFCNet(nn.Module):

    def __init__(self, num_layers, in_dim, out_dim, hidden_dim, use_bn=False):
        super(SimpleFCNet, self).__init__()
        self.num_layers = num_layers
        self.sigm = nn.Sigmoid()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.fc_hidd = nn.Linear(hidden_dim, hidden_dim)

        # self.use_bn = use_bn
        # self.bn = nn.BatchNorm1d()

        layers = []
        for i in range(self.num_layers + 2):
            # if self.use_bn:
            #     layers.append()
            if i == 0:
                layers.append(self.fc_in)
            elif i == self.num_layers + 1:
                layers.append(self.fc_out)
            else:
                layers.append(self.fc_hidd)
            layers.append(self.sigm)
        self.net = nn.Sequential(*layers)


    def forward(self, x):
        return self.net(x)

'''
x: driving forces produced by the model 
y: opinion vector obtained from the data (or change of opinion) 
timestep: an integer denoting the timestep of the simulation, used by summarywriter
writer: summary writer for training (global), none is used by default

we are invoking the training procedure in each simulation timestep

not using mini-batches
'''

def train(model, x, y, timestep, writer=None):
    num_epochs = 50
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    loss = nn.MSELoss()

    for i in tqdm(range(num_epochs), desc="training model ..."):
        def closure():
            optimizer.zero_grad()
            res = model(x)
            model_loss = loss(res, y) #TODO: create a meaningful loss function
            model_loss.backward()
            if writer is not None:
                writer.add_scalar('training loss time step {}'.format(timestep), model_loss, i)
            else:
                print("epoch {}: loss = {}".format(i, model_loss))
            return model_loss
        optimizer.step(closure())
