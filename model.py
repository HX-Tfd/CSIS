import torch.nn as nn
import torch

from tqdm import tqdm

class SimpleFCNet(nn.Module):

    def __init__(self, num_layers, in_dim, out_dim, hidden_dim):
        super(SimpleFCNet, self).__init__()
        self.num_layers = num_layers
        self.sigm = nn.Sigmoid()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.fc_hidd = nn.Linear(hidden_dim, hidden_dim)

        layers = []
        for i in range(self.num_layers + 2):
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


def train(model, x, y):
    num_epochs = 100
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    loss = nn.MSELoss()

    for i in tqdm(range(num_epochs), desc="training model ..."):
        res = model(x)
        model_loss = loss(res, y)
        model_loss.backward()
        optimizer.step()
        print("epoch {}: loss = {}".format(i, model_loss))