import copy

import torch
import torch.nn.functional as F
from torch import nn, optim


class MatchaNet(nn.Module):
    def __init__(self, input_dim, output_dim):#1次元の入力版
        super().__init__()

        self.online = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        #=============
        # cpt = torch.load("./trained_models/online_matcha_net_0_20231130_0.chkpt")
        # stdict_m = cpt["model"]
        # self.online.load_state_dict(stdict_m)
        #=============

        self.target = copy.deepcopy(self.online)

        # Q_target のパラメータは固定されます
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        input = input.to(torch.float)
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)        