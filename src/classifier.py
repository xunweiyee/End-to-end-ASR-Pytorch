from torch import nn as nn


class MLPCLassfier(nn.Module):
    def __init__(self, input_dim):
        super(MLPCLassfier, self).__init__()
        h1 = input_dim*2
        h2 = h1*2
        self.out_dim = h2*2
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h2),
            nn.ReLU(),
            nn.Linear(h2, h2),
            nn.ReLU(),
            nn.Linear(h2, self.out_dim),
        )

    def reshape_input(self, feature, group_size):
        down_sample_len = feature.size(1) // group_size
        feature = feature[:,:down_sample_len*group_size,:]
        reshape_feature = feature.reshape(feature.size(0) * down_sample_len, group_size*feature.size(2))
        return reshape_feature


    def forward(self, feature):


        return self.classifier(feature)