import torch
from torch import nn as nn
from torch.nn import functional as F


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


class CNNClassifier(nn.Module):
    def __init__(self, input_dim):
        super(CNNClassifier, self).__init__()
        self.hidden_dim_1 = hidden_dim_1 = input_dim // 16
        self.hidden_dim_2 = hidden_dim_2 = hidden_dim_1 * 2

        self.out_dim = input_dim // 4

        conv_hyperparams = {
            "kernel_size": (3,3),
            "dilation": (1,1),
            "stride": (1,1),
            "padding": (1,1),
        }

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=hidden_dim_1, **conv_hyperparams)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim_1, out_channels=hidden_dim_1, **conv_hyperparams)
        self.conv3 = nn.Conv2d(in_channels=hidden_dim_1, out_channels=hidden_dim_2, **conv_hyperparams)
        self.conv4 = nn.Conv2d(in_channels=hidden_dim_2, out_channels=hidden_dim_2, **conv_hyperparams)
        self.dense = nn.Linear(hidden_dim_2 * self.out_dim, self.out_dim)
        self.pool = nn.MaxPool2d((1,2))

    def reshape_input(self, feature, group_size):
        down_sample_len = feature.size(1) // group_size
        feature = feature[:,:down_sample_len*group_size,:]
        reshape_feature = feature.reshape(feature.size(0) * down_sample_len, group_size*feature.size(2))
        return reshape_feature

    def forward(self, feature):
        # Input size is varied - size is N timesteps x 384 features. Batch size = 1 and number of input channels = 1.
        # Therefore, inputs must be unsqueezed.
        feature = feature.unsqueeze(dim=0).unsqueeze(dim=1)

        # Format: input_channels x timesteps x features --> output_channels x new_timesteps x new_features
        feature = self.conv1(feature)  # 1 x N x 384 --> 24 x N x 384
        feature = F.relu(feature)
        feature = self.conv2(feature)  # 24 x N x 384 --> 24 x N x 384
        feature = F.relu(feature)
        feature = self.pool(feature)  # 24 x N x 384 --> 24 x N x 192
        feature = self.conv3(feature)  # 24 x N x 192 --> 48 x N x 192
        feature = F.relu(feature)
        feature = self.conv4(feature)  # 48 x N x 192 --> 48 x N x 192
        feature = F.relu(feature)
        feature = self.pool(feature)  # 48 x N x 192 --> 48 x N x 96

        feature = feature.transpose(0, 1)  # 48 x N x 96 --> N x 48 x 96
        feature = feature.view(-1, self.hidden_dim_2 * self.out_dim) # N x 48 x 96 --> N x (48 x 96)
        feature = torch.stack([
            self.dense(feature[index].unsqueeze(dim=0)) 
            for index in range(feature.size()[0])])  # N x (48 x 96) --> N x 96
        feature = feature.transpose(0, 1)  # N x 96 --> 96 x N
        return feature


