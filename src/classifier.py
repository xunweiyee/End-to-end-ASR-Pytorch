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


class CNNClassifier(nn.Module):
    def __init__(self, input_dim):
        super(CNNClassifier, self).__init__()
        hidden_dim_1 = 64
        hidden_dim_2 = hidden_dim_1 * 2

        self.out_dim = input_dim * 8
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=hidden_dim_1, kernel_size=(3,), stride=(1,), padding=(1,)),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim_1, out_channels=hidden_dim_1, kernel_size=(3,), stride=(1,), padding=(1,)),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=hidden_dim_1, out_channels=hidden_dim_2, kernel_size=(3,), stride=(1,), padding=(1,)),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim_2, out_channels=hidden_dim_2, kernel_size=(3,), stride=(1,), padding=(1,)),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Linear((input_dim // 4) * hidden_dim_2, self.out_dim),
        )


    def reshape_input(self, feature, group_size):
        down_sample_len = feature.size(1) // group_size
        feature = feature[:,:down_sample_len*group_size,:]
        reshape_feature = feature.reshape(feature.size(0) * down_sample_len, group_size*feature.size(2))
        return reshape_feature


    def forward(self, feature):
        print(f"{feature}\n\nSize:{feature.size()}")
        return self.classifier(feature)


