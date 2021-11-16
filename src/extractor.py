import torch
from torch import nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class VGGExtractor(nn.Module):
    ''' VGG extractor for ASR described in https://arxiv.org/pdf/1706.02737.pdf'''

    def __init__(self, input_dim):
        super(VGGExtractor, self).__init__()
        self.init_dim = 64
        self.hide_dim = 128
        in_channel, freq_dim, out_dim = self.check_dim(input_dim)
        self.in_channel = in_channel
        self.freq_dim = freq_dim
        self.out_dim = out_dim

        self.extractor = nn.Sequential(
            nn.Conv2d(in_channel, self.init_dim, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.init_dim, self.init_dim, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # Half-time dimension
            nn.Conv2d(self.init_dim, self.hide_dim, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hide_dim, self.hide_dim, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)  # Half-time dimension
        )

    def check_dim(self, input_dim):
        # Check input dimension, delta feature should be stack over channel.
        if input_dim % 13 == 0:
            # MFCC feature
            return int(input_dim/13), 13, (13//4)*self.hide_dim
        elif input_dim % 40 == 0:
            # Fbank feature
            return int(input_dim/40), 40, (40//4)*self.hide_dim
        else:
            raise ValueError(
            'Acoustic feature dimension for VGG should be 13/26/39(MFCC) or 40/80/120(Fbank) but got '+input_dim)

    def view_input(self, feature, feat_len):
        # downsample time
        # feat_len = feat_len//4
        feat_len = torch.div(feat_len, 4, rounding_mode='floor')
        # crop sequence s.t. t%4==0
        if feature.shape[1] % 4 != 0:
            feature = feature[:, :-(feature.shape[1] % 4), :].contiguous()
        bs, ts, ds = feature.shape
        # stack feature according to result of check_dim
        feature = feature.view(bs, ts, self.in_channel, self.freq_dim)
        feature = feature.transpose(1, 2)

        return feature, feat_len

    def forward(self, feature, feat_len):

        # Feature shape BSxTxD -> BS x CH(num of delta) x T x D(acoustic feature dim)
        # BSx128xT/4xD/4 -> BSxT/4x128xD/4
        #  BS x T/4 x 128 x D/4 -> BS x T/4 x 32D

        # Feature shape BSxTxD -> BS x CH(num of delta) x T x D(acoustic feature dim)
        feature, feat_len = self.view_input(feature, feat_len) #downsample on time
        # Foward
        feature = self.extractor(feature)
        # BSx128xT/4xD/4 -> BSxT/4x128xD/4
        feature = feature.transpose(1, 2)
        #  BS x T/4 x 128 x D/4 -> BS x T/4 x 32D
        feature = feature.contiguous().view(feature.shape[0], feature.shape[1], self.out_dim)

        return feature, feat_len


class MLPExtractor(nn.Module):
    '''
        A simple MLP extractor for acoustic feature down-sampling
        Every 4 frame will be convert to corresponding features by MLP
    '''

    def __init__(self, input_dim, out_dim):
        super(MLPExtractor, self).__init__()

        self.out_dim = out_dim
        self.hide_dim = input_dim * 3
        self.extractor = nn.Sequential(
            nn.Linear(input_dim, self.hide_dim),
            nn.ReLU(),
            nn.Linear(self.hide_dim, self.hide_dim),
            nn.ReLU(),
            nn.Linear(self.hide_dim, self.hide_dim),
            nn.ReLU(),
            nn.Linear(self.hide_dim, self.hide_dim),
            nn.ReLU(),
            nn.Linear(self.hide_dim, self.out_dim),
        )

    def reshape_input(self, feature, group_size):
        down_sample_len = feature.size(1) // group_size
        feature = feature[:,:down_sample_len*group_size,:]
        reshape_feature = feature.reshape(feature.size(0) * down_sample_len, group_size*feature.size(2))
        return reshape_feature


    def forward(self, feature, feat_len):
        bs = feature.size(0)
        feature = self.reshape_input(feature, group_size=4)
        raw_output = self.extractor(feature)

        reshape_output = raw_output.reshape(bs, raw_output.size(0) // bs, self.out_dim)
        return reshape_output, torch.div(feat_len,4, rounding_mode="floor")


class RNNExtractor(nn.Module):
    ''' A simple 2-layer RNN extractor for acoustic feature down-sampling'''

    def __init__(self, input_dim, out_dim):
        super(RNNExtractor, self).__init__()

        self.out_dim = out_dim
        self.layer = nn.RNN(input_dim, self.out_dim)
        # self.ln = nn.LayerNorm(self.out_dim)
        self.dp = nn.Dropout(p=0.2)
        self.linear = nn.Linear(self.out_dim, self.out_dim)

    def forward(self, feature, feat_len):

        # feat_len = feat_len//4
        feat_len = torch.div(feat_len, 4, rounding_mode='floor')
        feature, _ = self.layer(feature)
        # Normalization
        feature = self.dp(feature)
        
        # Downsample timestep
        sample_rate = 4
        if feature.shape[1] % sample_rate != 0:
            feature = feature[:, :-(feature.shape[1] % 4), :].contiguous()
        feature = feature[:, ::sample_rate, :].contiguous()

        return feature, feat_len


class ANNExtractor(nn.Module):
    ''' VGG extractor for ASR described in https://arxiv.org/pdf/1706.02737.pdf'''

    def __init__(self, input_dim):
        super(ANNExtractor, self).__init__()
        self.init_dim = 64
        self.hide_dim = 128
        in_channel, freq_dim, out_dim = self.check_dim(input_dim)
        self.in_channel = in_channel
        self.freq_dim = freq_dim
        self.out_dim = 640

        width = freq_dim


        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, width, kernel_size=1, bias=False),
            # nn.BatchNorm2d(width),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            AttentionConv(width, width, kernel_size=7, padding=3),
            # nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # Half-time dimension
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(width, 64, kernel_size=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2)  # Half-time dimension
        )

    def check_dim(self, input_dim):
        # Check input dimension, delta feature should be stack over channel.
        if input_dim % 13 == 0:
            # MFCC feature
            return int(input_dim/13), 13, (13//4)*self.hide_dim
        elif input_dim % 40 == 0:
            # Fbank feature
            return int(input_dim/40), 40, (40//4)*self.hide_dim
        else:
            raise ValueError(
            'Acoustic feature dimension for VGG should be 13/26/39(MFCC) or 40/80/120(Fbank) but got '+input_dim)

    def view_input(self, feature, feat_len):
        # downsample time
        # feat_len = feat_len//4
        feat_len = torch.div(feat_len, 4, rounding_mode='floor')
        # crop sequence s.t. t%4==0
        if feature.shape[1] % 4 != 0:
            feature = feature[:, :-(feature.shape[1] % 4), :].contiguous()
        bs, ts, ds = feature.shape
        # stack feature according to result of check_dim
        feature = feature.view(bs, ts, self.in_channel, self.freq_dim)
        feature = feature.transpose(1, 2)

        return feature, feat_len

    def forward(self, feature, feat_len):
        
        # Feature shape BSxTxD -> BSxCH(1)xT/4xD
        feature, feat_len = self.view_input(feature, feat_len) #downsample on time
        # Foward
        # BS x 1 x T/4 x D/4 -> BS x width x T/4 x D
        feature = self.conv1(feature)
        # BS x width x T/4 x D/4 -> BS x width x T/8 x D/2 (attention)
        feature = self.conv2(feature)

        # BS x width x T/8 x D/2 -> BS x 64 x T/8 x D/4
        feature = self.conv3(feature)
        feature = F.relu(feature)
        # BS x 64 x T/8 x D/4 -> BS x T/8 x 64 x 8D
        feature = feature.transpose(1, 2)
        # BS x T/8 x 64 x 8D -> BS x T/8 x 8D
        feature = feature.contiguous().view(feature.shape[0], feature.shape[1], -1)

        return feature, feat_len


class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)

        
class CNNExtractor(nn.Module):
    ''' A simple 2-layer CNN extractor for acoustic feature down-sampling'''

    def __init__(self, input_dim, out_dim):
        super(CNNExtractor, self).__init__()

        self.out_dim = out_dim
        self.extractor = nn.Sequential(
            nn.Conv1d(input_dim, out_dim, 4, stride=2, padding=1),
            nn.Conv1d(out_dim, out_dim, 4, stride=2, padding=1),
        )

    def forward(self, feature, feat_len):
        # Fixed down-sample ratio
        feat_len = torch.div(feat_len,4, rounding_mode="floor")
        # Channel first
        feature = feature.transpose(1,2)
        # Foward
        feature = self.extractor(feature)
        # Channel last
        feature = feature.transpose(1, 2)

        return feature, feat_len
