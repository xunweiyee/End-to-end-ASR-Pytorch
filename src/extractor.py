import torch
from torch import nn as nn

import logging
logger = logging.getLogger()
logging.basicConfig(level="INFO", format="%(filename)s: %(message)s")
# logging.basicConfig(level="INFO", format="%(levelname)s: %(filename)s: %(message)s")
logger.disabled = True


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
        logging.info(f"VGGExtractor: input_dim {input_dim}, in_channel {self.in_channel}, freq_dim {self.freq_dim}, out_dim {self.out_dim}")

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

        logging.info(f"VGGExtractor: feature, feat_len {feature.shape}, {feat_len}")
        # Feature shape BSxTxD -> BS x CH(num of delta) x T x D(acoustic feature dim)
        feature, feat_len = self.view_input(feature, feat_len) #downsample on time
        logging.info(f"VGGExtractor: downsampled to {feature.shape}, {feat_len}")
        # Foward
        feature = self.extractor(feature)
        logging.info(f"VGGExtractor: extracted {feature.shape}")
        # BSx128xT/4xD/4 -> BSxT/4x128xD/4
        feature = feature.transpose(1, 2)
        logging.info(f"VGGExtractor: feature.transpose {feature.shape}")
        #  BS x T/4 x 128 x D/4 -> BS x T/4 x 32D
        feature = feature.contiguous().view(feature.shape[0], feature.shape[1], self.out_dim)
        logging.info(f"VGGExtractor: feature transformed {feature.shape}\n")

        return feature, feat_len


class MLPExtractor(nn.Module):
    '''
        A simple MLP extractor for acoustic feature down-sampling
        Every 4 frame will be convert to corresponding features by MLP
    '''

    def __init__(self, input_dim, out_dim):
        super(MLPExtractor, self).__init__()

        logging.info(f"MLPExtractor: {input_dim}")
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

        logging.info(f"RNNExtractor: forward {feature.shape}, {feat_len}")
        # feat_len = feat_len//4
        feat_len = torch.div(feat_len, 4, rounding_mode='floor')
        feature, _ = self.layer(feature)
        # Normalization
        feature = self.dp(feature)
        logging.info(f"RNNExtractor: extracted {feature.shape}")
        
        # Downsample timestep
        sample_rate = 4
        if feature.shape[1] % sample_rate != 0:
            feature = feature[:, :-(feature.shape[1] % 4), :].contiguous()
        feature = feature[:, ::sample_rate, :].contiguous()
        logging.info(f"RNNExtractor: downsampled {feature.shape} {feat_len}\n")

        return feature, feat_len



# attention layer code inspired from: https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/4
import numpy as np
import torch.nn.functional as F

# attention layer code inspired from: https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/4
class ANNExtractor(nn.Module):
    def __init__(self, hidden_size, batch_first=True):
        super(ANNExtractor, self).__init__()

        self.out_dim = hidden_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)
        
        hidden_dim = hidden_size
        self.fc1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU()) 
        self.fc2 = nn.Linear(hidden_dim, 1)

    def get_mask(self):
        pass

    def forward(self, inputs, lengths):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]
            
        # apply attention layer
        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1) # (batch_size, hidden_size, 1)
                            )
    
        attentions = torch.softmax(F.relu(weights.squeeze()), dim=-1)

        # create mask based on the sentence lengths
        mask = torch.ones(attentions.size(), requires_grad=True).cuda()
        for i, l in enumerate(lengths):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0
        

        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(-1)  # sums per row
        
        attentions = masked.div(_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze().contiguous()
        # print(attentions.shape, representations.shape)
        z = self.fc1(representations)
        # z = self.fc2(z)

        # return z.unsqueeze(1), torch.ones(lengths.shape[0], dtype=int) # representations, attentions
        return representations, attentions