import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF

import numpy as np

import sklearn
from sklearn.neighbors import KDTree

import functools
from typing import Optional

from models.blocks import *


class KpconvEncoder(nn.Module):
    """
       Class defining KpconvEncoder
       """
    CHANNELS = [None, 16, 32, 64, 128, 256, 512, 1024, 2048]

    def __init__(self, config):
        super(KpconvEncoder, self).__init__()

        ############
        # Parameters
        ############

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.kpconv_in_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points

        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.kpconv_architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decoder(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2
        return

    def forward(self, batch):
        # Get input features

        # No.1: xyz as features
        # feat = torch.cat(batch['points'])
        # x = feat.clone().detach()

        # No.2: 1 as features
        x = batch['features'].clone().detach()

        # Loop over consecutive blocks
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)

        return x, skip_x, batch


class KpconvDecoder(nn.Module):
    """
       Class defining KpconvDecoder
       """

    def __init__(self, config):
        super(KpconvDecoder, self).__init__()

        ############
        # Parameters
        ############

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.kpconv_in_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []
        self.decoder_skip_dims = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.kpconv_architecture):
            if 'upsample' in block:
                start_i = block_i
                break

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.decoder_skip_dims.append(in_dim)

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.kpconv_architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.kpconv_architecture[start_i + block_i - 1]:
                in_dim += self.decoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decoder(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2
        return

    def forward(self, x, skip_x, batch):

        # Get input features
        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)

        # scores = self.detection_scores(batch, x)
        # features = F.normalize(x, p=2, dim=-1)

        return x


class KpconvFeatureExtractor(nn.Module):
    def __init__(self, config):
        super(KpconvFeatureExtractor, self).__init__()

        self.voxel_size = config.voxel_size
        # self.normalize_feature = normalize_feature

        # Initialize the backbone network
        self.encoder = KpconvEncoder(config)

        self.decoder = KpconvDecoder(config)

    def forward(self, batch):
        enc_feat, skip_x, batch = self.encoder(batch)
        dec_feat = self.decoder(enc_feat, skip_x, batch)
        kpconv_feat = dec_feat
        return kpconv_feat


class ResBlockBase(nn.Module):
    expansion = 1
    NORM_TYPE = 'BN'

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 D=3):
        super(ResBlockBase, self).__init__()

        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=stride, dimension=D)

        self.norm1 = get_norm_layer(self.NORM_TYPE, planes, bn_momentum=bn_momentum, D=D)

        self.conv2 = ME.MinkowskiConvolution(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            bias=False,
            dimension=D)

        self.norm2 = get_norm_layer(self.NORM_TYPE, planes, bn_momentum=bn_momentum, D=D)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = MEF.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = MEF.relu(out)

        return out


class ResBlockBN(ResBlockBase):
    NORM_TYPE = 'BN'


class ResBlockIN(ResBlockBase):
    NORM_TYPE = 'IN'


def get_res_block(norm_type,
                  inplanes,
                  planes,
                  stride=1,
                  dilation=1,
                  downsample=None,
                  bn_momentum=0.1,
                  D=3):
    if norm_type == 'BN':
        return ResBlockBN(inplanes, planes, stride, dilation, downsample, bn_momentum, D)

    elif norm_type == 'IN':
        return ResBlockIN(inplanes, planes, stride, dilation, downsample, bn_momentum, D)

    else:
        raise ValueError(f'Type {norm_type}, not defined')


def get_norm_layer(norm_type, num_feats, bn_momentum=0.05, D=-1):
    if norm_type == 'BN':
        return ME.MinkowskiBatchNorm(num_feats, momentum=bn_momentum)

    elif norm_type == 'IN':
        return ME.MinkowskiInstanceNorm(num_feats)

    else:
        raise ValueError(f'Type {norm_type}, not defined')


class SparseEncoder(ME.MinkowskiNetwork):
    CHANNELS = [None, 32, 64, 128, 256]

    def __init__(self,
                 in_channels=3,
                 out_channels=128,
                 bn_momentum=0.1,
                 conv1_kernel_size=9,
                 norm_type=None,
                 block_norm_type=None,
                 D=3):
        ME.MinkowskiNetwork.__init__(self, D)

        NORM_TYPE = norm_type
        BLOCK_NORM_TYPE = block_norm_type
        CHANNELS = self.CHANNELS

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=CHANNELS[1],
            kernel_size=conv1_kernel_size,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm1 = get_norm_layer(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, D=D)

        self.block1 = get_res_block(
            BLOCK_NORM_TYPE, CHANNELS[1], CHANNELS[1], bn_momentum=bn_momentum, D=D)

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[1],
            out_channels=CHANNELS[2],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)

        self.norm2 = get_norm_layer(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.block2 = get_res_block(
            BLOCK_NORM_TYPE, CHANNELS[2], CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.conv3 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[2],
            out_channels=CHANNELS[3],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm3 = get_norm_layer(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.block3 = get_res_block(
            BLOCK_NORM_TYPE, CHANNELS[3], CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.conv4 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[3],
            out_channels=CHANNELS[4],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm4 = get_norm_layer(NORM_TYPE, CHANNELS[4], bn_momentum=bn_momentum, D=D)

        self.block4 = get_res_block(
            BLOCK_NORM_TYPE, CHANNELS[4], CHANNELS[4], bn_momentum=bn_momentum, D=D)

    def forward(self, x, tgt_feature=False):
        skip_features = []
        out_s1 = self.conv1(x)
        out_s1 = self.norm1(out_s1)
        out_s1 = self.block1(out_s1)
        out = MEF.relu(out_s1)

        skip_features.append(out_s1)

        out_s2 = self.conv2(out)
        out_s2 = self.norm2(out_s2)
        out_s2 = self.block2(out_s2)
        out = MEF.relu(out_s2)

        skip_features.append(out_s2)

        out_s4 = self.conv3(out)
        out_s4 = self.norm3(out_s4)
        out_s4 = self.block3(out_s4)
        out = MEF.relu(out_s4)

        skip_features.append(out_s4)

        out_s8 = self.conv4(out)
        out_s8 = self.norm4(out_s8)
        out_s8 = self.block4(out_s8)
        out = MEF.relu(out_s8)
        return out, skip_features


class SparseDecoder(ME.MinkowskiNetwork):
    CHANNELS = [None, 32, 64, 128, 256]
    TR_CHANNELS = [None, 32, 64, 64, 128]

    def __init__(self,
                 out_channels=128,
                 bn_momentum=0.1,
                 norm_type=None,
                 block_norm_type=None,
                 D=3):
        ME.MinkowskiNetwork.__init__(self, D)

        NORM_TYPE = norm_type
        BLOCK_NORM_TYPE = block_norm_type
        TR_CHANNELS = self.TR_CHANNELS
        CHANNELS = self.CHANNELS

        self.conv4_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[4],
            out_channels=TR_CHANNELS[4],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)

        self.norm4_tr = get_norm_layer(NORM_TYPE, TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

        self.block4_tr = get_res_block(
            BLOCK_NORM_TYPE, TR_CHANNELS[4], TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

        self.conv3_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[3] + TR_CHANNELS[4],
            out_channels=TR_CHANNELS[3],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm3_tr = get_norm_layer(NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.block3_tr = get_res_block(
            BLOCK_NORM_TYPE, TR_CHANNELS[3], TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.conv2_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[2] + TR_CHANNELS[3],
            out_channels=TR_CHANNELS[2],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm2_tr = get_norm_layer(NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.block2_tr = get_res_block(
            BLOCK_NORM_TYPE, TR_CHANNELS[2], TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.conv1_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[1] + TR_CHANNELS[2],
            out_channels=TR_CHANNELS[1],
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)

        self.final = ME.MinkowskiConvolution(
            in_channels=TR_CHANNELS[1],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=True,
            dimension=D)

    def forward(self, x, skip_features):
        out = self.conv4_tr(x)
        out = self.norm4_tr(out)
        out = self.block4_tr(out)
        out_s4_tr = MEF.relu(out)

        out = ME.cat(out_s4_tr, skip_features[-1])

        out = self.conv3_tr(out)
        out = self.norm3_tr(out)
        out = self.block3_tr(out)
        out_s2_tr = MEF.relu(out)

        out = ME.cat(out_s2_tr, skip_features[-2])

        out = self.conv2_tr(out)
        out = self.norm2_tr(out)
        out = self.block2_tr(out)
        out_s1_tr = MEF.relu(out)

        out = ME.cat(out_s1_tr, skip_features[-3])

        out = self.conv1_tr(out)
        out = MEF.relu(out)
        out = self.final(out)

        return out


class MinkowskiFeatureExtractor(nn.Module):
    def __init__(self, config):
        super(MinkowskiFeatureExtractor, self).__init__()

        self.voxel_size = config.voxel_size

        self.encoder = SparseEncoder(in_channels=config.in_channels,
                                     conv1_kernel_size=config.conv1_kernel_size,
                                     bn_momentum=config.me_bn_momentum,
                                     norm_type="IN",
                                     block_norm_type="BN")

        self.decoder = SparseDecoder(out_channels=config.out_channels,
                                     bn_momentum=config.me_bn_momentum,
                                     norm_type="BN",
                                     block_norm_type="IN")

    def forward(self, st_1, st_2):
        enc_feat_1, skip_features_1 = self.encoder(st_1)
        enc_feat_2, skip_features_2 = self.encoder(st_2)

        dec_feat_1 = self.decoder(enc_feat_1, skip_features_1)
        dec_feat_2 = self.decoder(enc_feat_2, skip_features_2)

        return dec_feat_1, dec_feat_2


class V2PNet(nn.Module):
    def __init__(self, config):
        """
        Construct a model that, once trained, estimate the scene flow between
        two point clouds.

        Args:
            nb_iter: int. Number of iterations to unroll in the Sinkhorn algorithm.
            voxel_size: float. Voxel size when do voxelization.
        """
        super(V2PNet, self).__init__()

        # Mass regularisation
        self.gamma = torch.nn.Parameter(torch.zeros(1))

        # Entropic regularisation
        self.minkowski_extractor = MinkowskiFeatureExtractor(config)
        self.KPConv = KpconvFeatureExtractor(config)
        self.device = torch.device('cuda')
        self.mlp = nn.Linear(64, 32, bias=True)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, batch):
        ###############################
        #         KPConv Part         #
        ###############################
        kpconv_feat = self.KPConv(batch)

        ###############################
        #          Voxel Part         #
        ###############################
        sinput_0 = ME.SparseTensor(
            features=batch['features'][batch['sinput0_sel']].to(self.device),
            coordinates=batch['sinput0_C'].to(self.device),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)

        sinput_1 = ME.SparseTensor(
            features=batch['features'][batch['sinput1_sel'][0] + batch['stack_lengths'][0][0]].to(self.device),
            coordinates=batch['sinput1_C'].to(self.device),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)

        sfeats_0, sfeats_1 = self.minkowski_extractor(sinput_0, sinput_1)


        feats_0 = sfeats_0.slice(sinput_0).F
        feats_1 = sfeats_1.slice(sinput_1).F

        # restore voxel_feat to point_feat
        feats_0 = feats_0[batch['sinput0_map'][0]]
        feats_1 = feats_1[batch['sinput1_map'][0]]
        voxel2point_feat = torch.cat((feats_0, feats_1), dim=0)

        fus_features = torch.cat((voxel2point_feat, kpconv_feat), dim=1)
        features = self.mlp(fus_features)
        if self.training:
            features = self.dropout(features)

        # scores: from fused features to get scores
        scores = self.detection_scores(batch, features)

        features = F.normalize(features, p=2, dim=-1)
        return features, scores

    def detection_scores(self, inputs, features):
        neighbor = inputs['neighbors'][0]  # [n_points, n_neighbors]

        first_pcd_length, second_pcd_length = inputs['stack_lengths'][0]

        # add a fake point in the last row for shadow neighbors
        shadow_features = torch.zeros_like(features[:1, :])
        features = torch.cat([features, shadow_features], dim=0)
        shadow_neighbor = torch.ones_like(neighbor[:1, :]) * (first_pcd_length + second_pcd_length)
        neighbor = torch.cat([neighbor, shadow_neighbor], dim=0)

        # normalize the feature to avoid overflow
        features = features / (torch.max(features) + 1e-6)

        # local max score (saliency score)
        neighbor_features = features[neighbor, :]  # [n_points, n_neighbors, 64]
        neighbor_features_sum = torch.sum(neighbor_features, dim=-1)  # [n_points, n_neighbors]
        neighbor_num = (neighbor_features_sum != 0).sum(dim=-1, keepdims=True)  # [n_points, 1]
        neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))

        mean_features = torch.sum(neighbor_features, dim=1) / neighbor_num  # [n_points, 64]
        local_max_score = F.softplus(features - mean_features)  # [n_points, 64]

        # calculate the depth-wise max score
        depth_wise_max = torch.max(features, dim=1, keepdims=True)[0]  # [n_points, 1]
        depth_wise_max_score = features / (1e-6 + depth_wise_max)  # [n_points, 64]

        all_scores = local_max_score * depth_wise_max_score
        # use the max score among channel to be the score of a single point.
        scores = torch.max(all_scores, dim=1, keepdims=True)[0]  # [n_points, 1]

        # hard selection (used during test)
        if self.training is False:
            local_max = torch.max(neighbor_features, dim=1)[0]
            is_local_max = (features == local_max)
            detected = torch.max(is_local_max.float(), dim=1, keepdims=True)[0]
            scores = scores * detected
        return scores[:-1, :]

    def detection_voxel_scores(self, features):
        shadow_features = torch.zeros_like(features[:1, :])

        features = torch.cat([features, shadow_features], dim=0)

        # normalize the feature to avoid overflow
        features = (features / (torch.max(features) + 1e-6))

        mean_features = torch.sum(features.unsqueeze(1), dim=1) / len(features)   # [n_points, dim]

        max_score = F.softplus(features - mean_features)  # [n_points, dim]

        # calculate the depth-wise max score
        depth_wise_max = torch.max(features, dim=1, keepdims=True)[0]  # [n_points, 1]
        depth_wise_max_score = features / (1e-6 + depth_wise_max)  # [n_points, dim]

        all_scores = max_score * depth_wise_max_score
        # use the max score among channel to be the score of a single point.
        scores = torch.max(all_scores, dim=1, keepdims=True)[0]  # [n_points, 1]

        th1 = scores.mean() + (scores.max() - scores.mean()) / 2
        th2 = scores.mean()
        one = torch.ones_like(scores)

        th1_scores = torch.where(scores < th1, one, scores)
        th2_scores = torch.where(scores > th2, one, scores)

        scores = th1_scores * th2_scores * scores

        # hard selection (used during test)
        if self.training is False:
            detected = torch.max(features.float(), dim=1, keepdims=True)[0]
            scores = scores * detected
        return scores[:-1, :]


class NonMaxSuppression(torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr

    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1
        reliability, repeatability = reliability[0], repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        maxima *= (reliability >= self.rel_thr)

        return maxima.nonzero().t()[2:4]
