import argparse
import time
import os

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def str2bool(v):
    return v.lower() in ('true', '1')


experiment_id = "V2PNet-" + "3dmatch-" + time.strftime('%Y%m%d-%H%M')
# Snapshot Configurations
snapshot_arg = add_argument_group('Snapshot')
snapshot_arg.add_argument('--snapshot_dir', type=str,
                          default=f'./results/3dmatch/snapshot/{experiment_id}')
snapshot_arg.add_argument('--tboard_dir', type=str,
                          default=f'./results/3dmatch/tensorboard/{experiment_id}')
snapshot_arg.add_argument('--snapshot_interval', type=int, default=100)
snapshot_arg.add_argument('--save_dir', type=str, default=os.path.join(
    f'./results/3dmatch/snapshot/{experiment_id}',
    'models/'))

# Train Configurations
trainer_arg = add_argument_group('Trainer')

# Data loader configs(add from DeepGlobalRegistration)
trainer_arg.add_argument('--train_phase', type=str, default="train")
trainer_arg.add_argument('--val_phase', type=str, default="val")
trainer_arg.add_argument('--test_phase', type=str, default="test")

# trainer_arg.add_argument(
    # '--positive_pair_search_voxel_size_multiplier', type=float, default=1.5)
trainer_arg.add_argument(
    '--matching_search_voxel_size', type=float, default=0.03)
trainer_arg.add_argument('--hit_ratio_thresh', type=float, default=0.1)

# Network Configurations
net_arg = add_argument_group('Network')
net_arg.add_argument('--kpconv_architecture', type=int, default=['simple',
                                                                 'resnetb',
                                                                 'resnetb_strided',
                                                                 'resnetb',
                                                                 'resnetb',
                                                                 'resnetb_strided',
                                                                 'resnetb',
                                                                 'resnetb',
                                                                 'nearest_upsample',
                                                                 'unary',
                                                                 'nearest_upsample',
                                                                 'last_unary'
                                                                 ])
net_arg.add_argument('--in_points_dim', type=int, default=3)
net_arg.add_argument('--first_subsampling_dl', type=float, default=0.03)
net_arg.add_argument('--kpconv_in_dim', type=int, default=1, help='point input feature dimension')
net_arg.add_argument('--kpconv_out_dim', type=int, default=32, help='point output feature dimension')
net_arg.add_argument('--first_features_dim', type=int, default=128)
net_arg.add_argument('--conv_radius', type=float, default=2.5)
net_arg.add_argument('--deform_radius', type=float, default=5.0)
net_arg.add_argument('--num_kernel_points', type=int, default=15)
net_arg.add_argument('--KP_extent', type=float, default=2.0)
net_arg.add_argument('--KP_influence', type=str, default='linear')
net_arg.add_argument('--aggregation_mode', type=str, default='sum', choices=['closest', 'sum'])
net_arg.add_argument('--fixed_kernel_points', type=str, default='center', choices=['center', 'verticals', 'none'])
net_arg.add_argument('--use_batch_norm', type=str2bool, default=True)
net_arg.add_argument('--batch_norm_momentum', type=float, default=0.1)
net_arg.add_argument('--deformable', type=str2bool, default=False)
net_arg.add_argument('--modulated', type=str2bool, default=False)

# Sparce Conv Net
net_arg.add_argument('--conv1_kernel_size', type=int, default=7)
net_arg.add_argument('--in_channels', type=int, default=1)
net_arg.add_argument('--out_channels', type=int, default=32, help='Voxel feature dimension')
net_arg.add_argument('--me_bn_momentum', type=float, default=0.1)

# Loss Configurations
loss_arg = add_argument_group('Loss')
loss_arg.add_argument('--loss_dist_type', type=str, default='euclidean',
                      choices=['cosine', 'arccosine', 'sqeuclidean', 'euclidean', 'cityblock'])
loss_arg.add_argument('--desc_loss', type=str, default='circle', choices=['contrastive', 'circle'])
loss_arg.add_argument('--pos_margin', type=float, default=0.1)
loss_arg.add_argument('--neg_margin', type=float, default=1.4)
loss_arg.add_argument('--m', type=float, default=0.1)
loss_arg.add_argument('--log_scale', type=float, default=10)
loss_arg.add_argument('--safe_radius', type=float, default=0.1)
loss_arg.add_argument('--det_loss', type=str, default='score')
loss_arg.add_argument('--desc_loss_weight', type=float, default=1.0)
loss_arg.add_argument('--det_loss_weight', type=float, default=1.0)

# Optimizer Configurations
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'ADAM'])
opt_arg.add_argument('--max_epoch', type=int, default=200)
opt_arg.add_argument('--save_freq_epoch', type=int, default=1)
opt_arg.add_argument('--training_max_iter', type=int, default=3500)
opt_arg.add_argument('--iter_size', type=int, default=1, help='accumulate gradient')
opt_arg.add_argument('--val_max_iter', type=int, default=500)
opt_arg.add_argument('--lr', type=float, default=0.001)
opt_arg.add_argument('--weight_decay', type=float, default=1e-3)
opt_arg.add_argument('--adam_momentum', type=float, default=0.98)
opt_arg.add_argument('--sgd_momentum', type=float, default=0.98)

opt_arg.add_argument('--scheduler', type=str, default='ExpLR')
opt_arg.add_argument('--scheduler_gamma', type=float, default=0.95)
opt_arg.add_argument('--scheduler_interval', type=int, default=1)
opt_arg.add_argument('--grad_clip_norm', type=float, default=100.0)

# Dataset and dataloader configurations
data_arg = add_argument_group('Data')
data_arg.add_argument('--threedmatch_root', '--threed_match_dir', type=str, default='./dataset/3dmatch')
data_arg.add_argument('--num_node', type=int, default=64, help='number selected for calculate loss')
data_arg.add_argument('--downsample', type=float, default=0.03)
data_arg.add_argument('--self_augment', type=str2bool, default=False)
data_arg.add_argument('--augment_noise', type=float, default=0.005)
data_arg.add_argument('--augment_min_scale', type=float, default=0.8)
data_arg.add_argument('--augment_max_scale', type=float, default=1.2)
data_arg.add_argument('--augment_axis', type=int, default=1)
data_arg.add_argument('--augment_rotation', type=float, default=1.0, help='rotation angle = num * 2pi')
data_arg.add_argument('--augment_translation', type=float, default=1.0, help='translation = num (m)')
data_arg.add_argument('--batch_size', type=int, default=1)
data_arg.add_argument('--train_num_workers', type=int, default=16)
data_arg.add_argument('--val_num_workers', type=int, default=16)
data_arg.add_argument('--test_num_workers', type=int, default=16)

data_arg.add_argument('--dataset', type=str, default='3dmatch')
data_arg.add_argument('--voxel_size', type=float, default=0.025, help='0.025/0.05')
data_arg.add_argument('--base_radius', type=float, default=2.5, help='')
data_arg.add_argument('--overlap_threshold', type=float, default=None, help='')

# Other configurations
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--gpu_mode', type=str2bool, default=True)
misc_arg.add_argument('--verbose', type=str2bool, default=True)
misc_arg.add_argument('--pretrain_time', type=str, default='', help='20230101-0101')


def get_config():
    args = parser.parse_args()
    return args
