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


experiment_id = "V2PNet-" + "kitti-" + time.strftime('%Y%m%d-%H%M')
# snapshot configurations
snapshot_arg = add_argument_group('Snapshot')
snapshot_arg.add_argument('--snapshot_dir', type=str, default=f'./results/kitti/snapshot/{experiment_id}')
snapshot_arg.add_argument('--tboard_dir', type=str, default=f'./results/kitti/tensorboard/{experiment_id}')
snapshot_arg.add_argument('--snapshot_interval', type=int, default=50)
snapshot_arg.add_argument('--save_dir', type=str,
                          default=os.path.join(f'./results/kitti/snapshot/{experiment_id}', 'models/'))

# Network configurations
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
net_arg.add_argument('--first_features_dim', type=int, default=64)
net_arg.add_argument('--first_subsampling_dl', type=float, default=0.3)
net_arg.add_argument('--kpconv_in_dim', type=int, default=1, help='point input feature dimension')
net_arg.add_argument('--kpconv_out_dim', type=int, default=32, help='point output feature dimension')
net_arg.add_argument('--conv_radius', type=float, default=5.0)
net_arg.add_argument('--deform_radius', type=float, default=5.0)
net_arg.add_argument('--num_kernel_points', type=int, default=15)
net_arg.add_argument('--KP_extent', type=float, default=2.0)
net_arg.add_argument('--KP_influence', type=str, default='linear')
net_arg.add_argument('--aggregation_mode', type=str, default='sum', choices=['closest', 'sum'])
net_arg.add_argument('--fixed_kernel_points', type=str, default='center', choices=['center', 'verticals', 'none'])
net_arg.add_argument('--use_batch_norm', type=str2bool, default=True)
net_arg.add_argument('--batch_norm_momentum', type=float, default=0.98)
net_arg.add_argument('--deformable', type=str2bool, default=False)
net_arg.add_argument('--modulated', type=str2bool, default=False)

# Sparce Conv Net
net_arg.add_argument('--conv1_kernel_size', type=int, default=7)
net_arg.add_argument('--in_channels', type=int, default=1)
net_arg.add_argument('--out_channels', type=int, default=32, help='Voxel feature dimension')
net_arg.add_argument('--me_bn_momentum', type=float, default=0.98)
net_arg.add_argument('--D', type=int, default=3)

# Loss configurations
loss_arg = add_argument_group('Loss')
loss_arg.add_argument('--dist_type', type=str, default='euclidean')
loss_arg.add_argument('--desc_loss', type=str, default='circle', choices=['contrastive', 'circle'])
loss_arg.add_argument('--pos_margin', type=float, default=0.1)
loss_arg.add_argument('--neg_margin', type=float, default=1.4)
loss_arg.add_argument('--m', type=float, default=0.1)
loss_arg.add_argument('--log_scale', type=float, default=10)
loss_arg.add_argument('--safe_radius', type=float, default=1)
loss_arg.add_argument('--det_loss', type=str, default='score')
loss_arg.add_argument('--desc_loss_weight', type=float, default=1.0)
loss_arg.add_argument('--det_loss_weight', type=float, default=1.0)

# Optimizer configurations
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'ADAM'])
opt_arg.add_argument('--max_epoch', type=int, default=200)
opt_arg.add_argument('--training_max_iter', type=int, default=1000)
opt_arg.add_argument('--val_max_iter', type=int, default=100)
opt_arg.add_argument('--iter_size', type=int, default=1, help='accumulate gradient')
opt_arg.add_argument('--lr', type=float, default=0.1)
opt_arg.add_argument('--weight_decay', type=float, default=1e-6)
opt_arg.add_argument('--momentum', type=float, default=0.98)
opt_arg.add_argument('--scheduler', type=str, default='ExpLR')
opt_arg.add_argument('--scheduler_gamma', type=float, default=0.9)
opt_arg.add_argument('--scheduler_interval', type=int, default=1)
opt_arg.add_argument('--grad_clip_norm', type=float, default=100.0)

# Dataset and dataloader configurations
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default="kitti")
data_arg.add_argument('--kitti_root', type=str, default="dataset/kitti/")
data_arg.add_argument('--num_node', type=int, default=1024)
data_arg.add_argument('--self_augment', type=str2bool, default=False)
data_arg.add_argument('--augment_noise', type=float, default=0.01)
data_arg.add_argument('--augment_axis', type=int, default=1)
data_arg.add_argument('--augment_rotation', type=float, default=1.0, help='rotation angle = num * 2pi')
data_arg.add_argument('--augment_shift_range', type=float, default=2, help='translation = num (m)')
data_arg.add_argument('--batch_size', type=int, default=1)
data_arg.add_argument('--num_workers', type=int, default=0)

# Other configurations
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--gpu_mode', type=str2bool, default=True)
misc_arg.add_argument('--verbose', type=str2bool, default=True)
misc_arg.add_argument('--pretrain', type=str, default='', help='20230101-0101')

# ME configurations
me_arg = add_argument_group('ME')
me_arg.add_argument('--use_random_scale', type=str2bool, default=False)
me_arg.add_argument('--min_scale', type=float, default=0.8)
me_arg.add_argument('--max_scale', type=float, default=1.2)
me_arg.add_argument('--use_random_rotation', type=str2bool, default=True)
me_arg.add_argument('--rotation_range', type=float, default=360)
me_arg.add_argument('--voxel_size', type=float, default=0.3)
me_arg.add_argument(
    '--kitti_max_time_diff',
    type=int,
    default=3,
    help='max time difference between pairs (non inclusive)')
me_arg.add_argument('--kitti_date', type=str, default='2011_09_26')

# Other configurations
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--pretrain_time', type=str, default='', help='20220621-1029')

def get_config():
    args = parser.parse_args()
    return args
