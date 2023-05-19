import os
import json
import numpy as np
import logging
import argparse
import torch
import glob
from utils.timer import Timer, AverageMeter
import open3d as o3d

from config_kitti import get_config
from models.V2PNet import V2PNet
from easydict import EasyDict as edict
from dataloader.dataloader import get_dataloader

# Datasets
from dataloader.KITTI import KITTITestDataset


def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def make_open3d_feature(data, dim, npts):
    feature = o3d.pipelines.registration.Feature()
    feature.resize(dim, npts)
    feature.data = data.astype('d').transpose()
    return feature


def generate_features(model, dloader, args):
    dataloader_iter = dloader.__iter__()

    descriptor_path = f'./geometric_registration/kitti/model_best_{args.model_best_mode}/descriptors'
    keypoint_path = f'./geometric_registration/kitti/model_best_{args.model_best_mode}/keypoints'
    score_path = f'./geometric_registration/kitti/model_best_{args.model_best_mode}/scores'
    if not os.path.exists(descriptor_path):
        os.makedirs(descriptor_path)
    if not os.path.exists(keypoint_path):
        os.makedirs(keypoint_path)
    if not os.path.exists(score_path):
        os.makedirs(score_path)

    DATA_FILES = {
        'test': 'dataset/kitti/scene_list/test_kitti.txt',
    }

    success_meter, loss_meter, rte_meter, rre_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    feat_timer, reg_timer = Timer(), Timer()

    subset_names = open(DATA_FILES['test']).read().split()
    for dirname in subset_names:
        drive_id = int(dirname)
        print(f'Now, start to generate features in squ: {drive_id}')
        fnames = glob.glob(args.kitti_root + 'dataset/sequences/%02d/velodyne/*.bin' % drive_id)
        # pcdpath = args.kitti_root + 'sequences/%02d/velodyne/' % drive_id
        # num_frag = len([filename for filename in os.listdir(pcdpath) if filename.endswith('bin')])

        for i in range(len(fnames)):
            inputs = dataloader_iter.next()
            for k, v in inputs.items():  # load inputs to device.
                if type(v) == list:
                    inputs[k] = [item.to(config.device) for item in v]
                elif type(v) == torch.Tensor:
                    inputs[k] = v.to(config.device)
                elif type(v) == tuple:
                    inputs[k] = [item.to(config.device) for item in v]
                else:
                    inputs[k] = v
            feat_timer.tic()
            features, scores = model(inputs)
            feat_timer.toc()
            stack_lengths = inputs['stack_lengths'][0].cpu().detach().numpy()
            first_pcd_indices = np.arange(stack_lengths[0])
            second_pcd_indices = np.arange(stack_lengths[1]) + stack_lengths[0]

            if args.random_points:
                anc_keypoints_id = np.random.choice(stack_lengths[0], args.num_points)
                pos_keypoints_id = np.random.choice(stack_lengths[1], args.num_points) + stack_lengths[0]
                anc_points = inputs['points'][0][anc_keypoints_id].cpu().detach().numpy()
                pos_points = inputs['points'][0][pos_keypoints_id].cpu().detach().numpy()
                anc_features = features[anc_keypoints_id].cpu().detach().numpy()
                pos_features = features[pos_keypoints_id].cpu().detach().numpy()
                anc_scores = scores[anc_keypoints_id].cpu().detach().numpy()
                pos_scores = scores[pos_keypoints_id].cpu().detach().numpy()
            else:
                if args.num_points == 0:
                    scores_anc_pcd = scores[first_pcd_indices].cpu().detach().numpy()
                    scores_pos_pcd = scores[second_pcd_indices].cpu().detach().numpy()
                    anc_keypoints_id = np.argsort(scores_anc_pcd, axis=0).squeeze()
                    pos_keypoints_id = np.argsort(scores_pos_pcd, axis=0).squeeze() + stack_lengths[0]
                    anc_points = inputs['points'][0][anc_keypoints_id].cpu().detach().numpy()
                    anc_features = features[anc_keypoints_id].cpu().detach().numpy()
                    anc_scores = scores[anc_keypoints_id].cpu().detach().numpy()
                    pos_points = inputs['points'][0][pos_keypoints_id].cpu().detach().numpy()
                    pos_features = features[pos_keypoints_id].cpu().detach().numpy()
                    pos_scores = scores[pos_keypoints_id].cpu().detach().numpy()
                else:
                    scores_anc_pcd = scores[first_pcd_indices].cpu().detach().numpy()
                    scores_pos_pcd = scores[second_pcd_indices].cpu().detach().numpy()
                    anc_keypoints_id = np.argsort(scores_anc_pcd, axis=0)[-args.num_points:].squeeze()
                    pos_keypoints_id = np.argsort(scores_pos_pcd, axis=0)[-args.num_points:].squeeze() + stack_lengths[0]
                    anc_points = inputs['points'][0][anc_keypoints_id].cpu().detach().numpy()
                    anc_features = features[anc_keypoints_id].cpu().detach().numpy()
                    anc_scores = scores[anc_keypoints_id].cpu().detach().numpy()
                    pos_points = inputs['points'][0][pos_keypoints_id].cpu().detach().numpy()
                    pos_features = features[pos_keypoints_id].cpu().detach().numpy()
                    pos_scores = scores[pos_keypoints_id].cpu().detach().numpy()

            pcd0 = make_open3d_point_cloud(anc_points)
            pcd1 = make_open3d_point_cloud(pos_points)
            feat0 = make_open3d_feature(anc_features, 32, anc_features.shape[0])
            feat1 = make_open3d_feature(pos_features, 32, pos_features.shape[0])

            reg_timer.tic()
            distance_threshold = config.voxel_size * 1.0
            ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                pcd0, pcd1, feat0, feat1, True, distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3, [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
                ],
                o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000)
            )
            # print(ransac_result)
            T_ransac = ransac_result.transformation.astype(np.float32)

            #TODO: finish file save
            # np.savez(os.path.join(icp_save_path, filename),
            #          trans=T_ransac,
            #          anc_pts=anc_points,
            #          pos_pts=pos_points,
            #          anc_scores=anc_scores,
            #          pos_scores=pos_scores
            #          )
            reg_timer.toc()

            T_gth = inputs['trans'].cpu().detach().numpy()
            rte = np.linalg.norm(T_ransac[:3, 3] - T_gth[:3, 3])
            rre = np.arccos((np.trace(T_ransac[:3, :3].transpose() @ T_gth[:3, :3]) - 1) / 2)

            if rte < 2:
                rte_meter.update(rte)

            if not np.isnan(rre) and rre < np.pi / 180 * 5:
                rre_meter.update(rre * 180 / np.pi)

            if rte < 2 and not np.isnan(rre) and rre < np.pi / 180 * 5:
                success_meter.update(1)
            else:
                success_meter.update(0)
                logging.info(f"{fnames[i]} Failed with RTE: {rte}, RRE: {rre * 180 / np.pi}")

            # loss_meter.update(loss_ransac)

            if (i + 1) % 10 == 0:
                logging.info(
                    f"{i + 1} / {len(fnames)}: Feat time: {round(feat_timer.avg, 2)}," +
                    f" Reg time: {round(reg_timer.avg, 2)}, RTE: {round(rte_meter.avg, 4)}, STD: {round(rte_meter.std, 4)}" +
                    f" RRE: {round(rre_meter.avg, 4)}, STD: {round(rre_meter.std, 4)}, Success: {success_meter.sum} / {success_meter.count}" +
                    f" ({success_meter.avg * 100} %)"
                )
                feat_timer.reset()
                reg_timer.reset()

        logging.info(
            # f"Total loss: {loss_meter.avg}, RTE: {rte_meter.avg}, var: {rte_meter.var}," +
            f" RTE: {round(rte_meter.avg, 4)}, var: {round(rte_meter.var, 4)}, std:{round(rte_meter.std, 4)}" +
            f" RRE: {round(rre_meter.avg, 4)}, var: {round(rre_meter.var, 4)}, std:{round(rre_meter.std, 4)}, Success: {success_meter.sum} " +
            f"/ {success_meter.count} ({success_meter.avg * 100} %)"
        )


if __name__ == '__main__':
    print('############################### Prepare test ###############################')
    # set config
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_best_mode', default='acc', type=str, help='acc or loss')
    parser.add_argument('--kitti_root', type=str, default="dataset/kitti/")
    parser.add_argument('--random_points', default=False, action='store_true')
    parser.add_argument('--num_points', default=0, type=int, help="250, 500, 1000, 2500, 5000, 0(mean all)")
    parser.add_argument('--generate_features', default=True, action='store_true')
    args = parser.parse_args()

    # set log filename path
    if args.random_points:
        args.log_filename = f'geometric_registration/kitti/model_best_{args.model_best_mode}-rand-{args.num_points}.log'
    else:
        args.log_filename = f'geometric_registration/kitti/model_best_{args.model_best_mode}-pred-{args.num_points}.log'
    logging.basicConfig(level=logging.INFO,
                        filename=args.log_filename,
                        filemode='w',
                        format="")

    # load config
    config = get_config()
    dconfig = vars(config)
    config = edict(dconfig)

    # Set which gpu is going to be used
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
    config.device = torch.device('cuda')
    print(f"Load config from config_kitti")

    print('############################### Load model ###############################')

    # load model
    model = V2PNet(config)
    model.load_state_dict(torch.load(
        f'./pretrain_model/kitti/model_best_{args.model_best_mode}.pth')[
                              'state_dict'])
    print(f"Load weight from ./pretrain_model/kitti/model_best_{args.model_best_mode}.pth")

    model.eval()

    print('############################### Load testdata ###############################')
    # load dataset
    if args.generate_features:
        dset = KITTITestDataset(phase='test',
                                transform=None,
                                random_rotation=True,
                                random_scale=True,
                                random_trans=True,
                                manual_seed=False,
                                config=config)
        dloader, _ = get_dataloader(dataset=dset,
                                    batch_size=config.batch_size,
                                    shuffle=True,
                                    num_workers=config.num_workers)

        print('############################### Generate features ###############################')
        generate_features(model.cuda(), dloader, args)