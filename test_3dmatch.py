import os
import torch
import logging
import argparse
import numpy as np
import open3d as o3d
from easydict import EasyDict as edict
from models.V2PNet import V2PNet
from utils.timer import AverageMeter
from dataloader.ThreeDMatch import ThreeDMatchTestset
from dataloader.dataloader import get_dataloader
from geometric_registration.common import  get_keypts, get_desc, get_scores, loadlog, build_correspondence
from sklearn.neighbors import KDTree
from config_3dmatch import get_config

def find_mutually_nn_keypoints(ref_key, test_key, ref, test):
    ref_features = ref.data.T
    test_features = test.data.T
    ref_keypoints = np.asarray(ref_key.points)
    test_keypoints = np.asarray(test_key.points)
    n_samples = test_features.shape[0]

    ref_tree = KDTree(ref_features)
    test_tree = KDTree(test.data.T)
    test_NN_idx = ref_tree.query(test_features, return_distance=False)
    ref_NN_idx = test_tree.query(ref_features, return_distance=False)

    # find mutually closest points
    ref_match_idx = np.nonzero(
        np.arange(n_samples) == np.squeeze(test_NN_idx[ref_NN_idx])
    )[0]
    ref_matched_keypoints = ref_keypoints[ref_match_idx]
    test_matched_keypoints = test_keypoints[ref_NN_idx[ref_match_idx]]

    return np.transpose(ref_matched_keypoints), np.transpose(test_matched_keypoints)


def register_one_scene(inlier_ratio_threshold, distance_threshold, save_path, return_dict, scene):
    gt_matches = 0
    pred_matches = 0
    scorepath = f"{save_path}/scores/{scene}"
    inlier_num_meter, inlier_ratio_meter = AverageMeter(), AverageMeter()

    pcdpath = f"./dataset/3dmatch/{scene}/"
    resultpath = f"./geometric_registration/eva_result/model_best_{args.best_model_mode}_{args.num_points}/result/{scene}/model_best_{args.best_model_mode}_result"
    logpath = f"./geometric_registration/eva_result/model_best_{args.best_model_mode}_{args.num_points}/log_result/{scene}-evaluation"
    keyptspath = f"./geometric_registration/3dmatch/model_best_{args.best_model_mode}/keypoints/{scene}"
    descpath = f"./geometric_registration/3dmatch/model_best_{args.best_model_mode}/descriptors/{scene}"
    gtpath = f'./geometric_registration/gt_result/{scene}-evaluation/'
    gtLog = loadlog(gtpath)

    if not os.path.exists(logpath):
        os.makedirs(logpath)
    if not os.path.exists(resultpath):
        os.makedirs(resultpath)
    if not os.path.exists(logpath):
        os.makedirs(logpath)

    num_frag = len([filename for filename in os.listdir(pcdpath) if filename.endswith('ply')])
    for id1 in range(num_frag):
        for id2 in range(id1 + 1, num_frag):
            cloud_bin_s = f'cloud_bin_{id1}'
            cloud_bin_t = f'cloud_bin_{id2}'
            key = f"{id1}_{id2}"
            if key not in gtLog.keys():
                # skip the pairs that have less than 30% overlap.
                num_inliers = 0
                inlier_ratio = 0
                gt_flag = 0
            else:
                source_keypts = get_keypts(keyptspath, cloud_bin_s)
                target_keypts = get_keypts(keyptspath, cloud_bin_t)
                source_desc = get_desc(descpath, cloud_bin_s, 'V2PNet')
                target_desc = get_desc(descpath, cloud_bin_t, 'V2PNet')
                source_score = get_scores(scorepath, cloud_bin_s, 'V2PNet').squeeze()
                target_score = get_scores(scorepath, cloud_bin_t, 'V2PNet').squeeze()
                source_desc = np.nan_to_num(source_desc)
                target_desc = np.nan_to_num(target_desc)

                # randomly select 5000/2500/1000/500/250 keypts
                if args.random_points:
                    source_indices = np.random.choice(range(source_keypts.shape[0]), args.num_points)
                    target_indices = np.random.choice(range(target_keypts.shape[0]), args.num_points)
                else:
                    source_indices = np.argsort(source_score)[-args.num_points:]
                    target_indices = np.argsort(target_score)[-args.num_points:]

                source_keypts = source_keypts[source_indices, :]
                source_desc = source_desc[source_indices, :]
                target_keypts = target_keypts[target_indices, :]
                target_desc = target_desc[target_indices, :]

                corr = build_correspondence(source_desc, target_desc)

                gt_trans = gtLog[key]
                frag1 = source_keypts[corr[:, 0]]
                frag2_pc = o3d.geometry.PointCloud()
                frag2_pc.points = o3d.utility.Vector3dVector(target_keypts[corr[:, 1]])
                frag2_pc.transform(gt_trans)
                frag2 = np.asarray(frag2_pc.points)
                distance = np.sqrt(np.sum(np.power(frag1 - frag2, 2), axis=1))
                num_inliers = np.sum(distance < distance_threshold)
                inlier_ratio = num_inliers / len(distance)

                if inlier_ratio > inlier_ratio_threshold:
                    pred_matches += 1
                gt_matches += 1
                inlier_num_meter.update(num_inliers)
                inlier_ratio_meter.update(inlier_ratio)

                # calculate the transformation matrix using RANSAC, this is for Registration Recall.
                source_pcd = o3d.geometry.PointCloud()
                source_pcd.points = o3d.utility.Vector3dVector(source_keypts)
                target_pcd = o3d.geometry.PointCloud()
                target_pcd.points = o3d.utility.Vector3dVector(target_keypts)
                s_desc = o3d.pipelines.registration.Feature()
                s_desc.data = source_desc.T
                t_desc = o3d.pipelines.registration.Feature()
                t_desc.data = target_desc.T
                result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                    source_pcd, target_pcd, s_desc, t_desc, True,
                    0.05,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
                    [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                     o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.05)],
                    o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))

                # write the transformation matrix into .log file for evaluation.
                with open(os.path.join(logpath, f'model_best_{args.best_model_mode}.log'), 'a+') as f:
                    trans = result.transformation
                    trans = np.linalg.inv(trans)
                    s1 = f'{id1}\t {id2}\t  37\n'
                    f.write(s1)
                    f.write(f"{trans[0, 0]}\t {trans[0, 1]}\t {trans[0, 2]}\t {trans[0, 3]}\t \n")
                    f.write(f"{trans[1, 0]}\t {trans[1, 1]}\t {trans[1, 2]}\t {trans[1, 3]}\t \n")
                    f.write(f"{trans[2, 0]}\t {trans[2, 1]}\t {trans[2, 2]}\t {trans[2, 3]}\t \n")
                    f.write(f"{trans[3, 0]}\t {trans[3, 1]}\t {trans[3, 2]}\t {trans[3, 3]}\t \n")

                # write the result into resultpath so that it can be re-shown.
            s = f"{cloud_bin_s}\t{cloud_bin_t}\t{num_inliers}\t{inlier_ratio:.8f}\t{gt_matches}"
            with open(os.path.join(resultpath, f'{cloud_bin_s}_{cloud_bin_t}.rt.txt'), 'w+') as f:
                f.write(s)
    recall = pred_matches * 100.0 / gt_matches
    return_dict[scene] = [recall, inlier_num_meter.avg, inlier_ratio_meter.avg]
    logging.info(
        f"{scene}: Recall={recall:.2f}%, inlier ratio={inlier_ratio_meter.avg * 100:.2f}%, inlier num={inlier_num_meter.avg:.2f}")
    return recall, inlier_num_meter.avg, inlier_ratio_meter.avg


def generate_features(model, dloader, config):
    dataloader_iter = dloader.__iter__()

    descriptor_path = f'{save_path}/descriptors'
    keypoint_path = f'{save_path}/keypoints'
    score_path = f'{save_path}/scores'
    if not os.path.exists(descriptor_path):
        os.mkdir(descriptor_path)
    if not os.path.exists(keypoint_path):
        os.mkdir(keypoint_path)
    if not os.path.exists(score_path):
        os.mkdir(score_path)

    # generate descriptors
    recall_list = []
    for scene in dset.scene_list:
        descriptor_path_scene = os.path.join(descriptor_path, scene)
        keypoint_path_scene = os.path.join(keypoint_path, scene)
        score_path_scene = os.path.join(score_path, scene)
        if not os.path.exists(descriptor_path_scene):
            os.mkdir(descriptor_path_scene)
        if not os.path.exists(keypoint_path_scene):
            os.mkdir(keypoint_path_scene)
        if not os.path.exists(score_path_scene):
            os.mkdir(score_path_scene)
        pcdpath = f"./dataset/3dmatch/{scene}/"
        num_frag = len([filename for filename in os.listdir(pcdpath) if filename.endswith('ply')])
        # generate descriptors for each fragment
        for ids in range(num_frag):
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

            features, scores = model(inputs)

            pcd_size = inputs['stack_lengths'][0][0]

            # if test rotate invariance
            # pts = inputs['points'][0][-int(pcd_size):]
            # else
            pts = inputs['points'][0][:int(pcd_size)]

            features, scores = features[:int(pcd_size)], scores[:int(pcd_size)]

            np.save(f'{descriptor_path_scene}/cloud_bin_{ids}.V2PNet',
                    features.detach().cpu().numpy().astype(np.float32))
            np.save(f'{keypoint_path_scene}/cloud_bin_{ids}', pts.detach().cpu().numpy().astype(np.float32))
            np.save(f'{score_path_scene}/cloud_bin_{ids}', scores.detach().cpu().numpy().astype(np.float32))
            print(f"Generate cloud_bin_{ids} for {scene}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--chosen_snapshot', default='V2PNet-3dmatch-20221224-1801', type=str, help='snapshot dir')
    parser.add_argument('--best_model_mode', default='acc', help='acc or loss')
    parser.add_argument('--inlier_ratio_threshold', default=0.05, type=float)
    parser.add_argument('--distance_threshold', default=0.10, type=float)
    parser.add_argument('--random_points', default=False, action='store_true')
    parser.add_argument('--num_points', default=500, type=int)
    parser.add_argument('--generate_features', default=True, action='store_true')
    args = parser.parse_args()
    if args.random_points:
        log_filename = f'geometric_registration/model_best_{args.best_model_mode}-rand-{args.num_points}.log'
    else:
        log_filename = f'geometric_registration/model_best_{args.best_model_mode}-pred-{args.num_points}.log'
    logging.basicConfig(level=logging.INFO,
                        filename=log_filename,
                        filemode='w',
                        format="")

    # load config
    config = get_config()
    dconfig = vars(config)
    config = edict(dconfig)

    config.device = torch.device('cuda')

    print("==================================V2PNet Initialization==================================")
    # create model
    config.architecture = config.kpconv_architecture
    config.model = V2PNet(config).to(config.device)

    config.model.load_state_dict(torch.load(
        f'./pretrain_model/3dmatch/model_best_{args.best_model_mode}.pth')[
                                     'state_dict'])

    print(f"Load weight from ./pretrain_model/3dmatch/model_best_{args.best_model_mode}.pth")
    config.model.eval()

    save_path = f'./geometric_registration/3dmatch/model_best_{args.best_model_mode}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if args.generate_features:
        dset = ThreeDMatchTestset(root=config.threedmatch_root,
                                  phase='test',
                                  downsample=config.downsample,
                                  config=config,
                                  last_scene=False
                                  )
        dloader, _ = get_dataloader(dataset=dset,
                                    batch_size=config.batch_size,
                                    shuffle=False,
                                    num_workers=config.test_num_workers
                                    )
        print("==================================Start Generate Features==================================")
        generate_features(config.model, dloader, config)

    # register each pair of fragments in scenes using multiprocessing.
    scene_list = [
        '7-scenes-redkitchen',
        'sun3d-home_at-home_at_scan1_2013_jan_1',
        'sun3d-home_md-home_md_scan9_2012_sep_30',
        'sun3d-hotel_uc-scan3',
        'sun3d-hotel_umd-maryland_hotel1',
        'sun3d-hotel_umd-maryland_hotel3',
        'sun3d-mit_76_studyroom-76-1studyroom2',
        'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
    ]

    # No need for multiprocessing
    print("==================================Start Evaluate==================================")
    return_dict = dict()
    jobs = []
    for scene in scene_list:
        p = register_one_scene(args.inlier_ratio_threshold, args.distance_threshold, save_path, return_dict, scene)
        jobs.append(p)

    recalls = [v[0] for k, v in return_dict.items()]
    inlier_nums = [v[1] for k, v in return_dict.items()]
    inlier_ratios = [v[2] for k, v in return_dict.items()]

    logging.info("*" * 40)
    logging.info(recalls)
    logging.info(f"All 8 scene, average recall: {np.mean(recalls):.2f}%， STD:{np.std(recalls)}")
    logging.info(f"All 8 scene, average num inliers: {np.mean(inlier_nums):.2f}")
    logging.info(f"All 8 scene, average num inliers ratio: {np.mean(inlier_ratios) * 100:.2f}%")
