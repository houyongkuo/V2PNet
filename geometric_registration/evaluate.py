import open3d as o3d
import numpy as np
import time
import os
from utils import get_keypts, get_desc, get_scores, loadlog
from functools import partial


def build_correspondence(source_desc, target_desc):
    """
    Find the mutually closest point pairs in feature space.
    source and target are descriptor for 2 point cloud key points. [5000, 32]
    """

    distance = np.sqrt(2 - 2 * (source_desc @ target_desc.T))
    source_idx = np.argmin(distance, axis=1)
    source_dis = np.min(distance, axis=1)
    target_idx = np.argmin(distance, axis=0)
    target_dis = np.min(distance, axis=0)

    result = []
    for i in range(len(source_idx)):
        if target_idx[source_idx[i]] == i:
            result.append([i, source_idx[i]])
    return np.array(result)


def register2Fragments(id1, id2, keyptspath, descpath, scorespath, resultpath, logpath, gtLog, desc_name, inlier_ratio, distance_threshold, random_points):
    """
    Register point cloud {id1} and {id2} using the keypts location and descriptors.
    """
    cloud_bin_s = f'cloud_bin_{id1}'
    cloud_bin_t = f'cloud_bin_{id2}'
    write_file = f'{cloud_bin_s}_{cloud_bin_t}.rt.txt'
    if os.path.exists(os.path.join(resultpath, write_file)):
        return 0, 0, 0
    source_keypts = get_keypts(keyptspath, cloud_bin_s)
    target_keypts = get_keypts(keyptspath, cloud_bin_t)
    source_desc = get_desc(descpath, cloud_bin_s, desc_name)
    target_desc = get_desc(descpath, cloud_bin_t, desc_name)
    source_score = get_scores(scorespath, cloud_bin_s, desc_name).squeeze()
    target_score = get_scores(scorespath, cloud_bin_t, desc_name).squeeze()
    source_desc = np.nan_to_num(source_desc)
    target_desc = np.nan_to_num(target_desc)
    if random_points is False:
        # Select {num_keypts} points based on the scores. The descriptors and keypts are already sorted based on the detection score.
        source_indices = np.argsort(source_score)[-num_keypts:]
        target_indices = np.argsort(target_score)[-num_keypts:]

        source_keypts = source_keypts[source_indices, :]
        source_desc = source_desc[source_indices, :]
        target_keypts = target_keypts[target_indices, :]
        target_desc = target_desc[target_indices, :]
    if random_points is True:
        # Select {num_keypts} points randomly.
        source_indices = np.random.choice(range(source_keypts.shape[0]), num_keypts)
        target_indices = np.random.choice(range(target_keypts.shape[0]), num_keypts)
        source_keypts = source_keypts[source_indices, :]
        source_desc = source_desc[source_indices, :]
        target_keypts = target_keypts[target_indices, :]
        target_desc = target_desc[target_indices, :]
    key = f'{cloud_bin_s.split("_")[-1]}_{cloud_bin_t.split("_")[-1]}'
    if key not in gtLog.keys():
        # skip the pairs that have less than 30% overlap.
        num_inliers = 0
        inlier_ratio = 0
        gt_flag = 0
    else:
        # build correspondence set in feature space.
        corr = build_correspondence(source_desc, target_desc)

        # calculate the inlier ratio, this is for Feature Matching Recall.
        gt_trans = gtLog[key]
        frag1 = source_keypts[corr[:, 0]]
        frag2_pc = o3d.geometry.PointCloud()
        frag2_pc.points = o3d.utility.Vector3dVector(target_keypts[corr[:, 1]])
        frag2_pc.transform(gt_trans)
        frag2 = np.asarray(frag2_pc.points)
        distance = np.sqrt(np.sum(np.power(frag1 - frag2, 2), axis=1))
        num_inliers = np.sum(distance < distance_threshold)

        if num_inliers / len(distance) < inlier_ratio:
            print(key)
            print("num_corr:", len(corr), "inlier_ratio:", num_inliers / len(distance))

        inlier_ratio = num_inliers / len(distance)
        gt_flag = 1

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
             # o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 1000)
            o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000)
            # o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 10000))
        )
        # write the transformation matrix into .log file for evaluation.
        with open(os.path.join(logpath, f'{desc_name}_model_best_{model_best_mode}.log'), 'a+') as f:
            trans = result.transformation
            trans = np.linalg.inv(trans)
            s1 = f'{id1}\t {id2}\t  37\n'
            f.write(s1)
            f.write(f"{trans[0,0]}\t {trans[0,1]}\t {trans[0,2]}\t {trans[0,3]}\t \n")
            f.write(f"{trans[1,0]}\t {trans[1,1]}\t {trans[1,2]}\t {trans[1,3]}\t \n")
            f.write(f"{trans[2,0]}\t {trans[2,1]}\t {trans[2,2]}\t {trans[2,3]}\t \n")
            f.write(f"{trans[3,0]}\t {trans[3,1]}\t {trans[3,2]}\t {trans[3,3]}\t \n")

    # write the result into resultpath so that it can be re-shown.
    s = f"{cloud_bin_s}\t{cloud_bin_t}\t{num_inliers}\t{inlier_ratio:.8f}\t{gt_flag}"
    with open(os.path.join(resultpath, f'{cloud_bin_s}_{cloud_bin_t}.rt.txt'), 'w+') as f:
        f.write(s)
    return num_inliers, inlier_ratio, gt_flag


def read_register_result(resultpath, id1, id2):
    """
    Read the registration result of {id1} & {id2} from the resultpath
    Return values contain the inlier_number, inlier_ratio, flag(indicating whether this pair is a ground truth match).
    """
    cloud_bin_s = f'cloud_bin_{id1}'
    cloud_bin_t = f'cloud_bin_{id2}'
    with open(os.path.join(resultpath, f'{cloud_bin_s}_{cloud_bin_t}.rt.txt'), 'r') as f:
        content = f.readlines()
    nums = content[0].replace("\n", "").split("\t")[2:5]
    return nums


def deal_with_one_scene(inlier_ratio, distance_threshold, scene, random_points):
    """
    Function to register all the fragments pairs in one scene.
    """
    logpath = f"eva_result/model_best_{model_best_mode}_{num_keypts}/log_result/{scene}-evaluation"
    pcdpath = f"../dataset/3dmatch/{scene}/"
    keyptspath = f"3dmatch/model_best_{model_best_mode}/keypoints/{scene}"
    descpath = f"3dmatch/model_best_{model_best_mode}/descriptors/{scene}"
    scorespath = f"3dmatch/model_best_{model_best_mode}/scores/{scene}"
    gtpath = f'gt_result/{scene}-evaluation/'
    gtLog = loadlog(gtpath)
    resultpath = f"eva_result/model_best_{model_best_mode}_{num_keypts}/result/{scene}/{desc_name}_result_model_best_{model_best_mode}"

    if not os.path.exists(logpath):
        os.makedirs(logpath)
    if not os.path.exists(resultpath):
        os.makedirs(resultpath)
    if not os.path.exists(logpath):
        os.makedirs(logpath)

    # register each pair
    num_frag = len([filename for filename in os.listdir(pcdpath) if filename.endswith('ply')])
    print(f"Start Evaluate Descriptor {desc_name} for {scene}")
    start_time = time.time()
    for id1 in range(num_frag):
        for id2 in range(id1 + 1, num_frag):
                register2Fragments(id1, id2, keyptspath, descpath, scorespath, resultpath, logpath, gtLog, desc_name, inlier_ratio, distance_threshold, random_points)
    print(f"Finish Evaluation, time: {time.time() - start_time:.2f}s")


if __name__ == '__main__':
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

    inlier_ratio = 0.05  # 5%
    distance_threshold = 0.10  # 10cm

    desc_name = 'V2PNet'
    model_best_mode = 'acc'

    global num_keypts
    num_keypts = 500

    # multiprocessing to register each pair in each scene.
    # this part is time-consuming
    from multiprocessing import Pool

    pool = Pool(len(scene_list))
    func = partial(deal_with_one_scene, inlier_ratio, distance_threshold, random_points=False)
    pool.map(func, scene_list)
    pool.close()
    pool.join()

    # collect all the data and print the results.
    inliers_list = []
    recall_list = []
    inliers_ratio_list = []
    pred_match = 0
    gt_match = 0
    for scene in scene_list:
        # evaluate
        pcdpath = f"../dataset/3dmatch/{scene}/"
        resultpath = f"eva_result/model_best_{model_best_mode}_{num_keypts}/result/{scene}/{desc_name}_result_model_best_{model_best_mode}"
        num_frag = len([filename for filename in os.listdir(pcdpath) if filename.endswith('ply')])
        result = []
        for id1 in range(num_frag):
            for id2 in range(id1 + 1, num_frag):
                line = read_register_result(resultpath, id1, id2)
                result.append([int(line[0]), float(line[1]), int(line[2])])  # inlier_number, inlier_ratio, flag.
        result = np.array(result)
        gt_results = np.sum(result[:, 2] == 1)
        pred_results = np.sum(result[:, 1] > inlier_ratio)
        pred_match += pred_results
        gt_match += gt_results
        recall = float(pred_results / gt_results) * 100
        print(f"Correct Match {pred_results}, ground truth Match {gt_results}")
        print(f"Recall {recall}%")
        ave_num_inliers = np.sum(np.where(result[:, 2] == 1, result[:, 0], np.zeros(result.shape[0]))) / pred_results
        print(f"Average Num Inliners: {ave_num_inliers}")
        ave_inlier_ratio = np.sum(np.where(result[:, 2] == 1, result[:, 1], np.zeros(result.shape[0]))) / pred_results
        print(f"Average Num Inliner Ratio: {ave_inlier_ratio}")
        recall_list.append(recall)
        inliers_list.append(ave_num_inliers)
        inliers_ratio_list.append(ave_inlier_ratio)

    print("*" * 40)
    print(recall_list)
    # print(f"True Avarage Recall: {pred_match / gt_match * 100}%")
    print(f"Matching Recall Std: {np.std(recall_list)}")
    average_recall = sum(recall_list) / len(recall_list)
    print(f"All 8 scene, average recall: {average_recall}%")
    average_inliers = sum(inliers_list) / len(inliers_list)
    print(f"All 8 scene, average num inliers: {average_inliers}")
    average_inliers_ratio = sum(inliers_ratio_list) / len(inliers_list)
    print(f"All 8 scene, average num inliers ratio: {average_inliers_ratio}")
