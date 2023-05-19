import os
import os.path
from os.path import join, exists
import numpy as np
import pickle
import random
import torch
import open3d as o3d
from utils.pointcloud import make_open3d_point_cloud
import torch.utils.data as data
from scipy.spatial.distance import cdist
import MinkowskiEngine as ME


def rotation_matrix(augment_axis, augment_rotation):
    angles = np.random.rand(3) * 2 * np.pi * augment_rotation
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    # R = Rx @ Ry @ Rz
    if augment_axis == 1:
        return random.choice([Rx, Ry, Rz])
    return Rx @ Ry @ Rz


def translation_matrix(augment_translation):
    T = np.random.rand(3) * augment_translation
    return T


class ThreeDMatchDataset(data.Dataset):
    __type__ = 'descriptor'

    def __init__(self,
                 root,
                 phase='train',
                 num_node=16,
                 downsample=0.03,
                 self_augment=False,
                 augment_noise=0.0005,
                 augment_axis=1,
                 augment_rotation=1.0,
                 augment_translation=0.001,
                 config=None,
                 ):

        self.root = root
        self.phase = phase
        self.config = config
        self.num_node = num_node
        self.downsample = downsample
        self.self_augment = self_augment
        self.augment_noise = augment_noise
        self.augment_axis = augment_axis
        self.augment_rotation = augment_rotation
        self.augment_translation = augment_translation
        self.augment_min_scale = config.augment_min_scale
        self.augment_max_scale = config.augment_max_scale
        self.voxel_size = config.voxel_size
        # self.matching_search_voxel_size = \
            # config.voxel_size * config.positive_pair_search_voxel_size_multiplier
        self.matching_search_voxel_size = config.matching_search_voxel_size
        # self.curvature_radius = config.curvature_radius
        self.matching_radius = config.base_radius * config.voxel_size
        self.device = torch.device('cuda')

        # containers
        self.points = []
        self.src_to_tgt = {}

        # load data
        # pts means points, keypts means keypoints
        pts_filename = join(self.root, f'3DMatch_{phase}_{self.downsample:.3f}_points.pkl')
        keypts_filename = join(self.root, f'3DMatch_{phase}_{self.downsample:.3f}_keypts.pkl')

        if exists(pts_filename) and exists(keypts_filename):
            with open(pts_filename, 'rb') as file:
                pts_data = pickle.load(file)
                self.points = [*pts_data.values()]
                self.ids_list = [*pts_data.keys()]
            with open(keypts_filename, 'rb') as file:
                self.correspondences = pickle.load(file)
            print(f"Load PKL file from {pts_filename}")
        else:
            print("PKL file not found.")
            return

        for idpair in self.correspondences.keys():
            src = idpair.split("@")[0]
            tgt = idpair.split("@")[1]
            # add (key -> value)  src -> tgt
            if src not in self.src_to_tgt.keys():
                self.src_to_tgt[src] = [tgt]
            else:
                self.src_to_tgt[src] += [tgt]

    def __getitem__(self, index):
        src_id = list(self.src_to_tgt.keys())[index]

        if random.random() > 0.5:
            tgt_id = self.src_to_tgt[src_id][0]
        else:
            tgt_id = random.choice(self.src_to_tgt[src_id])

        src_ind = self.ids_list.index(src_id)
        tgt_ind = self.ids_list.index(tgt_id)
        src_pcd = make_open3d_point_cloud(self.points[src_ind])

        if self.self_augment:
            tgt_pcd = make_open3d_point_cloud(self.points[src_ind])
            N_src = self.points[src_ind].shape[0]
            N_tgt = self.points[src_ind].shape[0]
            corr = np.array([np.arange(N_src), np.arange(N_src)]).T
        else:
            tgt_pcd = make_open3d_point_cloud(self.points[tgt_ind])
            N_src = self.points[src_ind].shape[0]
            N_tgt = self.points[tgt_ind].shape[0]
            corr = self.correspondences[f"{src_id}@{tgt_id}"]
        if N_src > 50000 or N_tgt > 50000:
            return self.__getitem__(int(np.random.choice(self.__len__(), 1)))

        # data augmentation
        gt_trans = np.eye(4).astype(np.float32)

        # Rotation and Translation matrix
        R = rotation_matrix(self.augment_axis, self.augment_rotation)
        T = translation_matrix(self.augment_translation)

        gt_trans[0:3, 0:3] = R
        gt_trans[0:3, 3] = T

        tgt_pcd.transform(gt_trans)

        # get matching indices
        src_points = np.array(src_pcd.points)
        tgt_points = np.array(tgt_pcd.points)

        pcd_0 = make_open3d_point_cloud(src_points)
        pcd_1 = make_open3d_point_cloud(tgt_points)

        pcd_0.points = o3d.utility.Vector3dVector(np.array(pcd_0.points))
        pcd_1.points = o3d.utility.Vector3dVector(np.array(pcd_1.points))

        scale = self.augment_min_scale + (self.augment_max_scale - self.augment_min_scale) * random.random()
        tgt_pcd.scale(scale, center=tgt_pcd.get_center())
        src_pcd.scale(scale, center=src_pcd.get_center())

        if len(corr) > self.num_node:
            # choose num_node dim
            sel_corr = corr[np.random.choice(len(corr), self.num_node, replace=False)]
        else:
            sel_corr = corr

        sel_P_src = src_points[sel_corr[:, 0], :].astype(np.float32)
        sel_P_tgt = tgt_points[sel_corr[:, 1], :].astype(np.float32)

        dist_keypts = cdist(sel_P_src, sel_P_src)

        # Voxelization
        _, sel0, map0 = ME.utils.sparse_quantize(src_points / self.voxel_size, return_index=True, return_inverse=True)
        _, sel1, map1 = ME.utils.sparse_quantize(tgt_points / self.voxel_size, return_index=True, return_inverse=True)

        # Make point clouds using voxelized points
        pcd0 = make_open3d_point_cloud(src_points)
        pcd1 = make_open3d_point_cloud(tgt_points)

        # Select features and points using the returned voxelized indices
        pcd0.points = o3d.utility.Vector3dVector(np.array(pcd0.points)[sel0])
        pcd1.points = o3d.utility.Vector3dVector(np.array(pcd1.points)[sel1])

        # Get coords
        xyz0 = np.array(pcd0.points)
        xyz1 = np.array(pcd1.points)

        coords0 = np.floor(xyz0 / self.voxel_size)
        coords1 = np.floor(xyz1 / self.voxel_size)

        # Add noise
        src_rand = np.random.rand(src_points.shape[0], 3) * self.augment_noise
        tgt_rand = np.random.rand(tgt_points.shape[0], 3) * self.augment_noise

        src_points += src_rand
        tgt_points += tgt_rand

        pts0 = src_points
        pts1 = tgt_points

        pt_feat0 = np.ones_like(pts0[:, :1]).astype(np.float32)
        pt_feat1 = np.ones_like(pts1[:, :1]).astype(np.float32)

        if self.self_augment:
            pt_feat0[np.random.choice(pts0.shape[0], int(pts0.shape[0] * 0.99), replace=False)] = 0
            pt_feat1[np.random.choice(pts1.shape[0], int(pts1.shape[0] * 0.99), replace=False)] = 0

        voxel_feat0 = pt_feat0[sel0]
        voxel_feat1 = pt_feat1[sel1]

        return pts0, pts1, pt_feat0, pt_feat1, sel_corr, dist_keypts, gt_trans, coords0, coords1, voxel_feat0, voxel_feat1, sel0, sel1, map0, map1

    def __len__(self):
        return len(self.src_to_tgt.keys())

    def apply_transform(self, pts, trans):
        R = trans[0:3, 0:3]
        T = trans[0:3, 3]
        pts = pts @ R.T + T
        return pts

class ThreeDMatchTestset(data.Dataset):
    __type__ = 'descriptor'

    def __init__(self,
                 root,
                 phase='test',
                 downsample=0.03,
                 config=None,
                 last_scene=False):
        assert phase == 'test', "Supports only the test set."

        self.root = root
        self.phase = phase
        self.downsample = downsample
        self.config = config
        self.device = config.device

        # contrainer
        self.points = []
        self.voxel_coords = []
        self.ids_list = []
        self.num_test = 0
        self.voxel_size = config.voxel_size

        self.scene_list = [
            '7-scenes-redkitchen',
            'sun3d-home_at-home_at_scan1_2013_jan_1',
            'sun3d-home_md-home_md_scan9_2012_sep_30',
            'sun3d-hotel_uc-scan3',
            'sun3d-hotel_umd-maryland_hotel1',
            'sun3d-hotel_umd-maryland_hotel3',
            'sun3d-mit_76_studyroom-76-1studyroom2',
            'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
        ]

        if last_scene is True:
            self.scene_list = self.scene_list[-1:]
        for scene in self.scene_list:
            self.test_path = f'{self.root}/{scene}'
            pcd_list = [filename for filename in os.listdir(self.test_path) if filename.endswith('ply')]
            self.num_test += len(pcd_list)

            pcd_list = sorted(pcd_list, key=lambda x: int(x[:-4].split("_")[-1]))
            for i, ind in enumerate(pcd_list):
                pcd = o3d.io.read_point_cloud(join(self.test_path, ind))
                # points downsample
                downsample_pcd = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size=downsample)

                # Load points and labels
                points = np.array(downsample_pcd.points).astype(np.float32)

                self.points += [points]
                self.ids_list += [scene + '/' + ind]
        return

    def __getitem__(self, index):
        # points & feature
        pts = self.points[index].astype(np.float32)
        feat = np.ones_like(pts[:, :1]).astype(np.float32)

        # if test rotate invariance
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        T = np.identity(4)
        R = rotation_matrix(0, 1.0)
        T[:3, :3] = R
        pcd.transform(T)
        pts_rotate = np.asarray(pcd.points)

        # Voxelize xyz and feats
        coords = np.floor(pts / self.voxel_size)
        coords, sel, unique_map = ME.utils.sparse_quantize(coords, return_index=True, return_inverse=True)

        voxel_coords = torch.as_tensor(coords, dtype=torch.int32)
        voxel_feat = feat[sel]

        return pts, pts, feat, feat, np.array([]), np.array([]), np.array([]), voxel_coords, voxel_coords, voxel_feat, voxel_feat, sel, sel, unique_map, unique_map

    def __len__(self):
        return self.num_test