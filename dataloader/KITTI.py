# Basic libs
import os
import numpy as np
import time
import glob
import random
import pickle
import copy
import open3d as o3d
import pathlib

# Dataset parent class
import torch
import torch.utils.data as data
import MinkowskiEngine as ME
from scipy.spatial.distance import cdist
from scipy.linalg import expm, norm

kitti_icp_cache = {}
kitti_cache = {}


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


def get_matching_indices(source, target, trans, search_voxel_size, K=None):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    source_copy.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(target_copy)

    match_inds = []
    for i, point in enumerate(source_copy.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append((i, j))
    return match_inds


class KITTIDataset(data.Dataset):
    AUGMENT = None
    DATA_FILES = {
        'train': 'dataset/kitti/scene_list/train_kitti.txt',
        'val': 'dataset/kitti/scene_list/val_kitti.txt',
    }
    TEST_RANDOM_ROTATION = False
    IS_ODOMETRY = True
    MAX_TIME_DIFF = 3

    def __init__(self,
                 phase='train',
                 transform=None,
                 random_rotation=True,
                 random_scale=True,
                 random_trans=True,
                 manual_seed=False,
                 config=None):
        self.phase = phase
        self.files = []
        self.data_objects = []
        self.transform = transform
        self.voxel_size = config.voxel_size
        self.matching_search_voxel_size = config.voxel_size * 1.5
        # self.matching_search_voxel_size = \
        #     config.voxel_size * config.positive_pair_search_voxel_size_multiplier
        self.config = config
        self.random_trans = random_trans
        self.random_scale = random_scale
        self.min_scale = config.min_scale
        self.max_scale = config.max_scale
        self.random_rotation = random_rotation
        self.rotation_range = config.rotation_range
        self.augment_noise = config.augment_noise
        self.augment_shift_range = config.augment_shift_range
        self.num_node = config.num_node
        self.MIN_DIST = 10
        self.randg = np.random.RandomState()
        if manual_seed:
            self.reset_seed()

        # # For evaluation, use the odometry dataset training following the 3DFeat eval method
        if self.IS_ODOMETRY:
            self.root = config.kitti_root + 'dataset'
            self.random_rotation = random_rotation
        else:
            self.date = config.kitti_date
            self.root = os.path.join(config.kitti_root, self.date)

        self.icp_path = os.path.join(config.kitti_root, 'icp')
        pathlib.Path(self.icp_path).mkdir(parents=True, exist_ok=True)
        print(f"Find the subset {phase} data from {self.root}")

        # Use the kitti root
        self.max_time_diff = max_time_diff = config.kitti_max_time_diff

        subset_names = open(self.DATA_FILES[phase]).read().split()
        for dirname in subset_names:
            drive_id = int(dirname)
            inames = self.get_all_scan_ids(drive_id)
            for start_time in inames:
                for time_diff in range(2, max_time_diff):
                    pair_time = time_diff + start_time
                    if pair_time in inames:
                        self.files.append((drive_id, start_time, pair_time))

        self.max_time_diff = config.kitti_max_time_diff

        subset_names = open(self.DATA_FILES[phase]).read().split()
        if self.IS_ODOMETRY:
            for dirname in subset_names:
                drive_id = int(dirname)
                fnames = glob.glob(self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
                assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
                inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

                all_odo = self.get_video_odometry(drive_id, return_all=True)
                all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
                Ts = all_pos[:, :3, 3]
                pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
                pdist = np.sqrt(pdist.sum(-1))
                valid_pairs = pdist > self.MIN_DIST
                curr_time = inames[0]
                while curr_time in inames:
                    # Find the min index
                    next_time = np.where(valid_pairs[curr_time][curr_time:curr_time + 100])[0]
                    if len(next_time) == 0:
                        curr_time += 1
                    else:
                        # Follow https://github.com/yewzijian/3DFeatNet/blob/master/scripts_data_processing/kitti/process_kitti_data.m#L44
                        next_time = next_time[0] + curr_time - 1

                    if next_time in inames:
                        self.files.append((drive_id, curr_time, next_time))
                        curr_time = next_time + 1
        else:
            for dirname in subset_names:
                drive_id = int(dirname)
                fnames = glob.glob(self.root + '/' + self.date +
                                   '_drive_%04d_sync/velodyne_points/data/*.bin' % drive_id)
                assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
                inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

                all_odo = self.get_video_odometry(drive_id, return_all=True)
                all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
                Ts = all_pos[:, 0, :3]

                pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
                pdist = np.sqrt(pdist.sum(-1))

                for start_time in inames:
                    pair_time = np.where(
                        pdist[start_time][start_time:start_time + 100] > self.MIN_DIST)[0]
                    if len(pair_time) == 0:
                        continue
                    else:
                        pair_time = pair_time[0] + start_time

                    if pair_time in inames:
                        self.files.append((drive_id, start_time, pair_time))
        print(f"Finish the subset {phase} data loading.")

    def reset_seed(self, seed=0):
        print(f"Resetting the data loader seed to {seed}")
        self.randg.seed(seed)

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    def get_all_scan_ids(self, drive_id):
        if self.IS_ODOMETRY:
            fnames = glob.glob(self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
        else:
            fnames = glob.glob(self.root + '/' + self.date +
                               '_drive_%04d_sync/velodyne_points/data/*.bin' % drive_id)
        assert len(
            fnames) > 0, f"Make sure that the path {self.root} has drive id: {drive_id}"
        inames = [int(os.path.split(fname)[-1][:-4]) for fname in fnames]
        return inames

    @property
    def velo2cam(self):
        try:
            velo2cam = self._velo2cam
        except AttributeError:
            R = np.array([
                7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
            ]).reshape(3, 3)
            T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
            velo2cam = np.hstack([R, T])
            self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
        return self._velo2cam

    def get_video_odometry(self, drive, indices=None, ext='.txt', return_all=False):
        if self.IS_ODOMETRY:
            data_path = self.root + '/poses/%02d.txt' % drive
            if data_path not in kitti_cache:
                kitti_cache[data_path] = np.genfromtxt(data_path)
            if return_all:
                return kitti_cache[data_path]
            else:
                return kitti_cache[data_path][indices]
        else:
            data_path = self.root + '/' + self.date + '_drive_%04d_sync/oxts/data' % drive
            odometry = []
            if indices is None:
                fnames = glob.glob(self.root + '/' + self.date +
                                   '_drive_%04d_sync/velodyne_points/data/*.bin' % drive)
                indices = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            for index in indices:
                filename = os.path.join(data_path, '%010d%s' % (index, ext))
                if filename not in kitti_cache:
                    kitti_cache[filename] = np.genfromtxt(filename)
                odometry.append(kitti_cache[filename])

            odometry = np.array(odometry)
            return odometry

    def odometry_to_positions(self, odometry):
        if self.IS_ODOMETRY:
            T_w_cam0 = odometry.reshape(3, 4)
            T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            return T_w_cam0
        else:
            lat, lon, alt, roll, pitch, yaw = odometry.T[:6]

            R = 6378137  # Earth's radius in metres

            # convert to metres
            lat, lon = np.deg2rad(lat), np.deg2rad(lon)
            mx = R * lon * np.cos(lat)
            my = R * lat

            times = odometry.T[-1]
            return np.vstack([mx, my, alt, roll, pitch, yaw, times]).T

    def rot3d(self, axis, angle):
        ei = np.ones(3, dtype='bool')
        ei[axis] = 0
        i = np.nonzero(ei)[0]
        m = np.eye(3)
        c, s = np.cos(angle), np.sin(angle)
        m[i[0], i[0]] = c
        m[i[0], i[1]] = -s
        m[i[1], i[0]] = s
        m[i[1], i[1]] = c
        return m

    def pos_transform(self, pos):
        x, y, z, rx, ry, rz, _ = pos[0]
        RT = np.eye(4)
        RT[:3, :3] = np.dot(np.dot(self.rot3d(0, rx), self.rot3d(1, ry)), self.rot3d(2, rz))
        RT[:3, 3] = [x, y, z]
        return RT

    def get_position_transform(self, pos0, pos1, invert=False):
        T0 = self.pos_transform(pos0)
        T1 = self.pos_transform(pos1)
        return (np.dot(T1, np.linalg.inv(T0)).T if not invert else np.dot(
            np.linalg.inv(T1), T0).T)

    def _get_velodyne_fn(self, drive, t):
        if self.IS_ODOMETRY:
            fname = self.root + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
        else:
            fname = self.root + \
                    '/' + self.date + '_drive_%04d_sync/velodyne_points/data/%010d.bin' % (
                        drive, t)
        return fname

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        drive = self.files[idx][0]
        t0, t1 = self.files[idx][1], self.files[idx][2]
        all_odometry = self.get_video_odometry(drive, [t0, t1])
        positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]
        fname0 = self._get_velodyne_fn(drive, t0)
        fname1 = self._get_velodyne_fn(drive, t1)

        # XYZ and reflectance
        xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
        xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)

        xyz0 = xyzr0[:, :3]
        xyz1 = xyzr1[:, :3]

        key = '%d_%d_%d' % (drive, t0, t1)
        filename = self.icp_path + '/' + key + '.npy'
        if key not in kitti_icp_cache:
            if not os.path.exists(filename):
                if self.IS_ODOMETRY:
                    M = (self.velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
                         @ np.linalg.inv(self.velo2cam)).T
                else:
                    M = self.get_position_transform(positions[0], positions[1], invert=True).T
                xyz0_t = self.apply_transform(xyz0, M)
                pcd0 = make_open3d_point_cloud(xyz0_t)
                pcd1 = make_open3d_point_cloud(xyz1)
                reg = o3d.pipelines.registration.registration_icp(pcd0, pcd1, 0.2, np.eye(4),
                                                           o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                           o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
                pcd0.transform(reg.transformation)
                M2 = M @ reg.transformation

                np.save(filename, M2)
            else:
                M2 = np.load(filename)
            kitti_icp_cache[key] = M2
        else:
            M2 = kitti_icp_cache[key]
        trans = M2

        # get match
        pcd0 = make_open3d_point_cloud(xyz0)
        pcd1 = make_open3d_point_cloud(xyz1)
        pcd0 = o3d.geometry.PointCloud.voxel_down_sample(pcd0, self.voxel_size)
        pcd1 = o3d.geometry.PointCloud.voxel_down_sample(pcd1, self.voxel_size)

        matching_search_voxel_size = self.matching_search_voxel_size
        matches = get_matching_indices(pcd0, pcd1, trans, matching_search_voxel_size)
        if len(matches) < 1024:
            # raise ValueError(f"{drive}, {t0}, {t1}, {len(matches)}/{len(pcd0.points)}")
            print(f"Not enought corr: {drive}, {t0}, {t1}, {len(matches)}/{len(pcd0.points)}")
            return None, None, None, None, None, None

        xyz0 = np.array(pcd0.points)
        xyz1 = np.array(pcd1.points)

        # data augmentations: noise
        anc_noise = np.random.rand(xyz0.shape[0], 3) * self.augment_noise
        pos_noise = np.random.rand(xyz1.shape[0], 3) * self.augment_noise
        xyz0 += anc_noise
        xyz1 += pos_noise

        # data augmentations: rotation
        if self.random_rotation:
            xyz0 = self.rotate(xyz0, num_axis=1)
            xyz1 = self.rotate(xyz1, num_axis=1)

        # data augmentations: scale
        if self.random_scale:
            scale = self.min_scale + (self.max_scale - self.min_scale) * random.random()
            xyz0 = scale * xyz0
            xyz1 = scale * xyz1

        # data augmentations: translation
        if self.random_trans:
            xyz0 = xyz0 + np.random.uniform(-self.augment_shift_range, self.augment_shift_range, 3)
            xyz1 = xyz1 + np.random.uniform(-self.augment_shift_range, self.augment_shift_range, 3)

        # align the two point cloud into one corredinate system.
        matches = np.array(matches)
        anc_points = np.array(xyz0)
        pos_points = np.array(xyz1)

        anc_feat = np.ones_like(anc_points[:, :1]).astype(np.float32)
        pos_feat = np.ones_like(pos_points[:, :1]).astype(np.float32)

        if len(matches) > self.num_node:
            sel_matches = matches[np.random.choice(len(matches), self.num_node, replace=False)]
        else:
            sel_matches = matches

        sel_P_src = anc_points[sel_matches[:, 0], :].astype(np.float32)
        sel_P_tgt = pos_points[sel_matches[:, 1], :].astype(np.float32)
        dist_keypts = cdist(sel_P_src, sel_P_src)

        # Voxelization
        xyz0_th = torch.from_numpy(anc_points)
        xyz1_th = torch.from_numpy(pos_points)

        _, sel0, map0 = ME.utils.sparse_quantize(xyz0_th / self.voxel_size, return_index=True, return_inverse=True)
        _, sel1, map1 = ME.utils.sparse_quantize(xyz1_th / self.voxel_size, return_index=True, return_inverse=True)

        # Get features
        npts0 = len(sel0)
        npts1 = len(sel1)

        feats_train0, feats_train1 = [], []

        unique_xyz0_th = xyz0_th[sel0]
        unique_xyz1_th = xyz1_th[sel1]

        feats_train0.append(torch.ones((npts0, 1)))
        feats_train1.append(torch.ones((npts1, 1)))

        feats0 = torch.cat(feats_train0, 1)
        feats1 = torch.cat(feats_train1, 1)

        coords0 = torch.floor(unique_xyz0_th / self.voxel_size)
        coords1 = torch.floor(unique_xyz1_th / self.voxel_size)

        return anc_points, pos_points, anc_feat, pos_feat, sel_matches, dist_keypts, trans, coords0, coords1, feats0, feats1, sel0, sel1, map0, map1


    # Rotation matrix along axis with angle theta
    def M(self, axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    def sample_random_trans(self, pcd, randg, rotation_range=360):
        T = np.eye(4)
        R = self.M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
        T[:3, :3] = R
        T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
        return T

    def rotate(self, points, num_axis=1):
        if num_axis == 1:
            theta = np.random.rand() * 2 * np.pi
            axis = np.random.randint(3)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, -s], [s, c, -s], [s, s, c]], dtype=np.float32)
            R[:, axis] = 0
            R[axis, :] = 0
            R[axis, axis] = 1
            points = np.matmul(points, R)
        elif num_axis == 3:
            for axis in [0, 1, 2]:
                theta = np.random.rand() * 2 * np.pi
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s, -s], [s, c, -s], [s, s, c]], dtype=np.float32)
                R[:, axis] = 0
                R[axis, :] = 0
                R[axis, axis] = 1
                points = np.matmul(points, R)
        else:
            exit(-1)
        return points


class KITTITestDataset(data.Dataset):
    AUGMENT = None
    DATA_FILES = {
        'test': 'dataset/kitti/scene_list/test_kitti.txt'
    }
    TEST_RANDOM_ROTATION = False
    IS_ODOMETRY = True
    MAX_TIME_DIFF = 3

    def __init__(self,
                 phase='test',
                 transform=None,
                 random_rotation=False,
                 random_scale=False,
                 random_trans=False,
                 manual_seed=False,
                 config=None):
        self.phase = phase
        self.files = []
        self.data_objects = []
        self.transform = transform
        self.voxel_size = config.voxel_size
        self.matching_search_voxel_size = config.voxel_size * 1.5
        self.config = config
        self.random_trans = random_trans
        self.random_scale = random_scale
        self.min_scale = config.min_scale
        self.max_scale = config.max_scale
        self.random_rotation = random_rotation
        self.rotation_range = config.rotation_range
        self.augment_noise = config.augment_noise
        self.augment_shift_range = config.augment_shift_range
        self.num_node = config.num_node
        self.MIN_DIST = 10
        self.randg = np.random.RandomState()
        if manual_seed:
            self.reset_seed()

        # For evaluation, use the odometry dataset training following the 3DFeat eval method
        if self.IS_ODOMETRY:
            self.root = config.kitti_root + 'dataset'
            self.random_rotation = random_rotation
        else:
            self.date = config.kitti_date
            self.root = os.path.join(config.kitti_root, self.date)

        self.icp_path = os.path.join(config.kitti_root, 'icp')
        pathlib.Path(self.icp_path).mkdir(parents=True, exist_ok=True)
        print(f"Find the subset {phase} data from {self.root}")

        # Use the kitti root
        self.max_time_diff = max_time_diff = config.kitti_max_time_diff

        subset_names = open(self.DATA_FILES[phase]).read().split()
        for dirname in subset_names:
            drive_id = int(dirname)
            inames = self.get_all_scan_ids(drive_id)
            for start_time in inames:
                for time_diff in range(2, max_time_diff):
                    pair_time = time_diff + start_time
                    if pair_time in inames:
                        self.files.append((drive_id, start_time, pair_time))

        self.max_time_diff = config.kitti_max_time_diff

        subset_names = open(self.DATA_FILES[phase]).read().split()
        if self.IS_ODOMETRY:
            for dirname in subset_names:
                drive_id = int(dirname)
                fnames = glob.glob(self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
                assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
                inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

                all_odo = self.get_video_odometry(drive_id, return_all=True)
                all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
                Ts = all_pos[:, :3, 3]
                pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
                pdist = np.sqrt(pdist.sum(-1))
                valid_pairs = pdist > self.MIN_DIST
                curr_time = inames[0]
                while curr_time in inames:
                    # Find the min index
                    next_time = np.where(valid_pairs[curr_time][curr_time:curr_time + 100])[0]
                    if len(next_time) == 0:
                        curr_time += 1
                    else:
                        # Follow https://github.com/yewzijian/3DFeatNet/blob/master/scripts_data_processing/kitti/process_kitti_data.m#L44
                        next_time = next_time[0] + curr_time - 1

                    if next_time in inames:
                        self.files.append((drive_id, curr_time, next_time))
                        curr_time = next_time + 1
        else:
            for dirname in subset_names:
                drive_id = int(dirname)
                fnames = glob.glob(self.root + '/' + self.date +
                                   '_drive_%04d_sync/velodyne_points/data/*.bin' % drive_id)
                assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
                inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

                all_odo = self.get_video_odometry(drive_id, return_all=True)
                all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
                Ts = all_pos[:, 0, :3]

                pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
                pdist = np.sqrt(pdist.sum(-1))

                for start_time in inames:
                    pair_time = np.where(
                        pdist[start_time][start_time:start_time + 100] > self.MIN_DIST)[0]
                    if len(pair_time) == 0:
                        continue
                    else:
                        pair_time = pair_time[0] + start_time

                    if pair_time in inames:
                        self.files.append((drive_id, start_time, pair_time))

        if self.IS_ODOMETRY:
            # Remove problematic sequence
            for item in [
                (8, 15, 58),
            ]:
                if item in self.files:
                    self.files.pop(self.files.index(item))
        print(f"Finish the subset {phase} data loading.")

    def reset_seed(self, seed=0):
        print(f"Resetting the data loader seed to {seed}")
        self.randg.seed(seed)

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    def get_all_scan_ids(self, drive_id):
        if self.IS_ODOMETRY:
            fnames = glob.glob(self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
        else:
            fnames = glob.glob(self.root + '/' + self.date +
                               '_drive_%04d_sync/velodyne_points/data/*.bin' % drive_id)
        assert len(
            fnames) > 0, f"Make sure that the path {self.root} has drive id: {drive_id}"
        inames = [int(os.path.split(fname)[-1][:-4]) for fname in fnames]
        return inames

    @property
    def velo2cam(self):
        try:
            velo2cam = self._velo2cam
        except AttributeError:
            R = np.array([
                7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
            ]).reshape(3, 3)
            T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
            velo2cam = np.hstack([R, T])
            self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
        return self._velo2cam

    def get_video_odometry(self, drive, indices=None, ext='.txt', return_all=False):
        if self.IS_ODOMETRY:
            data_path = self.root + '/poses/%02d.txt' % drive
            if data_path not in kitti_cache:
                kitti_cache[data_path] = np.genfromtxt(data_path)
            if return_all:
                return kitti_cache[data_path]
            else:
                return kitti_cache[data_path][indices]
        else:
            data_path = self.root + '/' + self.date + '_drive_%04d_sync/oxts/data' % drive
            odometry = []
            if indices is None:
                fnames = glob.glob(self.root + '/' + self.date +
                                   '_drive_%04d_sync/velodyne_points/data/*.bin' % drive)
                indices = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            for index in indices:
                filename = os.path.join(data_path, '%010d%s' % (index, ext))
                if filename not in kitti_cache:
                    kitti_cache[filename] = np.genfromtxt(filename)
                odometry.append(kitti_cache[filename])

            odometry = np.array(odometry)
            return odometry

    def odometry_to_positions(self, odometry):
        if self.IS_ODOMETRY:
            T_w_cam0 = odometry.reshape(3, 4)
            T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            return T_w_cam0
        else:
            lat, lon, alt, roll, pitch, yaw = odometry.T[:6]

            R = 6378137  # Earth's radius in metres

            # convert to metres
            lat, lon = np.deg2rad(lat), np.deg2rad(lon)
            mx = R * lon * np.cos(lat)
            my = R * lat

            times = odometry.T[-1]
            return np.vstack([mx, my, alt, roll, pitch, yaw, times]).T

    def pos_transform(self, pos):
        x, y, z, rx, ry, rz, _ = pos[0]
        RT = np.eye(4)
        RT[:3, :3] = np.dot(np.dot(self.rot3d(0, rx), self.rot3d(1, ry)), self.rot3d(2, rz))
        RT[:3, 3] = [x, y, z]
        return RT

    def get_position_transform(self, pos0, pos1, invert=False):
        T0 = self.pos_transform(pos0)
        T1 = self.pos_transform(pos1)
        return (np.dot(T1, np.linalg.inv(T0)).T if not invert else np.dot(
            np.linalg.inv(T1), T0).T)

    def _get_velodyne_fn(self, drive, t):
        if self.IS_ODOMETRY:
            fname = self.root + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
        else:
            fname = self.root + \
                    '/' + self.date + '_drive_%04d_sync/velodyne_points/data/%010d.bin' % (
                        drive, t)
        return fname

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        drive = self.files[idx][0]
        t0, t1 = self.files[idx][1], self.files[idx][2]
        all_odometry = self.get_video_odometry(drive, [t0, t1])
        positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]

        fname0 = self._get_velodyne_fn(drive, t0)
        fname1 = self._get_velodyne_fn(drive, t1)

        # XYZ and reflectance
        xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
        xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)

        xyz0 = xyzr0[:, :3]
        xyz1 = xyzr1[:, :3]

        key = '%d_%d_%d' % (drive, t0, t1)
        filename = self.icp_path + '/' + key + '.npy'
        if key not in kitti_icp_cache:
            if not os.path.exists(filename):
                if self.IS_ODOMETRY:
                    M = (self.velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
                         @ np.linalg.inv(self.velo2cam)).T
                else:
                    M = self.get_position_transform(positions[0], positions[1], invert=True).T
                xyz0_t = self.apply_transform(xyz0, M)
                pcd0 = make_open3d_point_cloud(xyz0_t)
                pcd1 = make_open3d_point_cloud(xyz1)
                reg = o3d.pipelines.registration.registration_icp(pcd0, pcd1, 0.2, np.eye(4),
                                                           o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                           o3d.pipelines.registration.ICPConvergenceCriteria(
                                                               max_iteration=200))
                pcd0.transform(reg.transformation)
                M2 = M @ reg.transformation
                np.save(filename, M2)
            else:
                M2 = np.load(filename)
            kitti_icp_cache[key] = M2
        else:
            M2 = kitti_icp_cache[key]

        trans = M2

        pcd0 = make_open3d_point_cloud(xyz0)
        pcd1 = make_open3d_point_cloud(xyz1)
        pcd0 = o3d.geometry.PointCloud.voxel_down_sample(pcd0, self.voxel_size)
        pcd1 = o3d.geometry.PointCloud.voxel_down_sample(pcd1, self.voxel_size)

        anc_points = np.array(pcd0.points)
        pos_points = np.array(pcd1.points)

        anc_feat = np.ones_like(anc_points[:, :1]).astype(np.float32)
        pos_feat = np.ones_like(pos_points[:, :1]).astype(np.float32)

        # Voxelize xyz and feats
        anc_coords = np.floor(anc_points / self.voxel_size)
        pos_coords = np.floor(pos_points / self.voxel_size)
        anc_coords, anc_sel, anc_map = ME.utils.sparse_quantize(anc_coords, return_index=True, return_inverse=True)
        pos_coords, pos_sel, pos_map = ME.utils.sparse_quantize(pos_coords, return_index=True, return_inverse=True)

        anc_coords = torch.as_tensor(anc_coords, dtype=torch.int32)
        pos_coords = torch.as_tensor(pos_coords, dtype=torch.int32)
        anc_voxel_feat = anc_feat[anc_sel]
        pos_voxel_feat = pos_feat[pos_sel]

        return anc_points, pos_points, anc_feat, pos_feat, np.array([]), np.array([]), trans, anc_coords, pos_coords, anc_voxel_feat, pos_voxel_feat, anc_sel, pos_sel, anc_map, pos_map

    # Rotation matrix along axis with angle theta
    def M(self, axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    def sample_random_trans(self, pcd, randg, rotation_range=360):
        T = np.eye(4)
        R = self.M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
        T[:3, :3] = R
        T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
        return T