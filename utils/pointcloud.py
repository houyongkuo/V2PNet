import copy
import numpy as np
import math
import open3d as o3d
from utils.eval import find_nn_cpu
import torch
from scipy.spatial import cKDTree


def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def make_open3d_feature(data, dim, npts):
    feature = o3d.registration.Feature()
    feature.resize(dim, npts)
    feature.data = data.cpu().numpy().astype('d').transpose()
    return feature


def make_open3d_feature_from_numpy(data):
    assert isinstance(data, np.ndarray)
    assert data.ndim == 2

    feature = o3d.pipelines.registration.Feature()
    feature.resize(data.shape[1], data.shape[0])
    feature.data = data.astype('d').transpose()
    return feature


def prepare_pointcloud(filename, voxel_size):
    pcd = o3d.io.read_point_cloud(filename)
    T = get_random_transformation(pcd)
    pcd.transform(T)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    return pcd_down, T


def get_matching_indices(source, target, trans, search_voxel_size, K=None):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)

    # source_copy = o3d.geometry.PointCloud.voxel_down_sample(source_copy, voxel_size=0.03)
    # target_copy = o3d.geometry.PointCloud.voxel_down_sample(target_copy, voxel_size=0.03)

    source_copy.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(target_copy)

    match_inds0 = []
    for i, point in enumerate(source_copy.points):
        # Tuple[int, open3d.utility.IntVector, open3d.utility.DoubleVector]
        [_, idx, _] = pcd_tree.search_hybrid_vector_3d(point, search_voxel_size, 1)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds0.append((i, j))
    match0 = np.array(match_inds0)

    pcd_tree = o3d.geometry.KDTreeFlann(source_copy)
    match_inds1 = []
    for i, point in enumerate(target_copy.points):
        # Tuple[int, open3d.utility.IntVector, open3d.utility.DoubleVector]
        [_, idx, _] = pcd_tree.search_hybrid_vector_3d(point, search_voxel_size, 1)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds1.append((i, j))
    match1 = np.array(match_inds1)
    match1[:, [0, 1]] = match1[:, [1, 0]]
    import numpy_indexed as npi
    corr = npi.intersection(match0, match1)
    return corr


def valid_feat_ratio(pcd0, pcd1, feat0, feat1, trans_gth, thresh=0.1):
    pcd0_copy = copy.deepcopy(pcd0)
    pcd0_copy.transform(trans_gth)
    inds = find_nn_cpu(feat0, feat1, return_distance=False)
    dist = np.sqrt(((np.array(pcd0_copy.points) - np.array(pcd1.points)[inds]) ** 2).sum(1))
    return np.mean(dist < thresh)


def evaluate_feature_3dmatch(pcd0, pcd1, feat0, feat1, trans_gth, inlier_thresh=0.1):
    r"""Return the hit ratio (ratio of inlier correspondences and all correspondences).

  inliear_thresh is the inlier_threshold in meter.
  """
    if len(pcd0.points) < len(pcd1.points):
        hit = valid_feat_ratio(pcd0, pcd1, feat0, feat1, trans_gth, inlier_thresh)
    else:
        hit = valid_feat_ratio(pcd1, pcd0, feat1, feat0, np.linalg.inv(trans_gth), inlier_thresh)
    return hit


def get_matching_matrix(source, target, trans, voxel_size, debug_mode):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    source_copy.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(target_copy)
    matching_matrix = np.zeros((len(source_copy.points), len(target_copy.points)))

    for i, point in enumerate(source_copy.points):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(point, voxel_size * 1.5)
        if k >= 1:
            matching_matrix[i, idx[0]] = 1  # TODO: only the cloest?

    return matching_matrix


def get_random_transformation(pcd_input):
    def rot_x(x):
        out = np.zeros((3, 3))
        c = math.cos(x)
        s = math.sin(x)
        out[0, 0] = 1
        out[1, 1] = c
        out[1, 2] = -s
        out[2, 1] = s
        out[2, 2] = c
        return out

    def rot_y(x):
        out = np.zeros((3, 3))
        c = math.cos(x)
        s = math.sin(x)
        out[0, 0] = c
        out[0, 2] = s
        out[1, 1] = 1
        out[2, 0] = -s
        out[2, 2] = c
        return out

    def rot_z(x):
        out = np.zeros((3, 3))
        c = math.cos(x)
        s = math.sin(x)
        out[0, 0] = c
        out[0, 1] = -s
        out[1, 0] = s
        out[1, 1] = c
        out[2, 2] = 1
        return out

    pcd_output = copy.deepcopy(pcd_input)
    mean = np.mean(np.asarray(pcd_output.points), axis=0).transpose()
    xyz = np.random.uniform(0, 2 * math.pi, 3)
    R = np.dot(np.dot(rot_x(xyz[0]), rot_y(xyz[1])), rot_z(xyz[2]))
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:3, 3] = np.dot(-R, mean)
    T[3, 3] = 1
    return T


def pcd2xyz(pcd):
    return np.asarray(pcd.points).T


def extract_fpfh(pcd, voxel_size):
    radius_normal = voxel_size * 2
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return np.array(fpfh.data).T


def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
    feat1tree = cKDTree(feat1)
    dists, nn_inds = feat1tree.query(feat0, k=knn, n_jobs=-1)
    if return_distance:
        return nn_inds, dists
    else:
        return nn_inds


def Feat_Voxelization(feature, point_map, voxel_map):
    '''
    feature [num_point, feature dim]
    point_map [num_point] : voxel index in each point
    voxel_map [num_voxel] : Support point index in each voxel
    '''

    # global n
    point_list, voxel_list = [], []
    point_map = point_map.cpu().numpy()

    for n, index in enumerate(point_map):
        # point_list : point idx
        point_list.append(n)
        voxel_list.append(int(index))
    voxel_feat = torch.zeros([len(voxel_list), len(feature[1])], dtype=torch.float)
    
    voxel_list = np.array(voxel_list)
    for i, i_idx in enumerate(voxel_list):
        if np.sum(voxel_list == i_idx) == 1:
            voxel_feat[i_idx] = feature[i]
        elif np.sum(voxel_list == i_idx) != 1:
            index = [i for i, val in enumerate(voxel_list) if val == i_idx]
            mean_feat = torch.mean(feature[index], dim=0, keepdim=True)
            voxel_feat[i_idx] = mean_feat

    idx = torch.all(voxel_feat[..., :] == 0, axis=1)
    index = []
    for i in range(idx.shape[0]):
        if not idx[i].item():
            index.append(i)
    index = torch.tensor(index)

    voxel_feat = torch.index_select(voxel_feat, 0, index)
    assert len(voxel_feat) == len(voxel_map)

    return voxel_feat


def get_correspondences(ref_points, src_points, matching_radius):
    r"""Find the ground truth correspondences within the matching radius between two point clouds.

    Return correspondence indices [indices in ref_points, indices in src_points]
    """
    # src_points = apply_transform(src_points, transform)
    src_tree = cKDTree(src_points)
    indices_list = src_tree.query_ball_point(ref_points, matching_radius)
    corr_indices = np.array(
        [(i, j) for i, indices in enumerate(indices_list) for j in indices],
        dtype=np.long,
    )
    return corr_indices
