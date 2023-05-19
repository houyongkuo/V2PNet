import open3d as o3d
import os
import logging
import numpy as np
import torch

from utils.trajectory import CameraPose
from utils.pointcloud import compute_overlap_ratio, make_open3d_point_cloud, make_open3d_feature_from_numpy
import MinkowskiEngine as ME

def run_ransac(xyz0, xyz1, feat0, feat1, voxel_size):
  distance_threshold = voxel_size * 1.5
  result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
      xyz0, xyz1, feat0, feat1, distance_threshold,
      o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
          o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
          o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
      ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
  return result_ransac.transformation


def gather_results(results):
  traj = []
  for r in results:
    success = r[0]
    if success:
      traj.append(CameraPose([r[1], r[2], r[3]], r[4]))
  return traj


def gen_matching_pair(pts_num):
  matching_pairs = []
  for i in range(pts_num):
    for j in range(i + 1, pts_num):
      matching_pairs.append([i, j, pts_num])
  return matching_pairs


def read_data(feature_path, name):
  data = np.load(os.path.join(feature_path, name + ".npz"))
  xyz = make_open3d_point_cloud(data['xyz'])
  feat = make_open3d_feature_from_numpy(data['feature'])
  return data['points'], xyz, feat


def do_single_pair_matching(feature_path, set_name, m, voxel_size):
  i, j, s = m
  name_i = "%s_%03d" % (set_name, i)
  name_j = "%s_%03d" % (set_name, j)
  logging.info("matching %s %s" % (name_i, name_j))
  points_i, xyz_i, feat_i = read_data(feature_path, name_i)
  points_j, xyz_j, feat_j = read_data(feature_path, name_j)
  if len(xyz_i.points) < len(xyz_j.points):
    trans = run_ransac(xyz_i, xyz_j, feat_i, feat_j, voxel_size)
  else:
    trans = run_ransac(xyz_j, xyz_i, feat_j, feat_i, voxel_size)
    trans = np.linalg.inv(trans)
  ratio = compute_overlap_ratio(xyz_i, xyz_j, trans, voxel_size)
  logging.info(f"{ratio}")
  if ratio > 0.3:
    return [True, i, j, s, np.linalg.inv(trans)]
  else:
    return [False, i, j, s, np.identity(4)]

def extract_features(model,
                     xyz,
                     rgb=None,
                     normal=None,
                     voxel_size=0.05,
                     device=None,
                     skip_check=False,
                     is_eval=True):
  '''
  xyz is a N x 3 matrix
  rgb is a N x 3 matrix and all color must range from [0, 1] or None
  normal is a N x 3 matrix and all normal range from [-1, 1] or None

  if both rgb and normal are None, we use Nx1 one vector as an input

  if device is None, it tries to use gpu by default

  if skip_check is True, skip rigorous checks to speed up

  model = model.to(device)
  xyz, feats = extract_features(model, xyz)
  '''
  if is_eval:
    model.eval()

  if not skip_check:
    assert xyz.shape[1] == 3

    N = xyz.shape[0]
    if rgb is not None:
      assert N == len(rgb)
      assert rgb.shape[1] == 3
      if np.any(rgb > 1):
        raise ValueError('Invalid color. Color must range from [0, 1]')

    if normal is not None:
      assert N == len(normal)
      assert normal.shape[1] == 3
      if np.any(normal > 1):
        raise ValueError('Invalid normal. Normal must range from [-1, 1]')

  if device is None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  feats = []
  if rgb is not None:
    # [0, 1]
    feats.append(rgb - 0.5)

  if normal is not None:
    # [-1, 1]
    feats.append(normal / 2)

  if rgb is None and normal is None:
    feats.append(np.ones((len(xyz), 1)))

  feats = np.hstack(feats)

  # Voxelize xyz and feats
  coords = np.floor(xyz / voxel_size)
  coords, inds = ME.utils.sparse_quantize(coords, return_index=True)
  # Convert to batched coords compatible with ME
  coords = ME.utils.batched_coordinates([coords])
  return_coords = xyz[inds]

  feats = feats[inds]

  feats = torch.tensor(feats, dtype=torch.float32)
  coords = torch.tensor(coords, dtype=torch.int32)

  stensor = ME.SparseTensor(feats, coordinates=coords, device=device)

  return return_coords, model(stensor).F

