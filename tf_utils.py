import sys
sys.path.insert(0,"/usr/local/cython/")
import gtsam

import numpy as np

import torch
import pytorch3d.transforms as p3d_t

def vec3_to_pose2(vec):
    return gtsam.Pose2(vec[0], vec[1], vec[2])

def pose2_to_vec3(pose2):
    return [pose2.x(), pose2.y(), pose2.theta()]



def get_noise_model(factor_cov):

    factor_noise_model = None

    if (factor_cov.shape[0] <= 3):  # cov sigmas
        factor_noise_model = gtsam.noiseModel_Diagonal.Sigmas(factor_cov)
    elif (factor_cov.shape[0] == 9):  # cov matrix
        factor_cov = np.reshape(factor_cov, (3, 3))
        factor_noise_model = gtsam.noiseModel_Gaussian.Covariance(factor_cov)

    return factor_noise_model

def add_unary_factor(graph, keys, factor_cov, factor_meas):

    factor_noise_model = get_noise_model(factor_cov)
    factor_meas_pose = vec3_to_pose2(factor_meas)
    factor = gtsam.PriorFactorPose2(
        keys[0], factor_meas_pose, factor_noise_model)

    graph.push_back(factor)

    return graph

def wrap_to_pi(arr):
    arr_wrap = (arr + math.pi) % (2 * math.pi) - math.pi
    return arr_wrap

def add_binary_odom_factor(graph, keys, factor_cov, factor_meas):

    factor_noise_model = get_noise_model(factor_cov)
    factor_meas_pose = vec3_to_pose2(factor_meas)
    factor = gtsam.BetweenFactorPose2(
        keys[0], keys[1], factor_meas_pose, factor_noise_model)

    graph.push_back(factor)

    return graph

def tf2d_between(pose1, pose2, device=None, requires_grad=None):
    """
    Relative transform of pose2 in pose1 frame, i.e. T12 = T1^{1}*T2
    :param pose1: n x 3 tensor [x,y,yaw]
    :param pose2: n x 3 tensor [x,y,yaw]
    :return: pose12 n x 3 tensor [x,y,yaw]
    """

    num_data = pose1.shape[0]

    rot1 = torch.cat([torch.zeros(num_data, 1, device=device, requires_grad=requires_grad), torch.zeros(
        num_data, 1, device=device, requires_grad=requires_grad), pose1[:, 2][:, None]], 1)
    rot2 = torch.cat([torch.zeros(num_data, 1, device=device, requires_grad=requires_grad), torch.zeros(
        num_data, 1, device=device, requires_grad=requires_grad), pose2[:, 2][:, None]], 1)
    t1 = torch.cat([pose1[:, 0][:, None], pose1[:, 1]
                    [:, None], torch.zeros(num_data, 1, device=device, requires_grad=requires_grad)], 1)
    t2 = torch.cat([pose2[:, 0][:, None], pose2[:, 1]
                    [:, None], torch.zeros(num_data, 1, device=device, requires_grad=requires_grad)], 1)

    R1 = p3d_t.euler_angles_to_matrix(rot1, "XYZ")
    R2 = p3d_t.euler_angles_to_matrix(rot2, "XYZ")
    R1t = torch.inverse(R1)

    R12 = torch.matmul(R1t, R2)
    rot12 = p3d_t.matrix_to_euler_angles(R12, "XYZ")
    t12 = torch.matmul(R1t, (t2-t1)[:, :, None])
    t12 = t12[:, :, 0]

    tx = t12[:, 0][:, None]
    ty = t12[:, 1][:, None]
    yaw = rot12[:, 2][:, None]
    pose12 = torch.cat([tx, ty, yaw], 1)

    return pose12

def tf2d_compose(pose1, pose12):
    """
    Composing pose1 with pose12, i.e. T2 = T1*T12
    :param pose1: n x 3 tensor [x,y,yaw]
    :param pose12: n x 3 tensor [x,y,yaw]
    :return: pose2 n x 3 tensor [x,y,yaw]
    """

    num_data = pose1.shape[0]

    rot1 = torch.cat([torch.zeros(num_data, 1), torch.zeros(
        num_data, 1), pose1[:, 2][:, None]], 1)
    rot12 = torch.cat([torch.zeros(num_data, 1), torch.zeros(
        num_data, 1), pose12[:, 2][:, None]], 1)
    t1 = torch.cat([pose1[:, 0][:, None], pose1[:, 1]
                    [:, None], torch.zeros(num_data, 1)], 1)
    t12 = torch.cat([pose12[:, 0][:, None], pose12[:, 1]
                    [:, None], torch.zeros(num_data, 1)], 1)

    R1 = p3d_t.euler_angles_to_matrix(rot1, "XYZ")
    R12 = p3d_t.euler_angles_to_matrix(rot12, "XYZ")

    R2 = torch.matmul(R1, R12)
    rot2 = p3d_t.matrix_to_euler_angles(R2, "XYZ")
    t2 = torch.matmul(R1, t12[:, :, None]) + t1[:, :, None]
    t2 = t2[:, :, 0]

    tx = t2[:, 0][:, None]
    ty = t2[:, 1][:, None]
    yaw = rot2[:, 2][:, None]
    pose2 = torch.cat([tx, ty, yaw], 1)

    return pose2