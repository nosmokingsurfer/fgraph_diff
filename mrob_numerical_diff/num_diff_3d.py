import numpy as np
import mrob
from tqdm import tqdm 
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool, cpu_count
from sklearn.metrics.pairwise import cosine_similarity
from num_diff import find_factor_coord_idx, visualize_gradient, compare_gradients
import sys
from pathlib import Path


def info_matrix_from_triangular(elements):
    full_matrix = np.zeros((6, 6))
    upper_tri_indices = np.triu_indices(6)
    full_matrix[upper_tri_indices] = elements
    return full_matrix + np.triu(full_matrix, 1).T


def read_graph_toro_description_3d(toro_file):
    vertex_ini = {}
    factors = {}
    factors_dictionary = {}

    with open(toro_file, 'r') as file:
        for line in file:
            d = line.strip().split()
            if not d:
                continue
            if d[0] == 'VERTEX3':
                node_index = int(d[1])
                pose = np.array([float(v) for v in d[2:8]], dtype='float64')  # [x, y, z, roll, pitch, yaw] .Ln()
                vertex_ini[node_index] = pose
                factors_dictionary[node_index] = []
            elif d[0] == 'EDGE3':
                node_origin = int(d[1])
                node_target = int(d[2])
                meas = np.array([float(v) for v in d[3:9]], dtype='float64')  # [dx, dy, dz, droll, dpitch, dyaw] .Ln()
                info_values = [float(v) for v in d[9:]]
                info = info_matrix_from_triangular(info_values)
                factors[(node_origin, node_target)] = (meas, info)
                if node_target in factors_dictionary:
                    factors_dictionary[node_target].append(node_origin)
                else:
                    factors_dictionary[node_target] = [node_origin]
            elif d[0] == 'EDGE1':
                node_index = int(d[1])
                meas = np.array([float(v) for v in d[2:8]], dtype='float64')  # [x, y, z, roll, pitch, yaw]
                info_values = [float(v) for v in d[8:]]
                info = info_matrix_from_triangular(info_values)
                factors[(node_index, node_index)] = (meas, info)
                if node_index in factors_dictionary:
                    factors_dictionary[node_index].append(node_index)
                else:
                    factors_dictionary[node_index] = [node_index]
    return vertex_ini, factors, factors_dictionary


def compose_graph_3d(vertex_ini, factors, factors_dictionary, perturb_index_x=None, perturb_index_z=None, dx=0, dz=0):
    graph = mrob.FGraph()
    
    # adding nodes to the graph
    for idx, node_index in enumerate(sorted(vertex_ini.keys())):
        x = vertex_ini[node_index].copy()  # x is Ln() coordinates, shape=6: [x, y, z, roll, pitch, yaw]
        if perturb_index_x is not None and node_index == perturb_index_x[0]:
            x[perturb_index_x[1]] += dx
        pose = mrob.SE3(x)
        n = graph.add_node_pose_3d(pose)

    # adding factors to the graph using mapped indices
    for node_target in factors_dictionary:
        mrob_node_target = node_target
        for node_origin in factors_dictionary[node_target]:
            mrob_node_origin = node_origin
            measurement, information_matrix = factors[(node_origin, node_target)]
            obs = measurement.copy()
            if perturb_index_z is not None and (node_origin, node_target) == perturb_index_z[:2]:
                obs[perturb_index_z[2]] += dz
            obs_se3 = mrob.SE3(obs)
            if node_origin != node_target:
                graph.add_factor_2poses_3d(obs_se3, mrob_node_origin, mrob_node_target, information_matrix)
            else:
                graph.add_factor_1pose_3d(obs_se3, mrob_node_origin, information_matrix)

    return graph


def numerical_diff1_3d(toro_file, dz=1e-4):
    vertex_ini, factors, factors_dictionary = read_graph_toro_description_3d(toro_file)
    
    graph_0 = compose_graph_3d(vertex_ini, factors, factors_dictionary)
    graph_0.solve(mrob.LM, verbose=False)
    x_0 = graph_0.get_estimated_state()

    dim_x = len(x_0) * 6
    obs_dim = len(factors) * 6
    gradient = np.zeros((dim_x, obs_dim))
    factor_keys = list(factors.keys())
    
    for i in tqdm(range(obs_dim)):
        factor_idx, coord_idx = find_factor_coord_idx(i, coord_num=6)
        nodeOrigin, t = factor_keys[factor_idx]
        perturb_index = (nodeOrigin, t, coord_idx) 

        graph_new = compose_graph_3d(vertex_ini, factors, factors_dictionary, perturb_index_z=perturb_index, dz=dz)
        graph_new.solve(mrob.LM, verbose=False)
        x_new = graph_new.get_estimated_state()
        dx_new = np.zeros(6 * dim_x) #True only if states are 3D poses
        dx_new = []
        for k in range(len(x_new)):
            dx_new.append((mrob.SE3(x_new[k]) * mrob.SE3(x_0[k]).inv()).Ln() / dz)

        gradient[:, i] = np.array(dx_new).flatten()
    return gradient


def compute_chi2_matrix_3d(vertex_ini, factors, factors_dictionary, factor_keys, i_x, i_z, dx, dz):
    factor_idx_x, coord_idx_x = find_factor_coord_idx(i_x, coord_num=6)
    factor_idx_z, coord_idx_z = find_factor_coord_idx(i_z, coord_num=6)
    nodeOrigin_z, t_z = factor_keys[factor_idx_z]
    perturb_index_x = (factor_idx_x, coord_idx_x)
    perturb_index_z = (nodeOrigin_z, t_z, coord_idx_z)
    
    graph_pp = compose_graph_3d(vertex_ini, factors, factors_dictionary, 
                                perturb_index_x=perturb_index_x, perturb_index_z = perturb_index_z, dx=dx, dz=dz) # x+h, z+k
    graph_pm = compose_graph_3d(vertex_ini, factors, factors_dictionary, 
                                perturb_index_x=perturb_index_x, perturb_index_z = perturb_index_z, dx=dx, dz=-dz) # x+h, z-k
    graph_mp = compose_graph_3d(vertex_ini, factors, factors_dictionary, 
                                perturb_index_x=perturb_index_x, perturb_index_z = perturb_index_z, dx=-dx, dz=dz) # x-h, z+k
    graph_mm = compose_graph_3d(vertex_ini, factors, factors_dictionary, 
                                perturb_index_x=perturb_index_x, perturb_index_z = perturb_index_z, dx=-dx, dz=-dz) # x-h, z-k

    chi2_pp = graph_pp.chi2(evaluateResidualsFlag=True)
    chi2_pm = graph_pm.chi2(evaluateResidualsFlag=True)
    chi2_mp = graph_mp.chi2(evaluateResidualsFlag=True)
    chi2_mm = graph_mm.chi2(evaluateResidualsFlag=True)

    return (chi2_pp - chi2_pm - chi2_mp + chi2_mm) / 4 / dx / dz


def numerical_diff2_3d(toro_file, dx=1e-4, dz=1e-4):
    vertex_ini, factors, factors_dictionary = read_graph_toro_description_3d(toro_file)
    graph_0 = compose_graph_3d(vertex_ini, factors, factors_dictionary)
    x_0 = graph_0.get_estimated_state()
    dim_x = len(x_0) * 6
    dim_z = len(factors) * 6
    chi2_matrix = np.zeros((dim_x, dim_z))
    factor_keys = list(factors.keys())
    graph_0.solve(mrob.LM, verbose=False)
    hessian = graph_0.get_information_matrix().todense()
    
    for i_x in tqdm(range(dim_x)):
        for i_z in range(dim_z):
            chi2_matrix[i_x, i_z] = compute_chi2_matrix_3d(vertex_ini, factors, factors_dictionary, factor_keys, i_x, i_z, dx, dz)
            
    return -np.linalg.inv(hessian) @ chi2_matrix


if __name__ == "__main__":  
    toro_file = './out/spline_toro_graph_9.txt'
    dx = 1e-1
    dz = 1e-1
    
    gradient1 = numerical_diff1_3d(toro_file, dz=dz)
    visualize_gradient(gradient1, 'Gradient via direct method for 3D spline dataset', dx = None, dz=dz)
    
    gradient2 = numerical_diff2_3d(toro_file, dx=dx, dz=dz)
    visualize_gradient(gradient2, 'Gradient via direct method for 3D spline dataset', dx = dx, dz=dz)
    
    compare_gradients(gradient1, gradient2, dx=dx, dz=dz)