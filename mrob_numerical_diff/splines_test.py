import numpy as np
import mrob
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from tqdm import tqdm
import os
import pickle

from pathlib import Path

import sys
sys.path.insert(0,str(Path(".").resolve()))

from spline_dataset.spline_generation import generate_batch_of_splines
from spline_dataset.spline_dataloader import Spline_2D_Dataset
from graph_generator import ToRoContainer
from num_diff import mean_squared_error
from num_diff_3d import read_graph_toro_description_3d, compose_graph_3d


def compose_graph_3d_no_perturb(vertex_ini, factors, factors_dictionary, perturb_index_x=None, perturb_index_z=None, dx=0, dz=0):
    graph = mrob.FGraph()

    for node_index in sorted(vertex_ini.keys()):
        x = vertex_ini[node_index].copy()  # [x, y, z, roll, pitch, yaw]
        # if perturb_index_x is not None and node_index == perturb_index_x[0]:
        #     x[perturb_index_x[1]] += dx
        pose = mrob.SE3(x)
        graph.add_node_pose_3d(pose)

    for (node_origin, node_target), (meas, info) in factors.items():
        obs = meas.copy()
        # if perturb_index_z is not None and (node_origin, node_target) == perturb_index_z[:2]:
        #     obs[perturb_index_z[2]] += dz
        obs_se3 = mrob.SE3(obs)
        if node_origin != node_target:
            graph.add_factor_2poses_3d(obs_se3, node_origin, node_target, info)
        else:
            graph.add_factor_1pose_3d(obs_se3, node_origin, info)
    return graph


def integrate(R,p,v,omega,acc,dt):
    '''
    Integrates a one step integration by a pair [omega, acc], in body frame
    and R,v,p in global frame
    '''
    R_new = R @ mrob.geometry.SO3(omega * dt).R()
    v_new = v + dt * R_new @ acc
    p_new = p + dt * v_new
    return R_new,p_new,v_new


def populate_graph(sample, imu_step = 5, gps_step=10):

    W_odo = np.eye(6)*0.001
    W_gps = np.eye(6)

    graph = mrob.FGraph()
    toro_container = ToRoContainer()

    nodes_ids = []
    
    # adding first node
    T = sample['gt_se3'][0]
    t = time_stamp=sample['time'][0]
    n = graph.add_node_pose_3d(x=T)
    toro_container.add_node_pose_3d(n, T.Ln())
    nodes_ids.append((n,0))

    # adding all other nodes
    for idx in range(imu_step, len(sample['imu']), imu_step):
        T = sample['gt_se3'][idx]
        #injectiong noise

        # injecting (x,y) noise
        #T[:2,3] = T[:2,3] + np.random.randn(2)*0.25

        # putting update
        n = graph.add_node_pose_3d(x=mrob.SE3(T))
        toro_container.add_node_pose_3d(n, T.Ln()) #T,Ln only
        nodes_ids.append((n, idx))

    # reduced time matches to nodes timestamps
    reduced_time = np.array(range(0, len(sample['imu']), imu_step))*0.01

    # reducing IMU measuremnets
    reduced_imu = [] 
    for idx in range(imu_step, len(sample['imu']),imu_step):
        reduced_imu.append(sample['imu'][idx-imu_step:idx].mean(axis=0).squeeze())

    # adding all odometry factors for created nodes
    odometry_factors = []
    last_odo_factor_in_loop = False
    for i in range(0, len(nodes_ids) - 1):

        n, original_index = nodes_ids[i]

        src, dst = n, n + 1
        
        acc_x, acc_y, omega_z = reduced_imu[i]

        # relative pose betweeb two vertexes
        odo = (sample['gt_se3'][nodes_ids[i+1][1]].inv()*sample['gt_se3'][nodes_ids[i][1]]) #TODO https://github.com/prime-slam/mrob/blob/fix/numpy2.0_compat/src/FGraph/mrob/factors/factor2Poses3d.hpp
        assert src == n == nodes_ids[i][0], 'Mismatch in source indexes'
        assert dst == n + 1 == nodes_ids[i+1][0], 'Mismatch in destination indexes'
        
        graph.add_factor_2poses_3d(odo, src, dst, W_odo)
        odometry_factors.append((src,dst))
        
        toro_container.add_factor_2poses_3d(src, dst, odo.Ln(), W_odo) #only Ln array
        last_odo_factor_in_loop = i + 1 == len(nodes_ids) - 1
        
    # add to graph last odometry twist factor if neede
    if not last_odo_factor_in_loop:
        odo = (sample['gt_se3'][nodes_ids[-1][1]] * sample['gt_se3'][nodes_ids[-2][1]].inv())
        graph.add_factor_2poses_3d(odo, nodes_ids[-2][0], nodes_ids[-1][0], W_odo)
        toro_container.add_factor_2poses_3d(nodes_ids[-2][0], nodes_ids[-1][0], odo.Ln(), W_odo)
    
    # adding gps factors going along original IMU timestamps
    gps_factors = []
    last_gps_factor_in_loop = False
    for idx in range(0, len(sample['imu']), gps_step):
        
        gps_timestamp = idx*0.01
        # checking if there is a propriate node with timestamp just before the timestapms of GPS
        n_arg = np.argwhere(gps_timestamp >= reduced_time)
        if len(n_arg) > 0:
            n_arg = n_arg[-1][0]
            T = sample['gt_se3'][nodes_ids[n_arg][1]]
            graph.add_factor_1pose_3d(T, nodes_ids[n_arg][0], W_gps)
            toro_container.add_factor_1pose_3d(nodes_ids[n_arg][0], T.Ln(), W_gps)
            
            gps_factors.append(n_arg)
            last_gps_factor_in_loop = n_arg == len(nodes_ids) - 1 
    
    #adding last pose as GPS factor if needed
    if not last_gps_factor_in_loop:
        T = sample['gt_se3'][nodes_ids[-1][1]]
        graph.add_factor_1pose_3d(T, nodes_ids[-1][0], W_gps)
        gps_factors.append(nodes_ids[-1][0])
        toro_container.add_factor_1pose_3d(nodes_ids[-1][0], T.Ln(), W_gps)

    return graph, toro_container.get_lines()

if __name__ == "__main__":
    start_time = time()
    output_path = './out/'

    if not os.path.exists(output_path):
        os.makedirs(output_path,exist_ok=True)
    path_to_splines = output_path + 'splines/'

    number_of_splines = 10
    if not os.path.exists(path_to_splines):
        number_of_control_nodes = 10
        generate_batch_of_splines(path_to_splines, number_of_splines, number_of_control_nodes, 100)

    if not os.path.isfile(path_to_splines + f'spline_dataset_{number_of_splines}.pkl'):
        dataset = Spline_2D_Dataset(path_to_splines, window=1, enable_noise = not True)
        pickle.dump(dataset,open(path_to_splines + f'spline_dataset_{number_of_splines}.pkl','wb'))
    else:
        dataset = pickle.load(open(path_to_splines + f'spline_dataset_{number_of_splines}.pkl','rb'))

    # iterating through all splines in dataset
    for idx, sample in tqdm(enumerate(dataset)):
        #print(sample)
        toro_file = output_path+f'spline_toro_graph_{idx}.txt'

        # initializing graph 
        graph, toro_lines = populate_graph(sample, imu_step=100, gps_step=100)
        #print(toro_lines)

        with open(toro_file,'w') as f:
            f.writelines(toro_lines)
            f.close()

        # reading serialised graph from toro file and composing it back into mrob FGraph
        vertex_ini, factors, factors_dictionary = read_graph_toro_description_3d(toro_file)
        
        graph_0 = compose_graph_3d(vertex_ini, factors, factors_dictionary)
        
        # checking that serialized and deserialized graphs have the same states
        np.allclose(np.array(graph.get_estimated_state()), np.array(graph_0.get_estimated_state()))
        print('MSE:', mean_squared_error(np.array(graph.get_estimated_state()), np.array(graph_0.get_estimated_state())))

        initial_error = graph.chi2(True)
        print(initial_error)

    print(f"Elapsed time: {time() - start_time} secs")