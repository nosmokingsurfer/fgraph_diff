import numpy as np
import mrob
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from tqdm import tqdm
import os
import pickle

from pathlib import Path

from spline_dataset.spline_generation import generate_batch_of_splines
from spline_dataset.spline_dataloader import Spline_2D_Dataset

from num_diff import read_graph_toro_description, compose_graph


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

    W_odo = np.eye(6)*1e-2
    W_gps = np.eye(6)

    graph = mrob.FGraph()
    #TODO put toro container here

    nodes_ids = []
    
    # adding first node
    T = sample['gt_se3'][0].T()
    t = time_stamp=sample['time'][0]
    n = graph.add_node_pose_3d(x=mrob.SE3(T))
    nodes_ids.append((n,0))

    # adding all other nodes
    for idx in range(imu_step, len(sample['imu']), imu_step):
        T = sample['gt_se3'][idx].T()
        #injectiong noise

        # injecting (x,y) noise
        # T[:2,3] = T[:2,3] + np.random.randn(2)*0.25

        # putting update
        n = graph.add_node_pose_3d(x=mrob.SE3(T))
        # TODO add 3d pose (vertex3) into toro container
        nodes_ids.append((n, idx))

    # reduced time matches to nodes timestamps
    reduced_time = np.array(range(0, len(sample['imu']), imu_step))*0.01

    # reducing IMU measuremnets
    reduced_imu = [] 
    for idx in range(imu_step, len(sample['imu']),imu_step):
        reduced_imu.append(sample['imu'][idx-imu_step:idx].mean(axis=0).squeeze())

    # adding all odometry factors for created nodes
    odometry_factors = []
    for i in range(0, len(nodes_ids) - 1):

        n, original_index = nodes_ids[i]

        src, dst = n, n + 1
        
        acc_x, acc_y, omega_z = reduced_imu[i]

        # relative pose betweeb two vertexes
        odo = (sample['gt_se3'][idx + 1].inv()*sample['gt_se3'][idx])

        # TODO add 2pose 3d odometry factor to toro container
        # graph.add_factor .... #TODO place something here
        odometry_factors.append((src,dst))
    # TODO add to graph last odometry twist factor + put it into toro container

    # adding gps factors going along original IMU timestamps
    gps_factors = []
    for idx in range(0, len(sample['imu']), gps_step):
        
        gps_timestamp = idx*0.01
        # checking if there is a propriate node with timestamp just before the timestapms of GPS
        n_arg = np.argwhere(gps_timestamp >= reduced_time)
        if len(n_arg) > 0:
            n_arg = n_arg[-1][0]

            # TODO add GPS factor to graph
            # graph.add_factor_1pose_3d(obs=T, nodeId=n_arg, obsInvCov=W_gps)
            # TODO add GPS factor for 3D pose to toro container
            gps_factors.append(n_arg)
    
    #adding last pose as GPS factor

    T = sample['gt_se3'][nodes_ids[-1][1]]
    t = sample['time'][nodes_ids[-1][1]]

    # TODO add GPS factor for 3D pose to toro container
    # graph.add_factor_1pose_3d(obs=T, nodeId=nodes_ids[-1][0], obsInvCov=W_gps)
    gps_factors.append(nodes_ids[-1][0])

    return graph , "" #TODO output toro lines here as additional output

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
        print(sample)
        toro_file = output_path+f'spline_toro_graph_{idx}.txt'

        # initializing graph 
        graph, toro_lines = populate_graph(sample, imu_step=100, gps_step=100)
        print(toro_lines)

        with open(toro_file,'w') as f:
            f.writelines(toro_lines)
            f.close()

        # reading serialised graph from toro file and composing it back into mrob FGraph
        vertex_ini, factors, factors_dictionary = read_graph_toro_description(toro_file)
        
        graph_0 = compose_graph(vertex_ini, factors, factors_dictionary)

        # checking that serialized and deserialized graphs have the same states
        np.allclose(np.array(graph.get_estimated_state()),np.array(graph_0.get_estimated_state()))

        initial_error = graph.chi2(True)

    print(f"Elapsed time: {time() - start_time} secs")