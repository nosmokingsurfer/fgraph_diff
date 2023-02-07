from utils.vis_utils import vis_dataset_element
from multiprocessing import Pool
from attrdict import AttrDict
from tqdm import tqdm
import numpy as np
import os
import mrob
from data_utils import leo_dataset_to_toro, load_dataset

# import networkx as nx
from tqdm import tqdm

from pathlib import Path

# matplotlib.use('agg')
import matplotlib.pyplot as plt

import seaborn as sns

def toro_to_mrob(toro_file, i, j, delta):
    # Initialize data structures
    vertex_ini = {}
    factors = {}
    factors_dictionary = {}

    with open(toro_file, 'r') as file:
        for line in file:
            d = line.split()
            # read edges and vertex, in TORO format
            if d[0] == 'EDGE2':
                # EDGE2 id_origin   id_target   dx   dy   dth   I11   I12  I22  I33  I13  I23
                factors[int(d[1]), int(d[2])] = np.array(
                    [d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11]], dtype='float64')
                factors_dictionary[int(d[2])].append(int(d[1]))
            elif d[0] == 'VERTEX2':
                # VERTEX2 id x y theta
                # these are the initial guesses for node states
                vertex_ini[int(d[1])] = np.array(
                    [d[2], d[3], d[4]], dtype='float64')
                # create an empty list of pairs of nodes (factor) connected to each node
                factors_dictionary[int(d[1])] = []

            elif d[0] == 'EDGE1':
                factors[int(d[1]), int(d[1])] = np.array(
                    [d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10]], dtype='float64')
                factors_dictionary[int(d[1])].append(int(d[1]))

   # Initialize FG for solution
    graph = mrob.FGraph()
    x = vertex_ini[0]
    n = graph.add_node_pose_2d(x)
    # print('node 0 id = ', n) # id starts at 1

    N = len(vertex_ini)

    for t in range(1, N):
        x = vertex_ini[t]
        # print(x)
        n = graph.add_node_pose_2d(x)
        assert t == n, 'index on node is different from counter'

        # find factors to add. there must be 1 odom and other observations
        for nodeOrigin in factors_dictionary[n]:
            # inputs: obs, idOrigin, idTarget, invCov
            obs = factors[nodeOrigin, t][:3]
            covInv = np.zeros((3, 3))
            # on M3500 always diagonal information matrices
            covInv[0, 0] = factors[nodeOrigin, t][3]
            covInv[1, 1] = factors[nodeOrigin, t][5]
            covInv[2, 2] = factors[nodeOrigin, t][6]

            # covInv = np.linalg.inv(covInv)



            if nodeOrigin != n:
                # differentiating i-th odometry factors j-th covariance element
                if nodeOrigin == i:
                    covInv[j,j] += delta

                graph.add_factor_2poses_2d(obs, nodeOrigin, t, covInv)


            elif nodeOrigin == n:
                graph.add_factor_1pose_2d(obs, nodeOrigin, covInv)

    return graph


def mrob_solve_experiment(id, data, i,j,delta):

    if not os.path.exists("./out"):
        os.makedirs("./out", exist_ok=True)

    data = AttrDict(data)
    # print(id)

    # checking if cached unrolled data already exists
    unrolled_data_path = f"./out/{id}_unrolled_data.txt"
    if not os.path.exists(unrolled_data_path):

        unrolled_data = leo_dataset_to_toro(data)
        with open(unrolled_data_path, 'w') as f:
            f.writelines(unrolled_data)
            f.close()

#     # number of iterations for current experiment
#     n_steps = len(data.poses_gt)

#     # Initialize data structures
#     vertex_ini = {}
#     factors = {}
#     factors_dictionary = {}

#     # load file
    leo_dataset_file = f"./out/{id}_unrolled_data.txt"
#     with open(leo_dataset_file, 'r') as file:

#         for line in file:
#             d = line.split()
#             # read edges and vertex, in TORO format
#             if d[0] == 'EDGE2':
#                 # EDGE2 id_origin   id_target   dx   dy   dth   I11   I12  I22  I33  I13  I23
#                 factors[int(d[1]), int(d[2])] = np.array(
#                     [d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11]], dtype='float64')
#                 factors_dictionary[int(d[2])].append(int(d[1]))
#             elif d[0] == 'VERTEX2':
#                 # VERTEX2 id x y theta
#                 # these are the initial guesses for node states
#                 vertex_ini[int(d[1])] = np.array(
#                     [d[2], d[3], d[4]], dtype='float64')
#                 # create an empty list of pairs of nodes (factor) connected to each node
#                 factors_dictionary[int(d[1])] = []

#             elif d[0] == 'EDGE1':
#                 factors[int(d[1]), int(d[1])] = np.array(
#                     [d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10]], dtype='float64')
#                 factors_dictionary[int(d[1])].append(int(d[1]))

#    # Initialize FG for solution
#     graph = mrob.FGraph()
#     x = vertex_ini[0]
#     n = graph.add_node_pose_2d(x)
#     # print('node 0 id = ', n) # id starts at 1

#     # start events, we solve for each node, adding it and it corresponding factors
#     # in total takes 0.3s to read all datastructure
#     for t in range(1, n_steps):
#         x = vertex_ini[t]
#         # print(x)
#         n = graph.add_node_pose_2d(x)
#         assert t == n, 'index on node is different from counter'

#         # find factors to add. there must be 1 odom and other observations
#         for nodeOrigin in factors_dictionary[n]:
#             # inputs: obs, idOrigin, idTarget, invCov
#             obs = factors[nodeOrigin, t][:3]
#             covInv = np.zeros((3, 3))
#             # on M3500 always diagonal information matrices
#             covInv[0, 0] = factors[nodeOrigin, t][3]
#             covInv[1, 1] = factors[nodeOrigin, t][5]
#             covInv[2, 2] = factors[nodeOrigin, t][6]

#             # covInv = np.linalg.inv(covInv)

#             if nodeOrigin != n:
#                 graph.add_factor_2poses_2d(obs, nodeOrigin, t, covInv)

#             elif nodeOrigin == n:
#                 graph.add_factor_1pose_2d(obs, nodeOrigin, covInv)
    # exit(0)
    graph = toro_to_mrob(leo_dataset_file, i, j, delta)
    print(f'initial chi2 = {graph.chi2()}')
    # graph.print(True)
    graph.solve(mrob.LM, 50)
    print(f'chi2 = {graph.chi2()}')
    print('solution drawn')
    # graph.print(True)

    solution = graph.get_estimated_state()
    solution = np.array(solution).squeeze()

    return solution


def main():

    DATASET_PATH = str(Path("../leo/local/datasets/sim/nav2dfix/dataset_0000"))

    # reading the datasets
    datasets = load_dataset(DATASET_PATH, 'train')

    for i in tqdm(range(len(datasets))):
        experiment_id, data = datasets[i][0], datasets[i][1]

        mrob_solution = mrob_solve_experiment(experiment_id, data)

        np.savetxt(f"./out/{experiment_id}_mrob.txt", mrob_solution)

        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        vis_dataset_element(data, ax[0])
        ax[0].plot(mrob_solution[:, 0], mrob_solution[:, 1],
                   label='mrob trajectory')
        ax[0].legend()
        ax[0].grid()
        ax[0].axis('equal')

        ax[1].plot(np.unwrap(mrob_solution[:, 2]), label='mrob orientation')
        ax[1].legend()
        ax[1].grid()
        fig.suptitle(experiment_id)

        plt.savefig("./out/" + experiment_id + "_mrob.jpg")
        plt.show()

        plt.close('all')


def error_translational(traj1, traj2):
    t1= traj1[:,:2]
    t2 = traj2[:,:2]

    error = np.linalg.norm(t1-t2,axis=1)

    return error



def compute_gradient():
    DATASET_PATH = str(Path("../leo/local/datasets/sim/nav2dfix/dataset_0000"))

    # reading the datasets
    datasets = load_dataset(DATASET_PATH, 'train')

    # considering only first experiment
    experiment_id, data = datasets[0][0], datasets[0][1]

    # differentiating first 10 odometry factors
    # thus the result will be in 10x3 array
    delta = 1e-2

    K = 300 # how many factors will be differentiated

    num_gradient = np.zeros((K,3))

    # calculating error value in the starting point

    mrob_solution = mrob_solve_experiment(experiment_id, data, 0, 0, 0)

    np.savetxt(f"./out/{experiment_id}_mrob_{0}_{0}.txt", mrob_solution)


    gt_poses = np.array(data['poses_gt'])[:num_gradient.shape[0]]
    gt_traj = gt_poses[:,:2]
    pred_traj = mrob_solution[:num_gradient.shape[0],:2]

    error_0 = error_translational(gt_traj, pred_traj)


    for i in tqdm(range(num_gradient.shape[0])):
        for j in range(num_gradient.shape[1]):
            
            error_name = f"./out/{experiment_id}_mrob_{i}_{j}_error.txt"
            if os.path.exists(error_name):
                num_gradient[i][j] = (np.genfromtxt(error_name).mean() - error_0.mean())/delta
                continue
                

            mrob_solution = mrob_solve_experiment(experiment_id, data, i, j, delta)

            np.savetxt(f"./out/{experiment_id}_mrob_{i}_{j}.txt", mrob_solution)

            fig, ax = plt.subplots(2, 1, figsize=(10, 10))
            vis_dataset_element(data, ax[0])

            gt_poses = np.array(data['poses_gt'])[:num_gradient.shape[0]]
            gt_traj = gt_poses[:,:2]
            pred_traj = mrob_solution[:num_gradient.shape[0],:2]

            error = error_translational(gt_traj, pred_traj)
            np.savetxt(error_name, error)
            num_gradient[i][j] = (error.mean() - error_0.mean())/delta

            ax[0].plot(mrob_solution[:, 0], mrob_solution[:, 1], label='mrob trajectory')
            ax[0].legend()
            ax[0].grid()
            ax[0].axis('equal')

            ax[1].plot(np.unwrap(mrob_solution[:, 2]), label='mrob orientation')
            ax[1].legend()
            ax[1].grid()
            fig.suptitle(experiment_id + f" {i}_{j} iteration\n error = {error.mean()}")

            plt.savefig("./out/" + experiment_id + f"_mrob_{i}_{j}.jpg")
            # plt.show()

            plt.close('all')

    sns.heatmap(num_gradient, annot=True,  fmt=".3f")
    plt.title(f"Gradient of translational error with reference to odometry covariance elements.\n First {K} odometry factors were used.\n delta = {delta}")
    plt.xlabel("Covariance i-th coordinate")
    plt.ylabel("Odometry factor #")
    plt.show()

if __name__ == '__main__':
    # main()
    compute_gradient()
