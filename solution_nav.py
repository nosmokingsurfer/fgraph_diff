from vis_utils import vis_dataset_element, vec3_to_pose
from multiprocessing import Pool
from attrdict import AttrDict
from observation_models import ThetaNav2dFixedCov
from tqdm import tqdm
import numpy as np
import os
import glob
import mrob
from utils import dir_utils
from data_utils import leo_dataset_to_toro


from pathlib import Path
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt


DATASET_PATH = str(Path("../leo/local/datasets/sim/nav2dfix/dataset_0000"))



def load_dataset(dataset_path, dataset_mode="train"):

    files = glob.glob(f"{dataset_path}/{dataset_mode}/*.json")
    files.sort()

    datasets = []
    for f in files:
        print(f"Loading data from {f}")

        tmp = dir_utils.read_file_json(f, verbose=False)
        key = f"{Path(f).parts[-3]}_{Path(f).parts[-2]}_{Path(f).stem}"

        datasets.append((key, tmp))

    return datasets

# def process_experiment_gtsam(dataset, L=0):
#     id, experiment = dataset
#     experiment = AttrDict(experiment)
#     graph = gt

def process_experiment_mrob(dataset, L=0):

    if not os.path.exists("./out"):
        os.makedirs("./out", exist_ok=True)

    id, experiment = dataset
    experiment = AttrDict(experiment)
    print(id)

    # checking if cached unrolled data already exists
    unrolled_data_path = f"./out/{id}_unrolled_data.txt"
    if not os.path.exists(unrolled_data_path):

        unrolled_data = leo_dataset_to_toro(experiment)
        with open(unrolled_data_path,'w') as f:
            f.writelines(unrolled_data)
            f.close()

    # number of iterations for current experiment
    n_steps = len(experiment.poses_gt)
    # n_steps = 10

    # Initialize data structures
    vertex_ini = {}
    factors = {}
    factors_dictionary = {}

    # load file
    leo_dataset_file = f"./out/{id}_unrolled_data.txt"
    # with open('../benchmarks/M3500.txt', 'r') as file:
    with open(leo_dataset_file, 'r') as file:

        for line in file:
            d = line.split()
            # read edges and vertex, in TORO format
            if d[0] == 'EDGE2':
                # EDGE2 id_origin   id_target   dx   dy   dth   I11   I12  I22  I33  I13  I23
                factors[int(d[1]),int(d[2])] = np.array([d[3], d[4], d[5], d[6],d[7],d[8],d[9],d[10],d[11]],dtype='float64')
                factors_dictionary[int(d[2])].append(int(d[1]))
            elif d[0] == 'VERTEX2':
                # VERTEX2 id x y theta
                # these are the initial guesses for node states
                vertex_ini[int(d[1])] = np.array([d[2], d[3], d[4]],dtype='float64')
                # create an empty list of pairs of nodes (factor) connected to each node
                factors_dictionary[int(d[1])] = []
            
            elif d[0] == 'EDGE1':
                factors[int(d[1]), int(d[1])] = np.array([d[2],d[3], d[4], d[5], d[6],d[7],d[8],d[9],d[10]], dtype='float64')
                factors_dictionary[int(d[1])].append(int(d[1]))

    graph_matrix = np.zeros((n_steps,n_steps))

   # Initialize FG
    graph = mrob.FGraph()
    x = np.zeros(3)
    n = graph.add_node_pose_2d(x)
    print('node 0 id = ', n) # id starts at 1
    processing_time = []

    # start events, we solve for each node, adding it and it corresponding factors
    # in total takes 0.3s to read all datastructure
    for t in range(1,n_steps):
        x = vertex_ini[t]
        n = graph.add_node_pose_2d(x)
        assert t == n, 'index on node is different from counter'

        # find factors to add. there must be 1 odom and other observations
        connecting_nodes = factors_dictionary[n]

        for nodeOrigin in factors_dictionary[n]:
            # inputs: obs, idOrigin, idTarget, invCov
            obs = factors[nodeOrigin, t][:3]
            covInv = np.zeros((3,3))
            # on M3500 always diagonal information matrices
            covInv[0,0] = factors[nodeOrigin, t][3]
            covInv[1,1] = factors[nodeOrigin, t][5]
            covInv[2,2] = factors[nodeOrigin, t][6]

            if nodeOrigin != n:
                graph.add_factor_2poses_2d_odom(obs, nodeOrigin,t,covInv)
            elif nodeOrigin == n:
                graph.add_factor_1pose_2d(obs, nodeOrigin,covInv)
            # for end. no more loop inside the factors
    

    print('current initial chi2 = ', graph.chi2() )
    graph.solve(mrob.LM, 50)
    print('solution drawn')
    # graph.print(True)

    import networkx as nx

    G = nx.from_numpy_matrix(graph_matrix)
    nx.draw(G)

    res = graph.get_estimated_state()
    res = np.array(res).squeeze()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    vis_dataset_element(dataset, ax)
    ax.plot(res[:, 0], res[:, 1], label='estimated trajectory')
    # meas = np.array(gnss_measurements).squeeze()

    # ax.plot(meas[:, 0], meas[:, 1], label='measured')
    ax.legend()
    # plt.show()


    plt.savefig("./out/" + Path(dataset[0]).stem + ".jpg")

    plt.close('all')

    pass


def main():

    # reading the datasets
    datasets = load_dataset(DATASET_PATH, 'train')

    # process_experiment(datasets[0])

    with Pool(8) as p:
        res = [p.apply_async(process_experiment_mrob, args=([dataset,i]))
               for i, dataset in enumerate(datasets)]

        for r in res:
            r.get()


if __name__ == '__main__':
    main()
