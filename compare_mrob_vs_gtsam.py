from pathlib import Path

from data_utils import load_dataset

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt

DATASET_PATH = str(Path("./local/datasets/sim/nav2dfix/dataset_0000"))

def error_translational(traj1, traj2):
    t1= traj1[:,:2]
    t2 = traj2[:,:2]

    error = np.linalg.norm(t1-t2,axis=1)

    return error

import gtsam



if __name__ == "__main__":
    datasets = load_dataset(DATASET_PATH, "train")

    for dataset in tqdm(datasets):
        id, _ = dataset

        mrob_solution = np.genfromtxt(f'./out/{id}_mrob.txt')
        print(mrob_solution)
        gtsam_solution = np.genfromtxt(f'./out/{id}_gtsam.txt')
        print(gtsam_solution)

        trans_error = error_translational(mrob_solution, gtsam_solution)

        print(f"Translational error = {trans_error.mean()}")

        plt.plot(mrob_solution[:,0], mrob_solution[:,1], label='mrob')
        plt.plot(gtsam_solution[:,0], gtsam_solution[:,1], label='gtsam')
        plt.title(f"{id}\nerror = {trans_error.mean()}")
        plt.grid()
        plt.axis('equal')
        plt.legend()
        plt.show()

        # plt.savefig("./out/" + id + "_gtsam_vs_mrob.jpg")
        # plt.close('all')



        