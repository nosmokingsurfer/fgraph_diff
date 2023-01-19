import sys
sys.path.insert(0,"/usr/local/cython/")
import gtsam
import numpy as np

from pathlib import Path

from data_utils import load_dataset
import os
from utils import tf_utils
from tqdm import tqdm

import matplotlib.pyplot as plt


class GTSAM_graph():
    def __init__(self):
        # initizalizing the optimzator
        self.optimizer = self.init_isam2()
        self.graph = gtsam.NonlinearFactorGraph()
        self.init_vals = gtsam.Values()
        self.est_vals = gtsam.Values()

        self.sigma_inv_odom = np.array([1., 1., 1.])

        self.sigma_inv_gps = np.array([1., 1., 1.])

    def get_sigma_inv(self, factor_name):
        sigma_inv_val = getattr(self, "sigma_inv_{0}".format(factor_name))

        return sigma_inv_val


    def reset_graph(self):
        self.graph.resize(0)
        self.init_vals.clear()

    def optimizer_update(self):
        self.optimizer.update(self.graph, self.init_vals)
        self.est_vals = self.optimizer.calculateEstimate()

    def add_unary_factor(self, keys, factor_cov, factor_meas):

        factor_noise_model = tf_utils.get_noise_model(factor_cov)
        factor_meas_pose = tf_utils.vec3_to_pose2(factor_meas)
        factor = gtsam.PriorFactorPose2(keys[0], factor_meas_pose, factor_noise_model)

        self.graph.push_back(factor)


    def init_vars_step(self, tstep):
    
        key_tm1 = gtsam.symbol(ord('x'), tstep-1)
        key_t = gtsam.symbol(ord('x'), tstep)
        self.init_vals.insert(key_t, self.est_vals.atPose2(key_tm1))

    def add_first_pose_priors(self, data):

        prior_cov = np.array([1e-6, 1e-6, 1e-6])

    
        keys = [gtsam.symbol(ord('x'), 0)]
        self.init_vals.insert(keys[0], tf_utils.vec3_to_pose2(data['poses_gt'][0]))
        self.add_unary_factor(keys, prior_cov, data['poses_gt'][0])
    

        # self.optimizer_update()
        # self.reset_graph()

    def init_isam2(self):
        params_isam2 = gtsam.ISAM2Params()
        params_isam2.setRelinearizeThreshold(0.01)
        params_isam2.setRelinearizeSkip(10)

        return gtsam.ISAM2(params_isam2)


    def unary_factor_error(x, key_syms, key_ids, factor_inf, factor_meas, device=None, params=None):

        key_id = key_ids[0]
        key_id = key_id + int(0.5 * x.shape[0]) if (key_syms[0] == 'e') else key_id

        est_pose = (x[key_id, :]).view(1, -1)
        meas_pose = factor_meas.view(1, -1)

        err = tf_utils.tf2d_between(est_pose, meas_pose, device=device, requires_grad=True).view(-1, 1) # 3 x 1

        return err

    def binary_odom_factor_error(x, key_syms, key_ids, factor_inf, factor_meas, device=None, params=None):
        
        p1 = (x[key_ids[0], :]).view(1, -1) # n x 3
        p2 = (x[key_ids[1], :]).view(1, -1) # n x 3

        est_val = (tf_utils.tf2d_between(p1, p2, device=device, requires_grad=True)).view(-1)

        # err = (torch.sub(est_val, factor_meas)).view(-1, 1) # 3 x 1
        est_val = est_val.view(1, -1)
        factor_meas = factor_meas.view(1, -1)
        err = (tf_utils.tf2d_between(factor_meas, est_val, device=device, requires_grad=True)).view(-1, 1)

        return err


def get_optimizer_soln(num_poses, solver):

    poses_graph = solver.optimizer.calculateEstimate()
    pose_vec_graph = np.zeros((poses_graph.size(), 3))    

    num_poses
    keys = [[gtsam.symbol(ord('x'), i)] for i in range(0, num_poses)]

    key_vec = gtsam.gtsam.KeyVector()
    for key_idx in range(0, len(keys[0])):
        for pose_idx in range(0, num_poses):
            key = keys[pose_idx][key_idx]
            pose2d = poses_graph.atPose2(key)
            pose_vec_graph[key_idx * num_poses + pose_idx, :] = [pose2d.x(), pose2d.y(), pose2d.theta()]

            key_vec.push_back(key)

    mean = pose_vec_graph
    
    return mean


def gtsam_solve_experiment(data):
    solver_gtsam = GTSAM_graph()

    solver_gtsam.add_first_pose_priors(data)

    nsteps = len(data['poses_gt'])

    for tstep in range(1, nsteps):

        # filter out curr step factors
        factor_idxs = [idxs for idxs, keys in enumerate(data['factor_keyids']) if (max(keys) == tstep)]

        factor_keysyms = [data['factor_keysyms'][idx] for idx in factor_idxs]
        factor_keyids = [data['factor_keyids'][idx] for idx in factor_idxs]
        factor_meas = [data['factor_meas'][idx] for idx in factor_idxs]
        factor_names = [data['factor_names'][idx] for idx in factor_idxs]


        # compute factor costs
        for idx in range(0, len(factor_idxs)):

            key_syms, key_ids = factor_keysyms[idx], factor_keyids[idx]

            keys = [gtsam.symbol(ord(key_syms[i]), key_id)
                    for i, key_id in enumerate(key_ids)]
            factor_name = factor_names[idx]

            sigma_inv_val = solver_gtsam.get_sigma_inv(factor_name)
            factor_cov = np.reciprocal(np.sqrt(sigma_inv_val))

            if (factor_name == 'gps'):
                factor_noise_model = tf_utils.get_noise_model(factor_cov)
                factor_meas_pose_gps = tf_utils.vec3_to_pose2(factor_meas[idx][:3])
                factor = gtsam.PriorFactorPose2(keys[0], factor_meas_pose_gps, factor_noise_model)
                solver_gtsam.graph.push_back(factor)

                # solver_gtsam.graph = tf_utils.add_unary_factor(solver_gtsam.graph, keys, factor_cov, factor_meas[idx][0:3])
            elif (factor_name == 'odom'):
                factor_noise_model = tf_utils.get_noise_model(factor_cov)
                factor_meas_pose = tf_utils.vec3_to_pose2(factor_meas[idx][:3])
                factor = gtsam.BetweenFactorPose2(keys[0], keys[1], factor_meas_pose, factor_noise_model)
                solver_gtsam.graph.push_back(factor)
                # solver_gtsam.graph = tf_utils.add_binary_odom_factor(solver_gtsam.graph, keys, factor_cov, factor_meas[idx][0:3])


        # solver_gtsam.init_vars_step(tstep)

        key_tm1 = gtsam.symbol(ord('x'), tstep-1)
        key_t = gtsam.symbol(ord('x'), tstep)
        solver_gtsam.init_vals.insert(key_t, solver_gtsam.init_vals.atPose2(key_tm1))

        # optimize
    solver_gtsam.optimizer_update()

        # print(solver_gtsam.est_vals)

        # reset graph
        # solver_gtsam.reset_graph()

    gtsam_solution = get_optimizer_soln(nsteps, solver_gtsam)

    return gtsam_solution

# conducting small test with bare gtsam
if __name__ == "__main__":

    DATASET_PATH = str(Path("./local/datasets/sim/nav2dfix/dataset_0000"))

    if not os.path.exists("./out"):
        os.makedirs("./out", exist_ok=True)

    datasets = load_dataset(DATASET_PATH, 'train')

    for i in tqdm(range(len(datasets))):
        experiment_id = datasets[i][0]
        data = datasets[i][1]

        gtsam_solution = gtsam_solve_experiment(data)

        np.savetxt(f"./out/{experiment_id}_gtsam.txt",gtsam_solution)

        plt.plot(gtsam_solution[:,0], gtsam_solution[:,1], label=f"gtsam (isam2) solution")
        plt.grid()
        plt.axis('equal')
        plt.legend()
        plt.title(experiment_id)
        # plt.show()

        plt.savefig("./out/" + experiment_id + "_gtsam.jpg")

        plt.close('all')
            