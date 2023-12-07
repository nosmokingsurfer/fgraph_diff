import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

from spline_diff import generate_imu_data
from spline_generation import bspline
import glob
from tqdm import tqdm

# class that reads generated IMU data for spline curves on a 2d plane
class Spline_2D_Dataset(Dataset):
    def __init__(self, spline_path, window = 100):

        self.window = window

        self.bias_acc = np.array([0,0])
        self.Q_acc = np.array([0.05**2,0,0,0.05**2]).reshape(2,2)

        self.bias_w = np.array([0])
        self.Q_w = np.array([0.03**2]).reshape(1,1)

        paths = glob.glob(f'{spline_path}spline_*.txt')
        print(f"Found {len(paths)} splines in path: {spline_path}")

        B = len(paths) # this is our batch size

        self.data = np.zeros((B,window,3))#np.concatenate((acc,gyro),axis=1)

        self.slices = [[] for _ in range(B)]
        self.gt_odometry = [[] for _ in range(B)]
        self.gt_traj = [[] for _ in range(B)]
        self.gt_poses = [[] for _ in range(B)]
        self.gt_velocity = [[] for _ in range(B)]

        for b in tqdm(range(len(paths))):
            # 1 sample == 1 spline

            spline_points = np.genfromtxt(paths[b])

            self.gt_traj[b] = spline_points

            acc, gyro, tau, n, velocity = generate_imu_data(spline_points)
            # TODO inject noise:
            # -additive
            # -multiplicative

            tmp_data = np.concatenate((acc,gyro),axis=1)
            
            # splitting 1 track into several slices
            slice_num = tmp_data.shape[0] // self.window
            for i in range(slice_num):

                temp_slice = tmp_data[i*self.window : (i+1)*self.window]

                acc_noise = np.random.multivariate_normal(self.bias_acc,self.Q_acc,self.window)
                temp_slice[:,:2] = temp_slice[:,:2] + acc_noise

                omega_noise = np.random.multivariate_normal(self.bias_w, self.Q_w,self.window)
                temp_slice[:,2:] = temp_slice[:,2:] + omega_noise

                self.slices[b].append(temp_slice) # adding i-th slice into b-th track

                # need to rotate points using first tau vector orientation
                c = tau[i*self.window][0]
                s = tau[i*self.window][1]

                R = np.array([[c,-s],[s,c]])

                tmp = spline_points[i*self.window : (i + 1)*self.window]@R

                tmp = tmp - tmp[0]

                # calculating orientation increment

                tmp = np.concatenate((tmp,tau[i*self.window : (i + 1)*self.window]@R),axis=1)

                self.gt_odometry[b].append(tmp)

            self.gt_poses[b] = np.concatenate((self.gt_traj[b][:-1], tau),axis=1)[::window]
            self.gt_velocity[b] = velocity[::window]

        self.X = torch.tensor(np.array(self.slices),dtype=torch.float32)
        self.y = torch.tensor(np.array(self.gt_odometry),dtype=torch.float32)
        self.gt_traj = torch.tensor(np.array(self.gt_traj),dtype=torch.float32)
        self.gt_poses = torch.tensor(np.array(self.gt_poses),dtype=torch.float32)
        self.gt_velocity = torch.tensor(np.array(self.gt_velocity),dtype=torch.float32)
        


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self,idx):
        return self.X[idx], self.y[idx], self.gt_traj[idx], self.gt_poses[idx], self.gt_velocity[idx]


if __name__ == "__main__":
    dataset = Spline_2D_Dataset()

    print(f"{dataset.__len__()=}")
    print(f"{dataset.__getitem__(0)=}")

    for i in range(len(dataset)):
        x,y = dataset.__getitem__(i)

        plt.plot(x[:,0],label='acc_x')
        plt.plot(x[:,1],label='acc_y')
        plt.plot(x[:,2],label='omega_z')
        plt.grid()
        plt.legend()


        plt.figure()
        plt.plot(y[:,0],y[:,1],label='gt traj')
        plt.grid()
        plt.axis('equal')
        plt.legend()

        plt.show()

    N = len(dataset)

    for i in range(N):
        print(dataset.__getitem__(i))