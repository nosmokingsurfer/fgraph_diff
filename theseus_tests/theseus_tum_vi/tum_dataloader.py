import torch
torch.set_default_dtype(torch.float32)
import theseus as th
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from vis_utils import vis_xyz
from tqdm import tqdm

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

class TumViDataset(Dataset):
    def __init__(self, data_paths, imu_aug=False, window = 100, step = 50, train=True):
        self.data_paths = data_paths
        self.imu_aug = imu_aug


        self.imu_lengths = []
        self.imu_slices = []
        self.mocap_lenghts = []
        self.mocap_slices = []

        self.window = window # milliseconds
        self.step = step # milliseconds

        for path in self.data_paths:
            imu = pd.read_csv(path / "imu0/data.csv")
            mocap = pd.read_csv(path/"mocap0/data.csv")

            self.imu_lengths.append(len(imu))
            print(self.imu_lengths)

            self.mocap_lenghts.append(len(mocap))
            print(self.mocap_lenghts)

            # prepare imu slices indexes
            imu_slices = [(i, i+self.window) for i in range(0, len(imu) - self.window, self.step)]
            self.imu_slices.append((path,imu_slices))

    def __len__(self):

        return len(self.data_paths)

    def __getitem__(self, idx):
        #getting the IMU data
        path, sls = self.imu_slices[idx]
        imu = pd.read_csv(path/'imu0/data.csv')

        # getting mocap data
        mocap = pd.read_csv(path/'mocap0/data.csv')

        # vis_xyz([mocap.values[:,1:4]])


        X = []
        y = []
        slice_paths = []
        for sl in tqdm(sls):

            tmp_X = imu.iloc[sl[0]:sl[1]]

            #find closes timestamps in mocap that correspond to IMU slice first and last timestamp
            idx_left = find_nearest(mocap['#timestamp [ns]'].values, tmp_X.values[0,0])
            idx_right = find_nearest(mocap['#timestamp [ns]'].values, tmp_X.values[-1,0])

            # cropping mocap data
            tmp_y = mocap.iloc[idx_left: idx_right]

            # vis_xyz([tmp_y.values[:,1:4]])

            starting_pose = th.geometry.se3.SE3(torch.tensor(tmp_y.values[0,1:].astype(np.float32))).translation().tensor
            slice_paths.append(th.geometry.se3.SE3(torch.tensor(tmp_y.values[:,1:].astype(np.float32))).translation().tensor)

            finish_pose = slice_paths[-1][-1]

            tmp_y = finish_pose 

            tmp_X = torch.tensor(tmp_X.values[:,1:].astype(np.float32))
            

            if self.imu_aug : 
                R = th.rand_so3(1).tensor

                tmp_X[:,:3] = tmp_X[:,:3] @ R
                tmp_X[:,3:] = tmp_X[:,3:] @ R

                tmp_y = tmp_y @R

                tmp_y = tmp_y.squeeze(0)

            X.append(tmp_X)

            # first_abs_pose = th.geometry.se3.SE3(torch.tensor(tmp_y.values[0,1:],dtype=torch.float32))
            # last_abs_pose = th.geometry.se3.SE3(torch.tensor(tmp_y.values[-1,1:],dtype=torch.float32))
            # tmp_y = last_abs_pose.between(first_abs_pose).translation().tensor


            y.append(tmp_y)

            # using only gyro and acc data
        X = torch.tensor(torch.vstack(X)).reshape(len(sls),self.window,-1)

        y = torch.tensor(torch.vstack(y))

        # calculating incremental pose between first and last abs poses of the slice
   
        return X, y, slice_paths
