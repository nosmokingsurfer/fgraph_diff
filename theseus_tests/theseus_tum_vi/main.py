import theseus as th
import torch.nn as nn
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import matplotlib.pyplot as plt

import glob
import numpy as np

from typing import List


def get_weights_dict_from_model(
    mode_, my_nn_model_, values_, path_length_, print_stuff=False
):
    weights_dict = {}
    all_states_ = []
    # will compute weight for all cost weights in the path
    for i in range(path_length_):
        all_states_.append(values_[f"pose_{i}"])
    model_input_ = torch.cat(all_states_, dim=0)

    weights_ = my_nn_model_(model_input_)
    for i in range(path_length_):
        weights_dict[f"scale_between_{i}"] = weights_[i - 1, 1].view(1, 1)

    if print_stuff:
        with torch.no_grad():
            print("scale5", weights_dict["scale_gps_5"].item())
            print("scale45", weights_dict["scale_gps_45"].item())
            print("btwn5", weights_dict["scale_between_5"].item())
            print("btwn45", weights_dict["scale_between_45"].item())

    return weights_dict


# def run_model(
#     mode_,
#     my_nn_model_,
#     current_inputs_,
#     path_length_,
#     print_stuff=False,
# ):
#     # weights_ = get_weights_dict_from_model(
#     #     mode_,
#     #     my_nn_model_,
#     #     current_inputs_,
#     #     path_length_,
#     #     print_stuff=print_stuff,
#     # )
#     theseus_inputs_ = {}
#     theseus_inputs_.update(current_inputs_)
#     theseus_inputs_.update(weights_)

#     return theseus_inputs_



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


class SimpleNN(nn.Module):
    def __init__(self, in_size, out_size, hid_size=300, use_offset=False):
        super().__init__()
        # self.conv = nn
        self.fc = nn.Sequential(
            nn.Linear(in_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, out_size),
        )

    def forward(self, X):
        inputs = X

        if len(inputs.shape) == 3:
            N, _, _ = inputs.shape
            inputs = inputs.reshape(N, -1)
        else:
            inputs = inputs.flatten()

        return self.fc(inputs)



class TumViDataset(Dataset):
    def __init__(self, data_paths, imu_aug=False, train=True):
        self.data_paths = data_paths


        self.imu_lengths = []
        self.imu_slices = []
        self.mocap_lenghts = []
        self.mocap_slices = []

        self.window = 1000 # milliseconds
        self.step = 100 # milliseconds

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

        X = []
        y = []
        for sl in sls:

            tmp_X = imu.iloc[sl[0]:sl[1]]

            #find closes timestamps in mocap that correspond to IMU slice first and last timestamp
            idx_left = find_nearest(mocap['#timestamp [ns]'].values, tmp_X.values[0,0])
            idx_right = find_nearest(mocap['#timestamp [ns]'].values, tmp_X.values[-1,0])

            # cropping mocap data
            tmp_y = mocap.iloc[idx_left: idx_right]

            X.append(torch.tensor(tmp_X.values[:,1:],dtype=torch.float32))

            # first_abs_pose = th.geometry.se3.SE3(torch.tensor(tmp_y.values[0,1:],dtype=torch.float32))
            # last_abs_pose = th.geometry.se3.SE3(torch.tensor(tmp_y.values[-1,1:],dtype=torch.float32))
            # tmp_y = last_abs_pose.between(first_abs_pose).translation().tensor

            tmp_y = th.geometry.se3.SE3(torch.tensor(tmp_y.values[-1,1:],dtype=torch.float32)).translation().tensor

            y.append(tmp_y)

            # using only gyro and acc data
        X = torch.tensor(torch.vstack(X),dtype=torch.float32).reshape(len(sls),self.window,-1)

        y = torch.tensor(torch.vstack(y),dtype=torch.float32)

        # calculating incremental pose between first and last abs poses of the slice
   

        return X, y


def get_path_from_values(batch_size_, values_, path_length_):
    path = torch.empty(batch_size_, path_length_, 3, device='cpu')
    for i in range(path_length_):
        path[:, i, :3] = values_[f"pose_{i}"]
    return path

def get_initial_inputs(path_gt):
    inputs_ = {}
    for i, _ in enumerate(path_gt):
        inputs_[f"pose_{i}"] = (path_gt[i] + torch.randn(1)).unsqueeze(0)
    return inputs_


dataset_folder = "../../master_diploma/data/datasets/tum-vi//dataset-calib-imu1_512_16/mav0"

losses = []
best_loss = 100


if __name__ == "__main__":
    my_nn_model = SimpleNN(1000*6, 3, hid_size=100, use_offset=False).to('cpu')
    model_optimizer = torch.optim.Adam(my_nn_model.parameters(),lr=1e-3)

    data = TumViDataset([Path(dataset_folder)])

    measurements, path_gt = data.__getitem__(0)

    # print(my_nn_model(measurements[0]))

    # having measurments (N,window,6) and path_gt (N,3) here

    N, window, chanels = measurements.shape

    # between_cost_weights = []

    # # initializing the scalar weights for the odometry
    # for i in range(N-1):
    #     between_cost_weights.append()

    # initializing the poses for reconstruted trajectory
    poses = [] # optim vars
    for i in range(N):
        poses.append(th.Point3(name=f"pose_{i}"))

    # cost functions for every pose (basically errors)
    cost_functions: List[th.CostFunction] = []

    meas = [] # aux_vars
    for i in range(1,len(measurements)):
        meas.append(th.Point3(my_nn_model(measurements[i-1]),name = f"meas_{i}"))


    for i in range(1,len(measurements)):
        
        cost_functions.append(
            th.Between(
                poses[i-1],
                poses[i],
                meas[i-1],
                th.ScaleCostWeight(th.Variable(torch.ones(1,1),name=f"scale_between_{i}")),
                name=f"between_{i-1}"))

    objective = th.Objective()
    for cost_f in cost_functions:
        objective.add(cost_f)

    print("AUX vars")
    for k in objective.aux_vars.keys():
        print(f"{k}")

    print("Optim vars")
    for k in objective.optim_vars.keys():
        print(f"{k}")


    optimizer = th.LevenbergMarquardt(
        objective,
        th.CholeskyDenseSolver,
        max_iterations=40,
        step_size=0.3
    )

    state_estimator = th.TheseusLayer(optimizer)
    # state_estimator.to('cpu')

    for epoch in range(50):
        model_optimizer.zero_grad()

        theseus_inputs = get_initial_inputs(path_gt)

        tmp = my_nn_model(measurements)

        # tmp = [my_nn_model(sl).unsqueeze(0) for sl in measurements]

        for i in range(len(path_gt)-1):
            theseus_inputs[f"meas_{i+1}"] = tmp[i].reshape((1,-1))
        # theseus_inputs = run_model("not constant",
        #                            my_nn_model,
        #                            theseus_inputs,
        #                            len(path_gt),
        #                            print_stuff=epoch %10 ==0 and i == 0)
        
        objective.update(theseus_inputs)

        with torch.no_grad():
            if epoch % 10 == 0:
                print("Initial error:", objective.error_metric().mean().item())

            # for k,v in theseus_inputs.items():
            #     print(f"{k} : {v}")

        theseus_inputs, info = state_estimator.forward(theseus_inputs,optimizer_kwargs={'track_best_solution':True, 'verbose':True})
        print(info.best_solution)

        
        optimizer_path = get_path_from_values(objective.batch_size, theseus_inputs, N)

        mse_loss = F.mse_loss(optimizer_path, path_gt)


        loss = mse_loss

        loss = torch.mean(loss,dim=0)
        loss.backward()
        model_optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)

        if loss_value < best_loss:
            best_loss = loss_value
            best_solution = optimizer_path.detach()

        if epoch % 10 == 0:
            # if vis_flag:
            #     plot_path(
            #         optimizer_path.detach().cpu().numpy(),
            #         groundtruth_path.detach().cpu().numpy(),
            #     )
            print("Loss: ", loss.item())
            print("MSE error: ", mse_loss.item())
            print(f" ---------------- END EPOCH {epoch} -------------- ")

    
    # plt.plot(losses, label='loss')
    plt.semilogy(losses[1:],label='loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid()
    plt.show()

    
    # X, y_gt = data.__getitem__(0)

    # y_pred = model(X.flatten())

    # print(y_pred - y_gt)

    # print(y_pred)

    # objective = th.Objective()





