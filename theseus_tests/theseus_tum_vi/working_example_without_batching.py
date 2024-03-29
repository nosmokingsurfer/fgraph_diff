import theseus as th
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import List
import numpy as np


def plot_path(optimizer_path, groundtruth_path):
    plt.cla()
    # plt.gca().axis("equal")

    # plt.xlim(-25, 25)
    # plt.ylim(-10, 40)

    batch_idx = 0
    plt.plot(
        optimizer_path,
        linewidth=2,
        linestyle="-",
        color="tab:orange",
        label="optimizer",
    )
    plt.plot(
        groundtruth_path,
        linewidth=2,
        linestyle="-",
        color="tab:green",
        label="groundtruth",
    )
    plt.title(f"mean error {np.mean(np.array(optimizer_path) - np.array(groundtruth_path))}")
    plt.legend()
    plt.grid()
    plt.show()
    plt.pause(1e-12)


class SimpleNN(nn.Module):
    def __init__(self, in_size, out_size, hid_size=30, use_offset=False):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, out_size),
        )

    def forward(self, state_):
        return self.fc(state_)


def run_model(nn_model, current_inputs_):

    # getting predicted accelerations here
    predicted_acc = nn_model(current_inputs_)
    
    # doing manual double integration here
    theseus_inputs_ = {}
    for i in range(N-1):
        theseus_inputs_[f"predicted_odometry_{i}"] = th.SE2(torch.tensor([predicted_acc[i]**2/2 + predicted_acc[i], 0, 0],requires_grad=True).reshape(1,-1))

    return theseus_inputs_


N = 100 # number of trajectory points

# generating data
input_acc = torch.randn(N)
gt_vel = torch.cumsum(input_acc, dim=0)
gt_traj = torch.cumsum(gt_vel, dim=0)

model = SimpleNN(1, 1, hid_size=30)
model.train()
model

poses : List[th.SE2] = []
for i in range(N):
    poses.append(th.SE2(name=f"pose_{i}"))

cost_functions = []
for i in range(N-1):
    # odometry measurmenets will depend on NN output
    pred_acc = model(input_acc[i].reshape(1,))[0]  #<====== here we attach computational graph of NN to all the odometry factors via predicting acceleration
    meas_tensor = th.SE2(torch.tensor([pred_acc**2/2 + pred_acc, 0, 0]).reshape(1,-1), name=f"predicted_odometry_{i}")

    # meas_tensor = th.SE2(torch.tensor([gt_traj[i+1] - gt_traj[i], 0, 0]).reshape(1,-1))
    cost_between = th.ScaleCostWeight(1.0, name=f"scale_between_{i}")
    cost_functions.append(
                th.Between(poses[i], poses[i+1], meas_tensor,
                        cost_between,
                        name=f"between_{i}"))
    
# adding cost fuctors for absolute position
for i in range(N):
    gt_pose_tensor = th.SE2(torch.tensor([gt_traj[i],0,0]).reshape(1,-1), name=f"gt_pose_{i}")
    scale_gps = th.ScaleCostWeight(1.0, name=f"scale_gps_{i}")
    cost_functions.append(th.Difference(poses[i], gt_pose_tensor, scale_gps, name=f"gps_{i}"))

objective = th.Objective()
for cost in cost_functions:
    objective.add(cost)

optimizer = th.GaussNewton(
        objective,
        th.CholeskyDenseSolver,
        max_iterations=5,
        step_size=0.9,
    )

state_estimator = th.TheseusLayer(optimizer)

n_epoch = 100
inner_loop_iters = 1

model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)

losses = []
for epoch in tqdm(range(n_epoch)):
    model_optimizer.zero_grad()

    theseus_inputs = {}
    # predicted_odometry = run_model(model, input_acc[:-1].reshape(-1,1))

    # getting predicted accelerations here
    predicted_acc = model(input_acc[:-1].reshape(-1,1))
    
    # doing manual double integration here
    for i in range(N-1):
        theseus_inputs[f"predicted_odometry_{i}"] = th.SE2(torch.tensor([predicted_acc[i]**2/2 + predicted_acc[i], 0, 0],requires_grad=True).reshape(1,-1))

    # theseus_inputs.update(predicted_odometry)

    # here we update AUX variables (predicted incremental poses updated here)
    objective.update(theseus_inputs)

    theseus_inputs, _ = state_estimator.forward(
        theseus_inputs,
        optimizer_kwargs={
            "track_best_solution": True,
            "verbose" : True
        },
    )

    # here we transform trajectory from optimizator
    optimized_path = torch.empty(1,N,1)
    for i in range(N):
        optimized_path[:, i] = theseus_inputs[f"pose_{i}"][0,0]

    gt_path = torch.empty(1,N,1)
    for i in range(N):
        gt_path[:, i] = gt_traj[i]

    # calculating mse_loss function between trajectories
    mse_loss = F.mse_loss(optimized_path, gt_path)

    loss = torch.mean(mse_loss, dim=0) # <= averaging across the batch
    loss.backward(retain_graph=True) # we need to retain_grad to keep graph for gradietn calculation

    # updating model weights
    model_optimizer.step()

    # saving loss values
    loss_value = loss.item()
    losses.append(loss_value)
        # visualizing the GT and optimized trajectory
    if epoch % 1 == 0:
        pred_path = []

        for i in range(N):
            pred_path.append(optimized_path[0,i,0].detach().item())
        plot_path(pred_path, gt_traj)


print(losses)
    
