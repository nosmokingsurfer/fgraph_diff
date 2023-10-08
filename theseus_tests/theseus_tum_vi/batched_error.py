import theseus as th
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
from tqdm import tqdm
from typing import List

class SimpleNN(nn.Module):
    def __init__(self, in_size, out_size, hid_size=30, use_offset=False):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_size, hid_size, bias=False),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size, bias=False),
            nn.ReLU(),
            nn.Linear(hid_size, out_size, bias=False),
        )

    def forward(self, state_):
        return self.fc(state_)

# Generating data    
B = 16 # number of trajectories in batch
N = 10 # number of trajectory points

# generating random acceleration dt = 1
input_acc = torch.randn((B,N))
gt_vel = torch.cumsum(input_acc, dim=-1)
gt_traj = torch.cumsum(gt_vel, dim=-1)

model = SimpleNN(1, 1, hid_size=30)
model.train()
model

poses : List[th.SE2] = []
for i in range(N):
    poses.append(th.SE2(torch.zeros(B,3),name=f"pose_{i}"))

cost_functions = []
for i in range(N-1):
    # odometry measurmenets will depend on NN output
    meas_tensor = th.SE2(torch.zeros((B,3)), name=f"predicted_odometry_{i}")
    cost_between = th.ScaleCostWeight(torch.ones((B,1)), name=f"scale_between_{i}")
    cost_functions.append(
                th.Between(poses[i], poses[i+1], meas_tensor,
                        cost_between,
                        name=f"between_{i}"))

for i in range(N):
    gt_pose_tensor = th.SE2(F.pad(gt_traj[:,i].view(B,-1), pad=(0,2)), name=f"gt_pose_{i}")
    scale_gps = th.ScaleCostWeight(torch.ones((B,1)), name=f"scale_gps_{i}")
    cost_functions.append(th.Difference(poses[i], gt_pose_tensor, scale_gps, name=f"gps_{i}"))

objective = th.Objective()
for cost in cost_functions:
    objective.add(cost)

optimizer = th.GaussNewton(
        objective,
        th.CholeskyDenseSolver,
        max_iterations=10,
        step_size=0.1,
    )

state_estimator = th.TheseusLayer(optimizer)
model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)

losses = []
for epoch in tqdm(range(20)):
    model_optimizer.zero_grad()

    predicted_acc = model(input_acc.view(-1,1)).view(B,-1)

    theseus_inputs = {}
    for i in range(N-1):
        tmp = torch.zeros(B,4)
        tmp[:,2] =  1.0
        tmp[:,0] = 0.5*predicted_acc[:,i]**2 + predicted_acc[:,i]
        theseus_inputs[f"predicted_odometry_{i}"] = tmp

    objective.update(theseus_inputs)
    print(f"Objective error = {objective.error_metric().mean().item()}")

    theseus_output, _ = state_estimator.forward(
            theseus_inputs,
            optimizer_kwargs={
                "track_best_solution": True,
                "verbose": epoch % 1 == 0,
            },
        )

    optimized_path = torch.zeros((B,N))
    for i in range(N):
        optimized_path[:, i] = theseus_output[f"pose_{i}"][:,0]

    mse_loss = F.mse_loss(optimized_path, gt_traj,reduction='none')
    loss = mse_loss.mean(axis=1).mean()

    loss.backward()

    model_optimizer.step()

    losses.append(loss.item())

print(losses)
