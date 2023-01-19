import os
import numpy as np

import sys
sys.path.insert(0, "/usr/local/cython/")
import gtsam
import mrob

import matplotlib.pyplot as plt

from mrob_solver import toro_to_mrob
from compare_mrob_vs_gtsam import error_translational

import time

# creating the output path
if not os.path.exists("./out"):
    os.makedirs("./out", exist_ok=True)

leo_dataset_file = f"./benchmarks/M3500.txt"

#### GTSAM ####
# reading the M3500 data in TORO format with built-in reader
# graph - initialized graph
# initial - initial state of the system
[graph, initial] = gtsam.load2D(leo_dataset_file)

# do graph optimization

start = time.time()
result = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()
end = time.time()
gtsam_time = 1e0*(end - start)


# calculating chi2 according the following code in gtsam repo
# https://github.com/devbharat/gtsam/blob/5f15d264ed639cb0add335e2b089086141127fff/examples/SolverComparer.cpp#L92
dof = graph.size() - result.size()
gtsam_chi2 = 2*graph.error(result)/dof

# saving poses and plotting trajectory for the gtsam
poses = np.array([result.atPose2(key) for key in range(0, result.size())])

gtsam_solution = np.zeros((len(poses), 3))

for i in range(len(poses)):
    gtsam_solution[i][0] = poses[i].x()
    gtsam_solution[i][1] = poses[i].y()
    gtsam_solution[i][2] = poses[i].theta()

# saving (x,y,theta) in table
np.savetxt(f"./out/M3500_gtsam.txt", gtsam_solution)

#### MROB ###

# initializing mrob graph from TORO file M3500.txt
graph = toro_to_mrob(leo_dataset_file)

# print(f'initial chi2 = {graph.chi2()}')

start = time.time()
graph.solve(mrob.LM, 50)
end = time.time()
mrob_time = 1e0*(end - start)

mrob_solution = graph.get_estimated_state()
mrob_solution = np.array(mrob_solution).squeeze()

np.savetxt(f"./out/M3500_mrob.txt", mrob_solution)

# plotting obtained solutions for MROB and GTSAM
print("-"*80)
print(f"gtsam_chi2 = {gtsam_chi2}")
print(f'mrob_chi2 = {graph.chi2()}')
print(f'gtsam time[ms] = {gtsam_time*1e+3}')
print(f'mrob time[ms] = {mrob_time*1e+3}')

trans_error = error_translational(mrob_solution, gtsam_solution)

fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax[0].plot(gtsam_solution[:, 0], gtsam_solution[:, 1], label='gtsam trajectory')
ax[0].plot(mrob_solution[:,0], mrob_solution[:,1], label='mrob trajectory')
ax[0].legend()
ax[0].grid()
ax[0].axis('equal')

ax[1].plot(np.unwrap(gtsam_solution[:, 2]), label='theta angle from gtsam')
ax[1].plot(np.unwrap(gtsam_solution[:,2]), label='theta angle from mrob')
ax[1].legend()
ax[1].grid()
fig.suptitle(f"GTSAM vs MROB \nerror = {trans_error.mean()}")
plt.show()



