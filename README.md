# How to run **Manhattan test for MROB vs GTSAM comparison**
1. Run `gtsam_mrob_manhattan_3500.py` - to plot the solution trajectory and get console output

# How to run **LEO DATASET**
1. Download synthetic dataset using `download_local_files.sh`
2. run `gtsam_solver.py` - to get the solutions from gtsam
3. run `mrob_solver.py` - to get the solutions from mrob
4. run `compare_mrob_vs_gtsam.py` - plot trajectories from mrob and gtsam + translational error


# Output
All outputs will be stored in the folder `./out`