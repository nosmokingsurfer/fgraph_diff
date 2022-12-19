import numpy as np
import glob

from pathlib import Path

from utils import dir_utils

# http://www2.informatik.uni-freiburg.de/~stachnis/toro/
# Logfile Format
# A set of simple text messages to represent nodes and edges of the graph. Note that examples files are in the repository, see folder data.

# Format of the 2D graph files:

# Every line in the file specifies either one vertex or one edge

# The vertices are specified as follws: VERTEX2 id x y orientation (A 2D node in the graph)

# EDGE2 observed_vertex_id observing_vertex_id forward sideward rotate inf_ff inf_fs inf_ss inf_rr inf_fr inf_sr (A 2D-edge in the graph. inf_xx are the information matrix entries of the constraint)

# EQUIV id1 id2 (Equivalence constraints between nodes. It merges the node id1 and id2 wrt to the constraint between both vertices.)

# Format of the 3D graph files:

# Every line in the file specifies either one vertex or one edge

# The vertices are specified as follws: VETREX3 x y z phi theta psi

# The edges are specified as follows: EDGE3 observed_vertex_id observing_vertex_id x y z roll pitch yaw inf_11 inf_12 .. inf_16 inf_22 .. inf_66 (the information matrix is specified via its upper triangular block that means 21 values).

def leo_dataset_to_toro(data):
    # reding the vertexes
    # VERTEX2 2 2.043445 -0.060422 -0.026183
    result = "" # list of lines ot save

    gt_poses = data['poses_gt']

    for i in range(len(gt_poses)):
        result += f"VERTEX2 {i} {gt_poses[i][0]} {gt_poses[i][1]} {gt_poses[i][2]}\n"

    # print(result)

    # factor for GNSS
    # EDGE1 id_target x y theta I11   I12  I22  I33  I13  I23

    #factor for odometry
    # EDGE2 id_origin   id_target   dx   dy   dth   I11   I12  I22  I33  I13  I23
    # EDGE2 0 1 1.030390 0.011350 -0.012958 44.721360 0.000000 44.721360 44.721360 0.000000 0.000000
    
    odom_lines = ""
    gps_lines = ""

    for i in range(len(data["factor_keyids"])):
        # print(data['factor_keyids'])
        if data['factor_names'][i] == 'odom':
            origin = data['factor_keyids'][i][0]
            target = data['factor_keyids'][i][1]

            meas = data['factor_meas'][i]
            cov = np.array(data['factor_covs'][i]).reshape(3,3)

            odom_lines += f"EDGE2 {origin} {target} {meas[0]} {meas[1]} {meas[2]} {cov[0,0]} {cov[0,1]} {cov[1,1]} {cov[2,2]} {cov[0,2]} {cov[1,2]}\n"
            
        elif data['factor_names'][i] == 'gps':
            target = data['factor_keyids'][i][0]
            meas = data['factor_meas'][i]
            cov = np.array(data['factor_covs'][i]).reshape(3,3)
            gps_lines += f"EDGE1 {target} {meas[0]} {meas[1]} {meas[2]} {cov[0,0]} {cov[0,1]} {cov[1,1]} {cov[2,2]} {cov[0,2]} {cov[1,2]}\n"
        
    result += odom_lines + gps_lines


    return result

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