import mrob
import numpy as np
import matplotlib.pyplot as plt


def vec3_to_pose(vec):
    """
    transforming 3d vector (x,y,theta) to pose object wihtout exponential mapping
    """
    R = np.array([[np.cos(vec[2]), -np.sin(vec[2]), 0],
                  [np.sin(vec[2]), np.cos(vec[2]), 0],
                  [0, 0, 1]])
    t = np.array([vec[0], vec[1], 0]).reshape(3, 1)
    so3 = mrob.geometry.SO3(R)
    p = mrob.geometry.SE3(so3, t)

    return p


def vis_pose(pose, ax):
    """
    visualize single pose on a give axis handle
    """
    xyz = pose.t()
    ax.plot(xyz[0], xyz[1], marker='o', linewidth=2, color='black')
    angles = pose.Ln()[:3]

    axis_length = 2
    ax.arrow(xyz[0], xyz[1], axis_length*np.cos(angles[2]),
             axis_length*np.sin(angles[2]), color='red')
    ax.arrow(xyz[0], xyz[1], -axis_length*np.sin(angles[2]),
             axis_length*np.cos(angles[2]), color='green')


def vis_dataset_element(data,ax):
    """
    visualize trajectory defined by (x,y,theta) time series
    """
    traj = np.array(data['poses_gt'])
    # fig, ax = plt.subplots(1, 1)
    poses = []
    for i in range(len(traj)):
        p = vec3_to_pose(traj[i])
        if i % 10 == 0:
            vis_pose(p, ax)
            ax.text(traj[i, 0], traj[i, 1], s=str(i))

    ax.plot(traj[:, 0], traj[:, 1], linestyle='--',label='gt trajectory')

    ax.grid()
    ax.axis('equal')

    return poses
