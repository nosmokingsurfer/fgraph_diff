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

    return poses


# Example of gtsam built-in visualisation toola
# https://colab.research.google.com/drive/1j2HgxVUDUfQEseLmke4Tk_LKiyJ87QjQ#scrollTo=uwmKsg4OTbTH

# took some code from the gtsam github
# https://github.com/borglab/gtsam/blob/4.1.1/python/gtsam/utils/plot.py
# def plot_trajectory(
#         fignum: int,
#         values: Values,
#         scale: float = 1,
#         marginals: Marginals = None,
#         title: str = "Plot Trajectory",
#         axis_labels: Iterable[str] = ("X axis", "Y axis", "Z axis"),
# ) -> None:
#     """
#     Plot a complete 2D/3D trajectory using poses in `values`.
#     Args:
#         fignum: Integer representing the figure number to use for plotting.
#         values: Values containing some Pose2 and/or Pose3 values.
#         scale: Value to scale the poses by.
#         marginals: Marginalized probability values of the estimation.
#             Used to plot uncertainty bounds.
#         title: The title of the plot.
#         axis_labels (iterable[string]): List of axis labels to set.
#     """
#     fig = plt.figure(fignum)
#     if not fig.axes:
#         axes = fig.add_subplot(projection='3d')
#     else:
#         axes = fig.axes[0]

#     axes.set_xlabel(axis_labels[0])
#     axes.set_ylabel(axis_labels[1])
#     axes.set_zlabel(axis_labels[2])

#     # Plot 2D poses, if any
#     poses = gtsam.utilities.allPose2s(values)
#     for key in poses.keys():
#         pose = poses.atPose2(key)
#         if marginals:
#             covariance = marginals.marginalCovariance(key)
#         else:
#             covariance = None

#         plot_pose2_on_axes(axes,
#                            pose,
#                            covariance=covariance,
#                            axis_length=scale)

#     # Then 3D poses, if any
#     poses = gtsam.utilities.allPose3s(values)
#     for key in poses.keys():
#         pose = poses.atPose3(key)
#         if marginals:
#             covariance = marginals.marginalCovariance(key)
#         else:
#             covariance = None

#         plot_pose3_on_axes(axes, pose, P=covariance, axis_length=scale)

#     fig.suptitle(title)
#     fig.canvas.manager.set_window_title(title.lower())