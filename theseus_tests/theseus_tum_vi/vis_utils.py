import numpy as np
import open3d

# X here is the list of Nx3 numpy arrays


def vis_xyz(X):
    result = []
    result.append(open3d.geometry.TriangleMesh.create_coordinate_frame())

    for pcd in X:
        result.append(open3d.geometry.PointCloud())
        result[-1].points = open3d.utility.Vector3dVector(pcd)
        result[-1].paint_uniform_color(np.random.uniform(0, 1, 3))

    open3d.visualization.draw_geometries(result)
