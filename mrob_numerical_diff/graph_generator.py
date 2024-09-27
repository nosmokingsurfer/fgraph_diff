import mrob
import numpy as np
np.set_printoptions(precision=2,linewidth=160)


import matplotlib.pyplot as plt
from num_diff import read_graph_toro_description, compose_graph, numerical_diff1, numerical_diff2, visualize_gradient

class ToRoContainer():
    def __init__(self):
        self.toro_lines = ""

    def add_node(self, n , x):
        # VERTEX2 408 -16.166323 -21.968629 -1.454484
        self.toro_lines +=f"VERTEX2 {n} {x[0]:.6f} {x[1]:.6f} {x[2]:.6f}\n"

    def add_factor_1pose_2d(self, origin, meas, info):
        # EDGE2 1090 0.984655 0.062479 0.023806 44.721360 0.000000 44.721360 44.721360 0.000000 0.000000
        self.toro_lines += f"EDGE1 {origin} {meas[0]:.6f} {meas[1]:.6f} {meas[2]:.6f} {info[0,0]:.6f} {info[0,1]:.6f} {info[1,1]:.6f} {info[2,2]:.6f} {info[0,2]:.6f} {info[1,2]:.6f}\n"
    
    def add_factor_2poses_2d(self, meas, origin, target, info):
        # EDGE2 1089 1090 0.984655 0.062479 0.023806 44.721360 0.000000 44.721360 44.721360 0.000000 0.000000
        self.toro_lines += f"EDGE2 {origin} {target} {meas[0]:.6f} {meas[1]:.6f} {meas[2]:.6f} {info[0,0]:.6f} {info[0,1]:.6f} {info[1,1]:.6f} {info[2,2]:.6f} {info[0,2]:.6f} {info[1,2]:.6f}\n"

    def get_lines(self):
        return self.toro_lines
        

def generate_linear_random_graph(nodes: int = 5, gpsInfo = np.eye(3)*1e3, odoInfo = np.eye(3)*1e2):

    toro_container = ToRoContainer()

    # assert factors >= nodes*2
    graph = mrob.FGraph()

    x = np.random.randn(3)*1e-1
    n = graph.add_node_pose_2d(x, mrob.NODE_ANCHOR)
    toro_container.add_node(n,x)


    indexes = [n]
    for i in range(1, nodes):
        x = np.array([i,0,0]) + np.random.randn(3)*1e-1
        n = graph.add_node_pose_2d(x)
        toro_container.add_node(n,x)

        odoObs = np.array([1,0,0]) + np.random.randn(3)*1e-1

        graph.add_factor_2poses_2d(odoObs,indexes[-1],n,odoInfo)
        toro_container.add_factor_2poses_2d(odoObs,indexes[-1],n,odoInfo)

        obs= np.array([i,0,0] + np.random.randn(3)*1e-1)

        graph.add_factor_1pose_2d(obs,n,gpsInfo)
        toro_container.add_factor_1pose_2d(n,obs,gpsInfo)

        indexes.append(n)

    print('Current chi2 = ', graph.chi2() ) # re-evaluates the error, on print it is only the error on evalation before update
    
    return graph, toro_container.get_lines()

def print_grad(gradient):
    H,W = gradient.shape

    for i in range(H):
        print(f"{gradient[i,:]}\n")


if __name__ == "__main__":

    # setting deltas for nuerical diffs
    dx = 1e-1
    dz = 1e-1

    # generating random graph with odometry and gps factors
    # TODO generate gere from spline dataset
    # TODO add get/set functions to API to address factors with certain ID to read and write their parameters. the same for nodes
    toro_file = 'toro_graph.txt'
    graph, toro_lines = generate_linear_random_graph(3)


    print(toro_lines)

    with open(toro_file,'w') as f:
        f.writelines(toro_lines)
        f.close()

    # reading serialised graph from toro file and composing it back into mrob FGraph
    vertex_ini, factors, factors_dictionary = read_graph_toro_description(toro_file)
    
    graph_0 = compose_graph(vertex_ini, factors, factors_dictionary)

    # checking that serialized and deserialized graphs have the same states
    np.allclose(np.array(graph.get_estimated_state()),np.array(graph_0.get_estimated_state()))

    # computing first gradient with (x - x_gt)/dz after solve
    gradient1 = numerical_diff1(toro_file, dz=dz)
    
    visualize_gradient(gradient1, 'gradient #1', dx=None, dz=dz)

    # computing gradient using chi2 squares and explicit theorem
    gradient2 = numerical_diff2(toro_file,dx=dx,dz =dx)

    visualize_gradient(gradient2, 'gradient #2', dx=dx, dz=dz)

    # visualizing the diff between gradients
    plt.imshow(gradient1 - gradient2)
    plt.title('grad_1 - grad_2')
    plt.show()