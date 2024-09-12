import numpy as np
import mrob
from tqdm import tqdm 
import matplotlib.pyplot as plt
import seaborn as sns
import os


def read_graph_toro_description(toro_file):
    #Reads TORO file and returns vertices, edges, and factors.

    vertex_ini = {}
    factors = {}
    factors_dictionary = {}

    with open(toro_file, 'r') as file:
        for line in file:
            d = line.split()
            if d[0] == 'EDGE2':
                factors[int(d[1]), int(d[2])] = np.array([d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11]], dtype='float64')
                factors_dictionary[int(d[2])].append(int(d[1]))
                
            elif d[0] == 'VERTEX2':
                vertex_ini[int(d[1])] = np.array([d[2], d[3], d[4]], dtype='float64')
                factors_dictionary[int(d[1])] = []
    return vertex_ini, factors, factors_dictionary


def compose_graph(vertex_ini, factors, factors_dictionary, perturb_index=None, dz=0):
    graph = mrob.FGraph()
    N = len(vertex_ini)

    for t in range(N):
        x = vertex_ini[t]
        graph.add_node_pose_2d(x)

    for t in range(1, N):
        connecting_nodes = factors_dictionary[t]
        for nodeOrigin in connecting_nodes:
            obs = factors[nodeOrigin, t][:3].copy()

            # Perturb one element from (dx, dy, dtheta) based on perturb_index
            if perturb_index is not None and t == perturb_index[0]:
                obs[perturb_index[1]] += dz 

            covInv = np.zeros((3, 3))
            covInv[0, 0] = factors[nodeOrigin, t][3]
            covInv[1, 1] = factors[nodeOrigin, t][5]
            covInv[2, 2] = factors[nodeOrigin, t][6]

            graph.add_factor_2poses_2d(obs, nodeOrigin, t, covInv)

    return graph


def numerical_diff(toro_file, dz):
    vertex_ini, factors, factors_dictionary = read_graph_toro_description(toro_file)
    
    graph_0 = compose_graph(vertex_ini, factors, factors_dictionary)
    graph_0.solve()
    x_0 = graph_0.get_estimated_state()

    x_0 = np.array(x_0).flatten()

    obs_dim = len(factors) * 3  # 3 elements (dx, dy, dtheta) per factor
    gradient = np.zeros((len(x_0), obs_dim))

    for i in tqdm(range(obs_dim)):
        factor_idx = i // 3  
        coord_idx = i % 3   
        
        # Perturb the ith observation coordinate
        perturb_index = (factor_idx + 1, coord_idx) 

        # Compose the graph with perturbation
        graph_new = compose_graph(vertex_ini, factors, factors_dictionary, perturb_index=perturb_index, dz=dz)
        graph_new.solve(mrob.LM)
        x_new = graph_new.get_estimated_state()

        dx_new = (np.array(x_new).flatten() - x_0) / dz
        gradient[:, i] = dx_new

    return gradient


def simplify_toro_file(input_file, output_file, size):

    vertices = []
    verticies_ids = []
    edges = []
    
    with open(input_file, 'r') as f:
        for line in f:
            if line.startswith('VERTEX2') and len(vertices) < size:
                vertices.append(line)
                verticies_ids.append(line.split(' ')[1])
        f.close()
    with open(input_file, 'r') as f:
        for line in f:
            if line.startswith('EDGE2'):
                src,dst = line.split(' ')[1:3]
                if src in verticies_ids and dst in verticies_ids:
                    edges.append(line)
        f.close()


    with open(output_file, 'w') as f_out:
        f_out.writelines(vertices)
        f_out.writelines(edges)
    
    print('Vertices:', len(vertices), 'Edges:', len(edges))


def visualize_gradient(gradient):
    plt.figure(figsize=(10, 8))
    plt.imshow(gradient)
    plt.title('Gradients')
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.spy(gradient,precision=1e-5)
    plt.title('Gradients')
    plt.show()


input_file = './benchmarks/M3500.txt'
n = 100
simplified_file = f'./benchmarks/M{n}.txt'
simplify_toro_file(input_file, simplified_file, n)

gradient = numerical_diff(simplified_file, dz=1e-5)
visualize_gradient(gradient)