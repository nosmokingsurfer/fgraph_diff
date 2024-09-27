import numpy as np
import mrob
from tqdm import tqdm 
import matplotlib.pyplot as plt
import seaborn as sns
import os
from multiprocessing import Pool, cpu_count


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
            elif d[0] == 'EDGE1':
                factors[int(d[1]), int(d[1])] = np.array([d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10]], dtype='float64')
                factors_dictionary[int(d[1])].append(int(d[1]))
            elif d[0] == 'VERTEX2':
                vertex_ini[int(d[1])] = np.array([d[2], d[3], d[4]], dtype='float64')
                factors_dictionary[int(d[1])] = []
    return vertex_ini, factors, factors_dictionary


def compose_graph(vertex_ini, factors, factors_dictionary, perturb_index_x=None, perturb_index_z=None, dx=0, dz=0):
    graph = mrob.FGraph()
    N = len(vertex_ini)

    for t in range(N):
        x = vertex_ini[t].copy()
        #x = vertex_ini[t]
        if perturb_index_x is not None and t == perturb_index_x[0]:
            #print(f"Perturbing x[{t}][{perturb_index_x[1]}] by {dx}")
            x[perturb_index_x[1]] += dx
        if t == 0:
            n = graph.add_node_pose_2d(x, mrob.NODE_ANCHOR)
        else:   
            n = graph.add_node_pose_2d(x)
        assert t == n, 'index on node is different from counter'

        for nodeOrigin in factors_dictionary[n]:
            obs = factors[nodeOrigin, t][:3].copy()

            # Perturb one element from (dx, dy, dtheta) based on perturb_index (nodeOrigin, t, coord_idx) 
            if perturb_index_z is not None and (nodeOrigin, t) == perturb_index_z[:2]:
                #print(perturb_index_z)
                #print(f"Perturbing z[{nodeOrigin, t}][{perturb_index_z[2]}] by {dz}")
                obs[perturb_index_z[2]] += dz 

            covInv = np.zeros((3, 3))
            covInv[0, 0] = factors[nodeOrigin, t][3]
            covInv[1, 1] = factors[nodeOrigin, t][5]
            covInv[2, 2] = factors[nodeOrigin, t][6]

            if nodeOrigin != n:
                graph.add_factor_2poses_2d(obs, nodeOrigin, t, covInv)
            elif nodeOrigin == n:
                graph.add_factor_1pose_2d(obs, nodeOrigin, covInv)

    return graph


def numerical_diff1(toro_file, dz=1e-4):
    vertex_ini, factors, factors_dictionary = read_graph_toro_description(toro_file)
    
    graph_0 = compose_graph(vertex_ini, factors, factors_dictionary)
    graph_0.solve(mrob.LM, verbose=True)
    x_0 = graph_0.get_estimated_state()

    x_0 = np.array(x_0).flatten()

    obs_dim = len(factors) * 3  # 3 elements (dx, dy, dtheta) per factor
    gradient = np.zeros((len(x_0), obs_dim))
    factor_keys = list(factors.keys())
    
    for i in tqdm(range(obs_dim)):
        factor_idx, coord_idx = find_factor_coord_idx(i)
        
        nodeOrigin, t = factor_keys[factor_idx]
        perturb_index = (nodeOrigin, t, coord_idx) 

        # Compose the graph with perturbation
        graph_new = compose_graph(vertex_ini, factors, factors_dictionary, perturb_index_z=perturb_index, dz=dz)
        graph_new.solve(mrob.LM,verbose=False)
        x_new = graph_new.get_estimated_state()

        dx_new = (np.array(x_new).flatten() - x_0) / dz
        gradient[:, i] = dx_new
    return gradient


def find_factor_coord_idx(index, coord_num=3):
    return index // coord_num, index % coord_num


def compute_chi2_matrix(vertex_ini, factors, factors_dictionary, factor_keys, i_x, i_z, dx, dz):
    #vertex_ini, factors, factors_dictionary, factor_keys, chi2_matrix, i_x, i_z, dx, dz = params
    factor_idx_x, coord_idx_x = find_factor_coord_idx(i_x)
    factor_idx_z, coord_idx_z = find_factor_coord_idx(i_z)
    nodeOrigin_z, t_z = factor_keys[factor_idx_z]
    perturb_index_x = (factor_idx_x, coord_idx_x)
    perturb_index_z = (nodeOrigin_z, t_z, coord_idx_z)
    
    graph_pp = compose_graph(vertex_ini, factors, factors_dictionary, 
                                perturb_index_x=perturb_index_x, perturb_index_z = perturb_index_z, dx=dx, dz=dz) # x+h, z+k
    graph_pm = compose_graph(vertex_ini, factors, factors_dictionary, 
                                perturb_index_x=perturb_index_x, perturb_index_z = perturb_index_z, dx=dx, dz=-dz) # x+h, z-k
    graph_mp = compose_graph(vertex_ini, factors, factors_dictionary, 
                                perturb_index_x=perturb_index_x, perturb_index_z = perturb_index_z, dx=-dx, dz=dz) # x-h, z+k
    graph_mm = compose_graph(vertex_ini, factors, factors_dictionary, 
                                perturb_index_x=perturb_index_x, perturb_index_z = perturb_index_z, dx=-dx, dz=-dz) # x-h, z-k

    chi2_pp = graph_pp.chi2(evaluateResidualsFlag=True)
    chi2_pm = graph_pm.chi2(evaluateResidualsFlag=True)
    chi2_mp = graph_mp.chi2(evaluateResidualsFlag=True)
    chi2_mm = graph_mm.chi2(evaluateResidualsFlag=True)

    return (chi2_pp - chi2_pm - chi2_mp + chi2_mm) / 4 / dx / dz


def parallel_compute_chi2_matrix(args):
    vertex_ini, factors, factors_dictionary, factor_keys, i_x, i_z, dx, dz = args
    return compute_chi2_matrix(vertex_ini, factors, factors_dictionary, factor_keys, i_x, i_z, dx, dz)

def numerical_diff2(toro_file, dx=1e-1, dz=1e-1):
    vertex_ini, factors, factors_dictionary = read_graph_toro_description(toro_file)
    graph_0 = compose_graph(vertex_ini, factors, factors_dictionary)
    graph_0.solve()
    x_0 = graph_0.get_estimated_state()
    x_0 = np.array(x_0).flatten()
    dim_x = len(x_0)
    dim_z = len(factors) * 3  
    chi2_matrix = np.zeros((dim_x, dim_z))
    factor_keys = list(factors.keys())

    tasks = [(vertex_ini, factors, factors_dictionary, factor_keys, i_x, i_z, dx, dz)
             for i_x in range(dim_x) for i_z in range(dim_z)]
    
    with Pool(cpu_count()) as pool:
        result = list(tqdm(pool.imap(parallel_compute_chi2_matrix, tasks), total=len(tasks)))

    for idx, (i_x, i_z) in enumerate([(i_x, i_z) for i_x in range(dim_x) for i_z in range(dim_z)]):
        chi2_matrix[i_x, i_z] = result[idx]
    return chi2_matrix
        
    # for i_x in tqdm(range(dim_x)):
    #     for i_z in range(dim_z):
    #         chi2_matrix[i_x, i_z] = compute_chi2_matrix(vertex_ini, factors, factors_dictionary, factor_keys, i_x, i_z, dx, dz)
            
    # return chi2_matrix


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


def visualize_gradient(gradient, title, dx = None, dz=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    ax[0].imshow(gradient)
    ax[1].spy(gradient)
    plt.suptitle(f'{title}\n {dx=}, {dz=}')
    plt.show()

if __name__ == "__main__":
    input_file = './benchmarks/M3500.txt'
    n = 100
    simplified_file = f'./benchmarks/M{n}.txt'
    simplify_toro_file(input_file, simplified_file, n)

    dx = 1e-5
    dz = 1e-5
    
    #gradient = numerical_diff1(simplified_file, dz=dz)
    #visualize_gradient(gradient,'gradient #1',dx=None,dz=dz)

    chi2_matrix = numerical_diff2(simplified_file, dx=dx, dz=dz)
    visualize_gradient(chi2_matrix,'gradient #2', dx=dx, dz=dz)