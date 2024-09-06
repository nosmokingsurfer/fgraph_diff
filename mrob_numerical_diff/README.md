
## What to do


Implement folowing numerical diff function:

1) read_graph_toro_description(file with toro format)

2) compose graph(...) with only one obs coordinate changed





3) numerical_diff(toro_file):
    dz = 1e-5 

    graph_0 = compose_graph_without_z_perturbations()
    x_0 = graph_0.solve()

    gradient = dim(x) x dim(z) # matrix

    for i in len(obs_dim) # obs_dim = sum(dim(z_i))
    {
        graph_new = compose_graph_with perturbed ith observation coordinate()
        x_new = graph_new.solve()
        dx_new = (x_new - x_0)/dz
        gradient[:,i] = dx_new
    }