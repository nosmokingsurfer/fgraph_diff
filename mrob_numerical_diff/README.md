
## What to do


Implement folowing numerical diff function:

1) read_graph_toro_description(file with toro format)

2) compose graph(...) with only one obs coordinate changed


there are two options to get gradient - one via direct gradient of dx/dz the second one via mixed gradient d^2 chi2/dzdx


3) numerical_diff(toro_file):
    dz = 1e-5 

    graph_0 = compose_graph_without_z_perturbations()
    x_0 = graph_0.solve()

    gradient = dim(x) x dim(z) # matrix

    for i in len(obs_dim) # obs_dim = sum(dim(z_i))
    {
        graph_new = compose_graph_with perturbed ith observation coordinate()
        <!-- x_new = graph_new.chi2(True)
        help(chi2) -->
        dx_new = (x_new - x_0)/dz
        gradient[:,i] = dx_new
    }

d^2 chi2() / dzdx -> ?

d^2 chi2() / dzdx  = chi2(x_0,z_0) - 2* + 


graph.chi2() -> scalar


def num_diff_2(graph):
    graph.solve()

    x_0 = graph.get_estiamted_state()

    chi2_0 = graph.chi2()

    for j in range(dim(x)):
        for i in range(dim(z)):
            graph_++ = compose_graph(x_0 + dx[j], z + dz[i])
            graph_+- = ...
            graph_-+ = ...
            graph_-- = ...

            chi2++ = graph_++.chi2(params=True)
            chi2+- = graph_+-.chi2(...)


            #https://en.wikipedia.org/wiki/Finite_difference#Multivariate_finite_differences
            chi2[i,j] = (chi2++ - ...)/(4*dx*dz)
