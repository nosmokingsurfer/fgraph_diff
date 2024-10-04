import scipy
import scipy.interpolate as si

import matplotlib.pyplot as plt
import numpy as np

import os

from matplotlib.backend_bases import MouseButton

def bspline(cv, n=100, degree=3, periodic=False):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
        periodic: True - Curve is closed
                  False - Curve is open
    """

    # If periodic, extend the point array by count+degree+1
    cv = np.asarray(cv)
    count = len(cv)

    if periodic:
        factor, fraction = divmod(count+degree+1, count)
        cv = np.concatenate((cv,) * factor + (cv[:fraction],))
        count = len(cv)
        degree = np.clip(degree,1,degree)

    # If opened, prevent degree from exceeding count-1
    else:
        degree = np.clip(degree,1,count-1)


    # Calculate knot vector
    kv = None
    if periodic:
        kv = np.arange(0-degree,count+degree+degree-1,dtype='int')
    else:
        kv = np.concatenate(([0]*degree, np.arange(count-degree+1), [count-degree]*degree))


    # Calculate query range
    u = np.linspace(periodic,(count-degree),n)


    # Calculate result
    return np.array(si.splev(u, (kv,cv.T,degree))).T



def on_move(event):
    if event.inaxes:
        print(f'data coords {event.xdata} {event.ydata},',
              f'pixel coords {event.x} {event.y}')


def on_close(event):
    global spline_points

    np.savetxt("./theseus_tests/theseus_tum_vi/spline_points.txt", spline_points)

def on_click(event):
    if event.button is MouseButton.LEFT:
        # print('disconnecting callback')
        # plt.disconnect(binding_id)
        if None not in [event.xdata, event.ydata]:
            global control_points
            control_points = np.concatenate((control_points, np.array([[event.xdata, event.ydata]])))

    print(control_points)
    control[0].set_data(control_points.transpose())

    global spline_points
    spline_points = bspline(control_points, 100*len(control_points), 4)
    spline[0].set_data(spline_points[:,0], spline_points[:,1])

    plt.show()


def generate_batch_of_splines(out_path, B = 10, n_control_points = 100, n_pts_spline_segment = 100):
    # B - batch size
    # n_control_points - number of control verticies of B-spline
    # n_pts_spline_segment - number of points for each spline segement

    os.makedirs(out_path, exist_ok=True)
    # TODO clean up directory for each run

    for b in range(B):
        rnd_pts = np.random.uniform(-5,5,size=(n_control_points,2))

        rnd_pts[:,0] = np.linspace(0,20,n_control_points)

        spline_points = bspline(rnd_pts,n_control_points*n_pts_spline_segment,3)

        plt.plot(spline_points[:,0],spline_points[:,1])

        np.savetxt(f'{out_path}/spline_{b}.txt',spline_points)

    plt.grid()
    plt.title('Generated batch of trajectories')
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    
    fig, ax = plt.subplots()
    control_points = np.array([[0,0]])
    spline_points = np.array([[0,0]])

    control = ax.plot(control_points[:,0], control_points[:,1], 'x', color='black')
    spline = ax.plot([],[], label='spline', color='red')
    ax.axis('equal')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    binding_id = plt.connect('motion_notify_event', on_move)
    plt.connect('button_press_event', on_click)
    plt.connect('close_event', on_close)
    plt.grid()
    plt.legend()
    plt.show()