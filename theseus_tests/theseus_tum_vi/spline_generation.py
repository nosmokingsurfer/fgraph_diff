# import math

# import matplotlib.pyplot as plt
# import numpy as np

# from matplotlib.animation import FuncAnimation


# def beta_pdf(x, a, b):
#     return (x**(a-1) * (1-x)**(b-1) * math.gamma(a + b)
#             / (math.gamma(a) * math.gamma(b)))


# class UpdateDist:
#     def __init__(self, ax, prob=0.5):
#         self.success = 0
#         self.prob = prob
#         self.line, = ax.plot([], [], 'k-')
#         self.x = np.linspace(0, 1, 200)
#         self.ax = ax

#         # Set up plot parameters
#         self.ax.set_xlim(0, 1)
#         self.ax.set_ylim(0, 10)
#         self.ax.grid(True)

#         # This vertical line represents the theoretical value, to
#         # which the plotted distribution should converge.
#         self.ax.axvline(prob, linestyle='--', color='black')

#     def __call__(self, i):
#         # This way the plot can continuously run and we just keep
#         # watching new realizations of the process
#         if i == 0:
#             self.success = 0
#             self.line.set_data([], [])
#             return self.line,

#         # Choose success based on exceed a threshold with a uniform pick
#         if np.random.rand() < self.prob:
#             self.success += 1
#         y = beta_pdf(self.x, self.success + 1, (i - self.success) + 1)
#         self.line.set_data(self.x, y)
#         return self.line,

# # Fixing random state for reproducibility
# np.random.seed(19680801)


# fig, ax = plt.subplots()
# ud = UpdateDist(ax, prob=0.7)
# anim = FuncAnimation(fig, ud, frames=100, interval=100, blit=True)
# plt.show()

import scipy
import scipy.interpolate as si

import matplotlib.pyplot as plt
import numpy as np

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


fig, ax = plt.subplots()
control_points = np.array([[0,0]])
spline_points = np.array([[0,0]])

control = ax.plot(control_points[:,0], control_points[:,1], 'x', color='black')
spline = ax.plot([],[], label='spline', color='red')
ax.axis('equal')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)



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
    spline_points = bspline(control_points, 100*len(control_points), 3)
    spline[0].set_data(spline_points[:,0], spline_points[:,1])

    plt.show()

binding_id = plt.connect('motion_notify_event', on_move)
plt.connect('button_press_event', on_click)
plt.connect('close_event', on_close)
plt.grid()
plt.legend()
plt.show()