import numpy as np
import pytest

from spline_diff import generate_imu_data

import matplotlib.pyplot as plt


def generate_circle_data():
    R = 1.0

    t = np.linspace(0, 2*np.pi, 10000)
    xy = np.zeros((len(t),2))

    xy[:,0] = R*np.cos(t)
    xy[:,1] = R*np.sin(t)

    return xy

def test_imu_generation():
    data = generate_circle_data()

    plt.figure()
    plt.plot(data[:,0], data[:,1])

    acc, gyro = generate_imu_data(data)
    
    fig, ax = plt.subplots()

    ax.plot(acc[:,0],label='acc x')
    ax.plot(acc[:,1],label='acc y')
    plt.grid()
    plt.legend()

    ax2 = ax.twinx()
    ax2.plot(gyro, color='green',label='omega_z')

    plt.grid()
    plt.legend()


    plt.show()

    assert(len(acc) == len(gyro))
    