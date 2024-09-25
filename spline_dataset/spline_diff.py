import numpy as np
import matplotlib.pyplot as plt

def generate_imu_data(data):
    """
    calculating IMU data for provided spline points
    orientation is calculating using the tangent vector as a heading
    
    """

    sampling_rate = 100 #TODO parametrise this place
    dt = 1.0/sampling_rate 

    # plt.plot(data[:,0], data[:,1])
    # plt.plot(data[:,0], data[:,1],'x')

    velocity=np.diff(data,axis=0)/dt

    # plt.figure()
    # plt.plot(*velocity.T)

    vel_abs = np.linalg.norm(velocity,axis=1)

    # plt.figure()
    # plt.plot(vel_abs)

    # Frenet-Serret formulas
    # calculating tangent unitary vector
    tau = velocity/vel_abs[:,None]

    # plt.figure()
    # plt.plot(np.linalg.norm(tau,axis=1))

    # calculating norml unitary vector
    n = tau[:,::-1].copy()
    n[:,0] = -n[:,0]

    # calculatig the accelerometer measurmenets

    acc = np.diff(velocity, axis=0)/dt

    acc_measurments = np.zeros_like(tau)

    for i in range(len(tau)-1):
        acc_measurments[i,0] = np.dot(acc[i],tau[i])
        acc_measurments[i,1] = np.dot(acc[i],n[i])

    acc_measurments = acc_measurments[:-1]

    # plt.figure()
    # plt.plot(acc_measurments[:,0])
    # plt.plot(acc_measurments[:,1])

    gyro_measurments = np.zeros(len(tau)-1)

    for i in range(len(gyro_measurments)-1):
        gyro_measurments[i] = np.arcsin(np.cross(tau[i],tau[i+1]))/dt

    # plt.figure()
    # plt.plot(gyro_measurments)
    

    # plt.show()

    # TODO check differentiating results via integration

    time = dt*np.array(range(0,len(acc_measurments[:-1])))

    return acc_measurments[:-1], gyro_measurments[:-1].reshape(-1,1), tau, n, velocity, time

if __name__ == "__main__":
    log_path = './theseus_tests/theseus_tum_vi/spline_points.txt'
    data = np.genfromtxt(log_path)
    acc_measurments, gyro_measurments,_,_ = generate_imu_data(data)

    fig, ax = plt.subplots()

    ax.plot(acc_measurments[:,0],label='acc x')
    ax.plot(acc_measurments[:,1],label='acc y')
    plt.grid()
    plt.legend()

    ax2 = ax.twinx()
    ax2.plot(gyro_measurments, color='green',label='omega_z')

    plt.grid()
    plt.legend()

    plt.figure()

    plt.plot(data[:,0],data[:,1])

    plt.show()


    track = {
        'gt' : data,
        'acc' : acc_measurments,
        'gyro' : gyro_measurments
        }
    
    print(track)
