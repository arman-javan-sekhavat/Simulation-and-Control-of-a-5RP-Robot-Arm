#*****************************************************************************************************************
#=========================================  Author: Arman Javan Sekhavat =========================================
#*****************************************************************************************************************



import numpy as np

def interval_idx(dt_list, t):
    time_points = np.cumsum(np.hstack([0.0, dt_list]))
    k = np.argmax(np.cumsum(t - time_points))
    return k, t - time_points[k]


def Cubic(q_list, dt_list):
    n = q_list.shape[1]  # Number of joint variables
    N = dt_list.shape[0] # Number of path segments

    a = np.zeros(shape = (N, n, 4), dtype = np.float32) # Polynomial coefficients
    q_dot_list = np.zeros(shape = (N+1, n), dtype = np.float32)
    dq_list = np.diff(q_list, n = 1, axis = 0)

    for i in range(1, N):
        q_dot_list[i] = (dq_list[i-1]/dt_list[i-1] + dq_list[i]/dt_list[i])/2.0

    dt_list = np.transpose(np.broadcast_to(dt_list, shape = (n, N)))
    a[:, :, 0] = q_list[:-1]
    a[:, :, 1] = q_dot_list[:-1]
    a[:, :, 2] = +(3.0/dt_list**2)*dq_list - (2.0/dt_list)*q_dot_list[:-1] - (1.0/dt_list)*q_dot_list[+1:]
    a[:, :, 3] = -(2.0/dt_list**3)*dq_list + (1.0/dt_list**2)*(q_dot_list[:-1] + q_dot_list[+1:])

    return a

def CubicGenerator(a, dt_list, t):
    n = a.shape[1]
    k, t = interval_idx(dt_list, t)
    a = a[k]

    q = a[:, 0] + a[:, 1]*t + a[:, 2]*t**2 + a[:, 3]*t**3
    q_dot = a[:, 1] + 2*a[:, 2]*t + 3*a[:, 3]*t**2
    q_ddot = 2*a[:, 2] + 6*a[:, 3]*t
    return q, q_dot, q_ddot



def Quintic(q_list, dt_list):
    n = q_list.shape[1]  # Number of joint variables
    N = dt_list.shape[0] # Number of path segments

    a = np.zeros(shape = (N, n, 6), dtype = np.float32) # Polynomial coefficients
    q_dot_list = np.zeros(shape = (N+1, n), dtype = np.float32)
    q_ddot_list = np.zeros(shape = (N+1, n), dtype = np.float32)
    dq_list = np.diff(q_list, n = 1, axis = 0)

    for i in range(1, N):
        q_dot_list[i] = (dq_list[i-1]/dt_list[i-1] + dq_list[i]/dt_list[i])/2.0

    dt_list = np.transpose(np.broadcast_to(dt_list, shape = (n, N)))

    q_list_0 = q_list[:-1]
    q_list_f = q_list[+1:]

    q_dot_list_0 = q_dot_list[:-1]
    q_dot_list_f = q_dot_list[+1:]

    q_ddot_list_0 = q_ddot_list[:-1]
    q_ddot_list_f = q_ddot_list[+1:]

    a[:, :, 0] = q_list_0
    a[:, :, 1] = q_dot_list_0
    a[:, :, 2] = 0.5*q_ddot_list_0
    a[:, :, 3] = (0.5/dt_list**3)*(20*q_list_f - 20*q_list_0 - (8*q_dot_list_f + 12*q_dot_list_0)*dt_list - (3*q_ddot_list_0 - q_ddot_list_f)*dt_list**2)
    a[:, :, 4] = (0.5/dt_list**4)*(30*q_list_0 - 30*q_list_f + (14*q_dot_list_f + 16*q_dot_list_0)*dt_list + (3*q_ddot_list_0 - 2*q_ddot_list_f)*dt_list**2)
    a[:, :, 5] = (0.5/dt_list**5)*(12*q_list_f - 12*q_list_0 - (6*q_dot_list_f + 6*q_dot_list_0)*dt_list - (q_ddot_list_0 - q_ddot_list_f)*dt_list**2)

    return a


def QuinticGenerator(a, dt_list, t):
    n = a.shape[1]
    k, t = interval_idx(dt_list, t)
    a = a[k]

    q = a[:, 0] + a[:, 1]*t + a[:, 2]*t**2 + a[:, 3]*t**3 + a[:, 4]*t**4 + a[:, 5]*t**5
    q_dot = a[:, 1] + 2*a[:, 2]*t + 3*a[:, 3]*t**2 + 4*a[:, 4]*t**3 + 5*a[:, 5]*t**4
    q_ddot = 2*a[:, 2] + 6*a[:, 3]*t + 12*a[:, 4]*t**2 + 20*a[:, 5]*t**3

    return q, q_dot, q_ddot


def CartesianPath(t):
    
    w = np.pi
    R = 0.1

    x0 = 0.5
    y0 = 0.0
    z0 = 0.5

    X = np.array([x0, y0 + R*np.cos(w*t), z0 + R*np.sin(w*t)])
    X_dot = np.array([0.0, -R*w*np.sin(w*t), +R*w*np.cos(w*t)])
    X_ddot = np.array([0.0, -R*w*w*np.cos(w*t), -R*w*w*np.sin(w*t)])

    return X, X_dot, X_ddot