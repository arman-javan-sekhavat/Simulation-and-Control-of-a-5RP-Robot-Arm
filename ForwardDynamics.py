#*****************************************************************************************************************
#=========================================  Author: Arman Javan Sekhavat =========================================
#*****************************************************************************************************************

import jax
from jax import numpy as jnp
from jax import jit, vmap, grad, jacobian
from jax.numpy import cross, dot

@jit
def Transform_Matrix(alpha, a, d, theta):
    c_alpha = jnp.cos(alpha)
    s_alpha = jnp.sin(alpha)
    c_theta = jnp.cos(theta)
    s_theta = jnp.sin(theta)
    
    return jnp.array([[c_theta, -s_theta, 0, a],
               [s_theta*c_alpha, c_theta*c_alpha, -s_alpha, -s_alpha*d],
               [s_theta*s_alpha, c_theta*s_alpha, +c_alpha, +c_alpha*d],
               [0, 0, 0, 1]])


T_batch = jit(vmap(Transform_Matrix, in_axes = [0, 0, 0, 0]))

@jit
def Kinetic_Energy(q, q_dot):
    q_dim = q.shape[0]
    joint_type = ('R', 'R', 'R', 'R', 'R', 'P')
    mass = jnp.array([0.35690217, 0.45195861, 0.37484212, 0.25855864, 0.16089002, 0.01702190])

    P_C = jnp.array([[0, 0, -0.0314492], [+0.0987861, 0, 0], [+0.1119218, 0, 0], 
                     [0, 0.0735514, 0],  [0, 0, -0.1203123], [0, 0, -0.0560843]])
    
    I = jnp.array([jnp.diag(jnp.array([4440.59357e-7, 5006.70786e-7, 2814.25487e-7])),
                   jnp.diag(jnp.array([6288.91790e-7, 29455.03631e-7, 27693.25515e-7])),
                   jnp.diag(jnp.array([4327.61778e-7, 23921.73007e-7, 23635.71296e-7])),
                   jnp.diag(jnp.array([10088.49390e-7, 2599.35655e-7, 10393.03987e-7])),
                   jnp.diag(jnp.array([8087.03173e-7, 7875.32745e-7, 2264.69799e-7])),
                   jnp.diag(jnp.array([562.83931e-7, 562.83931e-7, 47.93806e-7]))])


#----------------------------------------  Constant Parameters
    L2 = 0.200
    L3 = 0.200
    d5 = 0.390

    alpha = jnp.array([0, +jnp.pi/2, 0, 0, -jnp.pi/2, 0])
    a = jnp.array([0, 0, L2, L3, 0, 0])
    d = jnp.array([0, 0, 0, 0, d5, q[5]])
    theta = jnp.array([q[0], q[1], q[2], q[3], q[4], 0])

    T = T_batch(alpha, a, d, theta)
    P = T[:, 0:3, 3]
    R = T[:, 0:3, 0:3]
    R_transpose = jnp.transpose(R, axes = (0, 2, 1))

    w = jnp.zeros(shape = (q_dim+1, 3))
    v = jnp.zeros(shape = (q_dim+1, 3))
    v_C = jnp.zeros(shape = (q_dim, 3))

    Z = jnp.array([0, 0, 1])

    K = 0.0

    #----------------------------------------  Velocity Propagation
    for i in range(q_dim):

        if joint_type[i] == 'R':
            w = w.at[i+1].set(R_transpose[i]@w[i] + q_dot[i]*Z)
            v = v.at[i+1].set(R_transpose[i]@(v[i] + cross(w[i], P[i])))
        else:
            w = w.at[i+1].set(R_transpose[i]@w[i])
            v = v.at[i+1].set(R_transpose[i]@(v[i] + cross(w[i], P[i])) + q_dot[i]*Z)
            
        v_C = v_C.at[i].set(v[i+1] + cross(w[i+1], P_C[i]))

        K += 0.5*(mass[i]*dot(v_C[i], v_C[i]) + jnp.transpose(w[i+1])@I[i]@w[i+1])

    return K


@jit
def Potential_Energy(q):
    q_dim = q.shape[0]
    U = 0.0
    g = jnp.array([0.000, 0.000, -9.810])
    mass = jnp.array([0.35690217, 0.45195861, 0.37484212, 0.25855864, 0.16089002, 0.01702190])
    P_C = jnp.array([[0, 0, -0.0314492], [+0.0987861, 0, 0], [+0.1119218, 0, 0], 
                     [0, 0.0735514, 0],  [0, 0, -0.1203123], [0, 0, -0.0560843]])

    #----------------------------------------  Constant Parameters
    L2 = 0.200
    L3 = 0.200
    d5 = 0.390

    alpha = jnp.array([0, +jnp.pi/2, 0, 0, -jnp.pi/2, 0])
    a = jnp.array([0, 0, L2, L3, 0, 0])
    d = jnp.array([0, 0, 0, 0, d5, q[5]])
    theta = jnp.array([q[0], q[1], q[2], q[3], q[4], 0])

    T = T_batch(alpha, a, d, theta)
    T_i_0 = jnp.eye(4, 4, dtype = jnp.float32)

    for i in range(q_dim):
        T_i_0 = T_i_0@T[i]
        R = T_i_0[0:3, 0:3]
        P = T_i_0[0:3, 3]
        U += -mass[i]*dot(g, R@P_C[i] + P)

    return U

@jit
def L(q, q_dot):
    return Kinetic_Energy(q, q_dot) - Potential_Energy(q)


dL_dq = jit(grad(L, argnums = 0))
dL_dq_dot = jit(grad(L, argnums = 1))
M = jit(jacobian(dL_dq_dot, argnums = 1))
C = jit(jacobian(dL_dq_dot, argnums = 0))

@jit
def ForwardDynamics(q, q_dot, tau):
    G = -dL_dq(q, q_dot)
    q_ddot = jnp.linalg.inv(M(q, q_dot))@(tau - C(q, q_dot)@q_dot - G)
    return q_ddot

#----------------------------------------------------------------------------------- Test
q = jnp.array([1.01, 2.18, -1.45, -2.58, -0.597, 0.0532])
q_dot = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
tau = jnp.array([-0.44, 0.64, -0.64, 0.75, -0.53, 0.28])

q_ddot = ForwardDynamics(q, q_dot, tau)
print(q_ddot)
