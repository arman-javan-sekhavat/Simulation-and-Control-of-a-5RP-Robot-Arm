#*****************************************************************************************************************
#=========================================  Author: Arman Javan Sekhavat =========================================
#*****************************************************************************************************************


import jax
from jax import numpy as jnp
from jax import jit, vmap, grad, jacobian
from jax.numpy import cross, dot


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


def Newton_Euler(q, q_dot, q_ddot):
    q_dim = q.shape[0]
    joint_type = ('R', 'R', 'R', 'R', 'R', 'P')
    mass = jnp.array([0.35690217, 0.45195861, 0.37484212, 0.25855864, 0.16089002, 0.01702190])

    # Positions of CMs relative to the link frames
    P_C = jnp.array([[0, 0, -0.0314492], [+0.0987861, 0, 0], [+0.1119218, 0, 0], 
                     [0, 0.0735514, 0],  [0, 0, -0.1203123], [0, 0, -0.0560843]])
    
    # Inertia tensors
    I = jnp.array([jnp.diag(jnp.array([4440.59357e-7, 5006.70786e-7, 2814.25487e-7])),
                   jnp.diag(jnp.array([6288.91790e-7, 29455.03631e-7, 27693.25515e-7])),
                   jnp.diag(jnp.array([4327.61778e-7, 23921.73007e-7, 23635.71296e-7])),
                   jnp.diag(jnp.array([10088.49390e-7, 2599.35655e-7, 10393.03987e-7])),
                   jnp.diag(jnp.array([8087.03173e-7, 7875.32745e-7, 2264.69799e-7])),
                   jnp.diag(jnp.array([562.83931e-7, 562.83931e-7, 47.93806e-7]))])


#----------------------------------------  Constant Parameters
    g  = 9.810
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
    w_dot = jnp.zeros(shape = (q_dim+1, 3))

    v_dot = jnp.zeros(shape = (q_dim+1, 3))
    v_dot = v_dot.at[0, 2].set(+g)
    
    v_dot_C = jnp.zeros(shape = (q_dim, 3))
    f = jnp.zeros(shape = (q_dim, 3))
    n = jnp.zeros(shape = (q_dim, 3))
    tau = jnp.zeros(shape = (q_dim,))

    Z = jnp.array([0, 0, 1])

#----------------------------------------  Outward Iteration
    for i in range(q_dim):

        if joint_type[i] == 'R':
            w = w.at[i+1].set(R_transpose[i]@w[i] + q_dot[i]*Z)
            w_dot = w_dot.at[i+1].set(R_transpose[i]@(w_dot[i] + q_dot[i]*cross(w[i], Z)) + q_ddot[i]*Z)
            v_dot = v_dot.at[i+1].set(R_transpose[i]@(cross(w_dot[i], P[i]) + cross(w[i], cross(w[i], P[i])) + v_dot[i]))
        else:
            w = w.at[i+1].set(R_transpose[i]@w[i])
            w_dot = w_dot.at[i+1].set(R_transpose[i]@w_dot[i])
            v_dot = v_dot.at[i+1].set((R_transpose[i]@(cross(w_dot[i], P[i]) + cross(w[i], cross(w[i], P[i])) + v_dot[i])
                        + 2*q_dot[i]*cross(w[i+1], Z) + q_ddot[i]*Z))
            
        v_dot_C = v_dot_C.at[i].set(cross(w_dot[i+1], P_C[i]) + cross(w[i+1], cross(w[i+1], P_C[i])) + v_dot[i+1])

#----------------------------------------  Inward Iteration
    F = mass[q_dim-1]*v_dot_C[q_dim-1]
    N = I[q_dim-1]@w_dot[q_dim] + cross(w[q_dim], I[q_dim-1]@w[q_dim])
    f = f.at[q_dim-1].set(F)
    n = n.at[q_dim-1].set(N + cross(P_C[q_dim-1], F))

    if joint_type[q_dim-1] == 'R':
        tau = tau.at[q_dim-1].set(dot(n[q_dim-1], Z))
    else:
        tau = tau.at[q_dim-1].set(dot(f[q_dim-1], Z))
    

    for i in range(q_dim - 2, -1, -1):
        F = mass[i]*v_dot_C[i]
        N = I[i]@w_dot[i+1] + cross(w[i+1], I[i]@w[i+1])

        f_new = F + R[i+1]@f[i+1]
        n_new = N + R[i+1]@n[i+1] + cross(P_C[i], F) + cross(P[i+1], R[i+1]@f[i+1])

        f = f.at[i].set(f_new)
        n = n.at[i].set(n_new)


        if joint_type[i] == 'R':
            tau = tau.at[i].set(dot(n[i], Z))
        else:
            tau = tau.at[i].set(dot(f[i], Z))
        
    return tau

Newton_Euler_jit = jit(Newton_Euler)
NE_batch = jit(vmap(jit(Newton_Euler), in_axes = [None, 0, 0]))


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

Kinetic_Energy_jit = jit(Kinetic_Energy)

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

def L(q, q_dot):
    return Kinetic_Energy(q, q_dot) - Potential_Energy(q)

dL_dq = grad(L, argnums = 0)
dL_dq_dot = grad(L, argnums = 1)

M = jacobian(dL_dq_dot, argnums = 1)
C = jacobian(dL_dq_dot, argnums = 0)

M_jit = jit(M)
C_jit = jit(C)


def Lagrange(q, q_dot, q_ddot):
    G = -dL_dq(q, q_dot)
    tau = M(q, q_dot)@q_ddot + C(q, q_dot)@q_dot + G
    return tau


Lagrange_jit = jit(Lagrange)


def M_and_C(q, q_dot):
    q_dim = q.shape[0]
    q_dot_batch = jnp.vstack([jnp.zeros(shape = (q_dim+1, q_dim)), q_dot])
    q_ddot_batch = jnp.vstack([jnp.zeros(shape = (1, q_dim)), jnp.eye(q_dim), jnp.zeros(shape = (q_dim,))])
    temp = NE_batch(q, q_dot_batch, q_ddot_batch)

    G_q = jnp.broadcast_to(temp[0], shape = (q_dim, q_dim))
    M = jnp.transpose(temp[+1:-1] - G_q)
    C = temp[-1]

    return M, C

M_and_C_jit = jit(M_and_C)


def G(q):
    q_dim = q.shape[0]
    Newton_Euler_jit(q, jnp.zeros(shape = (q_dim,)), jnp.zeros(shape = (q_dim,)))

G_jit = jit(G)