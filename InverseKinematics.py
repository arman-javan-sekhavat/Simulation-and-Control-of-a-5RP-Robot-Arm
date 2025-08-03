#*****************************************************************************************************************
#=========================================  Author: Arman Javan Sekhavat =========================================
#*****************************************************************************************************************



from math import pi, cos, sin, atan2, fabs
import numpy as np
import Circular

def Inverse(T):

    # constant length parameters
    L1 = 0.170
    L2 = 0.200
    L3 = 0.200
    d5 = 0.390

    ((r11, r12, r13), (r21, r22, r23), (r31, r32, r33)) = T[0:3, 0:3]
    x, y, z = T[0:3, 3]

    A = 1.0 - r33*r33

    theta1 = 0.5*atan2((+2*r13*r23)/A, (r13*r13 - r23*r23)/A)
    theta5 = 0.5*atan2((-2*r31*r32)/A, (r31*r31 - r32*r32)/A)

    u = np.array([r13, r23, -r31, r32], dtype = np.float32)
    v = np.array([cos(theta1), sin(theta1), cos(theta5), sin(theta5)], dtype = np.float32)
    i = np.argmax(np.fabs(v))

    c234 = r33
    s234 = u[i]/v[i]

    d6 = 0.0

    u = np.array([(d5 + d6)*r13 - x, (d5 + d6)*r23 - y], dtype = np.float32)
    v = np.array([cos(theta1), sin(theta1)], dtype = np.float32)
    i = np.argmax(np.fabs(v))

    K1 = u[i]/v[i]
    K2 = z - L1 - (d5 + d6)*r33

    C = Circular.Circular(L2, L3, K1, K2)


    # checking solution 1
    theta2 = C[0]
    theta3 = C[1] - C[0]
    theta4 = atan2(s234, c234) - C[1]

    if( (theta2 > 0.0 and theta2 < pi) and
        (theta3 > -(2.0/3.0)*pi and theta3 < +(2.0/3.0)*pi) and
        (theta4 > -pi and theta4 < 0.0)
        ):
        
        return (theta1, theta2, theta3, theta4, theta5, d6)
    

    # checking solution 2
    theta2 = C[2]
    theta3 = C[3] - C[2]
    theta4 = atan2(s234, c234) - C[3]

    if( (theta2 > 0.0 and theta2 < pi) and
        (theta3 > -(2.0/3.0)*pi and theta3 < +(2.0/3.0)*pi) and
        (theta4 > -pi and theta4 < 0.0)
        ):
        
        return (theta1, theta2, theta3, theta4, theta5, d6)

    
    
    
    

    return None

    