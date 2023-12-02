import numpy as np
from numba import njit

@njit
def sweep(fp,fm,dfdu,a,s,u,dx,x,n):
    d0 = 2/3
    d1 = 1/3
    eps = 1e-6
    
    for i in range(2,n-2):
        # Creating the left flux (\hat{f}_{j-1/2})
        # Positive flux
        fh_lp_0 = 0.5*fp(u[i],a) + 0.5*fp(u[i-1],a)
        fh_lp_1 = 1.5*fp(u[i-1],a) - 0.5*fp(u[i-2],a)
        b_lp_0 = (fp(u[i],a) - fp(u[i-1],a))**2
        b_lp_1 = (fp(u[i-1],a) - fp(u[i-2],a))**2
        a_lp_0 = d0/((eps + b_lp_0)**2)
        a_lp_1 = d1/((eps + b_lp_1)**2)
        w_lp_0 = a_lp_0/(a_lp_0 + a_lp_1)
        w_lp_1 = a_lp_1/(a_lp_0 + a_lp_1)
        fh_lp = w_lp_0*fh_lp_0 + w_lp_1*fh_lp_1
        # Negative flux
        fh_lm_0 = 0.5*fm(u[i-1],a) + 0.5*fm(u[i],a)
        fh_lm_1 = 1.5*fm(u[i],a) - 0.5*fm(u[i+1],a)
        b_lm_0 = (fm(u[i-1],a) - fm(u[i],a))**2
        b_lm_1 = (fm(u[i],a) - fm(u[i+1],a))**2
        a_lm_0 = d0/((eps + b_lm_0)**2)
        a_lm_1 = d1/((eps + b_lm_1)**2)
        w_lm_0 = a_lm_0/(a_lm_0 + a_lm_1)
        w_lm_1 = a_lm_1/(a_lm_0 + a_lm_1)
        fh_lm = w_lm_0*fh_lm_0 + w_lm_1*fh_lm_1
        fh_l = fh_lp + fh_lm # Final left flux
        # Creating the right flux (\hat{f}_{j+1/2})
        # Positive flux
        fh_rp_0 = 0.5*fp(u[i+1],a) + 0.5*fp(u[i],a)
        fh_rp_1 = 1.5*fp(u[i],a) - 0.5*fp(u[i-1],a)
        b_rp_0 = (fp(u[i+1],a) - fp(u[i],a))**2
        b_rp_1 = (fp(u[i],a) - fp(u[i-1],a))**2
        a_rp_0 = d0/((eps + b_rp_0)**2)
        a_rp_1 = d1/((eps + b_rp_1)**2)
        w_rp_0 = a_rp_0/(a_rp_0 + a_rp_1)
        w_rp_1 = a_rp_1/(a_rp_0 + a_rp_1)
        fh_rp = w_rp_0*fh_rp_0 + w_rp_1*fh_rp_1
        # Negative flux
        fh_rm_0 = 0.5*fm(u[i],a) + 0.5*fm(u[i+1],a)
        fh_rm_1 = 1.5*fm(u[i+1],a) - 0.5*fm(u[i+2],a)
        b_rm_0 = (fm(u[i],a) - fm(u[i+1],a))**2
        b_rm_1 = (fm(u[i+1],a) - fm(u[i+2],a))**2
        a_rm_0 = d0/((eps + b_rm_0)**2)
        a_rm_1 = d1/((eps + b_rm_1)**2)
        w_rm_0 = a_rm_0/(a_rm_0 + a_rm_1)
        w_rm_1 = a_rm_1/(a_rm_0 + a_rm_1)
        fh_rm = w_rm_0*fh_rm_0 + w_rm_1*fh_rm_1
        fh_r = fh_rp + fh_rm # Final right flux
        u[i] = 0.5*(u[i-1] + u[i+1]) + (dx*s(x[i]) - ((fh_r + a*(u[i+1] - u[i])/2) - (fh_l + a*(u[i] - u[i-1])/2)))/a
        a_maybe = dfdu(u[i])
        if(a_maybe > a):
            a = a_maybe

    # Backward Sweep
    for i in range(n-3,1,-1):
        # Creating the left flux (\hat{f}_{j-1/2})
        # Positive flux
        fh_lp_0 = 0.5*fp(u[i],a) + 0.5*fp(u[i-1],a)
        fh_lp_1 = 1.5*fp(u[i-1],a) - 0.5*fp(u[i-2],a)
        b_lp_0 = (fp(u[i],a) - fp(u[i-1],a))**2
        b_lp_1 = (fp(u[i-1],a) - fp(u[i-2],a))**2
        a_lp_0 = d0/((eps + b_lp_0)**2)
        a_lp_1 = d1/((eps + b_lp_1)**2)
        w_lp_0 = a_lp_0/(a_lp_0 + a_lp_1)
        w_lp_1 = a_lp_1/(a_lp_0 + a_lp_1)
        fh_lp = w_lp_0*fh_lp_0 + w_lp_1*fh_lp_1
        # Negative flux
        fh_lm_0 = 0.5*fm(u[i-1],a) + 0.5*fm(u[i],a)
        fh_lm_1 = 1.5*fm(u[i],a) - 0.5*fm(u[i+1],a)
        b_lm_0 = (fm(u[i-1],a) - fm(u[i],a))**2
        b_lm_1 = (fm(u[i],a) - fm(u[i+1],a))**2
        a_lm_0 = d0/((eps + b_lm_0)**2)
        a_lm_1 = d1/((eps + b_lm_1)**2)
        w_lm_0 = a_lm_0/(a_lm_0 + a_lm_1)
        w_lm_1 = a_lm_1/(a_lm_0 + a_lm_1)
        fh_lm = w_lm_0*fh_lm_0 + w_lm_1*fh_lm_1
        fh_l = fh_lp + fh_lm # Final left flux
        # Creating the right flux (\hat{f}_{j+1/2})
        # Positive flux
        fh_rp_0 = 0.5*fp(u[i+1],a) + 0.5*fp(u[i],a)
        fh_rp_1 = 1.5*fp(u[i],a) - 0.5*fp(u[i-1],a)
        b_rp_0 = (fp(u[i+1],a) - fp(u[i],a))**2
        b_rp_1 = (fp(u[i],a) - fp(u[i-1],a))**2
        a_rp_0 = d0/((eps + b_rp_0)**2)
        a_rp_1 = d1/((eps + b_rp_1)**2)
        w_rp_0 = a_rp_0/(a_rp_0 + a_rp_1)
        w_rp_1 = a_rp_1/(a_rp_0 + a_rp_1)
        fh_rp = w_rp_0*fh_rp_0 + w_rp_1*fh_rp_1
        # Negative flux
        fh_rm_0 = 0.5*fm(u[i],a) + 0.5*fm(u[i+1],a)
        fh_rm_1 = 1.5*fm(u[i+1],a) - 0.5*fm(u[i+2],a)
        b_rm_0 = (fm(u[i],a) - fm(u[i+1],a))**2
        b_rm_1 = (fm(u[i+1],a) - fm(u[i+2],a))**2
        a_rm_0 = d0/((eps + b_rm_0)**2)
        a_rm_1 = d1/((eps + b_rm_1)**2)
        w_rm_0 = a_rm_0/(a_rm_0 + a_rm_1)
        w_rm_1 = a_rm_1/(a_rm_0 + a_rm_1)
        fh_rm = w_rm_0*fh_rm_0 + w_rm_1*fh_rm_1
        fh_r = fh_rp + fh_rm # Final right flux
        u[i] = 0.5*(u[i-1] + u[i+1]) + (dx*s(x[i]) - ((fh_r + a*(u[i+1] - u[i])/2) - (fh_l + a*(u[i] - u[i-1])/2)))/a
        a_maybe = dfdu(u[i])
        if(a_maybe > a):
            a = a_maybe
    return u,a

@njit
def weno3(fp,fm,dfdu,s,x,u,dx):
    n = x.size
    
    a = np.max(np.abs(dfdu(u[2:-2])))
    u0 = u
    u1 = np.copy(u0)
    u1,a = sweep(fp,fm,dfdu,a,s,u1,dx,x,n) 

    u2 = np.copy(u1)
    u2,a = sweep(fp,fm,dfdu,a,s,u2,dx,x,n) 

    q1 = (np.linalg.norm(u2 - u1) + dx**3)/np.linalg.norm(u1 - u0)
    q2 = np.linalg.norm(u2 - u1) - dx**4
    iter = 0
    while((q1 < 1) or (q2 > 0)):
        u0 = np.copy(u1)
        u1 = np.copy(u2)
        u2,a = sweep(fp,fm,dfdu,a,s,u2,dx,x,n) 
        q1 = (np.linalg.norm(u2 - u1) + dx**3)/np.linalg.norm(u1 - u0)
        q2 = np.linalg.norm(u2 - u1) - dx**4
        iter += 1
    return iter, u2


