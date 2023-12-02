import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def sweep_helper(fp,fm,dfdu,gp,gm,dgdu,s,u,x,y,dx,dy,ax,ay,I,J,d0,d1,eps):
    for i in I:
        for j in J:
            # Creating the left flux (\hat{f}_{i-1/2})
            # Positive flux
            fh_lp_0 = 0.5*fp(u[i,j],ax) + 0.5*fp(u[i-1,j],ax)
            fh_lp_1 = 1.5*fp(u[i-1,j],ax) - 0.5*fp(u[i-2,j],ax)
            b_lp_0 = (fp(u[i,j],ax) - fp(u[i-1,j],ax))**2
            b_lp_1 = (fp(u[i-1,j],ax) - fp(u[i-2,j],ax))**2
            a_lp_0 = d0/((eps + b_lp_0)**2)
            a_lp_1 = d1/((eps + b_lp_1)**2)
            w_lp_0 = a_lp_0/(a_lp_0 + a_lp_1)
            w_lp_1 = a_lp_1/(a_lp_0 + a_lp_1)
            fh_lp = w_lp_0*fh_lp_0 + w_lp_1*fh_lp_1
            # Negative flux
            fh_lm_0 = 0.5*fm(u[i-1,j],ax) + 0.5*fm(u[i,j],ax)
            fh_lm_1 = 1.5*fm(u[i,j],ax) - 0.5*fm(u[i+1,j],ax)
            b_lm_0 = (fm(u[i-1,j],ax) - fm(u[i,j],ax))**2
            b_lm_1 = (fm(u[i,j],ax) - fm(u[i+1,j],ax))**2
            a_lm_0 = d0/((eps + b_lm_0)**2)
            a_lm_1 = d1/((eps + b_lm_1)**2)
            w_lm_0 = a_lm_0/(a_lm_0 + a_lm_1)
            w_lm_1 = a_lm_1/(a_lm_0 + a_lm_1)
            fh_lm = w_lm_0*fh_lm_0 + w_lm_1*fh_lm_1
            fh_l = fh_lp + fh_lm # Final left flux

            # Creating the right flux (\hat{f}_{i+1/2})
            # Positive flux
            fh_rp_0 = 0.5*fp(u[i+1,j],ax) + 0.5*fp(u[i,j],ax)
            fh_rp_1 = 1.5*fp(u[i,j],ax) - 0.5*fp(u[i-1,j],ax)
            b_rp_0 = (fp(u[i+1,j],ax) - fp(u[i,j],ax))**2
            b_rp_1 = (fp(u[i,j],ax) - fp(u[i-1,j],ax))**2
            a_rp_0 = d0/((eps + b_rp_0)**2)
            a_rp_1 = d1/((eps + b_rp_1)**2)
            w_rp_0 = a_rp_0/(a_rp_0 + a_rp_1)
            w_rp_1 = a_rp_1/(a_rp_0 + a_rp_1)
            fh_rp = w_rp_0*fh_rp_0 + w_rp_1*fh_rp_1
            # Negative flux
            fh_rm_0 = 0.5*fm(u[i,j],ax) + 0.5*fm(u[i+1,j],ax)
            fh_rm_1 = 1.5*fm(u[i+1,j],ax) - 0.5*fm(u[i+2,j],ax)
            b_rm_0 = (fm(u[i,j],ax) - fm(u[i+1,j],ax))**2
            b_rm_1 = (fm(u[i+1,j],ax) - fm(u[i+2,j],ax))**2
            a_rm_0 = d0/((eps + b_rm_0)**2)
            a_rm_1 = d1/((eps + b_rm_1)**2)
            w_rm_0 = a_rm_0/(a_rm_0 + a_rm_1)
            w_rm_1 = a_rm_1/(a_rm_0 + a_rm_1)
            fh_rm = w_rm_0*fh_rm_0 + w_rm_1*fh_rm_1
            fh_r = fh_rp + fh_rm # Final right flux

            # Creating the lower flux (\hat{g}_{j-1/2})
            # Positive flux
            gh_lp_0 = 0.5*gp(u[i,j],ay) + 0.5*gp(u[i,j-1],ay)
            gh_lp_1 = 1.5*gp(u[i,j-1],ay) - 0.5*gp(u[i,j-2],ay)
            b_lp_0 = (gp(u[i,j],ay) - gp(u[i,j-1],ay))**2
            b_lp_1 = (gp(u[i,j-1],ay) - gp(u[i,j-2],ay))**2
            a_lp_0 = d0/((eps + b_lp_0)**2)
            a_lp_1 = d1/((eps + b_lp_1)**2)
            w_lp_0 = a_lp_0/(a_lp_0 + a_lp_1)
            w_lp_1 = a_lp_1/(a_lp_0 + a_lp_1)
            gh_lp = w_lp_0*gh_lp_0 + w_lp_1*gh_lp_1
            # Negative flux
            gh_lm_0 = 0.5*gm(u[i,j-1],ay) + 0.5*gm(u[i,j],ay)
            gh_lm_1 = 1.5*gm(u[i,j],ay) - 0.5*gm(u[i,j+1],ay)
            b_lm_0 = (gm(u[i,j-1],ay) - gm(u[i,j],ay))**2
            b_lm_1 = (gm(u[i,j],ay) - gm(u[i,j+1],ay))**2
            a_lm_0 = d0/((eps + b_lm_0)**2)
            a_lm_1 = d1/((eps + b_lm_1)**2)
            w_lm_0 = a_lm_0/(a_lm_0 + a_lm_1)
            w_lm_1 = a_lm_1/(a_lm_0 + a_lm_1)
            gh_lm = w_lm_0*gh_lm_0 + w_lm_1*gh_lm_1
            gh_l = gh_lp + gh_lm # Final lower flux

            # Creating the upper flux (\hat{g}_{j+1/2})
            # Positive flux
            gh_rp_0 = 0.5*gp(u[i,j+1],ay) + 0.5*gp(u[i,j],ay)
            gh_rp_1 = 1.5*gp(u[i,j],ay) - 0.5*gp(u[i,j-1],ay)
            b_rp_0 = (gp(u[i,j+1],ay) - gp(u[i,j],ay))**2
            b_rp_1 = (gp(u[i,j],ay) - gp(u[i,j-1],ay))**2
            a_rp_0 = d0/((eps + b_rp_0)**2)
            a_rp_1 = d1/((eps + b_rp_1)**2)
            w_rp_0 = a_rp_0/(a_rp_0 + a_rp_1)
            w_rp_1 = a_rp_1/(a_rp_0 + a_rp_1)
            gh_rp = w_rp_0*gh_rp_0 + w_rp_1*gh_rp_1
            # Negative flux
            gh_rm_0 = 0.5*gm(u[i,j],ay) + 0.5*gm(u[i,j+1],ay)
            gh_rm_1 = 1.5*gm(u[i,j+1],ay) - 0.5*gm(u[i,j+2],ay)
            b_rm_0 = (gm(u[i,j],ay) - gm(u[i,j+1],ay))**2
            b_rm_1 = (gm(u[i,j+1],ay) - gm(u[i,j+2],ay))**2
            a_rm_0 = d0/((eps + b_rm_0)**2)
            a_rm_1 = d1/((eps + b_rm_1)**2)
            w_rm_0 = a_rm_0/(a_rm_0 + a_rm_1)
            w_rm_1 = a_rm_1/(a_rm_0 + a_rm_1)
            gh_rm = w_rm_0*gh_rm_0 + w_rm_1*gh_rm_1
            gh_r = gh_rp + gh_rm # Final upper flux

            u[i,j] = (-dy*((fh_r + (ax/2)*(u[i+1,j] - u[i,j])) - (fh_l + (ax/2)*(u[i,j] - u[i-1,j])) - (ax/2)*(u[i+1,j] + u[i-1,j])) - 
                      dx*((gh_r + (ay/2)*(u[i,j+1] - u[i,j])) - (gh_l + (ay/2)*(u[i,j] - u[i,j-1])) - (ay/2)*(u[i,j+1] + u[i,j-1])) + dx*dy*s(x[i],y[j]))/(dy*ax + dx*ay)
            
            ax_maybe = dfdu(u[i,j])
            if(ax_maybe > ax):
                ax = ax_maybe
            ay_maybe = dgdu(u[i,j])
            if(ay_maybe > ay):
                ay = ay_maybe

    return u,ax,ay

@njit
def sweep(fp,fm,dfdu,gp,gm,dgdu,s,u,x,y,dx,dy,n,m,ax,ay):
    d0 = 2/3
    d1 = 1/3
    eps = 1e-6
    I = np.arange(2,n-2)
    J = np.arange(2,m-2)
    u,ax,ay = sweep_helper(fp,fm,dfdu,gp,gm,dgdu,s,u,x,y,dx,dy,ax,ay,I,J,d0,d1,eps)
    u,ax,ay = sweep_helper(fp,fm,dfdu,gp,gm,dgdu,s,u,x,y,dx,dy,ax,ay,np.flip(I),J,d0,d1,eps)
    u,ax,ay = sweep_helper(fp,fm,dfdu,gp,gm,dgdu,s,u,x,y,dx,dy,ax,ay,I,np.flip(J),d0,d1,eps)
    u,ax,ay = sweep_helper(fp,fm,dfdu,gp,gm,dgdu,s,u,x,y,dx,dy,ax,ay,np.flip(I),np.flip(J),d0,d1,eps)
    return u,ax,ay

@njit
def weno3(fp,fm,dfdu,gp,gm,dgdu,s,x,y,u,dx,dy):
    n = x.size
    m = y.size

    ax = np.max(np.abs(dfdu(u[2:-2,2:-2])))
    ay = np.max(np.abs(dgdu(u[2:-2,2:-2])))

    u0 = u

    u1 = np.copy(u0)
    u1,ax,ay = sweep(fp,fm,dfdu,gp,gm,dgdu,s,u1,x,y,dx,dy,n,m,ax,ay) 

    u2 = np.copy(u1)
    u2,ax,ay = sweep(fp,fm,dfdu,gp,gm,dgdu,s,u2,x,y,dx,dy,n,m,ax,ay) 

    q1 = (np.linalg.norm(u2 - u1) + dx**3)/np.linalg.norm(u1 - u0)
    q2 = np.linalg.norm(u2 - u1) - dx**4
    iter = 0
    while((q1 < 1) or (q2 > 0)):
        u0 = np.copy(u1)
        u1 = np.copy(u2)
        u2,ax,ay = sweep(fp,fm,dfdu,gp,gm,dgdu,s,u2,x,y,dx,dy,n,m,ax,ay) 
        q1 = (np.linalg.norm(u2 - u1) + dx**3)/np.linalg.norm(u1 - u0)
        q2 = np.linalg.norm(u2 - u1) - dx**4
        iter += 1
    print(iter)
    return iter, u2


