import numpy as np
import matplotlib.pyplot as plt
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

# @njit
def weno3(fp,fm,dfdu,s,x,u,dx,save_freq):
    history = {} # saves the full results at specified times
    conditions = {} # saves the history of the stoppoing conditions 
    iter = 0 # start iteration counter
    order = 3
    n = x.size
    history[f'{iter}'] = np.copy(u[2:-2])
    a = np.max(np.abs(dfdu(u[2:-2])))

    u0 = u
    u1 = np.copy(u0)
    u1,a = sweep(fp,fm,dfdu,a,s,u1,dx,x,n) 
    iter += 1
    history[f'{iter}'] = np.copy(u1[2:-2])

    u2 = np.copy(u1)
    u2,a = sweep(fp,fm,dfdu,a,s,u2,dx,x,n) 
    iter += 1
    history[f'{iter}'] = np.copy(u2[2:-2])

    q1 = (np.linalg.norm(u2 - u1) + dx**order)/np.linalg.norm(u1 - u0) # this works better than the original paper
    # q2 = np.linalg.norm(u2 - u1) - dx**(order + 1) # from the original paper
    q2 = np.linalg.norm(u2 - u1) - dx**(order)
    while((q1 < 1) or (q2 > 0)):
        u0 = np.copy(u1)
        u1 = np.copy(u2)
        u2,a = sweep(fp,fm,dfdu,a,s,u2,dx,x,n) 
        q1 = (np.linalg.norm(u2 - u1) + dx**order)/np.linalg.norm(u1 - u0) # this works better than the original paper
        # q2 = np.linalg.norm(u2 - u1) - dx**(order + 1) # from the original paper
        q2 = np.linalg.norm(u2 - u1) - dx**(order)
        iter += 1
        if iter % save_freq == 0:
            history[f'{iter}'] = np.copy(u2[2:-2])
            conditions[f'{iter}'] = np.copy([q1, q2])
            # ###### Printing
            # print(conditions[f'{iter}'])
            # ###### Plotting
            # plt.plot(x[2:-2],u2[2:-2],'k-',label='Approximation')
            # plt.plot(x[2:-2],np.sin(x[2:-2]),'r--',label='Branch 1')
            # plt.plot(x[2:-2],-np.sin(x[2:-2]),'b--',label='Branch 2')
            # plt.title(f'Third Order WENO with Lax-Friedrichs Iteration {iter}')
            # plt.legend()
            # # plt.savefig(f'output_weno3_lf/plot_{iter}')
            # plt.show()
            # plt.close()
        
    history[f'{iter}'] = u2[2:-2]
    conditions[f'{iter}'] = [q1, q2]
    return iter, u2, history, conditions

######################################################################################################

# import numpy as np
# import matplotlib.pyplot as plt
# from numba import njit

# @njit
# def sweep(fp,fm,dfdu,a,s,u,dx,x,n):
#     d0 = 2/3
#     d1 = 1/3
#     eps = 1e-6
    
#     for i in range(2,n-2):
#         # Creating the left flux (\hat{f}_{j-1/2})
#         # Positive flux
#         fh_lp_0 = 0.5*fp(u[i],a) + 0.5*fp(u[i-1],a)
#         fh_lp_1 = 1.5*fp(u[i-1],a) - 0.5*fp(u[i-2],a)
#         b_lp_0 = (fp(u[i],a) - fp(u[i-1],a))**2
#         b_lp_1 = (fp(u[i-1],a) - fp(u[i-2],a))**2
#         a_lp_0 = d0/((eps + b_lp_0)**2)
#         a_lp_1 = d1/((eps + b_lp_1)**2)
#         w_lp_0 = a_lp_0/(a_lp_0 + a_lp_1)
#         w_lp_1 = a_lp_1/(a_lp_0 + a_lp_1)
#         fh_lp = w_lp_0*fh_lp_0 + w_lp_1*fh_lp_1
#         # Negative flux
#         fh_lm_0 = 0.5*fm(u[i-1],a) + 0.5*fm(u[i],a)
#         fh_lm_1 = 1.5*fm(u[i],a) - 0.5*fm(u[i+1],a)
#         b_lm_0 = (fm(u[i-1],a) - fm(u[i],a))**2
#         b_lm_1 = (fm(u[i],a) - fm(u[i+1],a))**2
#         a_lm_0 = d0/((eps + b_lm_0)**2)
#         a_lm_1 = d1/((eps + b_lm_1)**2)
#         w_lm_0 = a_lm_0/(a_lm_0 + a_lm_1)
#         w_lm_1 = a_lm_1/(a_lm_0 + a_lm_1)
#         fh_lm = w_lm_0*fh_lm_0 + w_lm_1*fh_lm_1
#         fh_l = fh_lp + fh_lm # Final left flux
#         # Creating the right flux (\hat{f}_{j+1/2})
#         # Positive flux
#         fh_rp_0 = 0.5*fp(u[i+1],a) + 0.5*fp(u[i],a)
#         fh_rp_1 = 1.5*fp(u[i],a) - 0.5*fp(u[i-1],a)
#         b_rp_0 = (fp(u[i+1],a) - fp(u[i],a))**2
#         b_rp_1 = (fp(u[i],a) - fp(u[i-1],a))**2
#         a_rp_0 = d0/((eps + b_rp_0)**2)
#         a_rp_1 = d1/((eps + b_rp_1)**2)
#         w_rp_0 = a_rp_0/(a_rp_0 + a_rp_1)
#         w_rp_1 = a_rp_1/(a_rp_0 + a_rp_1)
#         fh_rp = w_rp_0*fh_rp_0 + w_rp_1*fh_rp_1
#         # Negative flux
#         fh_rm_0 = 0.5*fm(u[i],a) + 0.5*fm(u[i+1],a)
#         fh_rm_1 = 1.5*fm(u[i+1],a) - 0.5*fm(u[i+2],a)
#         b_rm_0 = (fm(u[i],a) - fm(u[i+1],a))**2
#         b_rm_1 = (fm(u[i+1],a) - fm(u[i+2],a))**2
#         a_rm_0 = d0/((eps + b_rm_0)**2)
#         a_rm_1 = d1/((eps + b_rm_1)**2)
#         w_rm_0 = a_rm_0/(a_rm_0 + a_rm_1)
#         w_rm_1 = a_rm_1/(a_rm_0 + a_rm_1)
#         fh_rm = w_rm_0*fh_rm_0 + w_rm_1*fh_rm_1
#         fh_r = fh_rp + fh_rm # Final right flux
#         u[i] = 0.5*(u[i-1] + u[i+1]) + (dx*s(x[i]) - ((fh_r + a*(u[i+1] - u[i])/2) - (fh_l + a*(u[i] - u[i-1])/2)))/a
#         a_maybe = dfdu(u[i])
#         if(a_maybe > a):
#             a = a_maybe

#     # Backward Sweep
#     for i in range(n-3,1,-1):
#         # Creating the left flux (\hat{f}_{j-1/2})
#         # Positive flux
#         fh_lp_0 = 0.5*fp(u[i],a) + 0.5*fp(u[i-1],a)
#         fh_lp_1 = 1.5*fp(u[i-1],a) - 0.5*fp(u[i-2],a)
#         b_lp_0 = (fp(u[i],a) - fp(u[i-1],a))**2
#         b_lp_1 = (fp(u[i-1],a) - fp(u[i-2],a))**2
#         a_lp_0 = d0/((eps + b_lp_0)**2)
#         a_lp_1 = d1/((eps + b_lp_1)**2)
#         w_lp_0 = a_lp_0/(a_lp_0 + a_lp_1)
#         w_lp_1 = a_lp_1/(a_lp_0 + a_lp_1)
#         fh_lp = w_lp_0*fh_lp_0 + w_lp_1*fh_lp_1
#         # Negative flux
#         fh_lm_0 = 0.5*fm(u[i-1],a) + 0.5*fm(u[i],a)
#         fh_lm_1 = 1.5*fm(u[i],a) - 0.5*fm(u[i+1],a)
#         b_lm_0 = (fm(u[i-1],a) - fm(u[i],a))**2
#         b_lm_1 = (fm(u[i],a) - fm(u[i+1],a))**2
#         a_lm_0 = d0/((eps + b_lm_0)**2)
#         a_lm_1 = d1/((eps + b_lm_1)**2)
#         w_lm_0 = a_lm_0/(a_lm_0 + a_lm_1)
#         w_lm_1 = a_lm_1/(a_lm_0 + a_lm_1)
#         fh_lm = w_lm_0*fh_lm_0 + w_lm_1*fh_lm_1
#         fh_l = fh_lp + fh_lm # Final left flux
#         # Creating the right flux (\hat{f}_{j+1/2})
#         # Positive flux
#         fh_rp_0 = 0.5*fp(u[i+1],a) + 0.5*fp(u[i],a)
#         fh_rp_1 = 1.5*fp(u[i],a) - 0.5*fp(u[i-1],a)
#         b_rp_0 = (fp(u[i+1],a) - fp(u[i],a))**2
#         b_rp_1 = (fp(u[i],a) - fp(u[i-1],a))**2
#         a_rp_0 = d0/((eps + b_rp_0)**2)
#         a_rp_1 = d1/((eps + b_rp_1)**2)
#         w_rp_0 = a_rp_0/(a_rp_0 + a_rp_1)
#         w_rp_1 = a_rp_1/(a_rp_0 + a_rp_1)
#         fh_rp = w_rp_0*fh_rp_0 + w_rp_1*fh_rp_1
#         # Negative flux
#         fh_rm_0 = 0.5*fm(u[i],a) + 0.5*fm(u[i+1],a)
#         fh_rm_1 = 1.5*fm(u[i+1],a) - 0.5*fm(u[i+2],a)
#         b_rm_0 = (fm(u[i],a) - fm(u[i+1],a))**2
#         b_rm_1 = (fm(u[i+1],a) - fm(u[i+2],a))**2
#         a_rm_0 = d0/((eps + b_rm_0)**2)
#         a_rm_1 = d1/((eps + b_rm_1)**2)
#         w_rm_0 = a_rm_0/(a_rm_0 + a_rm_1)
#         w_rm_1 = a_rm_1/(a_rm_0 + a_rm_1)
#         fh_rm = w_rm_0*fh_rm_0 + w_rm_1*fh_rm_1
#         fh_r = fh_rp + fh_rm # Final right flux
#         u[i] = 0.5*(u[i-1] + u[i+1]) + (dx*s(x[i]) - ((fh_r + a*(u[i+1] - u[i])/2) - (fh_l + a*(u[i] - u[i-1])/2)))/a
#         a_maybe = dfdu(u[i])
#         if(a_maybe > a):
#             a = a_maybe
#     return u,a

# # @njit
# def weno3(fp,fm,dfdu,s,x,u,dx,save_freq):
#     history = {} # saves the full results at specified times
#     # L1_errors = []
#     iter = 0
#     order = 3
#     n = x.size
#     history[f'{iter}'] = u[2:-2]
#     a = np.max(np.abs(dfdu(u[2:-2])))

#     u0 = u
#     plt.plot(x,u0)
#     plt.plot(x,np.sin(x))
#     plt.title(f'{iter}')
#     plt.show()
#     u1 = np.copy(u0)
#     u1,a = sweep(fp,fm,dfdu,a,s,u1,dx,x,n) 
#     iter += 1
#     plt.plot(x,u1)
#     plt.plot(x,np.sin(x))
#     plt.title(f'{iter}')
#     plt.show()

#     u2 = np.copy(u1)
#     u2,a = sweep(fp,fm,dfdu,a,s,u2,dx,x,n) 
#     iter += 1
#     plt.plot(x,u2)
#     plt.plot(x,np.sin(x))
#     plt.title(f'{iter}')
#     plt.show()

#     q1 = (np.linalg.norm(u2 - u1) + dx**order)/np.linalg.norm(u1 - u0)
#     q2 = np.linalg.norm(u2 - u1) - dx**(order + 1)
#     while((q1 < 1) or (q2 > 0)):
#         u0 = np.copy(u1)
#         u1 = np.copy(u2)
#         u2,a = sweep(fp,fm,dfdu,a,s,u2,dx,x,n) 
#         q1 = (np.linalg.norm(u2 - u1) + dx**order)/np.linalg.norm(u1 - u0)
#         q2 = np.linalg.norm(u2 - u1) - dx**(order + 1)
#         iter += 1
#         if iter % save_freq == 0:
#             history[f'{iter}'] = u2[2:-2]
#             plt.plot(x,u2)
#             plt.plot(x,np.sin(x))
#             plt.title(f'{iter}')
#             plt.show()
        
#     history[f'{iter}'] = u2[2:-2]
#     return iter, u2, history

###########################################################################################################

# import numpy as np
# import matplotlib.pyplot as plt
# from numba import njit
# # import warnings
# # warnings.filterwarnings('error')
# # from IPython.display import clear_output
# # @njit
# def sweep_right_initial(fp,fm,dfdu,a,s,up,dx,x,n,iter):
#     d0 = 2/3
#     d1 = 1/3
#     eps = 1e-6
#     u = np.copy(up)
#     for i in range(2,n-2):
#         # Creating the left flux (\hat{f}_{j-1/2})
#         # Positive flux
#         fh_lp_0 = 0.5*fp(u[i],a) + 0.5*fp(u[i-1],a)
#         fh_lp_1 = 1.5*fp(u[i-1],a) - 0.5*fp(u[i-2],a)
#         b_lp_0 = (fp(u[i],a) - fp(u[i-1],a))**2
#         b_lp_1 = (fp(u[i-1],a) - fp(u[i-2],a))**2
#         a_lp_0 = d0/((eps + b_lp_0)**2)
#         a_lp_1 = d1/((eps + b_lp_1)**2)
#         w_lp_0 = a_lp_0/(a_lp_0 + a_lp_1)
#         w_lp_1 = a_lp_1/(a_lp_0 + a_lp_1)
#         fh_lp = w_lp_0*fh_lp_0 + w_lp_1*fh_lp_1
#         # Negative flux
#         fh_lm_0 = 0.5*fm(u[i-1],a) + 0.5*fm(u[i],a)
#         fh_lm_1 = 1.5*fm(u[i],a) - 0.5*fm(u[i+1],a)
#         b_lm_0 = (fm(u[i-1],a) - fm(u[i],a))**2
#         b_lm_1 = (fm(u[i],a) - fm(u[i+1],a))**2
#         a_lm_0 = d0/((eps + b_lm_0)**2)
#         a_lm_1 = d1/((eps + b_lm_1)**2)
#         w_lm_0 = a_lm_0/(a_lm_0 + a_lm_1)
#         w_lm_1 = a_lm_1/(a_lm_0 + a_lm_1)
#         fh_lm = w_lm_0*fh_lm_0 + w_lm_1*fh_lm_1
#         fh_l = fh_lp + fh_lm # Final left flux
#         # Creating the right flux (\hat{f}_{j+1/2})
#         # Positive flux
#         fh_rp_0 = 0.5*fp(u[i+1],a) + 0.5*fp(u[i],a)
#         fh_rp_1 = 1.5*fp(u[i],a) - 0.5*fp(u[i-1],a)
#         b_rp_0 = (fp(u[i+1],a) - fp(u[i],a))**2
#         b_rp_1 = (fp(u[i],a) - fp(u[i-1],a))**2
#         a_rp_0 = d0/((eps + b_rp_0)**2)
#         a_rp_1 = d1/((eps + b_rp_1)**2)
#         w_rp_0 = a_rp_0/(a_rp_0 + a_rp_1)
#         w_rp_1 = a_rp_1/(a_rp_0 + a_rp_1)
#         fh_rp = w_rp_0*fh_rp_0 + w_rp_1*fh_rp_1
#         # Negative flux
#         fh_rm_0 = 0.5*fm(u[i],a) + 0.5*fm(u[i+1],a)
#         fh_rm_1 = 1.5*fm(u[i+1],a) - 0.5*fm(u[i+2],a)
#         b_rm_0 = (fm(u[i],a) - fm(u[i+1],a))**2
#         b_rm_1 = (fm(u[i+1],a) - fm(u[i+2],a))**2
#         a_rm_0 = d0/((eps + b_rm_0)**2)
#         a_rm_1 = d1/((eps + b_rm_1)**2)
#         w_rm_0 = a_rm_0/(a_rm_0 + a_rm_1)
#         w_rm_1 = a_rm_1/(a_rm_0 + a_rm_1)
#         fh_rm = w_rm_0*fh_rm_0 + w_rm_1*fh_rm_1
#         fh_r = fh_rp + fh_rm # Final right flux
#         u[i] = 0.5*(u[i-1] + u[i+1]) + (dx*s(x[i]) - ((fh_r + a*(u[i+1] - u[i])/2) - (fh_l + a*(u[i] - u[i-1])/2)))/a
#         iter += 1
#         a_maybe = dfdu(u[i])
#         if(a_maybe > a):
#             a = a_maybe
#     return u,a,iter

# # @njit
# def sweep_right(fp,fm,dfdu,a,s,up,upp,dx,x,n,order,iter,flag):
#     d0 = 2/3
#     d1 = 1/3
#     eps = 1e-6
#     u = np.copy(up)
#     # Forward Sweep
#     for i in range(2,n-2):
#         # Creating the left flux (\hat{f}_{j-1/2})
#         # Positive flux
#         fh_lp_0 = 0.5*fp(u[i],a) + 0.5*fp(u[i-1],a)
#         fh_lp_1 = 1.5*fp(u[i-1],a) - 0.5*fp(u[i-2],a)
#         b_lp_0 = (fp(u[i],a) - fp(u[i-1],a))**2
#         b_lp_1 = (fp(u[i-1],a) - fp(u[i-2],a))**2
#         a_lp_0 = d0/((eps + b_lp_0)**2)
#         a_lp_1 = d1/((eps + b_lp_1)**2)
#         w_lp_0 = a_lp_0/(a_lp_0 + a_lp_1)
#         w_lp_1 = a_lp_1/(a_lp_0 + a_lp_1)
#         fh_lp = w_lp_0*fh_lp_0 + w_lp_1*fh_lp_1
#         # Negative flux
#         fh_lm_0 = 0.5*fm(u[i-1],a) + 0.5*fm(u[i],a)
#         fh_lm_1 = 1.5*fm(u[i],a) - 0.5*fm(u[i+1],a)
#         b_lm_0 = (fm(u[i-1],a) - fm(u[i],a))**2
#         b_lm_1 = (fm(u[i],a) - fm(u[i+1],a))**2
#         a_lm_0 = d0/((eps + b_lm_0)**2)
#         a_lm_1 = d1/((eps + b_lm_1)**2)
#         w_lm_0 = a_lm_0/(a_lm_0 + a_lm_1)
#         w_lm_1 = a_lm_1/(a_lm_0 + a_lm_1)
#         fh_lm = w_lm_0*fh_lm_0 + w_lm_1*fh_lm_1
#         fh_l = fh_lp + fh_lm # Final left flux
#         # Creating the right flux (\hat{f}_{j+1/2})
#         # Positive flux
#         fh_rp_0 = 0.5*fp(u[i+1],a) + 0.5*fp(u[i],a)
#         fh_rp_1 = 1.5*fp(u[i],a) - 0.5*fp(u[i-1],a)
#         b_rp_0 = (fp(u[i+1],a) - fp(u[i],a))**2
#         b_rp_1 = (fp(u[i],a) - fp(u[i-1],a))**2
#         a_rp_0 = d0/((eps + b_rp_0)**2)
#         a_rp_1 = d1/((eps + b_rp_1)**2)
#         w_rp_0 = a_rp_0/(a_rp_0 + a_rp_1)
#         w_rp_1 = a_rp_1/(a_rp_0 + a_rp_1)
#         fh_rp = w_rp_0*fh_rp_0 + w_rp_1*fh_rp_1
#         # Negative flux
#         fh_rm_0 = 0.5*fm(u[i],a) + 0.5*fm(u[i+1],a)
#         fh_rm_1 = 1.5*fm(u[i+1],a) - 0.5*fm(u[i+2],a)
#         b_rm_0 = (fm(u[i],a) - fm(u[i+1],a))**2
#         b_rm_1 = (fm(u[i+1],a) - fm(u[i+2],a))**2
#         a_rm_0 = d0/((eps + b_rm_0)**2)
#         a_rm_1 = d1/((eps + b_rm_1)**2)
#         w_rm_0 = a_rm_0/(a_rm_0 + a_rm_1)
#         w_rm_1 = a_rm_1/(a_rm_0 + a_rm_1)
#         fh_rm = w_rm_0*fh_rm_0 + w_rm_1*fh_rm_1
#         fh_r = fh_rp + fh_rm # Final right flux
#         u[i] = 0.5*(u[i-1] + u[i+1]) + (dx*s(x[i]) - ((fh_r + a*(u[i+1] - u[i])/2) - (fh_l + a*(u[i] - u[i-1])/2)))/a
#         iter += 1
#         a_maybe = dfdu(u[i])
#         if(a_maybe > a):
#             a = a_maybe
#         q1 = (np.abs(u[i] - up[i]) + dx**order)/np.abs(up[i] - upp[i])
#         q2 = np.abs(u[i] - up[i]) - dx**(order + 1)
#         if ((q1 > 1) and (q2 < 0)):
#             flag == False
#             break
#     return u,a,iter,flag
    
# # @njit
# def sweep_left(fp,fm,dfdu,a,s,up,upp,dx,x,n,order,iter,flag):
#     d0 = 2/3
#     d1 = 1/3
#     eps = 1e-6
#     u = np.copy(up)
#     # Backward Sweep
#     for i in range(n-3,1,-1):
#         # Creating the left flux (\hat{f}_{j-1/2})
#         # Positive flux
#         fh_lp_0 = 0.5*fp(u[i],a) + 0.5*fp(u[i-1],a)
#         fh_lp_1 = 1.5*fp(u[i-1],a) - 0.5*fp(u[i-2],a)
#         b_lp_0 = (fp(u[i],a) - fp(u[i-1],a))**2
#         b_lp_1 = (fp(u[i-1],a) - fp(u[i-2],a))**2
#         a_lp_0 = d0/((eps + b_lp_0)**2)
#         a_lp_1 = d1/((eps + b_lp_1)**2)
#         w_lp_0 = a_lp_0/(a_lp_0 + a_lp_1)
#         w_lp_1 = a_lp_1/(a_lp_0 + a_lp_1)
#         fh_lp = w_lp_0*fh_lp_0 + w_lp_1*fh_lp_1
#         # Negative flux
#         fh_lm_0 = 0.5*fm(u[i-1],a) + 0.5*fm(u[i],a)
#         fh_lm_1 = 1.5*fm(u[i],a) - 0.5*fm(u[i+1],a)
#         b_lm_0 = (fm(u[i-1],a) - fm(u[i],a))**2
#         b_lm_1 = (fm(u[i],a) - fm(u[i+1],a))**2
#         a_lm_0 = d0/((eps + b_lm_0)**2)
#         a_lm_1 = d1/((eps + b_lm_1)**2)
#         w_lm_0 = a_lm_0/(a_lm_0 + a_lm_1)
#         w_lm_1 = a_lm_1/(a_lm_0 + a_lm_1)
#         fh_lm = w_lm_0*fh_lm_0 + w_lm_1*fh_lm_1
#         fh_l = fh_lp + fh_lm # Final left flux
#         # Creating the right flux (\hat{f}_{j+1/2})
#         # Positive flux
#         fh_rp_0 = 0.5*fp(u[i+1],a) + 0.5*fp(u[i],a)
#         fh_rp_1 = 1.5*fp(u[i],a) - 0.5*fp(u[i-1],a)
#         b_rp_0 = (fp(u[i+1],a) - fp(u[i],a))**2
#         b_rp_1 = (fp(u[i],a) - fp(u[i-1],a))**2
#         a_rp_0 = d0/((eps + b_rp_0)**2)
#         a_rp_1 = d1/((eps + b_rp_1)**2)
#         w_rp_0 = a_rp_0/(a_rp_0 + a_rp_1)
#         w_rp_1 = a_rp_1/(a_rp_0 + a_rp_1)
#         fh_rp = w_rp_0*fh_rp_0 + w_rp_1*fh_rp_1
#         # Negative flux
#         fh_rm_0 = 0.5*fm(u[i],a) + 0.5*fm(u[i+1],a)
#         fh_rm_1 = 1.5*fm(u[i+1],a) - 0.5*fm(u[i+2],a)
#         b_rm_0 = (fm(u[i],a) - fm(u[i+1],a))**2
#         b_rm_1 = (fm(u[i+1],a) - fm(u[i+2],a))**2
#         a_rm_0 = d0/((eps + b_rm_0)**2)
#         a_rm_1 = d1/((eps + b_rm_1)**2)
#         w_rm_0 = a_rm_0/(a_rm_0 + a_rm_1)
#         w_rm_1 = a_rm_1/(a_rm_0 + a_rm_1)
#         fh_rm = w_rm_0*fh_rm_0 + w_rm_1*fh_rm_1
#         fh_r = fh_rp + fh_rm # Final right flux
#         u[i] = 0.5*(u[i-1] + u[i+1]) + (dx*s(x[i]) - ((fh_r + a*(u[i+1] - u[i])/2) - (fh_l + a*(u[i] - u[i-1])/2)))/a
#         iter += 1
#         a_maybe = dfdu(u[i])
#         if(a_maybe > a):
#             a = a_maybe
#         q1 = (np.abs(u[i] - up[i]) + dx**order)/np.abs(up[i] - upp[i])
#         q2 = np.abs(u[i] - up[i]) - dx**(order + 1)
#         if ((q1 > 1) and (q2 < 0)):
#             flag == False
#             break
#     return u,a,iter,flag

# # @njit
# def weno3(fp,fm,dfdu,s,x,u,dx,save_freq):

#     history = {} # saves the full results at specified times
#     # fig, ax = plt.subplots()
#     iter = 0
#     iter2 = 0
#     order = 3
#     n = x.size
#     history[f'{iter}'] = u[2:-2]
#     a = np.max(np.abs(dfdu(u[2:-2])))
#     flag = True
#     u0 = np.copy(u)

#     u1,a,iter = sweep_right_initial(fp,fm,dfdu,a,s,u0,dx,x,n,iter)
#     # history[f'{iter2}'] = u1[2:-2]
#     u2,a,iter,flag = sweep_left(fp,fm,dfdu,a,s,u1,u0,dx,x,n,order,iter,flag) 
#     # history[f'{iter2}'] = u2[2:-2]
#     iter2 += 1

#     plt.plot(x,u2)
#     # u0 = np.copy(u1)
#     # u1 = np.copy(u2)
#     # iter += 1
#     # u2,a = sweep_right(fp,fm,dfdu,a,s,u1,u0,dx,x,n,order) 
#     # history[f'{iter}'] = u2[2:-2]
#     # u0 = np.copy(u1)
#     # u1 = np.copy(u2)
#     # iter += 1
#     # u2,a = sweep_left(fp,fm,dfdu,a,s,u1,u0,dx,x,n,order) 
#     # history[f'{iter}'] = u2[2:-2]

#     # q1 = (np.linalg.norm(u2 - u1) + dx**order)/np.linalg.norm(u1 - u0)
#     # q2 = np.linalg.norm(u2 - u1) - dx**(order + 1)
#     while(flag):
#         u0 = np.copy(u1)
#         u1 = np.copy(u2)
#         u2,a,iter,flag = sweep_right(fp,fm,dfdu,a,s,u1,u0,dx,x,n,order,iter,flag)
#         if(flag):
#             u0 = np.copy(u1)
#             u1 = np.copy(u2)
#             u2,a,iter,flag = sweep_left(fp,fm,dfdu,a,s,u1,u0,dx,x,n,order,iter,flag)
#         iter2 += 1

#         if iter2 % save_freq == 0:
#             history[f'{iter2}'] = u2[2:-2]
#             plt.plot(x,u2)
#             plt.plot(x,np.sin(x))
#             plt.title(f'{iter2}')
#             plt.show()

        
#     history[f'{iter2}'] = u2[2:-2]
#     return iter, u2, history


############################################################################################################




