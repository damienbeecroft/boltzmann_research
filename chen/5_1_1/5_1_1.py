import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import pickle
import weno3_1d as w3


if __name__=="__main__":

    # Variables to be set before running ########################################################
    save_freq = 10 # Frequency at which the state of the system is saved
    save_me = True # Do you want to save the intermediate solution data for plotting?
    table_me = False # Do you want to save the relevant data and make a table?
    v = np.arange(4) # The number of different resolutions to resolve
    initial_resolution = 80 # The resolution of the grid
    b = 2 # Determines the coefficient of the initial condition
    #############################################################################################

    N = initial_resolution*2**v + 1
    s = njit(lambda x: np.sin(x)*np.cos(x))
    f = njit(lambda u: u**2/2)
    dfdu = njit(lambda u: u)

    @njit
    def fp(u,a): # positive flux splitting
        return 0.5*(f(u) + a*u)

    @njit
    def fm(u,a): # negative flux splitting
        return 0.5*(f(u) - a*u)

    histories = {}
    conditions = {}
    xs = {}
    sols = {}
    orders_1 = np.zeros(v.shape)
    orders_inf = np.zeros(v.shape)
    L1s = np.zeros(v.shape)
    Linfs = np.zeros(v.shape)
    iters = np.zeros(v.shape)

    L1_prev = np.nan
    Linf_prev = np.nan
    errors = {}
    i = 0
    for n in N:
        xb = np.linspace(0,np.pi,n)
        ub = b*np.sin(xb)
        dx = xb[1] - xb[0]
        x = np.concatenate(([-3*dx/2, -dx/2], xb + (dx/2), [xb[-1] + (3*dx/2)]))
        u = np.concatenate((np.sin(x[:2]),b*np.sin(x[2:-2]),np.sin(x[-2:])))
        iter, approx, history, condition_history = w3.weno3(fp,fm,dfdu,s,x,u,dx,save_freq)
        histories[f'{n}'] = history
        conditions[f'{n}'] = condition_history
        xs[f'{n}'] = x[2:-2]
        sol = np.sin(x[2:-2])
        sols[f'{n}'] = sol
        error = np.abs(sol - approx[2:-2])
        L1 = np.sum(error)/(n-1)
        Linf = np.max(error)
        orders_1[i] = np.log2(L1_prev/L1)
        orders_inf[i] = np.log2(Linf_prev/Linf)
        L1_prev = L1
        Linf_prev = Linf
        iters[i] = iter
        L1s[i] = L1
        Linfs[i] = Linf
        errors[f'{n}'] = error
        print(n)
        # plt.plot(x[2:-2],sol[2:-2],'k-',label='Approximation')
        # plt.plot(x[2:-2],np.sin(x[2:-2]),'r--',label='Branch 1')
        # plt.plot(x[2:-2],-np.sin(x[2:-2]),'b--',label='Branch 2')
        # plt.title(f'Third Order WENO with Lax-Friedrichs Iteration {iter}')
        # plt.legend()
        # # plt.savefig(f'output_weno3_lf/plot_{iter}')
        # plt.show()
        # plt.close()
        i+=1

    if save_me:
        with open('C:/Users/damie/OneDrive/UW/research/jingwei/chen/5_1_1/plotting/5_1_1_beta_2.pkl', 'wb') as foot:
            pickle.dump([sols,xs,histories,conditions], foot)
        
    if table_me:
        with open('5_1_1.pkl', 'wb') as foo:
            pickle.dump([errors,L1s,orders_1,Linfs,orders_inf,iters,N], foo)
    
    