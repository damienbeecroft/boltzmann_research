import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import pickle
import weno3_1d as w3

if __name__=="__main__":
    s = njit(lambda x: np.sin(x)*np.cos(x))
    f = njit(lambda u: u**2/2)
    dfdu = njit(lambda u: u)

    @njit
    def fp(u,a): # positive flux splitting
        return 0.5*(f(u) + a*u)

    @njit
    def fm(u,a): # negative flux splitting
        return 0.5*(f(u) - a*u)

    b = 2

    v = np.arange(7)
    orders_1 = np.zeros(v.shape)
    orders_inf = np.zeros(v.shape)
    L1s = np.zeros(v.shape)
    Linfs = np.zeros(v.shape)
    iters = np.zeros(v.shape)

    N = 40*2**v + 1

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
        iter, sol = w3.weno3(fp,fm,dfdu,s,x,u,dx)
        error = np.abs(np.sin(x[2:-2]) - sol[2:-2])
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
        
    with open('5_1_1.pkl', 'wb') as foo:
        pickle.dump([errors,L1s,orders_1,Linfs,orders_inf,iters,N], foo)
    
    