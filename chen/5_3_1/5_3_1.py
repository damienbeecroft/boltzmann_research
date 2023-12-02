import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import pickle
import weno3_2d as w3

if __name__=="__main__":
    s = njit(lambda x,y: np.sin((x+y)/np.sqrt(2))*np.cos((x+y)/np.sqrt(2)))
    f = njit(lambda u: u**2/(2*np.sqrt(2)))
    g = njit(lambda u: u**2/(2*np.sqrt(2)))
    dfdu = njit(lambda u: u/np.sqrt(2))
    dgdu = njit(lambda u: u/np.sqrt(2))

    @njit
    def fp(u,a): # positive flux splitting
        return 0.5*(f(u) + a*u)

    @njit
    def fm(u,a): # negative flux splitting
        return 0.5*(f(u) - a*u)

    @njit
    def gp(u,a): # positive flux splitting
        return 0.5*(g(u) + a*u)

    @njit
    def gm(u,a): # negative flux splitting
        return 0.5*(g(u) - a*u)
    
    b = 1.5

    v = np.arange(6)
    orders_1 = np.zeros(v.shape)
    orders_inf = np.zeros(v.shape)
    L1s = np.zeros(v.shape)
    Linfs = np.zeros(v.shape)
    iters = np.zeros(v.shape)

    N = 20*2**v + 1

    L1_prev = np.nan
    Linf_prev = np.nan
    errors = {}
    i = 0
    for n in N:
        # xb = np.array([[[x,y] for x in np.linspace(0,np.pi/np.sqrt(2),n)] for y in np.linspace(0,np.pi/np.sqrt(2),n)])
        xb = np.linspace(0,np.pi/np.sqrt(2),n)
        yb = np.linspace(0,np.pi/np.sqrt(2),n)
        dx = xb[1] - xb[0]
        dy = yb[1] - yb[0]
        x = np.concatenate(([-3*dx/2, -dx/2], xb + (dx/2), [xb[-1] + (3*dx/2)]))
        y = np.concatenate(([-3*dy/2, -dy/2], yb + (dy/2), [yb[-1] + (3*dy/2)]))
        coords =  np.array([[[x_,y_] for x_ in x] for y_ in y])
        u = np.array([[np.sin((x_ + y_)/np.sqrt(2)) for x_ in x] for y_ in y])
        u[2:-2,2:-2] = np.array([[b*np.sin((x_ + y_)/np.sqrt(2)) for x_ in x[2:-2]] for y_ in y[2:-2]])
        iter, sol = w3.weno3(fp,fm,dfdu,gp,gm,dgdu,s,x,y,u,dx,dy)
        true = np.sin((coords[2:-2,2:-2,0] + coords[2:-2,2:-2,1])/np.sqrt(2))
        error = np.abs(true - sol[2:-2,2:-2])
        L1 = np.sum(error)/((n-1)**2)
        Linf = np.max(error)
        orders_1[i] = np.log2(L1_prev/L1)
        orders_inf[i] = np.log2(Linf_prev/Linf)
        L1_prev = L1
        Linf_prev = Linf
        iters[i] = iter
        L1s[i] = L1
        Linfs[i] = Linf
        errors[f'{n}'] = error
        # print(n)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # true = np.sin((coords[2:-2,2:-2,0] + coords[2:-2,2:-2,1])/np.sqrt(2))
        # ax.plot_surface(coords[2:-2,2:-2,0], coords[2:-2,2:-2,1], sol[2:-2,2:-2], label='Approximation')
        # ax.plot_surface(coords[2:-2,2:-2,0], coords[2:-2,2:-2,1], true, label='True Solution')
        # plt.show()
        # plt.close()
        i+=1
        
    with open('./chen/5_3_1/5_3_1.pkl', 'wb') as foo:
        pickle.dump([errors,L1s,orders_1,Linfs,orders_inf,iters,N], foo)
    
    