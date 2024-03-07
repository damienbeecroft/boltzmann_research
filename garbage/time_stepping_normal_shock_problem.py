import numpy as np
import pickle as rick
from numba import njit
# from helpers import Qplus
from time_stepping_weno1_boltzmann import weno1_boltzmann

if __name__=="__main__":
    # This code replicates results from section 5.2 in 
    # "An adaptive dynamical low rank method for the nonlinear Boltzmann equation" by Hu and Wang

    # Variables to be set before running ########################################################
    save_me = True # Do you want to save the intermediate solution data for plotting?
    save_freq = 1 # Frequency at which the state of the system is saved
    C = 1 # Constant from rescaling of the loss collision term (not sure what this should be)
    Lx = 60 # The length of the spatial domain. The interval will be [-30,30]
    Nx = 100 # The resolution of the spatial grid
    Lv = 26.22 # Length of the velocity interval for both dimensions. The interval will be [-13.11,13.11]
    Nv = 32 # Number of collocation points in each dimension of the velocity domain
    Ntheta = 4 # Number of collocation points to integrate over for circle integral in collision operator
    v = 1 # Advection velocity of the equation
    R = 1; d = 2; gamma = 1; a = 0.5 # Refer to the paper for the significance of these variables
    ML = 1.4 # Mach number
    pL = 1 # Left density condition
    pR = 3 * ML**2 / (ML**2 + 2) # Right density condition
    p0 = lambda x: (np.tanh(a * x) + 1)/(2*(pR - pL)) + pL # Initial density profile
    uL = np.sqrt(2) * ML # Left bulk x-velocity condition
    uR = (pL * uL)/pR # Right bulk x-velocity condition
    u0 = lambda x: (np.tanh(a * x) + 1)/(2*(uR - uL)) + uL # Initial bulk x-velocity distribution
    # u0 = lambda x: np.array([(np.tanh(a * x) + 1)/(2*(uR - uL)) + uL],[np.zeros(x.shape)]) # Initial bulk x-velocity distribution
    TL = 1 # Left temperature condition
    TR = (4 * ML**2 - 1)/(3 * pR) # Right temperature condition
    T0 = lambda x: (np.tanh(a * x) + 1)/(2*(TR - TL)) + TL # Initial temperature profile
    f0 = lambda X, V1, V2: p0(X) * np.exp(-((V1 - u0(X))**2 + V2**2)/(2* R * T0(X))) / ((2 * np.pi * R * T0(X))**(d/2)) # Initial particle phase space distribution
    #############################################################################################

    # Initialize grids
    xb = np.linspace(-Lx/2,Lx/2,Nx) # spatial cell boundaries
    dx = xb[1] - xb[0]
    x_grid = np.concatenate(([-dx/2 + xb[0]], xb + (dx/2))) # spatial cell centers
    n = x_grid.size
    v_grid = np.linspace(-Lv/2,Lv/2,Nv)
    dt = (0.5 * dx)/(Lv/2)
    X, V1, V2 = np.meshgrid(x_grid,v_grid,v_grid,indexing='ij')

    p = p0(x_grid) # initial density on grid
    u1 = u0(x_grid) # initial bulk x-velocity on grid
    u2 = np.zeros(x_grid.shape) # initial bulk y-velocity on grid
    T = T0(x_grid) # initial temperature on grid
    f = f0(X,V1,V2)

    save_data = weno1_boltzmann(f,p,u1,u2,T)
    # save_data = weno1_boltzmann(f,p,u1,u2,T,pL,pR,uL,uR,TL,TR,v,C,n,dt,dx,Nv,R,Lv,Ntheta,V1,V2,v_grid,x_grid,save_me,save_freq)

    if save_me:
        with open('normal_shock_problem_data.pkl', 'wb') as foot:
            rick.dump(save_data, foot)






# Lx = np.array([-30,30]) # Spatial domain interval
# Lv = np.array([[-13.11,13.11],[-13.11,13.11]]) # Velocity domain interval
# Nv = np.array([32,32]) # The resolution of the velocity grid


# # Plot the matrix
# plt.imshow(f[:,Nx//2,:], cmap='viridis', interpolation='nearest')

# # Adding a colorbar to show the mapping from values to colors
# plt.colorbar()

# # Adding titles and labels if needed
# plt.title('Matrix Visualization')
# plt.xlabel('X Axis')
# plt.ylabel('Y Axis')

# # Show the plot
# plt.show()


    # fL = lambda v: pL * np.exp(-((v[0] - uL)**2 + v[1]**2)/(2* R * TL)) / ((2 * np.pi * R * TL)**(d/2))
    # fR = lambda v: pR * np.exp(-((v[0] - uR)**2 + v[1]**2)/(2* R * TR)) / ((2 * np.pi * R * TR)**(d/2))

































    #############################################################################################
    # N = initial_resolution*2**v + 1
    # s = njit(lambda x: np.sin(x)*np.cos(x))
    # f = njit(lambda u: u**2/2)
    # dfdu = njit(lambda u: u)


    # histories = {}
    # conditions = {}
    # xs = {}
    # sols = {}
    # orders_1 = np.zeros(v.shape)
    # orders_inf = np.zeros(v.shape)
    # L1s = np.zeros(v.shape)
    # Linfs = np.zeros(v.shape)
    # iters = np.zeros(v.shape)

    # L1_prev = np.nan
    # Linf_prev = np.nan
    # errors = {}
    # i = 0
    # for Nv in N:
    #     xb = np.linspace(0,np.pi,Nv)
    #     ub = b*np.sin(xb)
    #     dx = xb[1] - xb[0]
    #     x = np.concatenate(([-3*dx/2, -dx/2], xb + (dx/2), [xb[-1] + (3*dx/2)]))
    #     u = np.concatenate((np.sin(x[:2]),b*np.sin(x[2:-2]),np.sin(x[-2:])))
    #     iter, approx, history, condition_history = w3.weno3(fp,fm,dfdu,s,x,u,dx,save_freq)
    #     histories[f'{Nv}'] = history
    #     conditions[f'{Nv}'] = condition_history
    #     xs[f'{Nv}'] = x[2:-2]
    #     sol = np.sin(x[2:-2])
    #     sols[f'{Nv}'] = sol
    #     error = np.abs(sol - approx[2:-2])
    #     L1 = np.sum(error)/(Nv-1)
    #     Linf = np.max(error)
    #     orders_1[i] = np.log2(L1_prev/L1)
    #     orders_inf[i] = np.log2(Linf_prev/Linf)
    #     L1_prev = L1
    #     Linf_prev = Linf
    #     iters[i] = iter
    #     L1s[i] = L1
    #     Linfs[i] = Linf
    #     errors[f'{Nv}'] = error
    #     print(Nv)
    #     # plt.plot(x[2:-2],sol[2:-2],'k-',label='Approximation')
    #     # plt.plot(x[2:-2],np.sin(x[2:-2]),'r--',label='Branch 1')
    #     # plt.plot(x[2:-2],-np.sin(x[2:-2]),'b--',label='Branch 2')
    #     # plt.title(f'Third Order WENO with Lax-Friedrichs Iteration {iter}')
    #     # plt.legend()
    #     # # plt.savefig(f'output_weno3_lf/plot_{iter}')
    #     # plt.show()
    #     # plt.close()
    #     i+=1

    # if save_me:
    #     with open('', 'wb') as foot:
    #         pickle.dump([sols,xs,histories,conditions], foot)
        
    # if table_me:
    #     with open('', 'wb') as foo:
    #         pickle.dump([errors,L1s,orders_1,Linfs,orders_inf,iters,N], foo)
    
    