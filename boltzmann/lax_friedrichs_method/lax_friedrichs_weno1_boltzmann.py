import numpy as np
import pickle as rick
from numba import njit
import matplotlib.pyplot as plt
from helpers import Qplus, CBoltz2_Carl_Maxwell
from matplotlib.colors import LinearSegmentedColormap


# For setting breakpoints at warnings
import warnings
warnings.filterwarnings('error')

# Make relevant directories
import os

directory_path = 'C:/Users/damie/OneDrive/UW/research/jingwei/boltzmann/lax_friedrichs_method/lf_outputs/temp/'

if not os.path.exists(directory_path):
    os.makedirs(directory_path + 'plots/')
    for slice in [0,25,50,75,100]:
        os.makedirs(directory_path + f'slice{slice}/')

# Variables to be set before running ########################################################
save_me = True # Do you want to save the intermediate solution data for plotting?
save_freq = 1 # Frequency at which the state of the system is saved
max_iter = 10000 # Maximum number of iterations for the method
C = 1 # Constant from rescaling of the loss collision term (not sure what this should be)
Lx = 60 # The length of the spatial domain. The interval will be [-30,30]
Nx = 100 # The resolution of the spatial grid
Lv = 26.22 # Length of the velocity interval for both dimensions. The interval will be [-13.11,13.11]
Nv = 32 # Number of collocation points in each dimension of the velocity domain
Ntheta = 4 # Number of collocation points to integrate over for circle integral in collision operator
# v = 1 # Advection velocity of the equation
Boltz = 1; d = 2; gamma = 2; a = 0.5 # Refer to the paper for the significance of these variables
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
f0 = lambda X, V1, V2: p0(X) * np.exp(-((V1 - u0(X))**2 + V2**2)/(2 * Boltz * T0(X))) / ((2 * np.pi * Boltz * T0(X))**(d/2)) # Initial particle phase space distribution
# Initialize grids
xb = np.linspace(-Lx/2,Lx/2,Nx) # spatial cell boundaries
dx = xb[1] - xb[0]
x_grid = np.concatenate(([-dx/2 + xb[0]], xb + (dx/2))) # spatial cell centers
n = x_grid.size
dv = Lv / Nv
v_grid = np.arange(-Lv/2 + dv / 2, Lv/2, dv)
dt = (0.9 * dx)/(Lv/2)
S = Lv/(3 + np.sqrt(2))
# S = Lv/(3 * np.sqrt(2) + 1)
R = 2*S
X, V1, V2 = np.meshgrid(x_grid,v_grid,v_grid,indexing='ij')
V = V1[0,:,:]
#############################################################################################

# @njit
def sweep(f): # sweeping method for the Boltzmann equation

    Qplus_grid = get_Qplus_grid(f) # Implicit time-stepping
    # Q_grid = get_Q_grid(f) # Explicit time-stepping

    abs_V = np.abs(V)
    p = get_p(f)

    # Forward Sweep
    for i in range(1,n-1):
        f[i,:,:] = (Qplus_grid[i,:,:] + ((V + abs_V)/(2 * dx)) * f[i-1,:,:] - ((V - abs_V)/(2 * dx)) * f[i+1,:,:])/(abs_V/dx + C * p[i]) # Implicit time-stepping
        # f[i,:,:] = (Q_grid[i,:,:] + ((V + abs_V)/(2 * dx)) * f[i-1,:,:] - ((V - abs_V)/(2 * dx)) * f[i+1,:,:])/(abs_V/dx) # Implicit time-stepping

    f[n-1,16:,:] = (Qplus_grid[n-1,16:,:] + ((V[16:,:])/(2 * dx)) * f[n-2,16:,:])/(abs_V[16:,:]/dx + C * p[n-1]) # extrapolation boundary conditions for positive v

    Qplus_grid = get_Qplus_grid(f) # Implicit time-stepping
    # Q_grid = get_Q_grid(f) # Explicit time-stepping

    # Backward Sweep
    for i in range(n-2,0,-1):
        f[i,:,:] = (Qplus_grid[i,:,:] + ((V + abs_V)/(2 * dx)) * f[i-1,:,:] - ((V - abs_V)/(2 * dx)) * f[i+1,:,:])/(abs_V/dx + C * p[i]) # Implicit time-stepping
        # f[i,:,:] = (Q_grid[i,:,:] + ((V + abs_V)/(2 * dx)) * f[i-1,:,:] - ((V - abs_V)/(2 * dx)) * f[i+1,:,:])/(abs_V/dx) # Implicit time-stepping

    f[0,:16,:] = (Qplus_grid[0,:16,:] - ((V[:16,:])/(2 * dx)) * f[1,:16,:])/(abs_V[:16,:]/dx + C * p[0]) # extrapolation boundary conditions for positive v

    return f

# def print_f(f,iter):
#     j = 13
#     k = 13
#     p = get_p(f) 
#     u1, u2 = get_u(f,p)
#     T = get_T(f,p,u1,u2)
#     Qplus_grid = get_Qplus_grid(f) # Implicit time-stepping
#     Q_grid = get_Q_grid(f) # Explicit time-stepping

#     plt.figure(figsize=(10,5))
#     plt.subplots_adjust(bottom=0.3)  # Increase the value to add more space
#     plt.subplot(121)
#     plt.plot(x_grid,(p - pL)/(pR - pL),label = 'Normalized Density')
#     plt.plot(x_grid,(u1 - uR)/(uL - uR),label = 'Normalized Bulk Velocity 1')
#     plt.plot(x_grid,(T - TL)/(TR - TL),label = 'Normalized Temperature')
#     plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center')
#     plt.title('Physical Flow Values')
#     plt.subplot(122)
#     var = '{:.3f}'.format(V[j,k])
#     plt.plot(x_grid,f[:,j,k],label = f'Solution Slice with Velocity {var}')
#     plt.plot(x_grid,Q_grid[:,j,k], label = 'Explicit Collision')
#     plt.plot(x_grid,Qplus_grid[:,j,k] - C * p * f[:,j,k], label = 'Implicit Collision')
#     plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center')
#     plt.title(f'f[:,{j},{k}] and Q[:,{j},{k}]')
    
#     plt.suptitle(f'Data at Iteration {iter}')
#     plt.savefig(f'C:/Users/damie/OneDrive/UW/research/jingwei/boltzmann/lax_friedrichs_method/lf_outputs/output_implicit/{iter}')
#     print(np.max(f))

#     return

def print_f(f,iter):
    j = 17
    k = 17
    p = get_p(f) 
    u1, u2 = get_u(f,p)
    T = get_T(f,p,u1,u2)
    Q_grid = get_Q_grid(f) # Explicit time-stepping
    Qplus_grid = get_Qplus_grid(f) # Implicit time-stepping

    cmap1 = LinearSegmentedColormap.from_list(
    'custom_cmap', [(0, 'white'), (1, 'red')])
    cmap2 = LinearSegmentedColormap.from_list(
    'custom_cmap', [(0, 'blue'), (0.5, 'white'), (1, 'red')])

    for slice in [0,25,50,75,100]:
        min_Q = np.min(Q_grid[slice,:,:])
        max_Q = np.max(Q_grid[slice,:,:])
        Q_bd = np.max([-min_Q,max_Q])
        Qimp = Qplus_grid[slice,:,:] - C * p[slice] * f[slice,:,:]
        min_Qimp = np.min(Qimp)
        max_Qimp = np.max(Qimp)
        Qimp_bd = np.max([-min_Qimp,max_Qimp])
        min_f = np.min(f[slice,:,:])
        max_f = np.min(f[slice,:,:])
        f_bd = np.max([-min_f,max_f])
        fig, axs = plt.subplots(1,3,figsize=(15,5))
        # plt.subplot(121)
        var = '{:.3f}'.format(X[slice,0,0])
        var2 = '{:.3f}'.format(iter*dt)
        fig.suptitle(f'Slice through x={var}: Time = {var2}, Iteration = {iter}')
        im0 = axs[0].imshow(f[slice,:,:], extent = [-Lv/2,Lv/2,-Lv/2,Lv/2], cmap=cmap1, interpolation='nearest', vmin = 0,vmax = f_bd)
        axs[0].set_title('f')
        axs[0].set_ylabel(r'$v_1$')
        axs[0].set_xlabel(r'$v_2$')
        fig.colorbar(im0,ax=axs[0])
        im1 = axs[1].imshow(Q_grid[slice,:,:], extent = [-Lv/2,Lv/2,-Lv/2,Lv/2], cmap=cmap2, interpolation='nearest',vmin = -Q_bd,vmax = Q_bd)
        axs[1].set_title('Q')
        # axs[1].set_ylabel(r'$v_1$')
        axs[1].set_xlabel(r'$v_2$')
        fig.colorbar(im1,ax=axs[1])
        im2 = axs[2].imshow(Qplus_grid[slice,:,:] - C * p[slice] * f[slice,:,:], extent = [-Lv/2,Lv/2,-Lv/2,Lv/2], cmap=cmap2, interpolation='nearest',vmin = -Qimp_bd,vmax = Qimp_bd)
        axs[2].set_title(r'$Q^+ - C \rho f$')
        # axs[2].set_ylabel(r'$v_1$')
        axs[2].set_xlabel(r'$v_2$')
        fig.colorbar(im2,ax=axs[2])
        plt.savefig(f'C:/Users/damie/OneDrive/UW/research/jingwei/boltzmann/lax_friedrichs_method/lf_outputs/temp/slice{slice}/mat{iter}')
        plt.close()

    plt.figure(figsize=(10,5))
    plt.subplots_adjust(bottom=0.3)  # Increase the value to add more space
    plt.subplot(121)
    plt.plot(x_grid,(p - pL)/(pR - pL),label = 'Normalized Density')
    plt.plot(x_grid,(u1 - uR)/(uL - uR),label = 'Normalized Bulk Velocity 1')
    plt.plot(x_grid,(T - TL)/(TR - TL),label = 'Normalized Temperature')
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center')
    plt.xlabel('x')
    plt.title('Physical Flow Values')
    plt.subplot(122)
    var = '{:.3f}'.format(V[j,k])
    plt.plot(x_grid,f[:,j,k],label = f'f[:,{j},{k}] (Velocity = {var})')
    plt.plot(x_grid,Qplus_grid[:,j,k] - C * p * f[:,j,k], label = rf'$Q^+[:,{j},{k}] - C \rho f[:,{j},{k}]$')
    # plt.plot(x_grid,Q_grid[:,j,k], label = f'Q[:,{j},{k}]')
    plt.xlabel('x')
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center')
    plt.title(f'Phase Space Slice')
    plt.suptitle(f'Data at Time = {var2}, Iteration = {iter}')

    plt.savefig(f'C:/Users/damie/OneDrive/UW/research/jingwei/boltzmann/lax_friedrichs_method/lf_outputs/temp/plots/plot{iter}')
    plt.close()
    print(iter)
    return


# @njit  
def get_Q_grid(f): # get the gain term of the collision operator on the grid
    Q_grid = np.empty(f.shape)
    for i in range(f.shape[0]): # Iterate over the second dimension.
        Q_grid[i,:,:] = CBoltz2_Carl_Maxwell(f[i,:,:],Nv,R,Lv,Ntheta)
    return Q_grid

# @njit  
def get_Qplus_grid(f): # get the gain term of the collision operator on the grid
    Qplus_grid = np.empty(f.shape)
    for i in range(f.shape[0]):  # Iterate over the second dimension.
        Qplus_grid[i,:,:] = Qplus(f[i,:,:],Nv,R,Lv,Ntheta)
    return Qplus_grid

# @njit
def get_p(f): # get the density
    # plt.imshow(f[0,:,:], cmap='viridis', interpolation='nearest')
    p = np.trapz(f, x=v_grid, axis=1) # This integrates out V1 axis
    p = np.trapz(p, x=v_grid, axis=1) # This integrates out V2 axis
    return p

# @njit
def get_u(f,p): # get the bulk velocity
    # First dimension of the bulk velocity
    u1 = np.trapz(f * V1, x=v_grid, axis=1) # This integrates out V1 axis
    u1 = np.trapz(u1, x=v_grid, axis=1) # This integrates out V2 axis
    u1 = u1 / p # divide out the density

    # Second dimension of the bulk velocity
    u2 = np.trapz(f * V2, x=v_grid, axis=1) # This integrates out V1 axis
    u2 = np.trapz(u2, x=v_grid, axis=1) # This integrates out V2 axis
    u2 = u2 / p # divide out the density

    return u1, u2

# @njit
def get_T(f,p,u1,u2): # get the temperature
    nrg = 0.5 * f * (V1**2 + V2**2) # haha 'nrg' get it? I am so funny
    total_nrg = np.trapz(nrg, x=v_grid, axis=1) # This integrates out V1 axis
    total_nrg = np.trapz(total_nrg, x=v_grid, axis=1) # This integrates out V2 axis
    T = (total_nrg - 0.5 * p * (u1**2 + u2**2))/(Boltz*p)

    return T

# @njit
def weno1_boltzmann(f,p,u1,u2,T):
    iter = 0 # start iteration counter
    save_data = {} # saves data at specified times
    save_data['density'] = {} # saves the density function p at certain iterates
    save_data['bulk_velocity_1'] = {} # saves the bulk x-velocity function u1 at certain iterates
    save_data['bulk_velocity_2'] = {} # saves the bulk y-velocity function u2 at certain iterates
    save_data['temperature'] = {} # saves the temperature function T at certain iterates
    save_data['density'][iter] = p
    save_data['bulk_velocity_1'][iter] = u1
    save_data['bulk_velocity_2'][iter] = u2
    save_data['temperature'][iter] = T

    print_f(f,iter)

    f0 = f
    f1 = np.copy(f0)
    f1 = sweep(f1) # update phase space probability density
    # get pressure, bulk velocities, and temperature
    p = get_p(f1) 
    u1, u2 = get_u(f1,p)
    T = get_T(f1,p,u1,u2)
    # save pressure, bulk velocities, and temperature
    save_data['density'][iter] = p
    save_data['bulk_velocity_1'][iter] = u1
    save_data['bulk_velocity_2'][iter] = u2
    save_data['temperature'][iter] = T

    iter += 1
    print_f(f1,iter)

    f2 = np.copy(f1)
    f2 = sweep(f2) # update phase space probability density
    # get pressure, bulk velocities, and temperature
    p = get_p(f2) 
    u1, u2 = get_u(f1,p)
    T = get_T(f1,p,u1,u2)
    # save pressure, bulk velocities, and temperature
    save_data['density'][iter] = p
    save_data['bulk_velocity_1'][iter] = u1
    save_data['bulk_velocity_2'][iter] = u2
    save_data['temperature'][iter] = T

    iter += 1
    print_f(f2,iter)

    while(iter < max_iter):
        f0 = np.copy(f1)
        f1 = np.copy(f2)
        f2 = sweep(f2) # update phase space probability density
        p = get_p(f2)

        iter += 1

        if save_me and iter % save_freq == 0:
            u1, u2 = get_u(f1,p)
            T = get_T(f1,p,u1,u2)
            # save pressure, bulk velocities, and temperature
            save_data['density'][iter] = p
            save_data['bulk_velocity_1'][iter] = u1
            save_data['bulk_velocity_2'][iter] = u2
            save_data['temperature'][iter] = T

            print_f(f2,iter)
        
    save_data['density'][iter] = p
    save_data['bulk_velocity_1'][iter] = u1
    save_data['bulk_velocity_2'][iter] = u2
    save_data['temperature'][iter] = T
    save_data['final_iteration'] = iter

    return save_data

if __name__ == "__main__":
    p = p0(x_grid) # initial density on grid
    u1 = u0(x_grid) # initial bulk x-velocity on grid
    u2 = np.zeros(x_grid.shape) # initial bulk y-velocity on grid
    T = T0(x_grid) # initial temperature on grid
    f = f0(X,V1,V2)

    save_data = weno1_boltzmann(f,p,u1,u2,T)

    if save_me:
        with open('normal_shock_problem_data.pkl', 'wb') as foot:
            rick.dump(save_data, foot)


    # # # Collision Step
    # for i in range(1,n):
    #     f[i,16:,:] = f_star[i,16:,:] + dt * (Qplus_grid[i,16:,:] - C * p_star[i] * f_star[i,16:,:]) # Implicit Method
    #     # f[i,:,:] = f_star[i,:,:] + dt * Q_grid[i,:,:] # Explicit Method

    # for i in range(n-2,-1,-1):
    #     f[i,:16,:] = f_star[i,:16,:] + dt * (Qplus_grid[i,:16,:] - C * p_star[i] * f_star[i,:16,:]) # Implicit Method
    #     # f[i,:,:] = f_star[i,:,:] + dt * Q_grid[i,:,:] # Explicit Method


# import numpy as np
# from numba import njit
# import matplotlib.pyplot as plt
# from helpers import Qplus, CBoltz2_Carl_Maxwell

# # For setting breakpoints at warnings
# import warnings
# warnings.filterwarnings('error')

# # @njit
# def sweep(f,p,Nv,R,Lv,Ntheta,V,C,dx,n): # sweeping method for the Boltzmann equation
    
#     # Qplus_grid = get_Qplus_grid(f,Nv,R,Lv,Ntheta) # Implicit time-stepping
#     Q_grid = get_Q_grid(f,Nv,R,Lv,Ntheta) # Explicit time-stepping

#     abs_V = np.abs(V)

#     # Forward Sweep
#     for i in range(1,n-1):
#         # f[i,:,:] = (Qplus_grid[i,:,:] + ((V + abs_V)/(2 * dx)) * f[i-1,:,:] - ((V - abs_V)/(2 * dx)) * f[i+1,:,:])/(abs_V/dx + C * p[i]) # Implicit time-stepping
#         f[i,:,:] = (Q_grid[i,:,:] + ((V + abs_V)/(2 * dx)) * f[i-1,:,:] - ((V - abs_V)/(2 * dx)) * f[i+1,:,:])/(abs_V/dx) # Implicit time-stepping

#     f[-1,16:,:] = 2 * f[-2,16:,:] - f[-3,16:,:] # extrapolation boundary conditions for positive v

#     # Qplus_grid = get_Qplus_grid(f,Nv,R,Lv,Ntheta) # Implicit time-stepping
#     Q_grid = get_Q_grid(f,Nv,R,Lv,Ntheta) # Explicit time-stepping

#     # Backward Sweep
#     for i in range(n-2,0,-1):
#         # f[i,:,:] = (Qplus_grid[i,:,:] + ((V + abs_V)/(2 * dx)) * f[i-1,:,:] - ((V - abs_V)/(2 * dx)) * f[i+1,:,:])/(abs_V/dx + C * p[i]) # Implicit time-stepping
#         f[i,:,:] = (Q_grid[i,:,:] + ((V + abs_V)/(2 * dx)) * f[i-1,:,:] - ((V - abs_V)/(2 * dx)) * f[i+1,:,:])/(abs_V/dx) # Implicit time-stepping

#     f[0,:16,:] = 2 * f[1,:16,:] - f[2,:16,:] # extrapolation boundary conditions for negative v

#     return f

# # @njit  
# def get_Q_grid(f,Nv,R,Lv,Ntheta): # get the gain term of the collision operator on the grid
#     Q_grid = np.empty(f.shape)
#     for i in range(f.shape[0]): # Iterate over the second dimension.
#         Q_grid[i,:,:] = CBoltz2_Carl_Maxwell(f[i,:,:],Nv,R,Lv,Ntheta)
#     return Q_grid

# # @njit  
# def get_Qplus_grid(f,Nv,R,Lv,Ntheta): # get the gain term of the collision operator on the grid
#     Qplus_grid = np.empty(f.shape)
#     for i in range(f.shape[0]):  # Iterate over the second dimension.
#         Qplus_grid[i,:,:] = Qplus(f[i,:,:],Nv,R,Lv,Ntheta)
#     return Qplus_grid

# # @njit
# def get_p(f,v_grid): # get the density
#     # plt.imshow(f[0,:,:], cmap='viridis', interpolation='nearest')
#     p = np.trapz(f, x=v_grid, axis=1) # This integrates out V1 axis
#     p = np.trapz(p, x=v_grid, axis=1) # This integrates out V2 axis
#     return p

# # @njit
# def get_u(f,p,V1,V2,v_grid): # get the bulk velocity
#     # First dimension of the bulk velocity
#     u1 = np.trapz(f * V1, x=v_grid, axis=1) # This integrates out V1 axis
#     u1 = np.trapz(u1, x=v_grid, axis=1) # This integrates out V2 axis
#     u1 = u1 / p # divide out the density

#     # Second dimension of the bulk velocity
#     u2 = np.trapz(f * V2, x=v_grid, axis=1) # This integrates out V1 axis
#     u2 = np.trapz(u2, x=v_grid, axis=1) # This integrates out V2 axis
#     u2 = u2 / p # divide out the density

#     return u1, u2

# # @njit
# def get_T(f,p,u1,u2,R,V1,V2,v_grid): # get the temperature
#     nrg = 0.5 * f * (V1**2 + V2**2) # haha 'nrg' get it? I am so funny
#     total_nrg = np.trapz(nrg, x=v_grid, axis=1) # This integrates out V1 axis
#     total_nrg = np.trapz(total_nrg, x=v_grid, axis=1) # This integrates out V2 axis
#     T = (total_nrg - 0.5 * p * (u1**2 + u2**2))/(R*p)

#     return T

# # @njit
# def weno1_boltzmann(f,p,u1,u2,T,pL,pR,uL,uR,TL,TR,v,C,n,dx,Nv,R,Lv,Ntheta,V1,V2,v_grid,x_grid,save_me,save_freq):
#     iter = 0 # start iteration counter
#     order = 1 # order of the method
#     save_data = {} # saves data at specified times
#     save_data['density'] = {} # saves the density function p at certain iterates
#     save_data['bulk_velocity_1'] = {} # saves the bulk x-velocity function u1 at certain iterates
#     save_data['bulk_velocity_2'] = {} # saves the bulk y-velocity function u2 at certain iterates
#     save_data['temperature'] = {} # saves the temperature function T at certain iterates
#     save_data['stop_conditions'] = {} # saves the history of the stopping conditions 
#     save_data['density'][iter] = p
#     save_data['bulk_velocity_1'][iter] = u1
#     save_data['bulk_velocity_2'][iter] = u2
#     save_data['temperature'][iter] = T
#     # Qplus_grid = get_Qplus_grid(f,Nv,R,Lv,Ntheta)

#     plt.plot(x_grid,(p - pL)/(pR - pL),label = 'Normalized Density')
#     plt.plot(x_grid,(u1 - uR)/(uL - uR),label = 'Normalized Bulk Velocity 1')
#     plt.plot(x_grid,(T - TL)/(TR - TL),label = 'Normalized Temperature')      
#     plt.legend()
#     plt.title(f'Pressure, Bulk Velocity, and Temperature at Iteration {iter}')
#     plt.show()

#     f0 = f
#     f1 = np.copy(f0)
#     f1 = sweep(f1,p,Nv,R,Lv,Ntheta,V1[0,:,:],C,dx,n) # update phase space probability density
#     # Qplus_grid = get_Qplus_grid(f1,Nv,R,Lv,Ntheta)
#     # get pressure, bulk velocities, and temperature
#     p = get_p(f1,v_grid) 
#     u1, u2 = get_u(f1,p,V1,V2,v_grid)
#     T = get_T(f1,p,u1,u2,R,V1,V2,v_grid)
#     # save pressure, bulk velocities, and temperature
#     save_data['density'][iter] = p
#     save_data['bulk_velocity_1'][iter] = u1
#     save_data['bulk_velocity_2'][iter] = u2
#     save_data['temperature'][iter] = T

#     # # Makes everything probability distributions
#     # p_mass = np.trapz(p,x_grid) # total mass on the domain
#     # p = p/p_mass
#     # f1 = f1/p_mass

#     iter += 1

#     plt.plot(x_grid,(p - pL)/(pR - pL),label = 'Normalized Density')
#     plt.plot(x_grid,(u1 - uR)/(uL - uR),label = 'Normalized Bulk Velocity 1')
#     plt.plot(x_grid,(T - TL)/(TR - TL),label = 'Normalized Temperature')  
#     plt.legend()
#     plt.title(f'Pressure, Bulk Velocity, and Temperature at Iteration {iter}')
#     plt.show()

#     f2 = np.copy(f1)
#     f2 = sweep(f2,p,Nv,R,Lv,Ntheta,V1[0,:,:],C,dx,n) # update phase space probability density
#     # Qplus_grid = get_Qplus_grid(f2,Nv,R,Lv,Ntheta)
#     # get pressure, bulk velocities, and temperature
#     p = get_p(f2,v_grid) 
#     u1, u2 = get_u(f2,p,V1,V2,v_grid)
#     T = get_T(f2,p,u1,u2,R,V1,V2,v_grid)
#     # save pressure, bulk velocities, and temperature
#     save_data['density'][iter] = p
#     save_data['bulk_velocity_1'][iter] = u1
#     save_data['bulk_velocity_2'][iter] = u2
#     save_data['temperature'][iter] = T

#     # # Makes everything probability distributions
#     # p_mass = np.trapz(p,x_grid) # total mass on the domain
#     # p = p/p_mass
#     # f2 = f2/p_mass

#     iter += 1


#     plt.plot(x_grid,(p - pL)/(pR - pL),label = 'Normalized Density')
#     plt.plot(x_grid,(u1 - uR)/(uL - uR),label = 'Normalized Bulk Velocity 1')
#     plt.plot(x_grid,(T - TL)/(TR - TL),label = 'Normalized Temperature')
#     plt.legend()
#     plt.title(f'Pressure, Bulk Velocity, and Temperature at Iteration {iter}')
#     plt.show()

#     q1 = (np.linalg.norm(f2 - f1) + dx**order)/np.linalg.norm(f1 - f0)
#     # q2 = np.linalg.norm(f2 - f1) - dx**order
#     q2 = np.linalg.norm(f2 - f1) - dx**(order+1)
#     print(f'q1 < 1 is {q1 < 1} and q2 > 0 is {q2 > 0}\n')
#     while((q1 < 1) or (q2 > 0)):
#         f0 = np.copy(f1)
#         f1 = np.copy(f2)
#         f2 = sweep(f2,p,Nv,R,Lv,Ntheta,V1[0,:,:],C,dx,n) # update phase space probability density
#         # Qplus_grid = get_Qplus_grid(f2,Nv,R,Lv,Ntheta)
#         p = get_p(f2,v_grid)

#         # # Makes everything probability distributions
#         # p_mass = np.trapz(p,x_grid) # total mass on the domain
#         # p = p/p_mass
#         # f2 = f2/p_mass

#         q1 = (np.linalg.norm(f2 - f1) + dx**order)/np.linalg.norm(f1 - f0) 
#         # q2 = np.linalg.norm(f2 - f1) - dx**order
#         q2 = np.linalg.norm(f2 - f1) - dx**(order+1)
#         iter += 1
#         print(f'q1 < 1 is {q1 < 1} and q2 > 0 is {q2 > 0}\n')

#         if save_me and iter % save_freq == 0:
#             u1, u2 = get_u(f1,p,V1,V2,v_grid)
#             T = get_T(f1,p,u1,u2,R,V1,V2,v_grid)
#             # save pressure, bulk velocities, and temperature
#             save_data['stop_conditions'][iter] = [q1, q2]
#             save_data['density'][iter] = p
#             save_data['bulk_velocity_1'][iter] = u1
#             save_data['bulk_velocity_2'][iter] = u2
#             save_data['temperature'][iter] = T

#             plt.plot(x_grid,(p - pL)/(pR - pL),label = 'Normalized Density')
#             plt.plot(x_grid,(u1 - uR)/(uL - uR),label = 'Normalized Bulk Velocity 1')
#             plt.plot(x_grid,(T - TL)/(TR - TL),label = 'Normalized Temperature')  
#             plt.legend()
#             plt.title(f'Pressure, Bulk Velocity, and Temperature at Iteration {iter}')
#             plt.show()
        
#     save_data['stop_conditions'][iter] = [q1, q2]
#     save_data['density'][iter] = p
#     save_data['bulk_velocity_1'][iter] = u1
#     save_data['bulk_velocity_2'][iter] = u2
#     save_data['temperature'][iter] = T
#     save_data['final_iteration'] = iter

#     return save_data


