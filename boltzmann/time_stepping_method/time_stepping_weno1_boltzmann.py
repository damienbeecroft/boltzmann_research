import numpy as np
import pickle as rick
from numba import njit
import matplotlib.pyplot as plt
from helpers import Qplus, CBoltz2_Carl_Maxwell

# For setting breakpoints at warnings
import warnings
warnings.filterwarnings('error')

# Variables to be set before running ########################################################
save_me = True # Do you want to save the intermediate solution data for plotting?
save_freq = 10 # Frequency at which the state of the system is saved
max_iter = 1000 # Maximum number of iterations for the method
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
# Initialize grids
xb = np.linspace(-Lx/2,Lx/2,Nx) # spatial cell boundaries
dx = xb[1] - xb[0]
x_grid = np.concatenate(([-dx/2 + xb[0]], xb + (dx/2))) # spatial cell centers
n = x_grid.size
v_grid = np.linspace(-Lv/2,Lv/2,Nv)
dt = (0.5 * dx)/(Lv/2)
X, V1, V2 = np.meshgrid(x_grid,v_grid,v_grid,indexing='ij')
V = V1[0,:,:]
#############################################################################################

# @njit
def sweep(f): # sweeping method for the Boltzmann equation

    f_star = np.zeros(f.shape)
    f_star[0,:,:] = f[0,:,:]
    f_star[-1,:,:] = f[-1,:,:]
    V_plus = V[16:,:]
    V_minus = V[:16,:]

    # Advection Step
    for i in range(1,n):
        f_star[i,16:,:] = f[i,16:,:] - ((V_plus * dt)/dx) * (f[i,16:,:] - f[i-1,16:,:])

    for i in range(n-2,-1,-1):
        f_star[i,:16,:] = f[i,:16,:] - ((V_minus * dt)/dx) * (f[i+1,:16,:] - f[i,:16,:])

    # p_star = get_p(f_star)
    # Qplus_grid = get_Qplus_grid(f_star) # Implicit time-stepping
    Q_grid = get_Q_grid(f_star) # Explicit time-stepping

    # Collision Step
    for i in range(0,n):
        # f[i,:,:] = f_star[i,:,:] + dt * (Qplus_grid[i,:,:] - C * p_star[i] * f_star[i,:,:]) # Implicit Method
        f[i,:,:] = f_star[i,:,:] + dt * Q_grid[i,:,:] # Explicit Method

    # return f_star
    return f

def print_f(f,iter):
    j = 13
    k = 13
    p = get_p(f) 
    u1, u2 = get_u(f,p)
    T = get_T(f,p,u1,u2)
    Qplus_grid = get_Qplus_grid(f) # Implicit time-stepping
    Q_grid = get_Q_grid(f) # Explicit time-stepping

    plt.figure(figsize=(10,5))
    plt.subplots_adjust(bottom=0.3)  # Increase the value to add more space
    plt.subplot(121)
    plt.plot(x_grid,(p - pL)/(pR - pL),label = 'Normalized Density')
    plt.plot(x_grid,(u1 - uR)/(uL - uR),label = 'Normalized Bulk Velocity 1')
    plt.plot(x_grid,(T - TL)/(TR - TL),label = 'Normalized Temperature')
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center')
    plt.title('Physical Flow Values')
    plt.subplot(122)
    var = '{:.3f}'.format(V[j,k])
    plt.plot(x_grid,f[:,j,k],label = f'Solution Slice with Velocity {var}')
    plt.plot(x_grid,Q_grid[:,j,k], label = 'Explicit Collision')
    plt.plot(x_grid,Qplus_grid[:,j,k] - C * p * f[:,j,k], label = 'Implicit Collision')
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center')
    plt.title(f'f[:,{j},{k}] and Q[:,{j},{k}]')
    
    plt.suptitle(f'Data at Iteration {iter}')
    plt.savefig(f'C:/Users/damie/OneDrive/UW/research/jingwei/boltzmann/time_stepping_method/outputs/output_explicit/{iter}')
    plt.close()
    # print(np.max(f))
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
    T = (total_nrg - 0.5 * p * (u1**2 + u2**2))/(R*p)

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