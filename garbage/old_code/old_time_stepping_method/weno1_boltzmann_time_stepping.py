import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from helpers import Qplus, CBoltz2_Carl_Maxwell

# For setting breakpoints at warnings
import warnings
warnings.filterwarnings('error')

# @njit
def step(f,v_grid,v,Nv,R,Lv,Ntheta,C,dt,dx,n): # take one time step for the Boltzmann equation
    
    f_star = np.empty(f.shape)
    f_star[0,:,:] = f[0,:,:]

    # Compute f*, the intermediate step
    for i in range(1,n):
        f_star[i,:,:] = (1 - v*dt/dx)*f[i,:,:] + (v*dt/dx)*f[i-1,:,:]

    p_star = get_p(f_star,v_grid)
    # Qplus_star_grid = get_Qplus_grid(f_star,Nv,R,Lv,Ntheta) # Implicit time-stepping
    Q_star_grid = get_Q_grid(f_star,Nv,R,Lv,Ntheta) # Explicit time-stepping

    # Compute the update for f
    for i in range(1,n):
        # f[i,:,:] = dt*Qplus_star_grid[i,:,:] + (1 - C*dt*p_star[i])*f_star[i,:,:] # Implicit time-stepping
        f[i,:,:] = dt*Q_star_grid[i,:,:] + f_star[i,:,:] # Explicit time-stepping

    return f

# @njit  
def get_Q_grid(f,Nv,R,Lv,Ntheta): # get the gain term of the collision operator on the grid
    Q_grid = np.empty(f.shape)
    for i in range(f.shape[0]): # Iterate over the second dimension.
        Q_grid[i,:,:] = CBoltz2_Carl_Maxwell(f[i,:,:],Nv,R,Lv,Ntheta)
    return Q_grid

# @njit  
def get_Qplus_grid(f,Nv,R,Lv,Ntheta): # get the gain term of the collision operator on the grid
    Qplus_grid = np.empty(f.shape)
    for i in range(f.shape[0]): # Iterate over the second dimension.
        Qplus_grid[i,:,:] = Qplus(f[i,:,:],Nv,R,Lv,Ntheta)
    return Qplus_grid

# @njit
def get_p(f,v_grid): # get the density
    # plt.imshow(f[0,:,:], cmap='viridis', interpolation='nearest')
    p = np.trapz(f, x=v_grid, axis=1) # This integrates out V1 axis
    p = np.trapz(p, x=v_grid, axis=1) # This integrates out V2 axis
    return p

# @njit
def get_u(f,p,V1,V2,v_grid): # get the bulk velocity
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
def get_T(f,p,u1,u2,R,V1,V2,v_grid): # get the temperature
    nrg = 0.5 * f * (V1**2 + V2**2) # haha 'nrg' get it? I am so funny
    total_nrg = np.trapz(nrg, x=v_grid, axis=1) # This integrates out V1 axis
    total_nrg = np.trapz(total_nrg, x=v_grid, axis=1) # This integrates out V2 axis
    T = (total_nrg - 0.5 * p * (u1**2 + u2**2))/(R*p)

    return T

# @njit
def weno1_boltzmann_time_stepping(f,p,u1,u2,T,pL,pR,uL,uR,TL,TR,v,C,n,dt,dx,Nv,R,Lv,Ntheta,V1,V2,v_grid,x_grid,save_me,save_freq):
    iter = 0 # start iteration counter
    order = 1 # order of the method
    save_data = {} # saves data at specified times
    save_data['density'] = {} # saves the density function p at certain iterates
    save_data['bulk_velocity_1'] = {} # saves the bulk x-velocity function u1 at certain iterates
    save_data['bulk_velocity_2'] = {} # saves the bulk y-velocity function u2 at certain iterates
    save_data['temperature'] = {} # saves the temperature function T at certain iterates
    save_data['stop_conditions'] = {} # saves the history of the stopping conditions 
    save_data['density'][iter] = p
    save_data['bulk_velocity_1'][iter] = u1
    save_data['bulk_velocity_2'][iter] = u2
    save_data['temperature'][iter] = T

    plt.plot(x_grid,(p - pL)/(pR - pL),label = 'Normalized Density')
    plt.plot(x_grid,(u1 - uR)/(uL - uR),label = 'Normalized Bulk Velocity 1')
    plt.plot(x_grid,(T - TL)/(TR - TL),label = 'Normalized Temperature')      
    plt.legend()
    plt.title(f'Pressure, Bulk Velocity, and Temperature at Iteration {iter}')
    plt.show()

    f0 = f
    f1 = np.copy(f0)
    f1 = step(f1,v_grid,v,Nv,R,Lv,Ntheta,C,dt,dx,n) # update phase space probability density
    # get pressure, bulk velocities, and temperature
    p = get_p(f1,v_grid) 
    u1, u2 = get_u(f1,p,V1,V2,v_grid)
    T = get_T(f1,p,u1,u2,R,V1,V2,v_grid)
    # save pressure, bulk velocities, and temperature
    save_data['density'][iter] = p
    save_data['bulk_velocity_1'][iter] = u1
    save_data['bulk_velocity_2'][iter] = u2
    save_data['temperature'][iter] = T

    iter += 1

    plt.plot(x_grid,(p - pL)/(pR - pL),label = 'Normalized Density')
    plt.plot(x_grid,(u1 - uR)/(uL - uR),label = 'Normalized Bulk Velocity 1')
    plt.plot(x_grid,(T - TL)/(TR - TL),label = 'Normalized Temperature')  
    plt.legend()
    plt.title(f'Pressure, Bulk Velocity, and Temperature at Iteration {iter}')
    plt.show()

    f2 = np.copy(f1)
    f2 = step(f2,v_grid,v,Nv,R,Lv,Ntheta,C,dt,dx,n) # update phase space probability density
    # get pressure, bulk velocities, and temperature
    p = get_p(f2,v_grid) 
    u1, u2 = get_u(f2,p,V1,V2,v_grid)
    T = get_T(f2,p,u1,u2,R,V1,V2,v_grid)
    # save pressure, bulk velocities, and temperature
    save_data['density'][iter] = p
    save_data['bulk_velocity_1'][iter] = u1
    save_data['bulk_velocity_2'][iter] = u2
    save_data['temperature'][iter] = T

    iter += 1

    plt.plot(x_grid,(p - pL)/(pR - pL),label = 'Normalized Density')
    plt.plot(x_grid,(u1 - uR)/(uL - uR),label = 'Normalized Bulk Velocity 1')
    plt.plot(x_grid,(T - TL)/(TR - TL),label = 'Normalized Temperature')
    plt.legend()
    plt.title(f'Pressure, Bulk Velocity, and Temperature at Iteration {iter}')
    plt.show()

    q1 = (np.linalg.norm(f2 - f1) + dx**order)/np.linalg.norm(f1 - f0)
    # q2 = np.linalg.norm(f2 - f1) - dx**order
    q2 = np.linalg.norm(f2 - f1) - dx**(order+1)
    print(f'q1 < 1 is {q1 < 1} and q2 > 0 is {q2 > 0}\n')
    while(iter < 2000):
        f0 = np.copy(f1)
        f1 = np.copy(f2)
        f2 = step(f2,v_grid,v,Nv,R,Lv,Ntheta,C,dt,dx,n) # update phase space probability density
        p = get_p(f2,v_grid)

        iter += 1
        print(f'q1 < 1 is {q1 < 1} and q2 > 0 is {q2 > 0}\n')

        if save_me and iter % save_freq == 0:
            u1, u2 = get_u(f1,p,V1,V2,v_grid)
            T = get_T(f1,p,u1,u2,R,V1,V2,v_grid)
            # save pressure, bulk velocities, and temperature
            save_data['stop_conditions'][iter] = [q1, q2]
            save_data['density'][iter] = p
            save_data['bulk_velocity_1'][iter] = u1
            save_data['bulk_velocity_2'][iter] = u2
            save_data['temperature'][iter] = T

            plt.plot(x_grid,(p - pL)/(pR - pL),label = 'Normalized Density')
            plt.plot(x_grid,(u1 - uR)/(uL - uR),label = 'Normalized Bulk Velocity 1')
            plt.plot(x_grid,(T - TL)/(TR - TL),label = 'Normalized Temperature')  
            plt.legend()
            plt.title(f'Pressure, Bulk Velocity, and Temperature at Iteration {iter}')
            plt.show()
        
    save_data['stop_conditions'][iter] = [q1, q2]
    save_data['density'][iter] = p
    save_data['bulk_velocity_1'][iter] = u1
    save_data['bulk_velocity_2'][iter] = u2
    save_data['temperature'][iter] = T
    save_data['final_iteration'] = iter

    return save_data


    # # Test to ensure that the integration schemes are correct
    # plt.plot(x_grid,p,label = 'Pressure')
    # plt.plot(x_grid,u1,label = 'Bulk Velocity 1')
    # plt.plot(x_grid,u2,label = 'Bulk Velocity 2')
    # plt.plot(x_grid,T,label = 'Temperature')
    # p_temp = get_p(f,v_grid) 
    # u1_temp, u2_temp = get_u(f,p_temp,V1,V2,v_grid)
    # T_temp = get_T(f,p_temp,u1_temp,u2_temp,R,V1,V2,v_grid)
    # plt.plot(x_grid,p_temp,linestyle = 'dotted',color = 'k',label = 'Pressure Integrated')
    # plt.plot(x_grid,u1_temp,linestyle = 'dotted',color = 'k',label = 'Bulk Velocity 1 Integrated')
    # plt.plot(x_grid,u2_temp,linestyle = 'dotted',color = 'k',label = 'Bulk Velocity 2 Integrated')
    # plt.plot(x_grid,T_temp,linestyle = 'dotted',color = 'k',label = 'Temperature Integrated')
    # plt.legend()
    # plt.title('Pressure, Bulk Velocity, and Temperature')
    # plt.xlabel('x')
    # plt.show()


    # # Makes everything probability distributions
    # p_mass = np.trapz(p,x_grid) # total mass on the domain
    # p = p/p_mass
    # f2 = f2/p_mass