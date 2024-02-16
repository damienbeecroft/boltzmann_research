import numpy as np
from numba import njit
# from numba.scipy.fft import fft2, ifft2
from numpy.fft import fft2, ifft2
import time

# @njit
def sincc(x):
    eps = np.finfo(float).eps  # Machine epsilon for floating-point type
    return np.sin(x + eps) / (x + eps)

# @njit
def alpha2(s, R, L):
    return 2 * R * sincc(np.pi / L * R * s)

# @njit
def Qplus(f, N, R, L, Ntheta):
    """
    Carleman spectral method for the classical Boltzmann collision operator
    2D Maxwell molecule
    N # of Fourier modes: f(N,N), Q(N,N)
    theta: mid-point rule
    """
    # start_time = time.time()
    temp = np.concatenate((np.arange(0,N//2),np.arange(-N//2,0,1)))
    l1 = np.array([[row]*N for row in temp])
    l2 = l1.T
    
    # FFT2 = njit(fft2)
    FTf = fft2(f)

    QG = np.zeros((N, N), dtype=np.complex_)
    bb = np.zeros((N, N), dtype=np.complex_)

    wtheta = np.pi / Ntheta
    theta = np.arange(wtheta / 2, np.pi, wtheta)
    sig1 = np.cos(theta)
    sig2 = np.sin(theta)

    for q in range(Ntheta):
        aa1 = alpha2(l1 * sig1[q] + l2 * sig2[q], R, L)
        aa2 = alpha2(np.sqrt(l1**2 + l2**2 - (l1 * sig1[q] + l2 * sig2[q])**2), R, L)

        QG += 2 * wtheta * ifft2(aa1 * FTf) * ifft2(aa2 * FTf)
        bb += 2 * wtheta * aa1 * aa2

    Q = np.real(QG)

    return Q

def CBoltz2_Carl_Maxwell(f, N, R, L, Ntheta):
    """
    Carleman spectral method for the classical Boltzmann collision operator
    2D Maxwell molecule
    N # of Fourier modes: f(N,N), Q(N,N)
    theta: mid-point rule
    """
    start_time = time.time()
    temp = np.concatenate((np.arange(0,N//2),np.arange(-N//2,0,1)))
    l1 = np.array([[row]*N for row in temp])
    l2 = l1.T
    
    FTf = fft2(f)

    QG = np.zeros((N, N), dtype=np.complex_)
    bb = np.zeros((N, N), dtype=np.complex_)

    wtheta = np.pi / Ntheta
    theta = np.arange(wtheta / 2, np.pi, wtheta)
    sig1 = np.cos(theta)
    sig2 = np.sin(theta)

    for q in range(Ntheta):
        aa1 = alpha2(l1 * sig1[q] + l2 * sig2[q], R, L)
        aa2 = alpha2(np.sqrt(l1**2 + l2**2 - (l1 * sig1[q] + l2 * sig2[q])**2), R, L)

        QG += 2 * wtheta * ifft2(aa1 * FTf) * ifft2(aa2 * FTf)
        bb += 2 * wtheta * aa1 * aa2

    QL = f * ifft2(bb * FTf)

    Q = np.real(QG - QL)

    print(f'time of CBoltz2_Carl_Maxwell is {time.time() - start_time:.2f} sec')

    return Q
