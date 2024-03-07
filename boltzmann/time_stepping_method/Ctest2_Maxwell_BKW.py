import numpy as np
import matplotlib.pyplot as plt
# from helpers import Qplus
from helpers import CBoltz2_Carl_Maxwell

# Assuming alpha2 and CBoltz2_Carl_Maxwell functions are already defined as provided before

if __name__ == "__main__":
   gamma = 0
   b_gamma = 1 / (2 * np.pi)

   N = 64

   S = 3
   R = 2 * S

   L = (3 * np.sqrt(2) + 1) / 2 * S
   Ntheta = 4

   dv = 2 * L / N
   v = np.arange(-L + dv / 2, L, dv)
   vv1, vv2 = np.meshgrid(v, v)

   t = 0.5
   K = 1 - np.exp(-t / 8) / 2
   dK = np.exp(-t / 8) / 16

   f = 1 / (2 * np.pi * K**2) * np.exp(-(vv1**2 + vv2**2) / (2 * K)) * (2 * K - 1 + (1 - K) / (2 * K) * (vv1**2 + vv2**2))
   df = (-2 / K + (vv1**2 + vv2**2) / (2 * K**2)) * f \
      + 1 / (2 * np.pi * K**2) * np.exp(-(vv1**2 + vv2**2) / (2 * K)) * (2 - 1 / (2 * K**2) * (vv1**2 + vv2**2))
   df = df * dK
   extQ = df
   Max = 1 / (2 * np.pi) * np.exp(-(vv1**2 + vv2**2) / 2)

   # Plot f and Max
   plt.figure(figsize=(10, 5))
   plt.subplot(1, 2, 1)
   plt.plot(v, f[:, N // 2], 'k', linewidth=1.5, label='f')
   plt.plot(v, Max[:, N // 2], 'r', linewidth=1.5, label='Maxwellian')
   plt.title('f and Maxwellian')
   plt.legend()

   Q = CBoltz2_Carl_Maxwell(f, N, R, L, Ntheta) * b_gamma
   absmax = np.max(np.abs(extQ - Q))
   print(absmax)

   # Plot extQ and Q
   plt.subplot(1, 2, 2)
   plt.plot(v, extQ[:, N // 2], 'k', linewidth=1.5, label='extQ')
   plt.plot(v, Q[:, N // 2], 'o', label='Q')
   plt.title(f'Q(f) and extQ\nMax Abs Difference: {absmax:.2e}')
   plt.legend()

   plt.tight_layout()
   plt.show()
