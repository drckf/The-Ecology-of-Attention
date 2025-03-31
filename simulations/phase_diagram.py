import numpy as np
import matplotlib.pyplot as plt

# Parameters (example values)
s_J, s_l = 1.0, 1.2
A_JJ, A_ll, A_Jl = 1.0, 1.0, 0.5

# Grid of (omega_f, omega_i) values
f = np.linspace(0, 2.5, 50)
i = np.linspace(0, 2.5, 50)
F, I = np.meshgrid(f, i)

# Lotka-Volterra derivatives
dF = F * (s_J - A_Jl * I - A_JJ * F)
dI = I * (s_l - A_Jl * F - A_ll * I)

# Normalize for direction field
magnitude = np.sqrt(dF**2 + dI**2) + 1e-8
dF_norm = dF / magnitude
dI_norm = dI / magnitude

# Plotting
plt.figure(figsize=(6, 5))
plt.quiver(F, I, dF_norm, dI_norm, magnitude, cmap='viridis', scale=30)
plt.xlabel(r'$\omega_f$')
plt.ylabel(r'$\omega_i$')
plt.title('Phase portrait of gated LV dynamics')

# Optional: plot nullclines
omega_f_nullcline = (s_J - A_Jl * i) / A_JJ
omega_i_nullcline = (s_l - A_Jl * f) / A_ll
plt.plot(omega_f_nullcline, i, 'r--', label=r'$\dot{\omega}_f = 0$')
plt.plot(f, omega_i_nullcline, 'b--', label=r'$\dot{\omega}_i = 0$')
plt.xlim([0, 2.5])
plt.ylim([0, 2.5])
plt.legend()
plt.tight_layout()
plt.show()