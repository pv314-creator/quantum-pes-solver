import numpy as np
import scipy.linalg as lg 
import matplotlib.pyplot as plt

# Setting lattice size, disorder strength, and hopping
L = 500 #lattice size
W = 2.0 #disorder strength
t = 1.0 #hopping

# Random on-site energies drawn from a uniform distribution
np.random.seed(42)  # For reproducible results while testing
epsilon = np.random.uniform(-W/2, W/2, L)

# Create an array of -t values with length L - 1
off_diag = -t * np.ones(L - 1)

# Solves the tridiagonal system directly: H @ psi = E @ psi
# Returns sorted eigenvalues E and columns of eigenvectors psi
energies, wavefunctions = lg.eigh_tridiagonal(epsilon, off_diag)

# Calculate the IPR for every wavefunction column
ipr = np.sum(wavefunctions**4, axis=0)

# Calculate spacing between adjacent eigenvalues
spacings = np.diff(energies)



# Plotting the results
plt.figure(figsize=(12, 6)) 
plt.subplot(1, 2, 1)
plt.plot(energies, ipr, 'o', markersize=2)
plt.title('Inverse Participation Ratio (IPR) vs Energy')
plt.xlabel('Energy')
plt.ylabel('IPR')


plt.subplot(1, 2, 2)
plt.plot(energies, np.log(ipr), 'o', markersize=2)
plt.title('Log of IPR vs Energy')
plt.xlabel('Energy')
plt.ylabel('Log(IPR)')

plt.tight_layout()
plt.savefig('anderson_1d_plot.png', dpi=300)  
plt.show()

# Plotting the level spacing distribution
plt.figure(figsize=(6, 4))
plt.hist(spacings, bins=30, density=True, alpha=0.7, color='purple', edgecolor='black')
plt.title('Level Spacing Distribution')
plt.xlabel('Spacing (s)')
plt.ylabel('Probability Density')

plt.tight_layout()
plt.savefig('anderson_1d_spacing.png', dpi=300)  
plt.show()
