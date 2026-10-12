import numpy as np
import scipy.linalg as lg 
import scipy.sparse as sparse
import matplotlib.pyplot as plt

# Setting lattice size, disorder strength, and hopping
Nx = 20 #lattice size
Ny = 20 #lattice size
L = Nx * Ny  # 20 * 20 = 400
W = 2.0 #disorder strength
t = 1.0 #hopping

# Random on-site energies drawn from a uniform distribution
np.random.seed(42)  # For reproducible results while testing
epsilon = np.random.uniform(-W/2, W/2, L)

# Constructing the Sparse Hamiltonian Matrix for a 2D Grid
H = sparse.lil_matrix((L, L))

for x in range(Nx):
    for y in range(Ny):
        i = x * Ny + y
        
        #On-site energy on the diagonal
        H[i,i] = epsilon[i]

        #4 potential neighbor directions (dx, dy)
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # right, left, down, up

        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            #Check if neighbor is within bounds of the grid boundaries
            if 0 <= nx < Nx and 0 <= ny < Ny:
                j = nx * Ny + ny
                H[i,j] = -t  # hopping term

# Convert to CSR format for fast solving
H_csr = H.tocsr()
print("Hamiltonian shape:", H_csr.shape)
print("Number of non-zero elements:", H_csr.nnz)

from scipy.sparse.linalg import eigsh
# Find k eigenvalues and eigenvectors of the sparse Hamiltonian
k_states = 50
energies, wavefunctions = eigsh(H_csr, k=k_states, which='SM') # 'SM'  for smallest magnitude
print("Energies shape:", energies.shape)
print("Wavefunctions shape:", wavefunctions.shape)

# Calculate IPR for each of the 50 computed wavefunctions
ipr_2d = np.sum(wavefunctions**4, axis=0)

# Picking a specific state to visualize (e.g., the last one)
state_index = -1 
psi_vector = wavefunctions[:, state_index]

# Calculate probability density |psi|^2 and reshape back into Nx by Ny
prob_density = np.abs(psi_vector)**2
grid_profile = prob_density.reshape((Nx, Ny))

# Plotting the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(grid_profile, cmap='hot', interpolation='nearest')
plt.colorbar(label='Probability Density')
plt.title(f'Probability Density of State {state_index}')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

plt.subplot(1, 2, 2)
plt.plot(energies, ipr_2d, 'o', markersize=2)
plt.title('Inverse Participation Ratio (IPR) vs Energy')
plt.xlabel('Energy')
plt.ylabel('IPR')

plt.tight_layout()
plt.savefig('anderson_2d_plot.png', dpi=300)  
plt.show()

# Plotting the level spacing distribution
spacings_2d = np.diff(energies)
plt.figure(figsize=(6, 4))
plt.hist(spacings_2d, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.title('Level Spacing Distribution (2D)')
plt.xlabel('Spacing (s)')
plt.ylabel('Probability Density')

plt.tight_layout()
plt.savefig('anderson_2d_spacing.png', dpi=300)  
plt.show()







