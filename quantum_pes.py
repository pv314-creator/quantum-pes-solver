import numpy as np
import matplotlib.pyplot as plt

# Defining constants
# Dissociation Energy
D_e = 4.746
# Morse Parameter
Beta = 1.942
# Equilibrium bond length
r_e = 0.742
# Sato parameter
S = 0.20

# Defining morse potential components
def E_plus(r):
     return D_e * (np.exp(-2 * Beta * (r - r_e)) - 2 * np.exp(-Beta * (r - r_e)))
def E_minus(r):
     return D_e * (np.exp(-2 * Beta * (r - r_e)) + 2 * np.exp(-Beta * (r - r_e)))
def Q_val(r):
     return (E_plus(r) + E_minus(r))/2.0
def J_val(r):
    return (E_plus(r) - E_minus(r)) / 2.0

# Function that returns potential energy
def calculate_leps(r12, r23):
    r13 = r12 + r23
    Q12 = Q_val(r12)
    Q23 = Q_val(r23)
    Q13 = Q_val(r13)
    J12 = J_val(r12)
    J23 = J_val(r23)
    J13 = J_val(r13)
    return (Q12/(1+S)) + (Q23/(1+S)) + (Q13/(1+S)) - np.sqrt((J12**2/(1+S)**2) + (J23**2/(1+S)**2) + (J13**2/(1+S)**2) - (J12*J23 + J23*J13 + J12*J13)/((1+S)*(1+S)*(1+S)))

# Defining the gradient function
def get_gradient(r12, r23, h=1e-5):
    dV_dr12 = (calculate_leps(r12 + h, r23) - calculate_leps(r12 - h, r23)) / (2 * h)
    dV_dr23 = (calculate_leps(r12, r23 + h) - calculate_leps(r12, r23 - h)) / (2 * h)
    return dV_dr12, dV_dr23

# First derivative with respect to r12 evaluated at r12 = r23 = r
def g(r):
    dV_dr12, _ = get_gradient(r, r)
    return dV_dr12
# Second derivative (curvature) of g(r) using finite differences
def g_prime(r, h=1e-5):
    return (g(r + h) - g(r - h)) / (2 * h)
#Initial guess near center of plot (in Angstroms)
r_current = 1.0
#Convergence critiera 
tol = 1e-6
count = 100
h = 1e-5

print(f"{'Iter':<6}{'r_curr (Å)':<15}{'g(r)':<15}")
print("-" * 36)

for i in range(count):
    # Computing current derivative value
    g_val = g(r_current)
    print(f"{i:<6}{r_current:<15.6f}{g_val:<15.6e}")
    # Check if gradient is close enough to zero
    if abs(g_val) < tol:
        print(f"\nSuccessfully converged to the saddle point at r = {r_current:.6f} Å")
        break
    # Computing curvature
    gp_val = g_prime(r_current, h)
    # Newton-Raphson update step: r_{k+1} = r_k - g(r_k) / g'(r_k)
    r_current -= g_val / gp_val
# Final saddle point coordinates and energy
r_star = r_current
V_star = calculate_leps(r_star, r_star)

print(f"\nTransition State Coordinates: r12* = r23* = {r_star:.4f} Å")
print(f"Barrier Energy V*: {V_star:.4f} eV")

def compute_hessian(r12, r23, h=1e-5):
    # Unmixed second derivative: d2V / dr12^2
    H11 = (calculate_leps(r12 + h, r23) - 2 * calculate_leps(r12, r23) + calculate_leps(r12 - h, r23)) / (h ** 2)
    # Mixed second derivative: d2V / (dr12 dr23)
    V_pp = calculate_leps(r12 + h, r23 + h) 
    V_pm = calculate_leps(r12 + h, r23 - h)
    V_mp = calculate_leps(r12 - h, r23 + h) 
    V_mm = calculate_leps(r12 - h, r23 - h)  

    H12 = (V_pp - V_pm - V_mp + V_mm) / (4 * h**2)

    # Symmetry properties of the surface
    H21 = H12
    H22 = H11

    # Constructing the Hessian matrix
    H = np.array([[H11, H12], 
                  [H21, H22]])
    return H

H = compute_hessian(r_star, r_star)
eigenvalues, eigenvectors = np.linalg.eig(H)

print("Hessian Matrix:")
print(H)
print("\nEigenvalues:", eigenvalues)

# Finding index of the negative eigenvalue
unstable_idx = np.argmin(eigenvalues)
v_unstable = eigenvectors[:, unstable_idx]

# Normalizing the unstable eigenvector to ensure consistent step length
v_unstable /= np.linalg.norm(v_unstable)

# Helper function to perform Gradient Descent downhill
def trace_path(r_start, direction_sign, steps = 300, gamma = 0.001):
    path = []
    current_pos = np.array(r_start) + direction_sign * 0.01 * v_unstable
    path.append(current_pos.copy())

    for _ in range(steps):
        # Calculate 2D gradient at current position
        g1, g2 = get_gradient(current_pos[0], current_pos[1])
        grad = np.array([g1, g2])
        # Move downhill (opposite to gradient direction)
        current_pos -= gamma * grad
        path.append(current_pos.copy())
        # Stop if the gradient becomes flat (at the bottom of a valley)
        if np.linalg.norm(grad) < 1e-4:
            break

    return np.array(path)

# Trace forward (Products) and backward (Reactants)
path_products = trace_path([r_star, r_star], direction_sign=1.0, steps=800, gamma=0.003)
path_reactants = trace_path([r_star, r_star], direction_sign=-1.0, steps=800, gamma=0.003)

# Combine the paths to form the full IRC path
full_irc_path = np.vstack((path_reactants[::-1], np.array([[r_star, r_star]]), path_products))

# Extract r12 and r23 trajectories for plotting
r12_irc = full_irc_path[:, 0]  
r23_irc = full_irc_path[:, 1]


# Plotting the contour surface
r = np.linspace(0.5, 3.0, 200)
R12, R23 = np.meshgrid(r, r)
V = calculate_leps(R12, R23)
plt.figure(figsize=(7, 6))
plt.contourf(R12, R23, V, levels=50, cmap='viridis')
plt.colorbar(label='Potential Energy (eV)')
# Overlay the Intrinsic Reaction Coordinate (IRC) path
plt.plot(r12_irc, r23_irc, color='red', linewidth=2.5, linestyle='--', label='Minimum Energy Path (IRC)')
# Highlight the Transition State saddle point
plt.plot(r_star, r_star, marker='*', color='gold', markersize=15, markeredgecolor='black', label='Transition State')


plt.xlabel('r12 (Å)')
plt.ylabel('r23 (Å)')
plt.title('LEPS Potential Energy Surface (H + H2)')
plt.savefig('pes_plot.png', dpi=300)
plt.show()
