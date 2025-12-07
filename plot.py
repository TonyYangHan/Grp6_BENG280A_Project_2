import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Arc

# --- 1. Simulation Logic ---
def make_pgse_gradient(dt=0.1, n_delta=300, n_gap=200, G_amp=2.0, gamma=26.75):
    """Create PGSE gradient waveform"""
    n_big_delta = n_delta + n_gap
    n_total = n_big_delta + n_delta
    t = np.arange(n_total) * dt
    G_scalar = np.zeros(n_total)
    G_scalar[0:n_delta] = G_amp
    G_scalar[n_big_delta:n_big_delta + n_delta] = -G_amp
    delta = n_delta * dt
    big_delta = n_big_delta * dt
    return t, G_scalar, dt, delta, big_delta, gamma

def simulate_anisotropic_2d(D_x, D_y, theta_deg, n_spins=10000, G_amp=2.0, n_delta=300, n_gap=200, dt=0.1, gamma=26.75, seed=0):
    """Run 2D anisotropic diffusion simulation"""
    np.random.seed(seed)
    t, G_scalar, dt, delta, big_delta, gamma = make_pgse_gradient(dt=dt, n_delta=n_delta, n_gap=n_gap, G_amp=G_amp, gamma=gamma)
    n_total = len(t)
    theta = np.deg2rad(theta_deg)
    cos_th, sin_th = np.cos(theta), np.sin(theta)
    
    # Initialize particle positions and phase
    x = np.zeros(n_spins)
    y = np.zeros(n_spins)
    phi = np.zeros(n_spins)
    
    # Calculate diffusion step size
    step_x = np.sqrt(2 * D_x * dt)
    step_y = np.sqrt(2 * D_y * dt)
    
    # Time step loop
    for k in range(n_total):
        # Random walk
        x += step_x * np.random.randn(n_spins)
        y += step_y * np.random.randn(n_spins)
        
        # Gradient components
        Gx = G_scalar[k] * cos_th
        Gy = G_scalar[k] * sin_th
        
        # Accumulate phase
        phi += gamma * (Gx * x + Gy * y) * dt
        
    # Calculate signal
    S_sim = np.mean(np.exp(1j * phi))
    S_mag_sim = np.abs(S_sim)
    
    # Theoretical calculation (Stejskal-Tanner)
    D_eff = D_x * cos_th**2 + D_y * sin_th**2
    b_ms_cm2 = gamma**2 * G_amp**2 * delta**2 * (big_delta - delta/3.0)
    
    return S_mag_sim, b_ms_cm2, D_eff

# --- 2. Generate Data ---
# Set anisotropic diffusion coefficients (Dx >> Dy)
D_x = 1.5e-8
D_y = 5.0e-9
thetas = np.linspace(0, 360, 37)  # Angles from 0 to 360 degrees
sim_D_apps = []
theory_D_effs = []

# Run simulation for each angle
for th in thetas:
    S_mag, b_val, D_eff = simulate_anisotropic_2d(D_x, D_y, th, n_spins=10000)
    # Calculate Apparent Diffusion Coefficient (ADC)
    # D_app = -ln(S) / b
    D_app = -np.log(S_mag + 1e-12) / b_val
    sim_D_apps.append(D_app)
    theory_D_effs.append(D_eff)

sim_D_apps = np.array(sim_D_apps)
theory_D_effs = np.array(theory_D_effs)
D_scale = 1e8  

# --- 3. Create Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Schematic Diagram
ax1 = axes[0]
ax1.set_title("Simulation Setup: Anisotropic Diffusion & Gradient Rotation", fontsize=14, fontweight='bold', pad=15)
ax1.set_xlim(-2.5, 2.5)
ax1.set_ylim(-2.5, 2.5)
ax1.set_aspect('equal')
ax1.axis('off')

# Draw Ellipse (representing diffusion tensor)
ellipse = Ellipse((0, 0), width=4, height=2, angle=0, facecolor='#e6f2ff', alpha=0.7, edgecolor='#0066cc', lw=2)
ax1.add_patch(ellipse)
ax1.text(0, -0.2, 'Diffusion Tensor\n($D_x \gg D_y$)', ha='center', va='center', fontsize=12, color='#004d99', fontweight='bold')

# Draw Axes
ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax1.text(2.1, 0.1, 'X Axis (Fast)', fontsize=11, color='gray')
ax1.text(0.1, 1.2, 'Y Axis (Slow)', fontsize=11, color='gray')

# Draw Gradient Vector Arrow
theta_example = 45
rad = np.deg2rad(theta_example)
arrow_len = 1.8
ax1.arrow(0, 0, arrow_len*np.cos(rad), arrow_len*np.sin(rad), head_width=0.15, head_length=0.2, fc='#cc0000', ec='#cc0000', lw=3, zorder=10)
ax1.text(arrow_len*np.cos(rad)*1.15, arrow_len*np.sin(rad)*1.15, r'$\vec{G}$', color='#cc0000', fontsize=18, fontweight='bold')

# Draw Angle Arc
arc_patch = Arc((0,0), 1.5, 1.5, theta1=0, theta2=45, color='black', lw=1.5)
ax1.add_patch(arc_patch)
ax1.text(0.8, 0.3, r'$\theta$', fontsize=16)

# Add Description Text
desc_text = (
    "Setup:\n"
    "• $D_x$ (Fast): 1.5 $\\times 10^{-8}$ $cm^2/ms$\n"
    "• $D_y$ (Slow): 0.5 $\\times 10^{-8}$ $cm^2/ms$\n"
    "• Rotate Gradient $\\vec{G}$ from 0° to 360°"
)
ax1.text(-2.4, -2.4, desc_text, fontsize=11, bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))


# Subplot 2: Results Plot
ax2 = axes[1]
ax2.set_title("Result: Measured Signal vs. Gradient Angle", fontsize=14, fontweight='bold', pad=15)

# Plot Theory Line
ax2.plot(thetas, theory_D_effs * D_scale, 'k--', linewidth=2, label='Theoretical $D_{eff}$ (Stejskal-Tanner)')
# Plot Simulation Dots
ax2.plot(thetas, sim_D_apps * D_scale, 'o', color='#0066cc', markersize=8, alpha=0.8, label='Monte Carlo Simulation')

# Labels and Grid
ax2.set_xlabel('Gradient Direction $\\theta$ (degrees)', fontsize=12)
ax2.set_ylabel('Effective Diffusivity ($10^{-8} cm^2/ms$)', fontsize=12)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right', fontsize=11)

# Add key point annotations
# 0 degrees (Max Diffusion)
ax2.annotate('Parallel to Fiber (0°)\nFast Diffusion\nMax Signal Loss', 
             xy=(0, D_x*D_scale), xytext=(40, D_x*D_scale+0.2),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8), fontsize=10, ha='left')

# 90 degrees (Min Diffusion)
ax2.annotate('Perpendicular (90°)\nRestricted Diffusion\nMin Signal Loss', 
             xy=(90, D_y*D_scale), xytext=(90, D_y*D_scale-0.5),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8), fontsize=10, ha='center')

plt.tight_layout()
plt.show() 