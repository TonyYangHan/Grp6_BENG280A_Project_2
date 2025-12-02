import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def create_anisotropic_phantom(dim=128):
    """
    Create a cell phantom with anisotropic diffusion.
    Returns:
    - S0: Proton density map
    - D_tensor_map: Tensor map of shape (dim, dim, 2, 2)
    """
    S0 = np.zeros((dim, dim))
    # Initialize Tensor Map: Each pixel has a 2x2 matrix [[Dxx, Dxy], [Dyx, Dyy]]
    D_map = np.zeros((dim, dim, 2, 2))
    
    # Helper function: Create diagonal tensor (Principal axes aligned with X/Y)
    def get_tensor(dx, dy):
        return np.array([[dx, 0], [0, dy]])

    # --- 1. Define Diffusion Parameters (Unit: 10^-3 mm^2/s) ---
    # Isotropic: Dx = Dy
    T_water = get_tensor(3.0, 3.0)      # Free water
    T_cyto  = get_tensor(1.0, 1.0)      # Cytoplasm
    T_mem   = get_tensor(0.01, 0.01)    # Cell membrane (Barrier)
    T_channel = get_tensor(3.0, 3.0)    # Channel
    
    # Anisotropic: Dx != Dy
    # Organelle A: Vertical Fast, Horizontal Slow
    T_vertical = get_tensor(0.2, 2.5) 
    # Organelle B: Horizontal Fast, Vertical Slow
    T_horizontal = get_tensor(2.5, 0.2)

    # Coordinate grid
    x, y = np.meshgrid(np.arange(dim), np.arange(dim))
    
    # --- 2. Construct Cell Structure ---
    
    # A. Background (Extracellular fluid)
    S0[:, :] = 100
    D_map[:, :] = T_water
    
    # B. Cytoplasm (Large square)
    cell_size = 80
    c_start = (dim - cell_size) // 2
    c_end = c_start + cell_size
    cell_mask = (x >= c_start) & (x < c_end) & (y >= c_start) & (y < c_end)
    S0[cell_mask] = 150
    D_map[cell_mask] = T_cyto
    
    # C. Cell Membrane
    inner_start = c_start + 2
    inner_end = c_end - 2
    inner_mask = (x >= inner_start) & (x < inner_end) & (y >= inner_start) & (y < inner_end)
    membrane_mask = cell_mask & (~inner_mask)
    S0[membrane_mask] = 50
    D_map[membrane_mask] = T_mem
    
    # D. Channels
    # Open holes on top, bottom, left, and right
    mid = dim // 2
    # Top
    D_map[c_start:c_start+2, mid-2:mid+2] = T_channel
    # Bottom
    D_map[c_end-2:c_end, mid-2:mid+2] = T_channel
    # Left
    D_map[mid-2:mid+2, c_start:c_start+2] = T_channel
    # Right
    D_map[mid-2:mid+2, c_end-2:c_end] = T_channel
    
    # --- 3. Add Anisotropic Organelles ---
    
    # Organelle 1: Vertical strip (Vertical Fast) - Left side
    mask_org1 = (x > c_start+15) & (x < c_start+35) & (y > c_start+10) & (y < c_end-10)
    S0[mask_org1] = 160
    D_map[mask_org1] = T_vertical # Apply vertical tensor
    
    # Organelle 2: Horizontal strip (Horizontal Fast) - Right side
    mask_org2 = (x > c_start+45) & (x < c_end-15) & (y > c_start+30) & (y < c_start+50)
    S0[mask_org2] = 160
    D_map[mask_org2] = T_horizontal # Apply horizontal tensor

    return S0, D_map

def simulate_acquisition(S0, D_map, gx, gy, b_value=1000, snr=30):
    """
    Simulate acquisition for a single gradient direction: Signal -> K-Space -> Recon
    gx, gy: Components of the gradient direction vector (should be normalized)
    """
    dim = S0.shape[0]
    
    # 1. Generate Signal
    # Formula: S = S0 * exp(-b * g^T * D * g)
    # We perform matrix multiplication at each pixel: g.T @ D[i,j] @ g
    
    # Vectorized calculation:
    # term = gx*gx*Dxx + gy*gy*Dyy + 2*gx*gy*Dxy
    # D_map shape is (dim, dim, 2, 2)
    Dxx = D_map[:, :, 0, 0]
    Dyy = D_map[:, :, 1, 1]
    Dxy = D_map[:, :, 0, 1] # Symmetric matrix, Dyx = Dxy
    
    # Calculate projection of ADC (Apparent Diffusion Coefficient) along the gradient direction
    ADC_proj = (gx**2 * Dxx) + (gy**2 * Dyy) + (2 * gx * gy * Dxy)
    
    signal_ideal = S0 * np.exp(-b_value * ADC_proj * 1e-3)
    
    # 2. K-Space & Noise
    k_space = fftshift(fft2(signal_ideal))
    max_val = np.max(np.abs(k_space))
    noise_sigma = max_val / snr
    noise = np.random.normal(0, noise_sigma, k_space.shape) + 1j * np.random.normal(0, noise_sigma, k_space.shape)
    k_space_noisy = k_space + noise
    
    # 3. Recon (Inverse FFT)
    img_recon = np.abs(ifft2(ifftshift(k_space_noisy)))
    
    return img_recon

# --- Run Simulation ---
dim = 128
S0, D_tensor = create_anisotropic_phantom(dim)

# Simulate two extreme gradient directions
# 1. X-Gradient (Sensitive to horizontal diffusion only)
img_x_diff = simulate_acquisition(S0, D_tensor, gx=1, gy=0, b_value=1500)

# 2. Y-Gradient (Sensitive to vertical diffusion only)
img_y_diff = simulate_acquisition(S0, D_tensor, gx=0, gy=1, b_value=1500)

# --- Plotting Comparison ---
fig, axes = plt.subplots(1, 3, figsize=(15, 6))

# Phantom Structure Schematic (Showing Dxx)
axes[0].imshow(D_tensor[:,:,0,0], cmap='jet')
axes[0].set_title("Phantom: Horizontal Diffusivity (Dxx)\n(Red=Fast, Blue=Slow)")
# Annotations
axes[0].text(30, 64, 'Vertical Organelle\n(Low Dxx)', color='white', ha='center', fontsize=9)
axes[0].text(90, 40, 'Horizontal Org\n(High Dxx)', color='black', ha='center', fontsize=9)

# X-DWI
axes[1].imshow(img_x_diff, cmap='gray')
axes[1].set_title("DWI with X-Gradient (1,0)\nSensitive to Horizontal Diffusion")
axes[1].set_xlabel("Horizontal Org is DARK (Fast Diff)\nVertical Org is BRIGHT (Slow Diff)")

# Y-DWI
axes[2].imshow(img_y_diff, cmap='gray')
axes[2].set_title("DWI with Y-Gradient (0,1)\nSensitive to Vertical Diffusion")
axes[2].set_xlabel("Horizontal Org is BRIGHT (Slow Diff)\nVertical Org is DARK (Fast Diff)")

plt.tight_layout()
plt.show()