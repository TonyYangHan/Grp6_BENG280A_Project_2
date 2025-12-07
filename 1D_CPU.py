import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# This code simulates 1D diffusion under a PGSE sequence using CPU computation.

# Create PGSE gradient waveform
def make_pgse_gradient(dt=1.0, n_delta=50, n_gap=100, G_amp=0.02, gamma=1.0):
    n_big_delta = n_delta + n_gap
    n_total = n_big_delta + n_delta

    t = np.arange(n_total) * dt
    G = np.zeros(n_total)
    G[0:n_delta] = G_amp
    G[n_big_delta:n_big_delta + n_delta] = -G_amp

    delta = n_delta * dt
    big_delta = n_big_delta * dt
    
    return t, G, dt, delta, big_delta, gamma

# CPU core simulation function: Process a single batch of particles
def simulate_batch_cpu(D, batch_size, G, dt, gamma, L=None):
    n_total = len(G)

    # 1. Initialize position and phase arrays
    x = np.zeros(batch_size, dtype=np.float32) 
    phi = np.zeros(batch_size, dtype=np.float32)
    
    # Precompute standard deviation of diffusion step
    step_sigma = np.sqrt(2 * D * dt)
    
    # 2. Time step loop
    for k in range(n_total):
        # Generate random steps (Brownian motion using NumPy)
        noise = np.random.randn(batch_size).astype(np.float32)
        x += step_sigma * noise
        
        # Reflecting boundary conditions
        if L is not None:
            left, right = -L/2, L/2
            
            # Handle right boundary (x > right)
            mask_right = x > right
            if np.any(mask_right):
                x[mask_right] = 2 * right - x[mask_right]
            
            # Handle left boundary (x < left)
            mask_left = x < left
            if np.any(mask_left):
                x[mask_left] = 2 * left - x[mask_left]

        # Accumulate phase
        # G[k] is a scalar, multiplied by vector x
        phi += gamma * G[k] * x * dt

    # 3. Calculate complex signal sum for the batch
    batch_sum_signal = np.sum(np.exp(1j * phi))
    
    return batch_sum_signal

def run_simulation_main():
    D = 8e-9          # cm^2/ms
    G_amp = 2.0       # Gauss/cm
    n_delta = 300     # Pulse duration steps
    n_gap = 200       # Pulse gap steps
    dt = 0.1          # ms
    gamma = 26.75     # rad/(ms*Gauss)
    L = None          # cm (boundary size 40um)
    

    total_spins = 10**7  # <--- Suggest testing with 10^6 first, then switch back to 10**10
    
    # Batch size: CPU RAM is usually larger, so can set higher, e.g., 10^6 ~ 10^7
    batch_size = 10**7
    
    print("="*40)
    print(f"Starting CPU diffusion simulation")
    print(f"Total particles: {total_spins:.1e}")
    print(f"Batch size: {batch_size:.1e}")
    print(f"Estimated batches: {int(np.ceil(total_spins/batch_size))}")
    print("="*40)

    # Prepare gradient
    t, G, dt, delta, big_delta, gamma = make_pgse_gradient(
        dt=dt, n_delta=n_delta, n_gap=n_gap, G_amp=G_amp, gamma=gamma
    )
    
    # Initialize accumulator
    total_signal_sum = 0j
    particles_processed = 0
    num_batches = int(np.ceil(total_spins / batch_size))
    
    start_time = time.time()
    
    # --- Execute Batch Simulation ---
    for i in range(num_batches):
        # Calculate number of particles for current batch
        current_batch_n = min(batch_size, total_spins - particles_processed)
        
        # Call CPU core function
        batch_sum = simulate_batch_cpu(D, current_batch_n, G, dt, gamma, L)
        
        # Accumulate results
        total_signal_sum += batch_sum
        particles_processed += current_batch_n
        
        # Display progress
        if (i + 1) % 1 == 0 or (i + 1) == num_batches: # Update every batch since CPU is slower
            elapsed = time.time() - start_time
            percent = (particles_processed / total_spins) * 100
            # Estimate remaining time
            if i > 0:
                avg_time_per_batch = elapsed / (i + 1)
                remaining_batches = num_batches - (i + 1)
                eta = avg_time_per_batch * remaining_batches
                eta_str = f"{eta:.1f}s"
            else:
                eta_str = "Calculating..."
                
            print(f"\rProgress: {percent:5.1f}% | Batch {i+1}/{num_batches} | Time: {elapsed:.1f}s | ETA: {eta_str}", end="")

    print("\n" + "="*40)
    
    # --- Calculate Final Results ---
    S_sim = total_signal_sum / total_spins
    S_mag_sim = np.abs(S_sim)
    
    # Theoretical value (Stejskal-Tanner)
    b_value = (gamma**2) * (G_amp**2) * (delta**2) * (big_delta - delta/3.0)
    S_ST = np.exp(-b_value * D)
    conv_fct = 1e-5
    
    print("Simulation Results (Real Values - CPU):")
    print(f"  D       = {D}")
    print(f"  G_amp   = {G_amp}")
    print(f"  b-value = {b_value * conv_fct:.2e}")
    print("-" * 20)
    print(f"Simulated Signal |S| = {S_mag_sim:.6f}")
    print(f"Theoretical S_ST     = {S_ST:.6f} (Free Diffusion)")
    if L is not None:
        print(f"Note: Due to boundary L={L}, simulated value should be greater than free diffusion theory.")
    
    print(f"Relative Error (vs Free)   = {(S_mag_sim - S_ST)/S_ST:.2%}")

if __name__ == "__main__":
    run_simulation_main()