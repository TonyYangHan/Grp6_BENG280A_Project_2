import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time

# This code simulates 1D diffusion under a PGSE sequence using GPU computation.

# Create PGSE gradient waveform for GPU
def make_pgse_gradient(dt=1.0, n_delta=50, n_gap=100, G_amp=0.02, gamma=1.0):
    # This part has small data size, keep on CPU (numpy), transfer to GPU later
    n_big_delta = n_delta + n_gap
    n_total = n_big_delta + n_delta

    t = np.arange(n_total) * dt
    G = np.zeros(n_total)
    G[0:n_delta] = G_amp
    G[n_big_delta:n_big_delta + n_delta] = -G_amp

    delta = n_delta * dt
    big_delta = n_big_delta * dt
    
    # Convert to CuPy array for GPU usage
    G_gpu = cp.array(G)
    
    return t, G_gpu, dt, delta, big_delta, gamma

# Simulate a batch of particles on GPU
def simulate_batch_gpu(D, batch_size, G_gpu, dt, gamma, L=None):
    n_total = len(G_gpu)
    
    # Initialize arrays on GPU
    # float32 is faster and saves memory; use float64 if high precision is needed
    x = cp.zeros(batch_size, dtype=cp.float32) 
    phi = cp.zeros(batch_size, dtype=cp.float32)
    
    step_sigma = cp.sqrt(2 * D * dt)
    
    # Time step loop
    for k in range(n_total):
        # Generate random steps (Brownian motion)
        noise = cp.random.randn(batch_size, dtype=cp.float32)
        x += step_sigma * noise
        
        # Reflecting boundary conditions
        if L is not None:
            left, right = -L/2, L/2
            # Use CuPy vectorized operations for boundaries
            
            # Note: Optimized reflection logic for [-L/2, L/2]
            
            # Reflect if beyond right boundary
            mask_right = x > right
            x[mask_right] = 2 * right - x[mask_right]
            
            # Reflect if beyond left boundary
            mask_left = x < left
            x[mask_left] = 2 * left - x[mask_left]

        # Accumulate phase
        # G_gpu[k] is a scalar, multiplied by vector x
        phi += gamma * G_gpu[k] * x * dt

    # Calculate complex signal sum for the batch (not mean)
    # This allows summing all batches externally later
    batch_sum_signal = cp.sum(cp.exp(1j * phi))
    
    # Clear VRAM (optional, but helps stability in large loops)
    del x, phi, noise
    cp.get_default_memory_pool().free_all_blocks()
    
    return batch_sum_signal

def run_massive_simulation(
    total_spins=10**10, 
    batch_size=10**7,   # Adjust based on VRAM (10^7 takes ~hundreds MB to 1GB)
    D=8e-9, 
    G_amp=2.0, 
    n_delta=300, 
    n_gap=200, 
    dt=0.1, 
    gamma=26.75,
    L=None
):
    print(f"Starting simulation with {total_spins:.0e} particles...")
    print(f"Using GPU acceleration, batch size: {batch_size:.0e}")
    
    # Prepare gradient waveform (send to GPU)
    t, G_gpu, dt, delta, big_delta, gamma = make_pgse_gradient(
        dt=dt, n_delta=n_delta, n_gap=n_gap, G_amp=G_amp, gamma=gamma
    )
    
    total_signal_sum = 0j
    particles_processed = 0
    
    # Calculate number of batches needed
    num_batches = int(np.ceil(total_spins / batch_size))
    
    start_time = time.time()
    
    for i in range(num_batches):
        # Handle the last batch which might be smaller than batch_size
        current_batch = min(batch_size, total_spins - particles_processed)
        
        # Execute GPU simulation
        batch_sum = simulate_batch_gpu(D, current_batch, G_gpu, dt, gamma, L)
        
        # Transfer GPU result back to CPU and accumulate
        total_signal_sum += cp.asnumpy(batch_sum)
        particles_processed += current_batch
        
        # Progress bar
        if (i + 1) % 10 == 0 or (i + 1) == num_batches:
            elapsed = time.time() - start_time
            print(f"Progress: {particles_processed/total_spins*100:.1f}% | Batch {i+1}/{num_batches} | Time: {elapsed:.2f}s")

    # Calculate final average signal
    S_sim = total_signal_sum / total_spins
    S_mag_sim = np.abs(S_sim)
    
    # Calculate theoretical value (Stejskal-Tanner)
    b_value = (gamma**2) * (G_amp**2) * (delta**2) * (big_delta - delta/3.0)
    # Note: Assuming units are consistent for D and b_value
    S_ST = np.exp(-b_value * D)
    
    print("-" * 30)
    print(f"Simulation complete. Total spins: {total_spins}")
    print(f"|S_sim| (GPU)   = {S_mag_sim:.6f}")
    print(f"|S_ST| (Theory) = {S_ST:.6f}")
    print(f"b-value         = {b_value:.2e}")
    
    return S_mag_sim, b_value

# --- Execution ---
# Note: 10^10 is very large. Even with GPU, it may take minutes to tens of minutes.
# For testing, try 10^8 first, then switch to 10^10.

if __name__ == "__main__":
    # Demonstration with 10^8. Change to 10**10 for 10 billion particles.
    target_spins = 10**8 
    
    # Adjust batch_size based on your VRAM.
    # 10^7 is a conservative and efficient number (approx 200MB-500MB VRAM).
    run_massive_simulation(total_spins=target_spins, batch_size=10**7)