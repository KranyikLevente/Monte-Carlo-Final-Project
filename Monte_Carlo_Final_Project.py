import numpy as np
import matplotlib.pyplot as plt
import math
import random
import pandas as pd
from scipy.interpolate import interp1d

# --- CONSTANTS ---
ELECTRON_REST_ENERGY_MEV = 0.511 
PAIR_PRODUCTION_THRESHOLD = 1.022 

def load_xcom_data(filename='xs_raw.txt'):
    try:
        # skiprows=3: Skips the header rows
        data = np.loadtxt(filename, skiprows=3)
        
        energies = data[:, 0]
        sigma_compton = data[:, 1]
        sigma_photo = data[:, 2]
        # Total pair production is the sum of nuclear and electron field production
        sigma_pair = data[:, 3] + data[:, 4]
        
        return energies, sigma_compton, sigma_photo, sigma_pair

    except Exception as e:
        print(f"Error reading file: {e}. Using fallback data (testing only!).")
        # Fallback data for non-existent file (for testing purposes)
        return np.array([0.01, 10.0]), np.array([0.1, 0.01]), np.array([0.1, 0.01]), np.array([0.0, 0.0])

class XCOMDataFromFile:
    def __init__(self, filename='xs_raw.txt'):
        self.energies, self.s_compton, self.s_photo, self.s_pair = load_xcom_data(filename)
        # Create interpolation functions using linear kind
        self.f_compton = interp1d(self.energies, self.s_compton, kind='linear', fill_value="extrapolate")
        self.f_photo = interp1d(self.energies, self.s_photo, kind='linear', fill_value="extrapolate")
        self.f_pair = interp1d(self.energies, self.s_pair, kind='linear', fill_value="extrapolate")

    def get_cross_sections(self, energy_mev):
        sc = max(0.0, float(self.f_compton(energy_mev)))
        sph = max(0.0, float(self.f_photo(energy_mev)))
        # Pair production only happens above the threshold
        sp = max(0.0, float(self.f_pair(energy_mev))) if energy_mev >= PAIR_PRODUCTION_THRESHOLD else 0.0
        return sc, sph, sp


def isotropic_dir():
    """ Generates a normalized isotropic direction vector. """
    cos_theta = 2 * np.random.rand() - 1
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = 2 * np.pi * np.random.rand()
    return np.array([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        cos_theta
    ])

def photon_angle(energy_in):
    """ Samples the scattering angle and energy of the scattered photon using Kahn's method. """
    if energy_in <= 0:
        return 0.0, 0.0

    a = energy_in / ELECTRON_REST_ENERGY_MEV
    b = 1.0 / a
    c = 1.0 + 2.0 * a
    d = c / (9.0 + 2.0 * a)
    
    # Acceptance-Rejection Loop
    while True:
        r1, r2, r3 = random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)
        e = 1.0 + 2.0 * a * r1
        
        if r2 <= d:
            f = e
            g = 4.0 * (1.0/f - 1.0/(f*f))
        else:
            f = c / e
            g = 0.5 * ((1.0 + b - b * f)**2 + 1.0/f)

        if r3 < g:
            break

    cos_theta = np.clip(1.0 + b - b * f, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    energy_out = energy_in / f
    
    return theta, energy_out

def photon_direction(angle):
    """ Samples the new direction vector w.r.t the local Z-axis"""
    nz = np.cos(angle)
    rho = np.sin(angle)
    phi = random.uniform(0, 2 * np.pi)
    
    nx = rho * np.cos(phi)
    ny = rho * np.sin(phi)
    
    return np.array([nx, ny, nz])

def scatter_photon(energy_in):
    """ Determines the new direction and energy of the scattered photon. """
    angle, energy_out = photon_angle(energy_in)
    direction = photon_direction(angle)
    return direction, energy_out

def intersection(height: float, radius: float, point, direction):
    """ 
    Calculates the closest positive parameter intersection point between a ray 
    and the cylinder centered at the origin.
    """
    # Ensure point is a numpy array for vector operations if needed later
    point = np.array(point)
    direction = np.array(direction)

    x, y, z = point
    u, v, w = direction

    z_top = height / 2
    z_bottom = -height / 2
    EPSILON = 1e-6
    hit_t_values = []

    # Cylinder wall intersection
    A = u**2 + v**2
    B = 2 * (x * u + y * v)
    C = x**2 + y**2 - radius**2
    
    if abs(A) > EPSILON:
        discriminant = B**2 - 4 * A * C
        
        if discriminant >= -EPSILON:
            sqrt_delta = math.sqrt(max(0, discriminant))
            t1 = (-B - sqrt_delta) / (2 * A)
            t2 = (-B + sqrt_delta) / (2 * A)
            
            for t in [t1, t2]:
                if t > EPSILON:
                    z_hit = z + t * w
                    if z_bottom <= z_hit <= z_top:
                        hit_t_values.append(t)
    
    # Cap intersection
    if abs(w) > EPSILON:
        for z_plane in [z_top, z_bottom]:
            t = (z_plane - z) / w
            
            if t > EPSILON:
                x_hit = x + t * u
                y_hit = y + t * v
                if x_hit**2 + y_hit**2 <= radius**2 + EPSILON:
                    hit_t_values.append(t)

    if not hit_t_values:
        return None 

    t_closest = min(hit_t_values)
    t_farther = max(hit_t_values)
    
    hit_point = point + t_closest * direction
    out_point = point + t_farther * direction
    
    return hit_point, out_point, t_closest

def track_photon(start_position, start_direction, start_energy, height, radius, rho, XCOM_handler):
    """
    Simulates the path of a single photon (or annihilation photon) within the detector.
    Uses recursion to handle the two annihilation photons resulting from pair production.
    """
    
    current_position = np.array(start_position)
    current_direction = np.array(start_direction)
    current_energy = start_energy
    
    deposited_energy = 0.0
    
    active = True
    while active and current_energy > 0:
        
        # 1. Cross-section update (energy-dependent)
        compton_xs, photo_xs, pair_xs = XCOM_handler.get_cross_sections(current_energy)
        total_xs_mass = compton_xs + photo_xs + pair_xs 
        mu_total = total_xs_mass * rho
        
        if mu_total < 1e-12: 
            break
            
        mean_free_path = -math.log(random.random()) / mu_total

        # 2. Boundary Distance (d_out) update
        # Since current_position is inside, t_closest gives the distance to the exit point
        intersect_inside = intersection(height, radius, current_position, current_direction)
        
        if intersect_inside is None:
            distance_to_exit = 0 
        else:
            hit_point_tmp, out_point_tmp, distance_to_exit = intersect_inside
        
        # 3. Decision and movement
        if mean_free_path >= distance_to_exit:
            active = False # Escapes the detector
        else:
            # Collision occurs
            current_position = current_position + current_direction * mean_free_path
            
            random_num = random.random() * total_xs_mass
            
            # 4. INTERACTION SAMPLING
            if random_num < photo_xs:
                # Photoelectric Effect: All energy is deposited
                deposited_energy += current_energy
                current_energy = 0
            
            elif random_num < photo_xs + compton_xs:
                # Compton Scattering: Update direction and energy
                new_direction, energy_out = scatter_photon(current_energy)
                deposited_energy += current_energy - energy_out # Deposited part
                current_energy = energy_out
                current_direction = new_direction # Direction update
            
            else:
                # Pair Production
                if current_energy >= PAIR_PRODUCTION_THRESHOLD:
                    
                    # 4.1. Kinetic energy deposition
                    deposited_energy += (current_energy - PAIR_PRODUCTION_THRESHOLD)
                    
                    # 4.2. Simulate two annihilation photons (Recursion)
                    annihilation_energy = ELECTRON_REST_ENERGY_MEV
                    
                    # Photon 1: isotropic direction dir1
                    dir1 = isotropic_dir()
                    deposited_energy += track_photon(current_position, dir1, annihilation_energy, 
                                                     height, radius, rho, XCOM_handler)
                    
                    # Photon 2: opposite direction -dir1
                    dir2 = -dir1
                    deposited_energy += track_photon(current_position, dir2, annihilation_energy, 
                                                     height, radius, rho, XCOM_handler)
                    
                    current_energy = 0 # Original photon terminates
                else:
                    # Not enough energy for pair production (should be handled by XS sampling, but safety termination)
                    current_energy = 0 
    
    return deposited_energy

def energy_detection_calculation(height, radius, 
                                 num_of_particles, energy_source, FWHM, rho, point_source, XCOM_handler):
    
    # E_gamma and FWHM must be converted to MeV for internal calculation
    energy_source_mev = energy_source / 1000.0
    FWHM_mev = FWHM / 1000.0
    
    # Result collection
    detected_energies_measured = [] # Noisy data (for spectrum)
    detected_energies_pure = []     # Pure data (for efficiency)
    
    num_particles_reached_detector = 0
    
    for i in range(num_of_particles):
        direction = isotropic_dir()
        
        # 1. EXTERNAL CALL (ENTRY)
        intersect_result = intersection(height, radius, point_source, direction)
        
        if intersect_result is None:
            continue
            
        num_particles_reached_detector += 1
        
        hit_point, out_point_orig, distance_to_entry_point = intersect_result
        
        # 2. START TRACKING FROM ENTRY POINT
        total_deposited_pure = track_photon(hit_point, direction, energy_source_mev, 
                                            height, radius, rho, XCOM_handler)
        
        # 3. GAUSSIAN SMEARING AND RESULT COLLECTION
        if total_deposited_pure > 0:
             # Apply Gaussian noise
             sigma = FWHM_mev / 2.355
             measured = total_deposited_pure + random.gauss(0, sigma)
             detected_energies_measured.append(measured)
             
             detected_energies_pure.append(total_deposited_pure)

    # --- EFFICIENCY CALCULATION ---
    
    E_det = np.sum(detected_energies_pure)
    E_tot = num_of_particles * energy_source_mev
    E_int = num_particles_reached_detector * energy_source_mev

    total_efficiency = E_det / E_tot if E_tot > 0 else 0
    intrinsic_efficiency = E_det / E_int if E_int > 0 else 0

    return detected_energies_measured, total_efficiency, intrinsic_efficiency, num_particles_reached_detector

def run_simulation(params, num_particles, plot_spectrum=True):
    
    if plot_spectrum:
        print(f"\n--- Running Simulation: {params['label']} ---")
    
    # Load XCOM data
    XCOM_handler = XCOMDataFromFile(filename='xs_raw.txt') 
    
    # Call simulation
    results = energy_detection_calculation(
        height=params['h'],
        radius=params['R'],
        num_of_particles=num_particles,
        energy_source=params['E_gamma'],
        FWHM=params['FWHM'],
        rho=params['rho'],
        point_source=params['rs'],
        XCOM_handler=XCOM_handler
    )
    
    # Unpack results
    energies, eta_tot, eta_int, n_reached = results
    
    if plot_spectrum:
        print(f"Source Energy: {params['E_gamma']} keV")
        print(f"Particles Reached (N_int): {n_reached} / {num_particles}")
        print(f"Total Efficiency (eta_tot): {eta_tot:.4f}")
        print(f"Intrinsic Efficiency (eta_int): {eta_int:.4f}")

    # --- SPECTRUM PLOTTING (Only if requested) ---
    if plot_spectrum and energies:
        energies_keV = np.array(energies) * 1000 
        plt.figure(figsize=(10, 6))
        
        # Histogram
        E_max_plot = params['E_gamma'] * 1.1 
        plt.hist(energies_keV, bins=1024, range=(0, E_max_plot), histtype='stepfilled', color='darkblue', alpha=0.7)
        
        # Lines: Photopeak
        plt.axvline(params['E_gamma'], color='green', linestyle='-', linewidth=1.5, 
                    label=f"Photopeak ({params['E_gamma']:.1f} keV)") 
        
        # Lines: Compton Edge (Corrected calculation!)
        E_mev = params['E_gamma'] / 1000.0
        m_e = 0.511
        E_backscatter_mev = E_mev / (1 + (2 * E_mev) / m_e)
        E_edge_keV = (E_mev - E_backscatter_mev) * 1000
        
        plt.axvline(E_edge_keV, color='red', linestyle='--', linewidth=1.5, 
                    label=f'Compton Edge ({E_edge_keV:.1f} keV)')
        
        # Lines: Escape Peaks
        if params['E_gamma'] >= 1022:
            plt.axvline(params['E_gamma'] - 511, color='orange', linestyle=':', label='Single Escape')
            plt.axvline(params['E_gamma'] - 1022, color='purple', linestyle=':', label='Double Escape')
        
        plt.title(f"Gamma Spectrum - {params['label']}")
        plt.xlabel("Deposited Energy (keV)")
        plt.ylabel("Counts")
        plt.legend()
        plt.grid(True)
        plt.show()
    return eta_tot, eta_int

def main():
    NUM_PARTICLES = 100000 
    
    # --- BASIC PARAMETERS ---
    # Spectrum A (Cs-137)
    params_A = {
        'label': "Spectrum A (Cs-137)",
        'rs': (3.0, -3.0, 2.0),
        'E_gamma': 661.7,
        'R': 2.5, 'h': 3.0, 'rho': 3.67, 'FWHM': 6.0
    }
    
    # Spectrum B (Co-60)
    params_B = {
        'label': "Spectrum B (Co-60)",
        'rs': (4.0, 4.0, 0.0),
        'E_gamma': 1332.5,
        'R': 3.0, 'h': 5.0, 'rho': 3.67, 'FWHM': 8.0
    }

    # =========================================================================
    # TASK (i): Recording individual spectra
    # =========================================================================
    print("--- TASK (i) Start: Recording Spectra ---")
    # Here, histograms are plotted because plot_spectrum=True
    run_simulation(params_A, NUM_PARTICLES, plot_spectrum=True)
    run_simulation(params_B, NUM_PARTICLES, plot_spectrum=True)


    # =========================================================================
    # TASK (ii): Efficiency vs Position (with Spectrum A parameters)
    # =========================================================================
    print("\n--- TASK (ii) Start: Efficiency vs Position ---")
    
    pos_indices = []
    eff_tot_pos = []
    eff_int_pos = []
    
    start_pos = np.array([1.0, 3.5, 2.0])
    end_pos = np.array([-4.0, -1.5, 2.0])
    
    task_ii_params = params_A.copy()
    task_ii_params['label'] = "Task II Simulation"

    # 11 steps between coordinates
    for i in range(11):
        t = i / 10.0
        current_rs = start_pos + t * (end_pos - start_pos)
        
        task_ii_params['rs'] = tuple(current_rs)
        
        # Run without plotting
        e_tot, e_int = run_simulation(task_ii_params, NUM_PARTICLES, plot_spectrum=False)
        
        pos_indices.append(i)
        eff_tot_pos.append(e_tot)
        eff_int_pos.append(e_int)
        print(f"[Task ii] Step {i}: Pos={current_rs}, Eff_Tot={e_tot:.4f}")

    # Task (ii) Graph
    plt.figure(figsize=(10, 6))
    plt.plot(pos_indices, eff_tot_pos, 'o-', color='blue', label=r'Total Efficiency ($\eta_{tot}$)')
    plt.plot(pos_indices, eff_int_pos, 's-', color='red', label=r'Intrinsic Efficiency ($\eta_{int}$)')
    plt.title("Task (ii): Efficiency vs Source Position")
    plt.xlabel("Position Index (0: Start -> 10: End)")
    plt.ylabel("Efficiency")
    plt.legend()
    plt.grid(True)
    plt.show()


    # =========================================================================
    # TASK (iii): Efficiency vs Energy (with Spectrum B parameters)
    # 0.4 MeV -> 4.0 MeV, in 10 steps
    # =========================================================================
    print("\n--- TASK (iii) Start: Efficiency vs Energy ---")
    
    energies_plot = []
    eff_tot_en = []
    eff_int_en = []
    
    task_iii_params = params_B.copy()
    task_iii_params['label'] = "Task III Simulation"
    
    # 10 steps between energies 
    # Start: 400 keV, End: 4000 keV
    start_E = 400.0
    end_E = 4000.0
    steps = 10
    
    for i in range(steps):
        # Linear step calculation
        # i=0 -> 400, i=9 -> 4000
        current_E = start_E + i * (end_E - start_E) / (steps - 1)
        
        task_iii_params['E_gamma'] = current_E
        
        # Run without plotting
        e_tot, e_int = run_simulation(task_iii_params, NUM_PARTICLES, plot_spectrum=False)
        
        energies_plot.append(current_E / 1000.0) # Plotting in MeV, axis looks better
        eff_tot_en.append(e_tot)
        eff_int_en.append(e_int)
        print(f"[Task iii] Step {i}: E={current_E:.1f} keV, Eff_Int={e_int:.4f}")

    # Task (iii) Graph 
    plt.figure(figsize=(10, 6))
    plt.plot(energies_plot, eff_tot_en, 'o-', color='blue', label=r'Total Efficiency ($\eta_{tot}$)')
    plt.plot(energies_plot, eff_int_en, 's-', color='red', label=r'Intrinsic Efficiency ($\eta_{int}$)')
    plt.title("Task (iii): Efficiency vs Gamma Energy")
    plt.xlabel("Source Energy (MeV)")
    plt.ylabel("Efficiency")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\n--- All simulations finished successfully. ---")

if __name__ == '__main__':
    main()