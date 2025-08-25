#!/usr/bin/env python3
"""
Synthetic Aperture Sonar (SAS) Detection Range Model

This script models the detection range performance of SAS systems as a function of:
- Average electrical power (1-100W)
- Vehicle speed (0.1-3.0 m/s)
- Motion error (0.05-2.0% of vehicle speed)
- Target size (0.01-1.0m diameter)

Based on sonar equation fundamentals with SAS-specific modifications including:
- Effective aperture synthesis through vehicle motion
- Cavitation limits (calibrated from Urick 1975)
- Target strength frequency dependence for small targets
- Realistic seawater absorption (Francois & Garrison model)
- AUV propulsion power constraints

Author: SAS Model Development
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import warnings
warnings.filterwarnings('ignore')

class SASModel:
    """Synthetic Aperture Sonar detection range model"""
    
    def __init__(self):
        # Physical constants
        self.sound_speed = 1500  # m/s
        self.physical_aperture = 0.0254  # meters (1 inch)
        
        # System parameters
        self.transducer_efficiency = 0.35 
        # Estimated based on:
        #   ~15% electrical losses, 
        #   ~30% Piezo electro-mechanical transducer loss, 
        #   ~25% mechanical-acoustic losses,
        #   ~15% acoustic-radiation losses

        self.hotel_load = 30  # W
        self.detection_threshold = 12  # dB (SAS processing gain)
        
        # AUV drag parameters (from literature)
        self.seawater_density = 1027  # kg/m³
        self.frontal_area = 0.082  # m² (0.32m diameter AUV)
        self.drag_coefficient = 0.6 # Half sphere is 0.47, long cylinder is 0.8, streamlined body is 0.04
        self.propulsion_efficiency = 0.5
        self.max_power_budget = 500  # W total system
        
        # Frequency selection (Hz)
        self.frequencies = {
            '50 kHz': 50000,
            '75 kHz': 75000, 
            '100 kHz': 100000,
            '150 kHz': 150000,
            '200 kHz': 200000
        }
        
        self.colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#8dd1e1']
    
    def calculate_target_strength(self, frequency, target_diameter, 
                                      density_contrast=1.5, compressibility_contrast=0.1):
        """Proper physics-based target strength"""
        k = 2 * np.pi * frequency / self.sound_speed
        a = target_diameter / 2
        ka = k * a
        
        if ka < 0.3:  # Rayleigh regime
            # Proper f⁴ dependence
            ts = 10 * np.log10((9/4) * (ka)**4 * 
                            ((density_contrast - 1)/(density_contrast + 2))**2)
        elif ka > 10:  # Geometric regime  
            ts = 10 * np.log10(np.pi * a**2)
        else:  # Mie regime - needs proper series solution
            ts = self.mie_series_solution(ka, density_contrast, compressibility_contrast)
        
        return ts
    
    def calculate_effective_aperture(self, range_m, vehicle_speed):
        """
        Calculate SAS effective aperture length
        
        Args:
            range_m: Range to target (m)
            vehicle_speed: Vehicle speed (m/s)
        
        Returns:
            Effective aperture length (m)
        """
        # Effective aperture = vehicle travel distance during round trip time
        round_trip_time = (2 * range_m) / self.sound_speed
        travel_distance = vehicle_speed * round_trip_time
        
        # Limited by motion measurement precision and physical constraints
        max_aperture = min(travel_distance, 100)  # Cap at 100m for practical systems
        return max(self.physical_aperture, max_aperture)
    
    def calculate_cavitation_limit(self, frequency, depth=50):
        """
        Calculate cavitation-limited source level (calibrated from Urick 1975)
        
        Args:
            frequency: Acoustic frequency (Hz)
            depth: Water depth (m)
        
        Returns:
            Maximum source level before cavitation (dB re 1 μPa at 1m)
        """
        # Based on Urick (1975): "at 3 kHz at shallow depths, cavitation threshold 
        # is slightly more than 1 atm = 220 dB re 1 μPa"
        # Maximum levels can be 2-3× higher: ~226-230 dB re 1 μPa
        
        base_freq = 3000  # Hz (Urick's reference)
        max_operating_sl = 230  # dB re 1 μPa (practical limit allowing some cavitation)
        
        # Frequency dependence: cavitation threshold inversely related to frequency
        # Higher frequencies have lower pressure amplitudes but higher pressure gradients
        # Empirical scaling: threshold decreases ~6 dB per decade increase in frequency
        frequency_correction = -6 * np.log10(frequency / base_freq)
        
        # Depth correction: +6 dB per atmosphere (10m depth ≈ 1 atm)
        depth_correction = 6 * np.log10((depth + 10) / 10)
        
        return max_operating_sl + frequency_correction + depth_correction
    
    def calculate_seawater_absorption(self, frequency_hz):
        """
        Calculate seawater absorption coefficient (Francois & Garrison model)
        
        Args:
            frequency_hz: Frequency in Hz
        
        Returns:
            Absorption coefficient (dB/km)
        """
        f_khz = frequency_hz / 1000
        
        # Francois & Garrison approximation
        absorption1 = (0.11 * f_khz**2) / (1 + f_khz**2)
        absorption2 = (44 * f_khz**2) / (4100 + f_khz**2)  
        absorption3 = 2.75e-4 * f_khz**2
        
        return absorption1 + absorption2 + absorption3 + 0.003
    
    def calculate_motion_degradation(self, frequency, vehicle_speed, motion_error_percent):
        """
        Calculate motion-induced coherence degradation
        
        Args:
            frequency: Acoustic frequency (Hz)
            vehicle_speed: Vehicle speed (m/s)
            motion_error_percent: Motion error as % of vehicle speed
        
        Returns:
            Motion degradation (dB)
        """
        # Motion error as percentage of vehicle speed
        motion_error_meters = (motion_error_percent / 100) * vehicle_speed
        
        # Phase error due to imprecise motion measurement
        wavelength = self.sound_speed / frequency
        phase_error_radians = (2 * np.pi * motion_error_meters) / wavelength
        
        # More realistic motion degradation model
        # High-grade INS/DVL systems can maintain coherence better than simple exponential model
        phase_tolerance = 0.5  # radians (π/6 ≈ 30 degrees phase tolerance)
        degradation_factor = min(phase_error_radians / phase_tolerance, 2.0)
        
        # Gradual degradation rather than exponential
        motion_loss = 3 * degradation_factor  # 3dB per unit phase error
        
        return max(0, motion_loss)
    
    def calculate_energy_constraint(self, sonar_power, vehicle_speed):
        """
        Calculate energy constraint penalty from propulsion power
        
        Args:
            sonar_power: Sonar electrical power (W)
            vehicle_speed: Vehicle speed (m/s)
        
        Returns:
            Energy penalty (dB)
        """
        # AUV propulsion power: P = 0.5 * ρ * A * CD * v³ / η
        propulsion_power = (0.5 * self.seawater_density * self.frontal_area * 
                           self.drag_coefficient * vehicle_speed**3) / self.propulsion_efficiency
        
        # Total power budget constraint
        total_power = sonar_power + propulsion_power + self.hotel_load
        
        if total_power > self.max_power_budget:
            return -(total_power - self.max_power_budget) / 100  # More gradual penalty
        return 0
    
    def calculate_sas_range(self, electrical_power, frequency, vehicle_speed, 
                           motion_error_percent, target_size):
        """
        Calculate maximum SAS detection range using iterative sonar equation
        
        Args:
            electrical_power: Average electrical power (W)
            frequency: Acoustic frequency (Hz)
            vehicle_speed: Vehicle speed (m/s)
            motion_error_percent: Motion error as % of vehicle speed
            target_size: Target diameter (m)
        
        Returns:
            Maximum detection range (m)
        """
        # Start with initial guess
        range_m = 50  # Initial guess in meters
        converged = False
        
        # Iterative solution since effective aperture depends on range
        for iteration in range(10):
            if converged:
                break
                
            old_range = range_m
            
            # Calculate effective aperture at this range
            effective_aperture = self.calculate_effective_aperture(range_m, vehicle_speed)
            
            # SAS directivity index (much higher than physical aperture)
            wavelength = self.sound_speed / frequency
            sas_directivity = 10 * np.log10(4 * np.pi * effective_aperture**2 / wavelength**2)
            
            # Source level with transducer efficiency and cavitation limit
            theoretical_sl = (170.8 + 10 * np.log10(self.transducer_efficiency * electrical_power) 
                            + sas_directivity)
            cavitation_limit = self.calculate_cavitation_limit(frequency, 50)  # 50m depth
            source_level = min(theoretical_sl, cavitation_limit)
            
            # Target strength calculation
            target_strength = self.calculate_target_strength(frequency, target_size)
            
            # Transmission loss (one way for SAS due to multiple ping processing)
            absorption_coeff = self.calculate_seawater_absorption(frequency)
            transmission_loss = 20 * np.log10(range_m) + absorption_coeff * range_m / 1000
            
            # Ambient noise
            ambient_noise = 100 + 5 * np.log10(frequency / 1000)
            
            # Motion-induced degradation
            motion_degradation = self.calculate_motion_degradation(frequency, vehicle_speed, 
                                                                 motion_error_percent)
            
            # Energy constraint penalty
            energy_penalty = self.calculate_energy_constraint(electrical_power, vehicle_speed)
            
            # Solve for range using modified sonar equation
            signal_excess = (source_level + target_strength - transmission_loss - 
                           ambient_noise - self.detection_threshold - motion_degradation + 
                           energy_penalty)
            
            if signal_excess <= 0:
                range_m = max(10, range_m * 0.9)  # Reduce range if insufficient signal
            else:
                # Estimate new range based on signal excess
                new_range = range_m * (10 ** (signal_excess / (20 + absorption_coeff)))
                range_m = new_range
            
            # Check for convergence
            if abs(range_m - old_range) / old_range < 0.05:
                converged = True
        
        # Apply SAS-specific constraints
        final_effective_aperture = self.calculate_effective_aperture(range_m, vehicle_speed)
        max_sas_range = (final_effective_aperture * self.sound_speed) / (4 * vehicle_speed)
        
        return min(range_m, max_sas_range, 5000)  # Cap at 5km for practical systems

def create_interactive_plot():
    """Create interactive matplotlib plot with sliders"""
    
    model = SASModel()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.35)
    
    # Initial parameters
    initial_vehicle_speed = 2.0  # m/s
    initial_motion_error = 0.1   # % of vehicle speed
    initial_target_size = 0.1    # m diameter
    initial_log_scale = False
    
    # Power range: 1W to 100W
    powers = np.linspace(1, 100, 50)
    
    # Create sliders
    ax_speed = plt.axes([0.15, 0.25, 0.65, 0.03])
    ax_motion = plt.axes([0.15, 0.20, 0.65, 0.03]) 
    ax_target = plt.axes([0.15, 0.15, 0.65, 0.03])
    ax_log = plt.axes([0.15, 0.10, 0.65, 0.03])
    
    slider_speed = Slider(ax_speed, 'Vehicle Speed (m/s)', 0.1, 3.0, 
                         valinit=initial_vehicle_speed, valfmt='%.1f')
    slider_motion = Slider(ax_motion, 'Motion Error (% speed)', 0.05, 2.0,
                          valinit=initial_motion_error, valfmt='%.2f')
    slider_target = Slider(ax_target, 'Target Size (m)', 0.01, 1.0,
                          valinit=initial_target_size, valfmt='%.2f')
    slider_log = Slider(ax_log, 'Log Scale', 0, 1, valinit=0, valfmt='%d')
    
    def update_plot(val=None):
        """Update plot based on slider values"""
        ax.clear()
        
        # Get current slider values
        vehicle_speed = slider_speed.val
        motion_error = slider_motion.val  
        target_size = slider_target.val
        log_scale = slider_log.val > 0.5
        
        # Calculate ranges for each frequency
        for i, (freq_name, frequency) in enumerate(model.frequencies.items()):
            ranges = []
            for power in powers:
                range_m = model.calculate_sas_range(power, frequency, vehicle_speed,
                                                  motion_error, target_size)
                ranges.append(range_m)
            
            ax.plot(powers, ranges, label=freq_name, color=model.colors[i], linewidth=2)
        
        # Set scale and labels
        if log_scale:
            ax.set_yscale('log')
            ax.set_ylim(10, 10000)
        else:
            ax.set_yscale('linear')
        
        ax.set_xlabel('Average Electrical Power (W)', fontsize=12)
        ax.set_ylabel('Max Detection Range (m)', fontsize=12)
        ax.set_title(f'SAS Detection Range vs Power\n'
                    f'Target: {target_size:.2f}m, Speed: {vehicle_speed:.1f}m/s, '
                    f'Motion Error: {motion_error:.2f}%', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add analysis text
        ka_100khz = 2 * np.pi * (target_size/2) / (model.sound_speed / 100000)
        if ka_100khz > 10:
            regime = "Geometric (range-independent TS)"
        elif ka_100khz < 0.5:
            regime = "Rayleigh (TS ∝ f⁴)"
        else:
            regime = "Mie (complex oscillatory)"
            
        ax.text(0.02, 0.98, f'Target regime: {regime}\nka @ 100kHz: {ka_100khz:.2f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Connect sliders to update function
    slider_speed.on_changed(update_plot)
    slider_motion.on_changed(update_plot)
    slider_target.on_changed(update_plot)
    slider_log.on_changed(update_plot)
    
    # Initial plot
    update_plot()
    
    plt.show()

def generate_analysis_plots():
    """Generate static analysis plots showing key relationships"""
    
    model = SASModel()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Target size effect
    target_sizes = [0.05, 0.1, 0.2, 0.5, 1.0]
    powers = np.linspace(1, 100, 50)
    
    for target_size in target_sizes:
        ranges_100khz = []
        for power in powers:
            range_m = model.calculate_sas_range(power, 100000, 2.0, 0.1, target_size)
            ranges_100khz.append(range_m)
        
        ax1.plot(powers, ranges_100khz, label=f'{target_size}m target', linewidth=2)
    
    ax1.set_xlabel('Power (W)')
    ax1.set_ylabel('Range (m)')
    ax1.set_title('Target Size Effect (100 kHz, 2 m/s)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Frequency comparison for small targets
    frequencies = [50000, 100000, 150000, 200000]
    
    for freq in frequencies:
        ranges = []
        for power in powers:
            range_m = model.calculate_sas_range(power, freq, 2.0, 0.1, 0.1)
            ranges.append(range_m)
        
        ax2.plot(powers, ranges, label=f'{freq//1000} kHz', linewidth=2)
    
    ax2.set_xlabel('Power (W)')
    ax2.set_ylabel('Range (m)')
    ax2.set_title('Frequency Effect (0.1m target, 2 m/s)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Vehicle speed effect
    speeds = [0.5, 1.0, 2.0, 3.0]
    
    for speed in speeds:
        ranges = []
        for power in powers:
            range_m = model.calculate_sas_range(power, 100000, speed, 0.1, 0.1)
            ranges.append(range_m)
        
        ax3.plot(powers, ranges, label=f'{speed} m/s', linewidth=2)
    
    ax3.set_xlabel('Power (W)')
    ax3.set_ylabel('Range (m)')
    ax3.set_title('Vehicle Speed Effect (0.1m target, 100 kHz)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cavitation limits
    freqs = np.array([10, 20, 50, 100, 200, 500]) * 1000  # Hz
    powers = np.linspace(1, 100, 50)
    
    for freq in freqs:
        cavitation_limits = [model.calculate_cavitation_limit(freq) for _ in powers]
        theoretical_sls = []
        
        for power in powers:
            # Assume moderate effective aperture
            eff_aperture = 0.5  # meters
            wavelength = model.sound_speed / freq
            directivity = 10 * np.log10(4 * np.pi * eff_aperture**2 / wavelength**2)
            theoretical_sl = 170.8 + 10 * np.log10(0.5 * power) + directivity
            theoretical_sls.append(theoretical_sl)
        
        # Show where cavitation limiting occurs
        limited_powers = []
        limited_sls = []
        
        for i, power in enumerate(powers):
            if theoretical_sls[i] > cavitation_limits[i]:
                limited_powers.append(power)
                limited_sls.append(cavitation_limits[i])
        
        if limited_powers:
            ax4.plot(limited_powers, limited_sls, '--', label=f'{freq//1000} kHz (limited)',
                    linewidth=2)
    
    ax4.set_xlabel('Power (W)')
    ax4.set_ylabel('Source Level (dB re 1 μPa)')
    ax4.set_title('Cavitation Limits vs Power')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_model_summary():
    """Print summary of model equations and parameters"""
    
    print("="*80)
    print("SAS SONAR DETECTION RANGE MODEL SUMMARY")
    print("="*80)
    
    print("\nKEY EQUATIONS:")
    print("-" * 40)
    
    print("\n1. Fundamental Sonar Equation:")
    print("   SNR = SL + TS - TL - NL + DI - Motion_loss")
    print("   Detection when SNR ≥ DT = 12 dB")
    
    print("\n2. SAS Effective Aperture:")
    print("   L_eff = max(L_physical, min(v × 2R/c, 100))")
    print("   where v = vehicle speed, R = range, c = sound speed")
    
    print("\n3. Target Strength (size-dependent):")
    print("   TS = TS_geometric + (1-d) × 15×log₁₀(f/50000)")  
    print("   where d = target diameter, provides up to 15 dB advantage for small targets")
    
    print("\n4. Cavitation Limit (Urick 1975):")
    print("   SL_max = 230 - 6×log₁₀(f/3000) + 6×log₁₀((depth+10)/10)")
    print("   Based on empirical data: 3 kHz threshold ≈ 220 dB re 1 μPa")
    
    print("\n5. Seawater Absorption (Francois & Garrison):")
    print("   α = 0.11f²/(1+f²) + 44f²/(4100+f²) + 2.75×10⁻⁴f² + 0.003")
    print("   where f in kHz, α in dB/km")
    
    print("\nMODEL PARAMETERS:")
    print("-" * 40)
    print("• Physical aperture: 1 inch (0.0254 m)")
    print("• Transducer efficiency: 50%") 
    print("• Detection threshold: 12 dB (SAS processing gain)")
    print("• Hotel load: 30W (electronics)")
    print("• AUV drag: A=0.082m², C_D=0.2, η=0.5")
    print("• Motion error: 0.1% of vehicle speed (high-grade INS/DVL)")
    print("• Power budget: 800W total system limit")
    
    print("\nKEY INSIGHTS:")
    print("-" * 40)
    print("• Small targets (<0.5m) strongly favor higher frequencies")  
    print("• Cavitation limits prevent excessive power scaling")
    print("• Effective aperture grows with range and vehicle speed")
    print("• Energy budget constrains high-speed operations")
    print("• Model explains commercial frequency selection patterns")

if __name__ == "__main__":
    print("SAS Sonar Detection Range Model")
    print("===============================")
    print("\nOptions:")
    print("1. Interactive plot with sliders")
    print("2. Generate analysis plots")
    print("3. Print model summary")
    print("4. All of the above")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice in ['1', '4']:
        print("\nLaunching interactive plot...")
        create_interactive_plot()
    
    if choice in ['2', '4']:
        print("\nGenerating analysis plots...")
        generate_analysis_plots()
    
    if choice in ['3', '4']:
        print_model_summary()
    
    if choice not in ['1', '2', '3', '4']:
        print("Invalid choice. Running model summary...")
        print_model_summary()