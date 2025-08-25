#!/usr/bin/env python3
"""
Synthetic Aperture Sonar (SAS) Detection Range Model
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
        self.transducer_efficiency = 0.25  # Small AUV systems including all losses
        self.hotel_load = 150  # W (SAS processing)
        self.detection_threshold = 30  # dB (SNR for reliable detection in ocean clutter and multipath)
        
        # AUV parameters (keeping user-specified drag coefficient)
        self.seawater_density = 1027  # kg/m³
        self.frontal_area = 0.082  # m² (0.32m diameter AUV)
        self.drag_coefficient = 0.6  # Cylindrical with hemisphere nose
        self.propulsion_efficiency = 0.5
        self.max_power_budget = 500  # W total system
        
        # Environmental parameters
        self.water_temperature = 15  # °C
        self.salinity = 35  # ppt
        self.ph = 8.1  # typical seawater
        self.sea_state = 3  # moderate seas
        
        # Frequency selection (Hz)
        self.frequencies = {
            '50 kHz': 50000,
            '75 kHz': 75000, 
            '100 kHz': 100000,
            '150 kHz': 150000,
            '200 kHz': 200000
        }
        
        self.colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#8dd1e1']
    
    def mie_series_approximation(self, ka, density_contrast, compressibility_contrast):
        """
        Mie series approximation for intermediate ka values
        
        Args:
            ka: Size parameter (k * radius)
            density_contrast: Target/water density ratio
            compressibility_contrast: Target/water compressibility ratio
        
        Returns:
            Backscattering cross-section coefficient
        """
        # Acoustic contrasts
        g = density_contrast
        h = compressibility_contrast
        
        # Mie series coefficients (first few terms)
        a0 = (g - 1) / (g + 2)  # Monopole term
        a1 = 2 * (g - 1) * (1 - h) / (2 * g + 1 - h)  # Dipole term
        
        # For ka > 1, include higher order terms and proper oscillatory behavior
        if ka < 1:
            # Low ka limit - Rayleigh-like behavior
            sigma_bs = (9/4) * (ka**4) * (abs(a0)**2 + (ka**2/9) * abs(a1)**2)
        else:
            # Higher ka - include resonance and oscillations
            # This approximates the complex Mie oscillations
            oscillation = 1 + 0.3 * np.sin(2 * ka - np.pi/4) / np.sqrt(ka)
            
            # Transition from ka^4 to ka^2 behavior (approaching geometric limit)
            transition_factor = ka**2 / (1 + ka**2/4)  # Smooth transition
            
            sigma_bs = (9/4) * transition_factor * (abs(a0)**2 + abs(a1)**2/3) * oscillation
        
        return max(sigma_bs, 1e-10)  # Prevent zero values
    
    def calculate_target_strength(self, frequency, target_diameter, 
                                density_contrast=1.5, compressibility_contrast=0.8):
        """
        Calculate target strength using proper scattering physics
        
        Args:
            frequency: Acoustic frequency (Hz)
            target_diameter: Target diameter (m)
            density_contrast: Target/water density ratio
            compressibility_contrast: Target/water compressibility ratio
        
        Returns:
            Target strength (dB re 1 m²)
        """
        k = 2 * np.pi * frequency / self.sound_speed
        a = target_diameter / 2  # radius
        ka = k * a
        
        if ka < 0.5:  # True Rayleigh regime (more restrictive)
            # Proper f⁴ dependence for small targets
            g = density_contrast
            h = compressibility_contrast
            
            # Rayleigh scattering formula
            monopole_term = ((g - 1) / (g + 2))**2
            dipole_term = ((g - 1) * (1 - h) / (2*g + 1 - h))**2
            
            sigma_bs = (9/4) * (ka)**4 * (monopole_term + (ka**2/9) * dipole_term)
            
        elif ka > 15:  # Geometric regime
            # High frequency limit - approaches geometric cross-section
            # For a sphere: σ_bs = π*a² (physical cross-section)
            # Target strength = 10*log10(σ_bs / (4π))
            geometric_cross_section = np.pi * a**2
            sigma_bs = geometric_cross_section / (4 * np.pi)  # Normalize properly
            
        else:  # Mie/resonance regime (0.5 ≤ ka ≤ 15)
            # Use Mie series approximation
            sigma_bs = self.mie_series_approximation(ka, density_contrast, compressibility_contrast)
        
        # Convert to target strength
        ts = 10 * np.log10(max(sigma_bs, 1e-10))  # Prevent log(0)
        
        return ts
    
    def calculate_effective_aperture(self, range_m, vehicle_speed, motion_error_percent, frequency=100000):
        """
        Calculate SAS effective aperture with proper coherence constraints
        
        Args:
            range_m: Range to target (m)
            vehicle_speed: Vehicle speed (m/s)
            motion_error_percent: Motion error as % of vehicle speed
            frequency: Operating frequency (Hz) for wavelength-dependent coherence
        
        Returns:
            Effective aperture length (m)
        """
        # Maximum theoretical aperture from geometry
        round_trip_time = (2 * range_m) / self.sound_speed
        geometric_aperture = vehicle_speed * round_trip_time
        
        # More realistic coherence constraint
        # Motion error affects RELATIVE positioning between pings, not absolute position
        # High-grade INS maintains relative accuracy much better than absolute accuracy
        
        # Relative motion error between adjacent pings is much smaller than cumulative error
        ping_rate = 10  # Hz (typical SAS ping rate)
        ping_interval = 1.0 / ping_rate  # seconds
        relative_motion_error = (motion_error_percent / 100) * vehicle_speed * ping_interval
        
        # Use frequency-dependent coherence criterion
        # Each frequency uses its own wavelength for proper coherence calculation
        wavelength = self.sound_speed / frequency
        phase_tolerance = wavelength / 8  # Standard λ/8 coherence criterion
        
        # Maximum aperture length before phase decorrelation
        if relative_motion_error > 0:
            # Coherence length based on relative ping-to-ping errors, not cumulative
            coherence_limit = phase_tolerance / relative_motion_error * ping_interval * vehicle_speed
            # This gives the maximum aperture where phase errors accumulate to λ/4
        else:
            coherence_limit = 1000  # Very large if no motion error
        
        # Alternative coherence limit based on traditional approach but with looser criterion
        traditional_motion_error = (motion_error_percent / 100) * vehicle_speed
        traditional_coherence = phase_tolerance / traditional_motion_error  # λ/4 instead of λ/8
        
        # Use the less restrictive of the two approaches
        coherence_limit = max(coherence_limit, traditional_coherence)
        
        # More restrictive practical limits for commercial systems:
        # - INS drift accumulation
        # - Water current variations  
        # - Platform flex and vibrations
        # - Computational processing limits
        
        # Frequency-dependent stability limits (higher freq = shorter coherent aperture)
        freq_stability_factor = max(0.3, 100000 / frequency)  # Shorter apertures at higher freq
        practical_limit = min(geometric_aperture, 30 * freq_stability_factor)  # Even shorter practical limit
        
        # INS quality limit - even high-grade systems degrade over time
        ins_limit = min(geometric_aperture, 100)  # 100m maximum for high-grade INS
        
        # Effective aperture is minimum of all constraints
        effective_aperture = min(geometric_aperture, coherence_limit, practical_limit, ins_limit)
        
        return max(self.physical_aperture, effective_aperture)
    
    def calculate_sas_processing_gain(self, effective_aperture, frequency):
        """
        Calculate SAS-specific processing gains
        
        Args:
            effective_aperture: Effective synthetic aperture (m)
            frequency: Acoustic frequency (Hz)
        
        Returns:
            Processing gain (dB)
        """
        wavelength = self.sound_speed / frequency
        
        # Along-track beamforming gain (this is the main SAS advantage)
        along_track_gain = 10 * np.log10(effective_aperture / self.physical_aperture)
        
        # Multi-look processing gain (coherent integration)
        # Number of independent looks based on aperture sampling
        n_looks = max(1, int(effective_aperture / (wavelength/2)))  # Nyquist sampling
        coherent_integration_gain = 10 * np.log10(min(n_looks, 50))  # More conservative cap
        
        # SAS sidelobe suppression benefit
        sidelobe_suppression = 2  # dB (more conservative)
        
        total_gain = along_track_gain + coherent_integration_gain + sidelobe_suppression
        
        # REALISTIC processing gain limits for commercial SAS
        # Real systems are limited by:
        # - Imperfect motion compensation
        # - Multipath interference  
        # - Platform vibrations
        # - Processing limitations
        freq_penalty = max(0, 2 * np.log10(frequency / 50000))  # Penalty above 50kHz
        
        # Processing gain limits for practical systems
        # Commercial SAS systems achieve 8-12dB typical, 15dB maximum under ideal conditions
        return min(total_gain - freq_penalty, 12)  # 12dB maximum gain
    
    def calculate_cavitation_limit(self, frequency, depth=50):
        """
        Cavitation-limited source level
        
        Args:
            frequency: Acoustic frequency (Hz)
            depth: Water depth (m)
        
        Returns:
            Maximum source level before cavitation (dB re 1 μPa at 1m)
        """
        # Base cavitation threshold (at 10 kHz, sea level)
        # Higher frequencies have HIGHER cavitation thresholds
        base_freq = 10000  # Hz
        base_threshold = 220  # dB re 1 μPa
        
        # Frequency dependence: threshold increases with frequency
        # Higher frequencies create lower pressure amplitudes for same intensity
        frequency_correction = 3 * np.log10(frequency / base_freq)  # +3dB per decade
        
        # Depth correction: +6 dB per atmosphere
        depth_correction = 6 * np.log10((depth + 10) / 10)
        
        # Conservative operating limit (allow some micro-bubbles)
        operating_margin = 5  # dB below inception threshold
        
        max_sl = base_threshold + frequency_correction + depth_correction - operating_margin
        
        return max_sl
    
    def calculate_seawater_absorption(self, frequency_hz):
        """
        Francois & Garrison model with environmental parameters
        
        Args:
            frequency_hz: Frequency in Hz
        
        Returns:
            Absorption coefficient (dB/km)
        """
        f_khz = frequency_hz / 1000
        T = self.water_temperature  # °C
        S = self.salinity  # ppt
        
        # Temperature and salinity corrections
        temp_factor = 1 + 0.0383 * (T - 20)
        salinity_factor = 1 + 0.0268 * (S - 35) / 35
        
        # Boric acid relaxation
        f1 = 0.78 * np.sqrt(S/35) * np.exp(T/26)
        absorption1 = (0.106 * f1 * f_khz**2) / (f1**2 + f_khz**2)
        
        # Magnesium sulfate relaxation
        f2 = 42 * np.exp(T/17)
        absorption2 = (0.52 * (S/35) * f2 * f_khz**2) / (f2**2 + f_khz**2)
        
        # Pure water absorption
        absorption3 = 0.00049 * f_khz**2 * np.exp(-(T-27)/17)
        
        total_absorption = (absorption1 + absorption2 + absorption3) * temp_factor * salinity_factor
        
        return total_absorption
    
    def wenz_noise_model(self, frequency):
        """
        Ambient noise model based on Wenz curves
        
        Args:
            frequency: Acoustic frequency (Hz)
        
        Returns:
            Noise spectral density (dB re 1 μPa²/Hz)
        """
        f_khz = frequency / 1000
        
        # Wenz noise components
        # Thermal noise (dominant at high frequencies)
        thermal_noise = -15 + 20 * np.log10(f_khz)
        
        # Sea state noise (dominant at mid frequencies)
        # Sea state 3: moderate seas, 3-4 foot waves
        sea_noise = 50 + 7.5 * self.sea_state - 20 * np.log10(f_khz)
        
        # Shipping noise (dominant at low frequencies)
        shipping_noise = 76 - 20 * np.log10(f_khz)
        
        # Wind noise
        wind_speed = 5 + 2.5 * self.sea_state  # Approximate wind from sea state
        wind_noise = 44 + 0.5 * wind_speed - 15 * np.log10(f_khz)
        
        # Combine noise sources (energy addition)
        combined_noise = 10 * np.log10(
            10**(thermal_noise/10) + 
            10**(sea_noise/10) + 
            10**(shipping_noise/10) + 
            10**(wind_noise/10)
        )
        
        # Convert to 1 Hz bandwidth
        return combined_noise
    
    def calculate_motion_degradation(self, frequency, vehicle_speed, motion_error_percent, 
                                   effective_aperture):
        """
        Calculate motion-induced coherence degradation for SAS
        
        Args:
            frequency: Acoustic frequency (Hz)
            vehicle_speed: Vehicle speed (m/s)
            motion_error_percent: Motion error as % of vehicle speed
            effective_aperture: Effective synthetic aperture (m)
        
        Returns:
            Motion degradation (dB)
        """
        # Motion degradation model for commercial systems
        # Real SAS systems suffer significant degradation from:
        # - Imperfect INS measurements
        # - Platform vibrations
        # - Water current effects
        # - Multipath propagation
        
        motion_error_m = (motion_error_percent / 100) * vehicle_speed
        wavelength = self.sound_speed / frequency
        
        # Phase error from motion uncertainty
        phase_error = (2 * np.pi * motion_error_m) / wavelength
        
        # Real systems have significant residual errors even after compensation
        # Commercial INS + DVL typically leaves 20-30% residual error
        residual_factor = 0.25  # 25% residual error after compensation
        effective_phase_error = phase_error * residual_factor
        
        # Coherence loss from phase errors
        coherence = np.exp(-(effective_phase_error**2) / 2)
        coherence_loss = -10 * np.log10(max(coherence, 0.01))
        
        # Additional degradation sources
        multipath_loss = 2.0  # dB (seafloor multipath)
        platform_vibration_loss = 1.5  # dB (engine/propeller vibrations)
        aperture_decorrelation = min(3.0, 0.3 * np.log10(effective_aperture / self.physical_aperture))
        
        total_degradation = coherence_loss + multipath_loss + platform_vibration_loss + aperture_decorrelation
        
        return min(total_degradation, 15)  # Much higher realistic degradation
    
    def calculate_energy_constraint(self, sonar_power, vehicle_speed):
        """
        Calculate energy constraint penalty
        """
        propulsion_power = (0.5 * self.seawater_density * self.frontal_area * 
                           self.drag_coefficient * vehicle_speed**3) / self.propulsion_efficiency
        
        total_power = sonar_power + propulsion_power + self.hotel_load
        
        if total_power > self.max_power_budget:
            return -(total_power - self.max_power_budget) / 50  # Gradual penalty
        return 0
    
    def calculate_sas_range(self, electrical_power, frequency, vehicle_speed, 
                           motion_error_percent, target_size):
        """
        Calculate maximum SAS detection range using sonar equation
        
        Args:
            electrical_power: Average electrical power (W)
            frequency: Acoustic frequency (Hz)
            vehicle_speed: Vehicle speed (m/s)
            motion_error_percent: Motion error as % of vehicle speed
            target_size: Target diameter (m)
        
        Returns:
            Maximum detection range (m)
        """
        # Iterative solution since effective aperture depends on range
        range_m = 100  # Initial guess
        converged = False
        
        for iteration in range(15):  # More iterations for stability
            if converged:
                break
                
            old_range = range_m
            
            # Calculate range-dependent parameters
            effective_aperture = self.calculate_effective_aperture(range_m, vehicle_speed, 
                                                                 motion_error_percent, frequency)
            
            # Source level calculation
            wavelength = self.sound_speed / frequency
            transmit_di = 10 * np.log10(4 * np.pi * effective_aperture**2 / wavelength**2)
            
            theoretical_sl = (170.8 + 10 * np.log10(self.transducer_efficiency * electrical_power) 
                            + transmit_di)
            cavitation_limit = self.calculate_cavitation_limit(frequency, 50)
            source_level = min(theoretical_sl, cavitation_limit)
            
            # Target strength with proper physics
            target_strength = self.calculate_target_strength(frequency, target_size)
            
            # Two-way transmission loss
            absorption_coeff = self.calculate_seawater_absorption(frequency)
            transmission_loss = 40 * np.log10(range_m) + 2 * absorption_coeff * range_m / 1000
            
            # Receiver directivity (SAS beamforming)
            receive_di = 10 * np.log10(effective_aperture / self.physical_aperture)
            
            # Noise model
            ambient_noise = self.wenz_noise_model(frequency)
            
            # SAS processing gain
            processing_gain = self.calculate_sas_processing_gain(effective_aperture, frequency)
            
            # Motion degradation
            motion_degradation = self.calculate_motion_degradation(frequency, vehicle_speed, 
                                                                 motion_error_percent, 
                                                                 effective_aperture)
            
            # Energy constraint
            energy_penalty = self.calculate_energy_constraint(electrical_power, vehicle_speed)
            
            # Sonar equation with realism penalty
            
            # Small target detection penalty - reflects real-world challenges
            # - Seafloor clutter increases relative to target strength for small targets
            # - Volume reverberation masking
            # - Resolution cell competition
            if target_size < 0.1:  # Targets smaller than 10cm
                clutter_penalty = 10 * np.log10(0.1 / target_size)  # Penalty increases as target gets smaller
                small_target_penalty = min(clutter_penalty, 20)  # Cap at 20dB
            else:
                small_target_penalty = 0
                
            signal_excess = (source_level + target_strength - transmission_loss + 
                           receive_di - ambient_noise + processing_gain - 
                           self.detection_threshold - motion_degradation + energy_penalty - 
                           small_target_penalty)
            
            # More stable range adjustment to prevent oscillation
            if signal_excess <= -3:  # Clearly insufficient signal
                range_m = max(10, range_m * 0.85)
            elif signal_excess >= 3:  # Clearly sufficient signal  
                scaling_factor = min(1.15, 10**(signal_excess / 80))  # Less aggressive scaling
                range_m = min(range_m * scaling_factor, 5000)
            else:
                # Near threshold: use small adjustments to prevent oscillation
                if signal_excess < 0:
                    range_m = max(10, range_m * 0.98)  # Small reduction
                else:
                    range_m = min(range_m * 1.02, 5000)  # Small increase
            
            # More stringent convergence check near threshold
            convergence_tolerance = 0.02 if abs(signal_excess) > 3 else 0.005
            if abs(range_m - old_range) / max(old_range, 1) < convergence_tolerance:
                converged = True
        
        # Final range constraints
        final_effective_aperture = self.calculate_effective_aperture(range_m, vehicle_speed,
                                                                   motion_error_percent, frequency)
        
        # Note: The geometric constraint was limiting detection range
        # Aperture formation time is NOT the same as detection range
        # Detection range should be limited by signal-to-noise ratio, not aperture geometry
        
        return min(range_m, 8000)  # Practical maximum range

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
        
        # Add scattering regime analysis
        k_100khz = 2 * np.pi * 100000 / model.sound_speed
        ka_100khz = k_100khz * (target_size/2)
        
        if ka_100khz < 0.3:
            regime = "Rayleigh (TS ∝ f⁴)"
        elif ka_100khz > 10:
            regime = "Geometric (frequency independent)"
        else:
            regime = "Mie/Resonance (complex)"
            
        ax.text(0.4, 0.95, f'Scattering regime: {regime}\nka @ 100kHz: {ka_100khz:.2f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))
    
    # Connect sliders to update function
    slider_speed.on_changed(update_plot)
    slider_motion.on_changed(update_plot)
    slider_target.on_changed(update_plot)
    slider_log.on_changed(update_plot)
    
    # Initial plot
    update_plot()
    
    plt.show()

if __name__ == "__main__":
    create_interactive_plot()
    