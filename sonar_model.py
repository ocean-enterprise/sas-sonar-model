#!/usr/bin/env python3
"""
Synthetic Aperture Sonar (SAS) Detection Range Model
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.special import spherical_jn, spherical_yn
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
warnings.filterwarnings('ignore')

import tqdm

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
        self.bandwidth = 20000  # Hz (10 kHz bandwidth for range resolution)
        
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
        Proper Mie series for acoustic scattering from a fluid sphere
        
        Args:
            ka: Size parameter (k * radius)
            density_contrast: Target/water density ratio (ρ_target/ρ_water)
            compressibility_contrast: Target/water compressibility ratio (κ_target/κ_water)
        
        Returns:
            Backscattering cross-section normalized by πa² (dimensionless)
        """
        g = density_contrast
        h = compressibility_contrast
        c_ratio = np.sqrt(h / g)
        ka_internal = ka / c_ratio
        
        n_max = max(int(ka + 4 * ka**(1/3) + 10), 5)
        
        backscatter_sum = 0.0
        for n in range(n_max):
            jn_ka = spherical_jn(n, ka)
            yn_ka = spherical_yn(n, ka)
            jn_prime_ka = spherical_jn(n, ka, derivative=True)
            yn_prime_ka = spherical_yn(n, ka, derivative=True)
            
            jn_kai = spherical_jn(n, ka_internal)
            jn_prime_kai = spherical_jn(n, ka_internal, derivative=True)
            
            numerator = (g * jn_prime_kai * jn_ka - jn_kai * jn_prime_ka)
            denominator = (g * jn_prime_kai * (jn_ka + 1j * yn_ka) - 
                          jn_kai * (jn_prime_ka + 1j * yn_prime_ka))
            
            if abs(denominator) > 1e-10:
                a_n = -numerator / denominator
                backscatter_sum += (2 * n + 1) * (-1)**n * a_n
        
        sigma_bs = abs(backscatter_sum)**2 / ka**2
        
        return max(sigma_bs, 1e-10)
    
    def calculate_target_strength(self, frequency, target_diameter, 
                                density_contrast=1.5, compressibility_contrast=0.8):
        """
        Calculate target strength using Mie scattering physics
        
        Args:
            frequency: Acoustic frequency (Hz)
            target_diameter: Target diameter (m)
            density_contrast: Target/water density ratio
            compressibility_contrast: Target/water compressibility ratio
        
        Returns:
            Target strength (dB re 1 m²)
        """
        k = 2 * np.pi * frequency / self.sound_speed
        a = target_diameter / 2
        ka = k * a
        
        sigma_bs_normalized = self.mie_series_approximation(ka, density_contrast, compressibility_contrast)
        
        sigma_bs_m2 = sigma_bs_normalized * np.pi * a**2
        
        ts = 10 * np.log10(max(sigma_bs_m2, 1e-20))
        
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
    
    def calculate_noise_level(self, frequency, bandwidth=None):
        """
        Calculate total noise level including bandwidth effects
        
        Args:
            frequency: Acoustic frequency (Hz)
            bandwidth: Receiver bandwidth (Hz), uses self.bandwidth if None
        
        Returns:
            Total noise level (dB re 1 μPa)
        """
        if bandwidth is None:
            bandwidth = self.bandwidth
            
        ambient_noise_spectral = self.wenz_noise_model(frequency)
        total_noise = ambient_noise_spectral + 10 * np.log10(bandwidth)
        
        return total_noise
    
    def calculate_volume_reverberation(self, frequency, range_m, effective_aperture):
        """
        Calculate volume reverberation from ensonified water volume
        
        Args:
            frequency: Acoustic frequency (Hz)
            range_m: Range to target (m)
            effective_aperture: Effective synthetic aperture length (m)
        
        Returns:
            Volume reverberation level (dB re 1 μPa)
        """
        wavelength = self.sound_speed / frequency
        
        # Beamwidth in radians (along-track and across-track)
        along_track_beamwidth = wavelength / effective_aperture
        across_track_beamwidth = wavelength / self.physical_aperture
        
        # Range resolution from bandwidth
        range_resolution = self.sound_speed / (2 * self.bandwidth)
        
        # Ensonified volume (cone approximation)
        volume = (np.pi * range_m**2 * along_track_beamwidth * across_track_beamwidth * 
                 range_resolution / 4)
        
        # Volume backscattering coefficient (frequency dependent)
        # Typical values: -80 to -90 dB for clean ocean water
        # Increases with frequency and particulate matter
        sigma_v_db = -85 + 1.5 * np.log10(frequency / 100000)
        sigma_v = 10**(sigma_v_db / 10)
        
        # Volume reverberation level
        volume_reverb = 10 * np.log10(max(sigma_v * volume, 1e-20))
        
        return volume_reverb
    
    def calculate_seafloor_clutter(self, frequency, range_m, effective_aperture, grazing_angle_deg=30):
        """
        Calculate seafloor clutter level from resolution cell
        
        Args:
            frequency: Acoustic frequency (Hz)
            range_m: Range to target (m)
            effective_aperture: Effective synthetic aperture length (m)
            grazing_angle_deg: Grazing angle to seafloor (degrees)
        
        Returns:
            Seafloor clutter level (dB re 1 μPa)
        """
        wavelength = self.sound_speed / frequency
        grazing_angle_rad = np.radians(grazing_angle_deg)
        
        # Range resolution
        range_resolution = self.sound_speed / (2 * self.bandwidth)
        
        # Along-track resolution (SAS resolution)
        along_track_resolution = (range_m * wavelength) / (2 * effective_aperture)
        
        # Resolution cell area on seafloor
        clutter_area = range_resolution * along_track_resolution / np.sin(grazing_angle_rad)
        
        # Seafloor backscattering coefficient (Lambert's law approximation)
        # Typical values: -30 to -10 dB depending on seafloor type
        # Sand: -25 dB, Mud: -35 dB, Rock: -15 dB
        sigma_0_db = -25 + 10 * np.log10(np.sin(grazing_angle_rad)**2)
        sigma_0 = 10**(sigma_0_db / 10)
        
        # Clutter level
        clutter_level = 10 * np.log10(max(sigma_0 * clutter_area, 1e-20))
        
        return clutter_level
    
    def calculate_directivity_index(self, frequency, effective_aperture, aperture_width=None):
        """
        Calculate proper directivity index for SAS array
        
        Args:
            frequency: Acoustic frequency (Hz)
            effective_aperture: Effective synthetic aperture length (m)
            aperture_width: Physical aperture width (m), uses self.physical_aperture if None
        
        Returns:
            Directivity index (dB)
        """
        wavelength = self.sound_speed / frequency
        
        if aperture_width is None:
            aperture_width = self.physical_aperture
        
        # Along-track directivity (from synthetic aperture)
        # DI = 10*log10(4π*A/λ²) for a uniformly illuminated aperture
        along_track_di = 10 * np.log10(2 * effective_aperture / wavelength)
        
        # Across-track directivity (from physical aperture)
        across_track_di = 10 * np.log10(2 * aperture_width / wavelength)
        
        # Total directivity (sum in dB space for orthogonal dimensions)
        total_di = along_track_di + across_track_di
        
        # Apply realistic array efficiency factor
        # Real arrays have tapering, element spacing, and other non-idealities
        array_efficiency = 0.7
        total_di = total_di + 10 * np.log10(array_efficiency)
        
        return max(total_di, 0)
    
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
            
            # Improved directivity index
            receive_di = self.calculate_directivity_index(frequency, effective_aperture)
            
            # Improved noise model with bandwidth
            ambient_noise = self.calculate_noise_level(frequency, self.bandwidth)
            
            # Additional noise sources
            volume_reverberation = self.calculate_volume_reverberation(frequency, range_m, 
                                                                      effective_aperture)
            seafloor_clutter = self.calculate_seafloor_clutter(frequency, range_m, 
                                                               effective_aperture, 
                                                               grazing_angle_deg=30)
            
            # Combined noise level (energy addition)
            total_noise = 10 * np.log10(
                10**(ambient_noise/10) + 
                10**(volume_reverberation/10) + 
                10**(seafloor_clutter/10)
            )
            
            # SAS processing gain
            processing_gain = self.calculate_sas_processing_gain(effective_aperture, frequency)
            
            # Motion degradation
            motion_degradation = self.calculate_motion_degradation(frequency, vehicle_speed, 
                                                                 motion_error_percent, 
                                                                 effective_aperture)
            
            # Energy constraint
            energy_penalty = self.calculate_energy_constraint(electrical_power, vehicle_speed)
            
            # Improved sonar equation
            signal_excess = (source_level + target_strength - transmission_loss + 
                           receive_di - total_noise + processing_gain - 
                           self.detection_threshold - motion_degradation + energy_penalty)
            
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

            # final_effective_aperture = self.calculate_effective_aperture(range_m, vehicle_speed, motion_error_percent, frequency)
        
        return min(range_m, 8000)  # Practical maximum range

def _calculate_range_worker(args):
    """Worker function for multiprocessing pool"""
    target_size, frequency, electrical_power, vehicle_speed, motion_error = args
    model = SASModel()
    return model.calculate_sas_range(
        electrical_power, frequency, vehicle_speed, motion_error, target_size
    )

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

def create_3d_surface_plot():
    """Create 3D surface plot of detection range vs target size and frequency"""
    
    model = SASModel()
    
    # Fixed parameters for 3D plot
    fixed_electrical_power = 50.0  # W
    fixed_vehicle_speed = 2.0      # m/s  
    fixed_motion_error = 0.1       # % of vehicle speed
    
    # Define parameter ranges
    target_diameters = np.logspace(-3, 0, 100)
    frequencies = np.linspace(10000, 200000, 100)
    
    # Create meshgrid
    TARGET_MESH, FREQ_MESH = np.meshgrid(target_diameters, frequencies)
    
    # Calculate detection range for each combination using multiprocessing
    print(f"Calculating detection ranges for 3D surface using {cpu_count()} cores...")
    RANGE_MESH = np.zeros_like(TARGET_MESH)
    
    # Prepare arguments for all calculations
    args_list = []
    for i in range(TARGET_MESH.shape[0]):
        for j in range(TARGET_MESH.shape[1]):
            target_size = TARGET_MESH[i, j]
            frequency = FREQ_MESH[i, j]
            args_list.append((target_size, frequency, fixed_electrical_power, 
                            fixed_vehicle_speed, fixed_motion_error))
    
    # Calculate in parallel
    with Pool() as pool:
        results = pool.map(_calculate_range_worker, args_list)
    
    # Reshape results back into mesh
    RANGE_MESH = np.array(results).reshape(TARGET_MESH.shape)
    print("100% complete")
    
    # Create 3D surface plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize data for better 3D visualization
    # Use log scale for target sizes to handle the wide range
    LOG_TARGET_MESH = np.log10(TARGET_MESH)
    NORM_FREQ_MESH = (FREQ_MESH - frequencies.min()) / (frequencies.max() - frequencies.min())
    NORM_RANGE_MESH = (RANGE_MESH - RANGE_MESH.min()) / (RANGE_MESH.max() - RANGE_MESH.min())
    
    # Create surface plot with normalized coordinates
    surf = ax.plot_surface(LOG_TARGET_MESH, NORM_FREQ_MESH, NORM_RANGE_MESH, 
                          cmap='viridis', alpha=0.8, linewidth=0, antialiased=True,
                          rstride=1, cstride=1)
    
    # Add contour lines on the normalized surface
    contours = ax.contour(LOG_TARGET_MESH, NORM_FREQ_MESH, NORM_RANGE_MESH, 
                         levels=8, colors='black', alpha=0.4, linewidths=0.8)
    
    # Set up custom tick labels to show actual values
    # X-axis (target diameter) - log scale
    log_target_ticks = np.log10([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    target_tick_labels = ['0.01', '0.02', '0.05', '0.1', '0.2', '0.5', '1.0']
    ax.set_xticks(log_target_ticks)
    ax.set_xticklabels(target_tick_labels)
    
    # Y-axis (frequency) - normalized scale
    freq_ticks = np.linspace(0, 1, 6)
    freq_values = frequencies.min() + freq_ticks * (frequencies.max() - frequencies.min())
    freq_labels = [f'{int(f/1000)}' for f in freq_values]
    ax.set_yticks(freq_ticks)
    ax.set_yticklabels(freq_labels)
    
    # Z-axis (range) - normalized scale  
    range_ticks = np.linspace(0, 1, 6)
    range_values = RANGE_MESH.min() + range_ticks * (RANGE_MESH.max() - RANGE_MESH.min())
    range_labels = [f'{int(r)}' for r in range_values]
    ax.set_zticks(range_ticks)
    ax.set_zticklabels(range_labels)
    
    # Set labels and title
    ax.set_xlabel('Target Diameter (m)', fontsize=12, labelpad=10)
    ax.set_ylabel('Frequency (kHz)', fontsize=12, labelpad=10)
    ax.set_zlabel('Detection Range (m)', fontsize=12, labelpad=10)
    
    ax.set_title(f'SAS Detection Range Surface\n'
                f'Power: {fixed_electrical_power}W, Speed: {fixed_vehicle_speed}m/s, '
                f'Motion Error: {fixed_motion_error}%', fontsize=14, pad=20)
    
    # Add colorbar mapped to actual range values
    mappable = plt.cm.ScalarMappable(cmap='viridis')
    mappable.set_array(RANGE_MESH)
    colorbar = fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=20)
    colorbar.set_label('Detection Range (m)', fontsize=11)
    
    # Set equal aspect ratio for better 3D interaction
    ax.set_box_aspect([1,1,0.8])  # Make Z slightly shorter for better viewing
    
    # Improve viewing angle
    ax.view_init(elev=25, azim=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add text box with scattering regime information
    textstr = '\n'.join([
        'Scattering Regimes:',
        '• Small targets (ka < 0.5): Rayleigh (TS ∝ f⁴)',  
        '• Medium targets (0.5 ≤ ka ≤ 15): Mie/Resonance',
        '• Large targets (ka > 15): Geometric'
    ])
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text2D(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
              verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig, ax

def create_cmap_2d_plot():
    """Create 2D colormap plot as an alternative view"""
    
    model = SASModel()
    
    # Fixed parameters
    fixed_electrical_power = 50.0
    fixed_vehicle_speed = 1.0
    fixed_motion_error = 0.1
    
    # Define parameter ranges (same as 3D plot for consistency)
    target_diameters = np.logspace(-2.7, 1.3, 200)
    frequencies = np.linspace(20000, 200000, 150)
    
    # Create meshgrid
    TARGET_MESH, FREQ_MESH = np.meshgrid(target_diameters, frequencies)
    RANGE_MESH = np.zeros_like(TARGET_MESH)
    
    thread_count = cpu_count() // 2 
    print(f"Calculating detection ranges for 2D plot using {thread_count} cores...")
    
    # Prepare arguments for all calculations
    args_list = []
    for i in range(TARGET_MESH.shape[0]):
        for j in range(TARGET_MESH.shape[1]):
            target_size = TARGET_MESH[i, j]
            frequency = FREQ_MESH[i, j]
            args_list.append((target_size, frequency, fixed_electrical_power,
                            fixed_vehicle_speed, fixed_motion_error))
    
    # Calculate in parallel
    results = []
    with Pool(thread_count) as pool:
        for r in tqdm.tqdm(pool.imap(_calculate_range_worker, args_list), total=len(args_list)):
            results.append(r)
    
    # Reshape results back into mesh
    RANGE_MESH = np.array(results).reshape(TARGET_MESH.shape)
    print("100% complete")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.pcolormesh(TARGET_MESH, FREQ_MESH/1000, RANGE_MESH, 
                       cmap='viridis', shading='gouraud')
    
    colorbar = fig.colorbar(im, ax=ax)
    colorbar.set_label('Detection Range (m)', fontsize=11)
    
    ax.set_xscale('log')
    
    ax.set_xlabel('Target Diameter (m)', fontsize=12)
    ax.set_ylabel('Frequency (kHz)', fontsize=12)
    ax.set_title(f'SAS Detection Range\n'
                f'Power: {fixed_electrical_power}W, Speed: {fixed_vehicle_speed}m/s, '
                f'Motion Error: {fixed_motion_error}%', fontsize=14)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add scattering regime boundaries as curves
    # For a given target diameter d and frequency f:
    # ka = π*d*f/c
    # So for constant ka: f = ka*c/(π*d)
    
    # Calculate boundary curves    
    # ka = 0.5 boundary (Rayleigh to Mie/Resonance transition)
    ka_05_freq = (0.5 * model.sound_speed) / (np.pi * target_diameters)
    
    # ka = 15 boundary (Mie/Resonance to Geometric transition)
    ka_15_freq = (15 * model.sound_speed) / (np.pi * target_diameters)
    
    # Plot the boundary curves
    mask_05 = (ka_05_freq >= frequencies.min()) & (ka_05_freq <= frequencies.max())
    if np.any(mask_05):
        ax.plot(target_diameters[mask_05], ka_05_freq[mask_05]/1000, 
               'r--', linewidth=2.5, alpha=0.8, label='ka = 0.5 (Rayleigh→Mie)')
    
    mask_15 = (ka_15_freq >= frequencies.min()) & (ka_15_freq <= frequencies.max())
    if np.any(mask_15):
        ax.plot(target_diameters[mask_15], ka_15_freq[mask_15]/1000, 
               'orange', linestyle='--', linewidth=2.5, alpha=0.8, 
               label='ka = 15 (Mie→Geometric)')
    
    # Add text labels for the three regimes
    # Find good positions for labels within the plot, these are not automatically placed
    ax.text(0.004, 30, 'Rayleigh\n(TS ∝ f⁴)', fontsize=11, 
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7),
           ha='center', va='center')
    
    ax.text(0.015, 80, 'Mie/Resonance\n(complex)', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
           ha='center', va='center')
    
    ax.text(0.5, 145, 'Geometric\n(TS ≈ constant)', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
           ha='center', va='center')
    
    # Update legend
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    return fig, ax

if __name__ == "__main__":
    
    # Create both 3D surface and 2D contour plots
    print("Creating 3D surface plot...")
    # fig_3d, ax_3d = create_3d_surface_plot()
    
    print("\nCreating 2D colormap plot...")
    fig_2d, ax_2d = create_cmap_2d_plot()
    
    # Show interactive 2D plot as well
    print("\nCreating interactive plot...")
    create_interactive_plot()
    
    # Display all plots
    plt.show()
    