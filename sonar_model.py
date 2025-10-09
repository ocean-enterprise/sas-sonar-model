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
warnings.filterwarnings('ignore')

import tqdm

colormap = 'gist_earth'

class SASModel:
    """Synthetic Aperture Sonar detection range model"""
    
    def __init__(self):
        # Physical constants
        self.sound_speed = 1500  # m/s
        self.physical_aperture = 0.0254  # meters (1 inch)
        
        # System parameters
        self.transducer_efficiency = 0.25  # Small AUV systems including all electrical and mechanical losses
        self.hotel_load = 150  # W (SAS processing)
        self.detection_threshold = 30  # dB (SNR for reliable detection in ocean clutter and multipath)
        self.bandwidth = 10000  # Hz 
        
        # AUV parameters
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
        
        # SAS carrier frequencies for analysis (Hz)
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
        Mie series approximation for acoustic scattering from a fluid sphere
        Based on Anderson (1950) and Faran (1951)
        
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
            
            # Acoustic scattering coefficients with boundary conditions
            numerator = (g * jn_prime_kai * jn_ka - jn_kai * jn_prime_ka)
            denominator = (g * jn_prime_kai * (jn_ka + 1j * yn_ka) - 
                          jn_kai * (jn_prime_ka + 1j * yn_prime_ka))
            
            if abs(denominator) > 1e-10:
                a_n = -numerator / denominator
                backscatter_sum += (2 * n + 1) * (-1)**n * a_n
        
        sigma_bs = abs(backscatter_sum)**2 / ka**2
        
        return max(sigma_bs, 1e-10)
    
    def rayleigh_scattering(self, ka, density_contrast, compressibility_contrast):
        """
        Rayleigh scattering approximation for small particles (ka < 0.5)
        Using Anderson (1950) formulation for acoustic scattering
        
        Args:
            ka: Size parameter (k * radius)
            density_contrast: Target/water density ratio
            compressibility_contrast: Target/water compressibility ratio
        
        Returns:
            Backscattering cross-section normalized by πa² (dimensionless)
        """
        g = density_contrast
        h = compressibility_contrast
        
        # Rayleigh formula for acoustic scattering
        # σ_bs/πa² = (ka)⁴/9 × [f₁ - f₂]²
        # where f₁ = (g-1)/(2g+1) and f₂ = (h-1)/(1+2(h-1)/3)
        
        f1 = (g - 1) / (2 * g + 1)
        f2 = (h - 1) / (1 + 2 * (h - 1) / 3)
        
        sigma_bs = (ka**4 / 9) * (f1 - f2)**2
        
        return max(sigma_bs, 1e-10)
    
    def geometric_approximation(self):
        """
        Geometric approximation for large particles (ka > 15)
        
        Returns:
            Backscattering cross-section normalized by πa² (dimensionless)
        """
        return 1.0
    
    def calculate_target_strength(self, frequency, target_diameter, 
                                density_contrast=1.5, compressibility_contrast=0.8):
        """
        Calculate target strength using appropriate scattering regime
        
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
        
        if ka < 0.5:
            sigma_bs_normalized = self.rayleigh_scattering(ka, density_contrast, compressibility_contrast)
        elif ka > 15:
            sigma_bs_normalized = self.geometric_approximation()
        else:
            sigma_bs_normalized = self.mie_series_approximation(ka, density_contrast, compressibility_contrast)
        
        sigma_bs_m2 = sigma_bs_normalized * np.pi * a**2
        
        ts = 10 * np.log10(max(sigma_bs_m2, 1e-20))
        
        return ts
    
    def calculate_effective_aperture(self, range_m, vehicle_speed, motion_error_percent, frequency=100000):
        """
        Calculate SAS effective aperture with cumulative coherence constraints
        
        Args:
            range_m: Range to target (m)
            vehicle_speed: Vehicle speed (m/s)
            motion_error_percent: Motion error as % of vehicle speed (RMS per second)
            frequency: Operating frequency (Hz) for wavelength-dependent coherence
        
        Returns:
            Effective aperture length (m)
        """
        # Maximum theoretical aperture from geometry
        round_trip_time = (2 * range_m) / self.sound_speed
        geometric_aperture = vehicle_speed * round_trip_time
        
        # Motion error accumulation over aperture synthesis
        # Motion errors accumulate as random walk over synthetic aperture
        wavelength = self.sound_speed / frequency
        phase_tolerance = wavelength / 8  # Standard λ/8 coherence criterion
        
        # RMS motion error per second
        motion_error_per_sec = (motion_error_percent / 100) * vehicle_speed
        
        # For aperture of length L, synthesis time is T = L / v
        # Cumulative RMS error: σ_cumulative = σ_per_sec × sqrt(T)
        # Phase error must satisfy: 2π × σ_cumulative / λ < π/4 (λ/8 criterion)
        # This gives: σ_per_sec × sqrt(L/v) < λ/8
        # Solving for L: L < (λ/8 / σ_per_sec)² × v
        
        if motion_error_per_sec > 0:
            coherence_limit = (phase_tolerance / motion_error_per_sec)**2 * vehicle_speed
        else:
            coherence_limit = 10000  # Very large if no motion error
        
        # Practical limits from system stability
        # Higher frequencies have reduced coherence due to:
        # - Increased sensitivity to platform motion
        # - Shorter wavelengths make phase errors more critical
        # - Environmental effects (water turbulence, multipath)
        max_practical_aperture = 50.0  # meters for stable platforms
        freq_degradation = np.exp(-(frequency - 50000) / 100000)  # Exponential decay
        practical_limit = max_practical_aperture * (0.3 + 0.7 * freq_degradation)
        
        # Effective aperture is minimum of all constraints
        effective_aperture = min(geometric_aperture, coherence_limit, practical_limit)
        
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
        
        # Along-track beamforming gain (main SAS advantage)
        # This represents coherent integration over synthetic aperture
        along_track_gain = 10 * np.log10(effective_aperture / self.physical_aperture)
        
        # Additional incoherent processing gain from multi-look averaging
        # This is ONLY applicable if doing separate incoherent averaging after SAR processing
        # For N independent looks: gain = 10*log10(sqrt(N)) = 5*log10(N)
        # Typical SAS systems use 4-8 independent looks
        n_independent_looks = 4  # Conservative estimate
        multilook_gain = 5 * np.log10(n_independent_looks)  # sqrt(N) improvement
        
        # Sidelobe suppression from windowing
        sidelobe_suppression = 1.5  # dB (conservative)
        
        # Total processing gain
        total_gain = along_track_gain + multilook_gain + sidelobe_suppression
        
        # Practical systems achieve 10-15 dB typical processing gain
        if total_gain > 15:
            print(f"Warning: Processing gain is higher than realistic: {total_gain} dB > 15 dB")
        return min(total_gain, 15)  # 15 dB maximum realistic gain
    
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
        base_freq = 10000  # Hz
        base_threshold = 220  # dB re 1 μPa
        
        # Frequency dependence: threshold increases with frequency
        frequency_correction = 2.5 * np.log10(frequency / base_freq)  # 2.5 dB per decade
        
        # Depth correction: +6 dB per atmosphere (~10m depth)
        depth_correction = 6 * np.log10((depth + 10) / 10)
        
        # Conservative operating margin
        operating_margin = 5  # dB below inception threshold
        
        max_sl = base_threshold + frequency_correction + depth_correction - operating_margin
        
        return max_sl
    
    def calculate_seawater_absorption(self, frequency_hz):
        """
        Francois & Garrison model with environmental parameters including pH
        http://resource.npl.co.uk/acoustics/techguides/seaabsorption/
        
        Args:
            frequency_hz: Frequency in Hz
        
        Returns:
            Absorption coefficient (dB/km)
        """
        f_khz = frequency_hz / 1000
        T = self.water_temperature  # °C
        S = self.salinity  # ppt
        pH = self.ph
        
        # Boric acid relaxation frequency (with pH dependence)
        A1 = 8.86 / self.sound_speed * 10**(0.78 * pH - 5)
        P1 = 1  # Pressure in atmospheres (shallow water)
        f1 = 2.8 * np.sqrt(S / 35) * 10**(4 - 1245 / (T + 273))
        
        # Boric acid absorption
        absorption1 = (A1 * P1 * f1 * f_khz**2) / (f1**2 + f_khz**2)
        
        # Magnesium sulfate relaxation
        A2 = 21.44 * S / self.sound_speed * (1 + 0.025 * T)
        P2 = 1 - 1.37e-4 * 50 + 6.2e-9 * 50**2  # Depth = 50m
        f2 = 8.17 * 10**(8 - 1990 / (T + 273)) / (1 + 0.0018 * (S - 35))
        
        # Magnesium sulfate absorption
        absorption2 = (A2 * P2 * f2 * f_khz**2) / (f2**2 + f_khz**2)
        
        # Pure water absorption
        A3 = 4.937e-4 - 2.59e-5 * T + 9.11e-7 * T**2 - 1.5e-8 * T**3
        absorption3 = A3 * P2 * f_khz**2
        
        total_absorption = absorption1 + absorption2 + absorption3
        
        return total_absorption
    
    def wenz_noise_model(self, frequency):
        """
        Ambient noise model based on Wenz curves
        https://studylib.net/doc/11095773/ambient-noise-the-background-noise-of-the-sea
        
        Args:
            frequency: Acoustic frequency (Hz)
        
        Returns:
            Noise spectral density (dB re 1 μPa²/Hz)
        """
        f_khz = frequency / 1000
        
        # Wenz noise components (all in dB re 1 μPa²/Hz)
        # Thermal noise (dominant at high frequencies)
        thermal_noise = -15 + 20 * np.log10(f_khz)
        
        # Sea state noise (dominant at mid frequencies)
        sea_noise = 50 + 7.5 * self.sea_state - 20 * np.log10(f_khz)
        
        # Shipping noise (dominant at low frequencies)
        shipping_noise = 76 - 20 * np.log10(f_khz)
        
        # Wind noise
        wind_speed = 5 + 2.5 * self.sea_state
        wind_noise = 44 + 0.5 * wind_speed - 15 * np.log10(f_khz)
        
        # Combine noise sources (power addition in linear domain)
        combined_noise = 10 * np.log10(
            10**(thermal_noise/10) + 
            10**(sea_noise/10) + 
            10**(shipping_noise/10) + 
            10**(wind_noise/10)
        )
        
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
        
        # Beamwidth in radians
        along_track_beamwidth = wavelength / effective_aperture
        across_track_beamwidth = wavelength / self.physical_aperture
        
        # Range resolution from bandwidth
        range_resolution = self.sound_speed / (2 * self.bandwidth)
        
        # Ensonified volume
        volume = (np.pi * range_m**2 * along_track_beamwidth * across_track_beamwidth * 
                 range_resolution / 4)
        
        # Volume backscattering coefficient (frequency and depth dependent)
        # Base value for clean ocean water at 100 kHz
        sigma_v_base_db = -85
        freq_dependence = 1.5 * np.log10(frequency / 100000)  # Rayleigh scattering
        sigma_v_db = sigma_v_base_db + freq_dependence
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
        
        # Seafloor backscattering coefficient with frequency dependence
        # Base value for sand at 30° grazing angle
        sigma_0_base = -25  # dB
        angle_dependence = 10 * np.log10(np.sin(grazing_angle_rad)**2)
        # Roughness scattering increases with frequency
        freq_dependence = 2 * np.log10(frequency / 100000)
        sigma_0_db = sigma_0_base + angle_dependence + freq_dependence
        sigma_0 = 10**(sigma_0_db / 10)
        
        # Clutter level
        clutter_level = 10 * np.log10(max(sigma_0 * clutter_area, 1e-20))
        
        return clutter_level
    
    def calculate_directivity_index(self, frequency, effective_aperture, aperture_width=None):
        """
        Calculate directivity index for SAS array
        
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
        
        # Directivity for rectangular aperture: DI = 10*log10(4π*A/λ²)
        # where A = L × W (aperture area)
        aperture_area = effective_aperture * aperture_width
        total_di = 10 * np.log10(4 * np.pi * aperture_area / wavelength**2)
        
        # Apply realistic array efficiency factor
        array_efficiency = 0.7  # Accounts for tapering, element spacing, etc.
        total_di = total_di + 10 * np.log10(array_efficiency)
        
        return max(total_di, 0)
    
    def calculate_motion_degradation(self, frequency, vehicle_speed, motion_error_percent, 
                                   effective_aperture):
        """
        Calculate motion-induced coherence degradation for SAS
        
        Args:
            frequency: Acoustic frequency (Hz)
            vehicle_speed: Vehicle speed (m/s)
            motion_error_percent: Motion error as % of vehicle speed (RMS per second)
            effective_aperture: Effective synthetic aperture (m)
        
        Returns:
            Motion degradation (dB)
        """
        motion_error_per_sec = (motion_error_percent / 100) * vehicle_speed
        wavelength = self.sound_speed / frequency
        
        # Aperture synthesis time
        synthesis_time = effective_aperture / vehicle_speed
        
        # Cumulative RMS phase error over aperture (random walk)
        cumulative_motion_error = motion_error_per_sec * np.sqrt(synthesis_time)
        phase_error_rms = (2 * np.pi * cumulative_motion_error) / wavelength
        
        # Residual error after motion compensation (INS + autofocus)
        # High-grade tactical INS: 5-10% residual
        # Medium-grade navigation INS: 15-25% residual
        residual_factor = 0.15  # 15% residual after compensation
        effective_phase_error = phase_error_rms * residual_factor
        
        # Coherence loss from Gaussian phase errors
        coherence = np.exp(-(effective_phase_error**2) / 2)
        coherence_loss = -10 * np.log10(max(coherence, 0.01))
        
        # Additional degradation sources
        multipath_loss = 1.5  # dB (seafloor/surface multipath)
        platform_vibration_loss = 1.0  # dB (engine/propeller vibrations)
        
        # Aperture decorrelation (longer apertures harder to maintain coherence)
        aperture_factor = min(2.0, 0.2 * np.log10(effective_aperture / self.physical_aperture))
        
        total_degradation = coherence_loss + multipath_loss + platform_vibration_loss + aperture_factor
        
        if total_degradation > 12:
            print(f"Warning: Motion degradation is higher than realistic: {total_degradation} dB > 12 dB")

        return min(total_degradation, 12)  # Realistic maximum degradation
    
    def calculate_energy_constraint(self, sonar_power, vehicle_speed):
        """
        Calculate energy constraint penalty
        """
        propulsion_power = (0.5 * self.seawater_density * self.frontal_area * 
                           self.drag_coefficient * vehicle_speed**3) / self.propulsion_efficiency
        
        total_power = sonar_power + propulsion_power + self.hotel_load
        
        if total_power > self.max_power_budget:
            return -(total_power - self.max_power_budget) / 50
        return 0
    
    def calculate_sas_range(self, electrical_power, frequency, vehicle_speed, 
                           motion_error_percent, target_size):
        """
        Calculate maximum SAS detection range using sonar equation
        Uses bisection method for robust convergence
        
        Args:
            electrical_power: Average electrical power (W)
            frequency: Acoustic frequency (Hz)
            vehicle_speed: Vehicle speed (m/s)
            motion_error_percent: Motion error as % of vehicle speed
            target_size: Target diameter (m)
        
        Returns:
            Maximum detection range (m)
        """
        # Bisection search for range where signal excess = 0
        range_min = 10.0
        range_max = 8000.0
        tolerance = 5.0  # meters
        
        def calculate_signal_excess(range_m):
            """Calculate signal excess at given range"""
            # Range-dependent parameters
            effective_aperture = self.calculate_effective_aperture(
                range_m, vehicle_speed, motion_error_percent, frequency
            )
            
            # Source level
            wavelength = self.sound_speed / frequency
            transmit_di = 10 * np.log10(4 * np.pi * effective_aperture**2 / wavelength**2)
            theoretical_sl = (170.8 + 10 * np.log10(self.transducer_efficiency * electrical_power) 
                            + transmit_di)
            cavitation_limit = self.calculate_cavitation_limit(frequency, 50)
            source_level = min(theoretical_sl, cavitation_limit)
            
            # Target strength
            target_strength = self.calculate_target_strength(frequency, target_size)
            
            # Transmission loss (spherical spreading + absorption)
            absorption_coeff = self.calculate_seawater_absorption(frequency)
            transmission_loss = 40 * np.log10(range_m) + 2 * absorption_coeff * range_m / 1000
            
            # Receive directivity
            receive_di = self.calculate_directivity_index(frequency, effective_aperture)
            
            # Noise and reverberation
            ambient_noise = self.calculate_noise_level(frequency, self.bandwidth)
            volume_reverberation = self.calculate_volume_reverberation(
                frequency, range_m, effective_aperture
            )
            seafloor_clutter = self.calculate_seafloor_clutter(
                frequency, range_m, effective_aperture, grazing_angle_deg=30
            )
            
            # Combined noise (power addition)
            total_noise = 10 * np.log10(
                10**(ambient_noise/10) + 
                10**(volume_reverberation/10) + 
                10**(seafloor_clutter/10)
            )
            
            processing_gain = self.calculate_sas_processing_gain(effective_aperture, frequency)

            motion_degradation = self.calculate_motion_degradation(
                frequency, vehicle_speed, motion_error_percent, effective_aperture
            )
            
            energy_penalty = self.calculate_energy_constraint(electrical_power, vehicle_speed)
            
            # Sonar equation
            signal_excess = (source_level + target_strength - transmission_loss + 
                           receive_di - total_noise + processing_gain - 
                           self.detection_threshold - motion_degradation + energy_penalty)
            
            return signal_excess
        
        # Bisection search
        for iteration in range(30):
            range_mid = (range_min + range_max) / 2
            
            if range_max - range_min < tolerance:
                break
            
            signal_excess_mid = calculate_signal_excess(range_mid)
            
            if signal_excess_mid > 0:
                # Can detect farther
                range_min = range_mid
            else:
                # Too far, reduce range
                range_max = range_mid
        
        return min((range_min + range_max) / 2, 8000)

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
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.35)
    
    initial_vehicle_speed = 2.0 # m/s
    initial_motion_error = 0.1 # % of speed RMS
    initial_target_size = 0.1 # m
    initial_log_scale = False
    
    powers = np.linspace(1, 100, 50)
    
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
        ax.clear()
        
        vehicle_speed = slider_speed.val
        motion_error = slider_motion.val  
        target_size = slider_target.val
        log_scale = slider_log.val > 0.5
        
        for i, (freq_name, frequency) in enumerate(model.frequencies.items()):
            ranges = []
            for power in powers:
                range_m = model.calculate_sas_range(power, frequency, vehicle_speed,
                                                  motion_error, target_size)
                ranges.append(range_m)
            
            ax.plot(powers, ranges, label=freq_name, color=model.colors[i], linewidth=2)
        
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
        
        k_100khz = 2 * np.pi * 100000 / model.sound_speed
        ka_100khz = k_100khz * (target_size/2)
        
        if ka_100khz < 0.5:
            regime = "Rayleigh (TS ∝ f⁴)"
        elif ka_100khz > 15:
            regime = "Geometric (frequency independent)"
        else:
            regime = "Mie/Resonance (complex)"
            
        ax.text(0.4, 0.95, f'Scattering regime: {regime}\nka @ 100kHz: {ka_100khz:.2f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))
    
    slider_speed.on_changed(update_plot)
    slider_motion.on_changed(update_plot)
    slider_target.on_changed(update_plot)
    slider_log.on_changed(update_plot)
    
    update_plot()
    
    plt.show()

def create_3d_surface_plot():
    """Create 3D surface plot of detection range vs target size and frequency"""
    
    model = SASModel()
    
    fixed_electrical_power = 50.0
    fixed_vehicle_speed = 2.0
    fixed_motion_error = 0.1
    
    target_diameters = np.logspace(-3, 0, 100)
    frequencies = np.linspace(10000, 200000, 100)
    
    TARGET_MESH, FREQ_MESH = np.meshgrid(target_diameters, frequencies)
    
    print(f"Calculating detection ranges for 3D surface using {cpu_count()} cores...")
    RANGE_MESH = np.zeros_like(TARGET_MESH)
    
    args_list = []
    for i in range(TARGET_MESH.shape[0]):
        for j in range(TARGET_MESH.shape[1]):
            target_size = TARGET_MESH[i, j]
            frequency = FREQ_MESH[i, j]
            args_list.append((target_size, frequency, fixed_electrical_power, 
                            fixed_vehicle_speed, fixed_motion_error))
    
    with Pool() as pool:
        results = pool.map(_calculate_range_worker, args_list)
    
    RANGE_MESH = np.array(results).reshape(TARGET_MESH.shape)
    print("100% complete")
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    LOG_TARGET_MESH = np.log10(TARGET_MESH)
    NORM_FREQ_MESH = (FREQ_MESH - frequencies.min()) / (frequencies.max() - frequencies.min())
    NORM_RANGE_MESH = (RANGE_MESH - RANGE_MESH.min()) / (RANGE_MESH.max() - RANGE_MESH.min())
    
    surf = ax.plot_surface(LOG_TARGET_MESH, NORM_FREQ_MESH, NORM_RANGE_MESH, 
                          cmap=colormap, alpha=0.8, linewidth=0, antialiased=True,
                          rstride=1, cstride=1)
    
    contours = ax.contour(LOG_TARGET_MESH, NORM_FREQ_MESH, NORM_RANGE_MESH, 
                         levels=8, colors='black', alpha=0.4, linewidths=0.8)
    
    log_target_ticks = np.log10([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    target_tick_labels = ['0.01', '0.02', '0.05', '0.1', '0.2', '0.5', '1.0']
    ax.set_xticks(log_target_ticks)
    ax.set_xticklabels(target_tick_labels)
    
    freq_ticks = np.linspace(0, 1, 6)
    freq_values = frequencies.min() + freq_ticks * (frequencies.max() - frequencies.min())
    freq_labels = [f'{int(f/1000)}' for f in freq_values]
    ax.set_yticks(freq_ticks)
    ax.set_yticklabels(freq_labels)
    
    range_ticks = np.linspace(0, 1, 6)
    range_values = RANGE_MESH.min() + range_ticks * (RANGE_MESH.max() - RANGE_MESH.min())
    range_labels = [f'{int(r)}' for r in range_values]
    ax.set_zticks(range_ticks)
    ax.set_zticklabels(range_labels)
    
    ax.set_xlabel('Target Diameter (m)', fontsize=12, labelpad=10)
    ax.set_ylabel('Frequency (kHz)', fontsize=12, labelpad=10)
    ax.set_zlabel('Detection Range (m)', fontsize=12, labelpad=10)
    
    ax.set_title(f'SAS Detection Range Surface\n'
                f'Power: {fixed_electrical_power}W, Speed: {fixed_vehicle_speed}m/s, '
                f'Motion Error: {fixed_motion_error}%', fontsize=14, pad=20)
    
    mappable = plt.cm.ScalarMappable(cmap=colormap)
    mappable.set_array(RANGE_MESH)
    colorbar = fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=20)
    colorbar.set_label('Detection Range (m)', fontsize=11)
    
    ax.set_box_aspect([1,1,0.8])
    ax.view_init(elev=25, azim=45)
    ax.grid(True, alpha=0.3)
    
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
    
    fixed_electrical_power = 50.0
    fixed_vehicle_speed = 1.0
    fixed_motion_error = 0.1
    
    target_diameters = np.logspace(-2.7, 1.3, 400)
    frequencies = np.linspace(40000, 200000, 300)
    
    TARGET_MESH, FREQ_MESH = np.meshgrid(target_diameters, frequencies)
    RANGE_MESH = np.zeros_like(TARGET_MESH)
    
    thread_count = cpu_count() // 2 
    print(f"Calculating detection ranges for 2D plot using {thread_count} cores...")
    
    args_list = []
    for i in range(TARGET_MESH.shape[0]):
        for j in range(TARGET_MESH.shape[1]):
            target_size = TARGET_MESH[i, j]
            frequency = FREQ_MESH[i, j]
            args_list.append((target_size, frequency, fixed_electrical_power,
                            fixed_vehicle_speed, fixed_motion_error))
    
    results = []
    with Pool(thread_count) as pool:
        for r in tqdm.tqdm(pool.imap(_calculate_range_worker, args_list), total=len(args_list)):
            results.append(r)
    
    RANGE_MESH = np.array(results).reshape(TARGET_MESH.shape)
    print("100% complete")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.pcolormesh(TARGET_MESH, FREQ_MESH/1000, RANGE_MESH, 
                       cmap=colormap, shading='gouraud')
    
    colorbar = fig.colorbar(im, ax=ax)
    colorbar.set_label('Detection Range (m)', fontsize=11)
    
    ax.set_xscale('log')
    
    ax.set_xlabel('Target Diameter (m)', fontsize=12)
    ax.set_ylabel('Frequency (kHz)', fontsize=12)
    ax.set_title(f'SAS Detection Range\n'
                f'Power: {fixed_electrical_power}W, Speed: {fixed_vehicle_speed}m/s, '
                f'Motion Error: {fixed_motion_error}%', fontsize=14)
    
    ax.grid(True, alpha=0.3)
    
    # Scattering regime boundaries
    ka_05_freq = (0.5 * model.sound_speed) / (np.pi * target_diameters)
    ka_15_freq = (15 * model.sound_speed) / (np.pi * target_diameters)
    
    mask_05 = (ka_05_freq >= frequencies.min()) & (ka_05_freq <= frequencies.max())
    if np.any(mask_05):
        ax.plot(target_diameters[mask_05], ka_05_freq[mask_05]/1000, 
               'r--', linewidth=2.5, alpha=0.8, label='ka = 0.5 (Rayleigh→Mie)')
    
    mask_15 = (ka_15_freq >= frequencies.min()) & (ka_15_freq <= frequencies.max())
    if np.any(mask_15):
        ax.plot(target_diameters[mask_15], ka_15_freq[mask_15]/1000, 
               'orange', linestyle='--', linewidth=2.5, alpha=0.8, 
               label='ka = 15 (Mie→Geometric)')
    
    ax.text(0.003, 45, 'Rayleigh\n(TS ∝ f⁴)', fontsize=11, 
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7),
           ha='center', va='center')
    
    ax.text(0.015, 80, 'Mie/Resonance\n(complex)', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
           ha='center', va='center')
    
    ax.text(0.5, 145, 'Geometric\n(TS ≈ constant)', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
           ha='center', va='center')
    
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    return fig, ax

if __name__ == "__main__":
    print("Creating 3D surface plot...")
    fig_3d, ax_3d = create_3d_surface_plot()

    print("Creating 2D colormap plot...")
    fig_2d, ax_2d = create_cmap_2d_plot()
    
    print("\nCreating interactive plot...")
    create_interactive_plot()
    
    plt.show()