import logging
import os
import math
import numpy as np
import scipy.stats as stats
from scipy.special import chdtri
from scipy.fft import fft, rfft
from sps_common.constants import FREQ_BOTTOM, FREQ_TOP, DM_CONSTANT
from sps_databases import db_api
from matplotlib import pyplot as plt

log = logging.getLogger(__name__)

import numpy as np
import numpy.random as rand

phis = np.linspace(0, 1, 1024)
"""These values come from counting by eye a sample of 200 out of the 1208 pulsars
in the TPA dataset. Each represents the mean fraction of pulsars that have x number
of subpulses."""
mean_zeros = 0.522394
mean_ones = 0.40298
mean_twos = 0.064676
mean_threes = 0.00995

kernels = np.load(os.path.dirname(__file__) + "/kernels.npy")
kernel_scaling = np.load(os.path.dirname(__file__) + "/kernels.meta.npy")

def gaussian(mu, sig):
    x = np.linspace(0, 1, 1024)
    return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))

def lorentzian(phi, gamma, x0=0.5):
    return (gamma / ((phi - x0) ** 2 + gamma**2)) / np.pi

def sinc(x):
    """Sinc function."""
    return np.sin(x) / x

def generate(noise=False):
    """
    This function generates a random pulse profile to inject.

    Inputs:
    -------
            noise: bool
                whether or not the pulse should be distorted by white noise
    """
    u = rand.choice(np.linspace(0.01, 0.99, 1000))
    #inverse sampling theorem for an exponential distribution with lambda = 1/15
    gamma = -15*np.log(1 - u)/2/360    
    prof = lorentzian(phis, gamma)
    prof /= max(prof)
    subpulses = rand.choice(range(4), p=(mean_zeros, mean_ones, mean_twos, mean_threes))

    # interpulse?
    # the chances of having an interpulse are approximately 23/(1208 - 23), as in TPA
    roll = rand.choice(range(1208 - 23))
    if roll < 23:
        u = rand.choice(np.linspace(0.01, 0.99, 1000))
        #inverse sampling theorem for an exponential distribution with lambda = 1/15
        gamma_interpulse = -15*np.log(1 - u)/2/360
        x0_inter = rand.normal(0.5, 10 / 360)
        interpulse = lorentzian(phis, gamma_interpulse, x0_inter)
        interpulse *= rand.choice(np.linspace(0.4, 0.8)) / max(interpulse)
        # roll interpulse to correct location at ~180 deg from main pulse
        np.roll(interpulse, 512)
        prof += interpulse

    # subpulses
    for i in range(subpulses):
        u = rand.choice(np.linspace(0.01, 0.99, 1000))
        #inverse sampling theorem for an exponential distribution with lambda = 1/15
        gamma_sub = -15*np.log(1 - u)/2/360
        x0_sub = 0.5 + rand.normal(0, 0.1)
        subpulse = lorentzian(phis, gamma_sub, x0_sub)
        subpulse *= rand.choice(np.linspace(0.4, 0.8)) / max(subpulse)
        prof += subpulse

    # working on this-- the standard deviation isn't correct
    if noise:
        prof += rand.normal(0, 1, len(phis))

    return prof

def x_to_chi2(x, df):
    '''This function returns the approximate equivalent chi2 value for a normal distribution sigma.
    This function returns results with no greater than 0.01% error.

    Inputs:
    --------
        x (float): value in a normal distribution
        df (int) : degrees of freedom in your chi-squared distribution

    Returns:
    --------
        chi2 (float): approximate chi-squared value
    '''
    
    if x <= 7.3:
        mean = 0.0
        sd = 1.0
        
        # Calculate the cumulative distribution function (CDF) of normal distribution
        p = stats.norm.cdf(x, loc=mean, scale=sd)
        
        # Calculate the cumulative distribution function (CDF) of chi-squared distribution
        chi2 = stats.chi2.ppf(p, df)

        return chi2
    
    else:
        #A&S 26.2.23. This is the inverse of the operation for logp in PRESTO
        A = 2.515517
        B = 0.802853
        C = 0.010328
        D = 1.432788
        E = 0.189269
        F = 0.001308
        const = np.array([F, E - x*F, D - C - x*E, 1 - B - x*D, -(x + A)])
        t = np.roots(const)[1]
        prob = np.exp(-t**2/2)
        
        chi2 = chdtri(df, prob)

        return chi2


class Injection:
    """This class allows pulse injection."""

    def __init__(self, pspec_obj, phase_prof, sigma, true_f, true_dm):
        self.pspec = pspec_obj.power_spectra
        self.ndays = pspec_obj.num_days
        self.pspec_obj = pspec_obj
        self.birdies = pspec_obj.bad_freq_indices
        self.f = true_f
        self.deltaDM = self.onewrap_deltaDM() 
        self.true_dm = true_dm
        self.trial_dms = self.pspec_obj.dms
        self.true_dm_trial = np.argmin(np.abs(self.trial_dms - self.true_dm))
        self.phase_prof = phase_prof
        self.sigma = sigma
        self.power_threshold = 1
    
    def onewrap_deltaDM(self):
        """Return the deltaDM where the dispersion smearing is one pulse period in duration"""
        deltaDM =  1 / (1.0 / FREQ_BOTTOM**2 - 1.0 / FREQ_TOP**2) / self.f / DM_CONSTANT
        return deltaDM

    def sigma_to_power(self):
        '''This function converts an input Gaussian sigma to an approximately equivalent power. In order to do so,
        we have to estimate the number of summed harmonics in our final profile. This code was largely written by 
        Scott Ransom.

        Inputs:
        -------
            nharm (int): number of harmonics before the Nyquist cutoff frequency

        Returns:
        -------
            power (int): approximate equivalent power
        '''
        # Compute the normalized powers of the injected profile (Sum of pows == 1)
        prof_fft = rfft(self.phase_prof)[1:]
        norm_pows = np.abs(prof_fft)**2.0
        maxpower = norm_pows.sum()
        norm_pows /= maxpower
        assert(math.isclose(norm_pows.sum(), 1.0))
        Nallharms = len(norm_pows)

        # Compute the theoretical power in the harmonics where we will inject
        # a significant amount of our power (i.e. > 1%)
        Nsignif = int(((norm_pows/norm_pows.max())>0.01).sum())
        log.info(f'Nsignif = {Nsignif}')
        power = x_to_chi2(self.sigma, 2*Nsignif*self.ndays)/2
        log.info(f'Total theoretical power = {power}')

        # Now compute the theoretical power for each harmonic to inject. We will
        # add this value to the current stack. Note that we are subtracting the
        # predicted amount of power due to the means of the power spectra.

        power -= Nsignif * self.ndays
        scaled_fft = prof_fft[:Nsignif] * np.sqrt(power / maxpower)
        log.info(f'Sum of scaled_fft power = {np.sum(np.abs(scaled_fft)**2)}')

        return scaled_fft, Nsignif

    def disperse(self, prof_fft, kernels, kernel_scaling):
        '''
        This function disperses an input pulse profile over a range of -2*deltaDM to 2*deltaDM according
        to the algorithm specified above.

        Inputs:
        -------
                kernels (arr)  : 2D array containing the smeared impulse function kernels
                kernel_scaling (arr): a 1D array containing the labels of kernels in units of DM/deltaDM
        Returns:
        --------
                dispersed_prof_fft (arr): a 2D array of size (len(DM_labels), len(prof)) containing the
                                          dispersed profile values
        '''
        

        #i is our index referring to the DM_labels in the target power spectrum
        #find the starting index, where the DM scale is -2
        i_min = np.argmin(np.abs((self.true_dm -2*self.deltaDM) - self.trial_dms))
        log.info(f'Starting DM: {self.trial_dms[i_min]}')
        i0 = np.argmin(np.abs(self.true_dm - self.trial_dms))
        #find the stopping index, where the DM scale is +2
        i_max = np.argmin(np.abs((self.true_dm + 2*self.deltaDM) - self.trial_dms))
        log.info(f'Stopping DM: {self.trial_dms[i_max]}')
        dispersed_prof_fft = np.zeros((len(self.trial_dms), len(prof_fft)), dtype = 'complex_')
        dms = self.trial_dms[i_min:i_max + 1]

        for i in range(i_min, i_max + 1):
            key = np.argmin(np.abs(np.abs((self.trial_dms[i] - self.true_dm)/self.deltaDM) - kernel_scaling))
            dispersed_prof_fft[i] = prof_fft * kernels[key, :len(prof_fft)]
            #log.info(f'DM = {self.trial_dms[i]}, first harm power = {np.abs(dispersed_prof_fft[i, 1])**2}')

        return dispersed_prof_fft, i0

    def harmonics(self, prof_fft, df, n_harm):
        """
        This function calculates the array of frequency-domain harmonics for a given
        pulse profile.

        Inputs:
        _______
                prof (ndarray)   : pulse phase profile
                df (float)       : frequency bin width in target spectrum
                n_harm (int)     : the number of harmonics before the Nyquist frequency
        Returns:
        ________
                harmonics (ndarray) : Fourier-transformed harmonics of the profile convolved with
                                        [cycles] number of Delta functions
        """
        harmonics = np.zeros((4*n_harm))
        bins = np.zeros((4*n_harm)).astype(int)

        #now evaluate sinc-modified power at each of the first 10 harmonics
        for i in range(n_harm):
            
            f_harm = (i+1)*self.f
            bin_true = f_harm/df
            bin_below = np.floor(bin_true)
            bin_above = np.ceil(bin_true)

            #use 2 bins on either side
            current_bins = np.array([bin_below - 1, bin_below, bin_above, bin_above + 1])
            bins[i*4:i*4+4] = current_bins
            amplitude = prof_fft[i]*sinc(np.pi*(bin_true - current_bins))
            harmonics[i*4:i*4+4] = np.abs(amplitude)**2

        return bins, harmonics

    def injection(self):
        """
        This function creates the fake power spectrum and then interpolates it onto the
        range of the real power spectrum.

        Returns:
        _______
                fake_pspec (array): 2D power grid of form (trial DM, frequency)
                bins (array)      : 1D array of bin indices at which the pulse was injected
        """

        # pull frequency bins from target power spectrum
        freqs = self.pspec_obj.freq_labels
        df = freqs[1] - freqs[0]
        f_nyquist = np.floor(freqs[-1] / 2)
        n_harm = int(np.floor(f_nyquist / self.f))
        scaled_prof_fft, Nsignif = self.sigma_to_power()
        log.info(f'Theoretical power at true DM on top of distribution = {np.sum(np.abs(scaled_prof_fft)**2)}')
        
        if Nsignif < n_harm:
            n_harm = Nsignif

        log.info(f"Injecting {n_harm} harmonics.")

        dispersed_prof_fft, i0 = self.disperse(scaled_prof_fft, kernels, kernel_scaling)
        log.info(f'Power injected at true DM prior to sinc-interpolation: {np.sum(np.abs(dispersed_prof_fft[i0])**2)}')
        
        harms = []
        dms = []
        
        for i in range(len(dispersed_prof_fft)):
            bins, harm = self.harmonics(dispersed_prof_fft[i], df, n_harm)
            harms.append(harm)
            dms.append(i)

        harms = np.asarray(harms)
        log.info(f'shape of harms = {harms.shape}')

        log.info(f'Actual power injected at true DM on top of distribution = {np.sum(harms[i0])}')

        return harms, bins, dms


def main(pspec, injection_profile="random", num_injections=1):
    """
    This function runs the injection.

    Inputs:
    ______
            injection_profile (str or tuple): either a string specifying the key that references the
                                              profile in the dictionary defaults, or a tuple with
                                              custom injection profile parameters of the format
                                              (pulse profile, sigma, frequency,
                                              DM)
            num_injections (int)            : provided if injection_profile == 'random.' How many profiles
                                              to randomly generate.

    Returns:
    --------
            bins (list)                     : 2D list of bins with injected power
            dms (list)                      : 1D list of DMs with injected power
    """
    default_freq = rand.choice(
        np.linspace(0.1, 100, 10000), num_injections, replace=False
    )
    default_dm = rand.choice(np.linspace(10, 200, 10000), num_injections, replace=False)
    default_sigma = rand.choice(np.linspace(5, 20, 1000), num_injections, replace = False)

    defaults = {
        "gaussian": (gaussian(0.5, 0.025), 20, default_freq[0], 121.4375),
        "subpulse": (
            gaussian(0.5, 0.025) + 0.5 * gaussian(0.6, 0.015),
            20,
            default_freq[0],
            121.4375,
        ),
        "interpulse": (
            gaussian(0.5, 0.025) + 0.8 * gaussian(0.1, 0.02),
            20,
            default_freq[0],
            121.4375,
        ),
        "faint": (gaussian(0.5, 0.025), 10, default_freq[0], 121.4375),
        "high-DM": (gaussian(0.5, 0.025), 20, default_freq[0], 212.3),
        "slow": (gaussian(0.5, 0.025), 20, 3.27, 121.4375),
        "fast": (gaussian(0.5, 0.025), 20, 70.26, 121.4375),
    }

    injection_profiles = []

    if type(injection_profile) == str and injection_profile != "random":
        injection_profile = defaults[injection_profile]
        injection_profiles.append(injection_profile)
        if injection_profile != "slow" and injection_profile != "fast":
            log.info(f"Your randomly assigned frequency is {default_freq} Hz.")

    elif injection_profile == "random":
        for i in range(num_injections):
            pulse = generate()
            injection_profiles.append([pulse, default_sigma[i], default_freq[i], default_dm[i]])

    else:
        injection_profiles.append(injection_profile)

    i = 0
    dms = []
    bins = []
    # If the power is 0 nothing should be injected
    # Here I just check the zero bins in DM0 at the start and set them all to 0 at the end
    # There are probably easier ways to do this
    zero_bins = pspec.power_spectra[0, :] == 0
    for injection_profile in injection_profiles:
        injection, bins_temp, dms_temp = Injection(
            pspec, *injection_profile
        ).injection()
        if len(injection) == 0:
            log.info("Pulsar too weak.")
            continue
        parameters = np.array(
            [injection_profile[1], injection_profile[2], injection_profile[3]]
        )
        np.savetxt(f"Injection_{i}_params.txt", parameters)
        np.savetxt(f"Injection_{i}_profile.txt", injection_profile[0])
        dms.append(dms_temp)
        bins.append(bins_temp)
        # Just using pspec.power_spectra[dms_temp,:][:, bins_temp] will return the slice but
        # not change the object
        injected_indices = np.ix_(dms_temp, bins_temp)
        log.info(f'Mean of pre-injection indices = {np.mean(pspec.power_spectra[injected_indices])}')
        pspec.power_spectra[injected_indices] += injection.astype(
            pspec.power_spectra.dtype
        )
     
        log.info("Replacing power spectrum with injected power spectrum")
        i += 1
    pspec.power_spectra[:, zero_bins] = 0

    return bins, dms
