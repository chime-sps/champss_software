import logging

import numpy as np
from ps_processes.processes.clustering import Clusterer
from scipy.fft import rfft
from sps_common.constants import FREQ_BOTTOM, FREQ_TOP
from sps_common.interfaces import PowerSpectraDetectionClusters, SearchAlgorithm
from sps_common.interfaces.ps_processes import PowerSpectra
from sps_common.interfaces.utilities import (
    harmonic_sum,
    powersum_at_sigma,
    sigma_sum_powers,
)
from sps_databases import db_api, db_utils

log = logging.getLogger(__name__)

import numpy as np
import numpy.random as rand

phis = np.linspace(0, 1, 1024)
"""
These values come from counting by eye a sample of 200 out of the 1208 pulsars in the
TPA dataset.

Each represents the mean fraction of pulsars that have x number of subpulses.
"""
mean_zeros = 0.522394
mean_ones = 0.40298
mean_twos = 0.064676
mean_threes = 0.00995
"""These values come from approximating the FWHMs given for interpulses in the TPA
dataset as a Gaussian distribution with a mean and a standard deviation.
"""
mean_gamma_interpulse = 0.021472222 / 2
mean_std_interpulse = 0.020875266 / 2


def gaussian(mu, sig):
    x = np.linspace(0, 1, 1024)
    return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))


def lorentzian(phi, gamma, x0=0.5):
    return (gamma / ((phi - x0) ** 2 + gamma**2)) / np.pi


"""These FWHM bins come from the Pareto distribution of the FWHM distributions of
the TPA dataset. Divided by 360 because they are given in degrees of phase."""
bins = (
    np.array(
        [
            [
                5.25300000e00,
                8.05500000e00,
                6.18700000e00,
                9.92300000e00,
                1.17910000e01,
                1.64610000e01,
                1.45930000e01,
                2.48670000e01,
                2.58010000e01,
                2.01970000e01,
                2.29990000e01,
                3.42070000e01,
                4.54150000e01,
                3.79430000e01,
                5.38210000e01,
                8.74450000e01,
                1.24805000e02,
                4.63490000e01,
                1.17333000e02,
                5.47550000e01,
            ],
            [
                1.20038722e-01,
                1.00677637e-01,
                9.97095839e-02,
                9.48693127e-02,
                6.58276865e-02,
                6.09874153e-02,
                6.00193606e-02,
                4.54985481e-02,
                4.54985481e-02,
                4.45304934e-02,
                4.45304934e-02,
                4.06582769e-02,
                3.48499521e-02,
                2.51694097e-02,
                2.51694097e-02,
                2.22652467e-02,
                1.93610837e-02,
                1.74249755e-02,
                1.64569219e-02,
                1.64569219e-02,
            ],
        ]
    )
    / 360
)
gammas = bins[0] / 2
weights = bins[1]
# normalize
weights /= sum(weights)


def generate(noise=False, shape=gaussian):
    """
    This function generates a random pulse profile to inject.

    Inputs:
    -------
            noise: bool
                whether or not the pulse should be distorted by white noise
    """
    gamma = rand.choice(gammas, p=weights)
    prof = lorentzian(phis, gamma)
    prof /= max(prof)
    subpulses = rand.choice(range(4), p=(mean_zeros, mean_ones, mean_twos, mean_threes))

    # interpulse?
    # the chances of having an interpulse are approximately 23/(1208 - 23), as in TPA
    roll = rand.choice(range(1208 - 23))
    if roll < 23:
        gamma_interpulse = rand.normal(mean_gamma_interpulse, mean_std_interpulse)
        x0_inter = rand.normal(0.5, 10 / 360)
        interpulse = lorentzian(phis, gamma_interpulse, x0_inter)
        interpulse *= rand.choice(np.linspace(0.4, 0.8)) / max(interpulse)
        # roll interpulse to correct location at ~180 deg from main pulse
        np.roll(interpulse, 512)
        prof += interpulse

    # subpulses
    for i in range(subpulses):
        gamma_sub = rand.choice(gammas, p=weights)
        x0_sub = 0.5 + rand.normal(0, 0.1)
        subpulse = lorentzian(phis, gamma_sub, x0_sub)
        subpulse *= rand.choice(np.linspace(0.4, 0.8)) / max(subpulse)
        prof += subpulse

    sigma = rand.choice(np.linspace(5, 25))

    # working on this-- the standard deviation isn't correct
    if noise:
        prof += rand.normal(0, 1, len(phis))

    return prof, sigma


class Injection:
    """This class allows pulse injection."""

    def __init__(self, pspec_obj, phase_prof, sigma, true_f, true_dm):
        self.pspec = pspec_obj.power_spectra
        self.pspec_obj = pspec_obj
        self.birdies = pspec_obj.bad_freq_indices
        self.f = true_f
        self.true_dm = true_dm
        self.trial_dms = self.pspec_obj.dms
        self.phase_prof = phase_prof
        self.sigma = sigma

    def get_power(self):
        """
        This function converts an SNR in standard deviation to an SNR in power.

        From Scott Ransom's PRESTO suite.
        """
        return self.sigma**2 / 2.0 + np.log(np.sqrt(np.pi / 2) * self.sigma)

    def disperse(self, trial_DM):
        """
        This function "dedisperses" a pulse profile according to some error from the
        true DM.

        Inputs:
        _______
                trial_dm (float)    : test DM

        Returns:
        ________
                dispersed_phase_prof: a DM-smeared 1D phase profile of a pulse
        """

        DM_err = self.true_dm - trial_DM

        # connect to database and find number of spectral channels in observation
        obs = db_api.get_observation(self.pspec_obj.obs_id[0])
        nchans = db_api.get_pointing(obs.pointing_id).nchans

        # create frequency array
        freqs = np.linspace(FREQ_TOP, FREQ_BOTTOM, nchans, endpoint=False)
        freq_ref = np.max(freqs)

        # define constants
        kDM = 1 / (2.41e-4)  # in MHz^2 s cm^3 pc^-1
        dt = 2.56 * 512 * 0.75 * 1e-6  # s

        # calculate dispersion delay
        delay = kDM * DM_err * (1 / freqs**2 - 1 / freq_ref**2)

        # calculate how much to shift bins
        dd_binshift = (delay // dt).astype("int")

        # create 2D pulse profile
        pulse2D = np.zeros((len(freqs), len(self.phase_prof)))
        pulse2D[:] = self.phase_prof

        # apply dispersion delay to each spectral channel of 2D pulse profile
        for i in range(len(freqs)):
            pulse2D[i] = np.roll(pulse2D[i], dd_binshift[i])

        # average all spectral channels to get 1D dispersed pulse
        dispersed_phase_prof = pulse2D.mean(0)

        return dispersed_phase_prof

    def harmonics(self, prof, df, n_harm, weights):
        """
        This function calculates the array of frequency-domain harmonics for a given
        pulse profile.

        Inputs:
        _______
                prof (ndarray)   : pulse phase profile
                df (float)       : frequency bin width in target spectrum
                n_harm (int)     : the number of harmonics before the Nyquist frequency
                weights (arr)    : the weight of each harmonic, calculated at the true DM
        Returns:
        ________
                harmonics (ndarray) : Fourier-transformed harmonics of the profile convolved with
                                        [cycles] number of Delta functions
        """
        harmonics = np.zeros(4 * n_harm)
        bins = np.zeros(4 * n_harm).astype(int)
        # take the fft of the pulse
        prof_fft = rfft(prof)[1:]

        def sinc(x):
            """Sinc function."""
            return np.sin(x) / x

        # now evaluate sinc-modified power at each of the first 10 harmonics
        for i in range(n_harm):
            f_harm = (i + 1) * self.f
            bin_true = f_harm / df
            bin_below = int(np.floor(bin_true))
            bin_above = int(np.ceil(bin_true))
            # use 2 bins on either side
            current_bins = np.array(
                [bin_below - 1, bin_below, bin_above, bin_above + 1]
            )
            bins[i * 4 : (i + 1) * 4] = current_bins
            amplitude = prof_fft[i] * sinc(np.pi * (bin_true - current_bins))
            harmonics[i * 4 : (i + 1) * 4] = np.abs(amplitude) ** 2

        harmonics *= weights

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
        prof_fft = rfft(self.phase_prof)[1:]
        weight = self.get_power() / np.sum(np.abs(prof_fft) ** 2)
        log.info(f"Injecting {n_harm} harmonics.")
        # weights = self.get_weights(n_harm)
        fake_pspec = np.zeros((len(self.trial_dms), len(freqs)))

        for i in range(len(self.trial_dms)):
            # disperse input pulse and FFT
            dispersed_prof = self.disperse(self.trial_dms[i])
            # take FFT
            bins, harm = self.harmonics(dispersed_prof, df, n_harm, weight)
            # add to fake power spectrum grid
            fake_pspec[i, bins] = harm

        return fake_pspec, bins


def main(pspec, injection_profile="random", num_injections=1):
    """
    This function runs the injection.

    Inputs:
    ______
            injection_profile (str or tuple): either a string specifying the key that references the
                                              profile in the dictionary defaults, or a tuple with
                                              custom injection profile parameters of the format
                                              (target power spectrum, pulse profile, sigma, frequency,
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
            pulse, sigma = generate()
            injection_profiles.append([pulse, sigma, default_freq[i], default_dm[i]])
    i = 0
    dms = []
    bins = []
    for injection_profile in injection_profiles:
        injection, bins_temp = Injection(pspec, *injection_profile).injection()
        log.info("Replacing power spectrum with injected power spectrum")
        parameters = np.array(
            [injection_profile[1], injection_profile[2], injection_profile[3]]
        )
        np.savetxt(f"Injection_{i}_params.txt", parameters)
        np.savetxt(f"Injection_{i}_profile.txt", injection_profile[0])
        dms.append(injection_profile[3])
        bins.extend(bins_temp)
        pspec.power_spectra[:] += injection.astype(pspec.power_spectra.dtype)
        i += 1
    # below is not working
    # print(pspec.bad_freq_indices)
    # for birdie in pspec.bad_freq_indices:
    #    pspec.power_spectra[birdie] += np.zeros(1).astype(pspec.power_spectra.dtype)

    return bins, dms