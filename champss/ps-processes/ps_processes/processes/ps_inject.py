import logging
import math
import os

import numpy as np
import scipy.stats as stats
from scipy.fft import rfft
from scipy.special import chdtri
from sps_common.constants import DM_CONSTANT, FREQ_BOTTOM, FREQ_TOP
from sps_common.interfaces.utilities import sigma_sum_powers

log = logging.getLogger(__name__)

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


def generate_pulse(noise=False):
    """
    This function generates a random pulse profile to inject.

    Inputs:
    -------
            noise: bool
                whether or not the pulse should be distorted by white noise
    """
    u = rand.uniform(0.01, 0.99, 1000)
    # inverse sampling theorem for an exponential distribution with lambda = 1/15
    gamma = -15 * np.log(1 - u) / 2 / 360
    prof = lorentzian(phis, gamma)
    prof /= max(prof)
    subpulses = rand.choice(range(4), p=(mean_zeros, mean_ones, mean_twos, mean_threes))

    # interpulse?
    # the chances of having an interpulse are approximately 23/(1208 - 23), as in TPA
    roll = rand.choice(range(1208 - 23))
    if roll < 23:
        u = rand.choice(np.linspace(0.01, 0.99, 1000))
        # inverse sampling theorem for an exponential distribution with lambda = 1/15
        gamma_interpulse = -15 * np.log(1 - u) / 2 / 360
        x0_inter = rand.normal(0.5, 10 / 360)
        interpulse = lorentzian(phis, gamma_interpulse, x0_inter)
        interpulse *= rand.choice(np.linspace(0.4, 0.8)) / max(interpulse)
        # roll interpulse to correct location at ~180 deg from main pulse
        np.roll(interpulse, 512)
        prof += interpulse

    # subpulses
    for i in range(subpulses):
        u = rand.choice(np.linspace(0.01, 0.99, 1000))
        # inverse sampling theorem for an exponential distribution with lambda = 1/15
        gamma_sub = -15 * np.log(1 - u) / 2 / 360
        x0_sub = 0.5 + rand.normal(0, 0.1)
        subpulse = lorentzian(phis, gamma_sub, x0_sub)
        subpulse *= rand.choice(np.linspace(0.4, 0.8)) / max(subpulse)
        prof += subpulse

    # working on this-- the standard deviation isn't correct
    if noise:
        prof += rand.normal(0, 1, len(phis))

    return prof


def x_to_chi2(x, df):
    """
    This function returns the approximate equivalent chi2 value for a normal
    distribution sigma. This function returns results with no greater than 0.01% error.

    Inputs:
    --------
        x (float): value in a normal distribution
        df (int) : degrees of freedom in your chi-squared distribution

    Returns:
    --------
        chi2 (float): approximate chi-squared value
    """

    if x <= 7.3:
        mean = 0.0
        sd = 1.0

        # Calculate the cumulative distribution function (CDF) of normal distribution
        p = stats.norm.cdf(x, loc=mean, scale=sd)

        # Calculate the cumulative distribution function (CDF) of chi-squared distribution
        chi2 = stats.chi2.ppf(p, df)

        return chi2 / 2

    else:
        # A&S 26.2.23. This is the inverse of the operation for logp in PRESTO
        A = 2.515517
        B = 0.802853
        C = 0.010328
        D = 1.432788
        E = 0.189269
        F = 0.001308
        const = np.array([F, E - x * F, D - C - x * E, 1 - B - x * D, -(x + A)])
        t = np.roots(const)[1]
        prob = np.exp(-(t**2) / 2)

        chi2 = chdtri(df, prob)

        # normalize to CHAMPSS power spectrum by dividing by 2
        return chi2 / 2


class Injection:
    """This class allows pulse injection."""

    def __init__(
        self,
        pspec_obj,
        full_harm_bins,
        DM,
        frequency,
        profile,
        sigma,
        scale_injections=False,
    ):
        self.pspec = pspec_obj.power_spectra
        self.ndays = pspec_obj.num_days
        self.pspec_obj = pspec_obj
        self.birdies = pspec_obj.bad_freq_indices
        self.f = frequency
        self.deltaDM = self.onewrap_deltaDM()
        self.true_dm = DM
        self.trial_dms = self.pspec_obj.dms
        self.true_dm_trial = np.argmin(np.abs(self.trial_dms - self.true_dm))
        self.phase_prof = profile
        self.sigma = sigma
        self.power_threshold = 1
        self.full_harm_bins = full_harm_bins
        self.rescale_to_expected_sigma = scale_injections
        self.use_rfi_information = True

    def onewrap_deltaDM(self):
        """Return the deltaDM where the dispersion smearing is one pulse period in
        duration.
        """
        deltaDM = 1 / (1.0 / FREQ_BOTTOM**2 - 1.0 / FREQ_TOP**2) / self.f / DM_CONSTANT
        return deltaDM

    def sigma_to_power(self, n_harm, df):
        """
        This function converts an input Gaussian sigma to an approximately equivalent
        power. In order to do so, we have to estimate the number of summed harmonics in
        our final profile. This code was largely written by Scott Ransom.

        Inputs:
        -------
            n_harm (int): number of harmonics before the Nyquist cutoff frequency
        Returns:
        -------
            scaled_fft (arr): the first Nsignif harmonics of the FFT of the pulse profile, scaled so that the sum
            of the powers will equal the input sigma, and not including the zeroth harmonic
            n_harm (int): The actually inected number of harmonics.
        """
        # Compute the normalized powers of the injected profile (Sum of pows == 1)
        prof_fft = rfft(self.phase_prof)[1:]
        norm_pows = np.abs(prof_fft) ** 2.0
        maxpower = norm_pows.sum()
        norm_pows /= maxpower
        # check the sum of powers is 1
        assert math.isclose(norm_pows.sum(), 1.0)
        Nallharms = len(norm_pows)

        # Compute the theoretical power in the harmonics where we will inject
        # a significant amount of our power (i.e. > 1%)
        Nsignif = int(((norm_pows / norm_pows.max()) > 0.01).sum())

        power = x_to_chi2(self.sigma, 2 * Nsignif * self.ndays)

        # Now compute the theoretical power for each harmonic to inject. We will
        # add this value to the current stack. Note that we are subtracting the
        # predicted amount of power due to the means of the power spectra.

        power -= Nsignif * self.ndays

        # if there are fewer significant harmonics than there are harmonics that fit
        # then use Nsignif
        # otherwise only take harmonics that fit before the Nyquist cutoff frequency
        if Nsignif < n_harm:
            n_harm = Nsignif

        scaled_fft = prof_fft[:n_harm] * np.sqrt(power / maxpower)

        return scaled_fft, n_harm

    def disperse(self, prof_fft, kernels, kernel_scaling):
        """
        This function disperses an input pulse profile over a range of -2*deltaDM to
        2*deltaDM according to the algorithm specified above.

        A more detailed walk-through of the algorithm can be found in reyniersquillace/RadioTime/Kernel_Dedispersion.

        Inputs:
        -------
                prof_fft (arr) : FFT array to disperse, not including the zeroth harmonic
                kernels (arr)  : 2D array containing the smeared impulse function kernels
                kernel_scaling (arr): a 1D array containing the labels of kernels in units of DM/deltaDM
        Returns:
        --------
                dispersed_prof_fft (arr): a 2D array of size (len(DM_labels), len(prof)) containing the
                                          dispersed profile values
                target_dm_idx (ndarray) : dm bins of the injection
        """

        # i is our index referring to the DM_labels in the target power spectrum
        # find the starting index, where the DM scale is -2

        i_min = np.argmin(np.abs((self.true_dm - 2 * self.deltaDM) - self.trial_dms))
        log.info(f"Starting DM: {self.trial_dms[i_min]}")
        i0 = np.argmin(np.abs(self.true_dm - self.trial_dms))
        # find the stopping index, where the DM scale is +2
        i_max = np.argmin(np.abs((self.true_dm + 2 * self.deltaDM) - self.trial_dms))
        log.info(f"Stopping DM: {self.trial_dms[i_max]}")

        # reminder; we are inclusive of i_max because that's the last index into which we want to inject
        target_dm_idx = np.arange(i_min, i_max + 1)
        dms = self.trial_dms[target_dm_idx]
        dispersed_prof_fft = np.zeros((len(dms), len(prof_fft)), dtype="complex_")

        # load the dispersion kernels and multiply our pulse profile by them
        for i in range(len(dms)):
            key = np.argmin(
                np.abs(np.abs((dms[i] - self.true_dm) / self.deltaDM) - kernel_scaling)
            )
            dispersed_prof_fft[i] = prof_fft * kernels[key, 1 : len(prof_fft) + 1]

        return dispersed_prof_fft, target_dm_idx

    def harmonics(self, prof_fft, df, N=4):
        """
        This function calculates the array of frequency-domain harmonics for a given
        pulse profile.

        Inputs:
        _______
                prof_fft (ndarray): FFT of pulse profile, not including zeroth harmonic
                df (float)        : frequency bin width in target spectrum
                n_harm (int)      : the number of harmonics before the Nyquist frequency
                N (int)           : number of bins over which to sinc-interpolate the harmonic
        Returns:
        ________
                bins (ndarray) : frequency bins of the injection
                harmonics (ndarray) : Fourier-transformed harmonics of the profile convolved with
                                        [cycles] number of Delta functions
        """
        n_harm = len(prof_fft)
        # Because of the rapid drop-off of the sinc function, adding further interpolation bins gives a negligible increase
        # in power fidelity. Hence the default is 4.
        harmonics = np.zeros(N * n_harm)
        bins = np.zeros(N * n_harm).astype(int)

        # now evaluate sinc-modified power at each of the first 10 harmonics
        for i in range(n_harm):
            f_harm = (i + 1) * self.f
            bin_true = f_harm / df
            bin_below = np.floor(bin_true)
            bin_above = np.ceil(bin_true)

            # use 2 bins on either side
            current_bins = np.array(
                [bin_below - 1, bin_below, bin_above, bin_above + 1]
            )
            bins[i * N : (i + 1) * N] = current_bins
            amplitude = prof_fft[i] * sinc(np.pi * (bin_true - current_bins))
            harmonics[i * N : (i + 1) * N] = np.abs(amplitude) ** 2

        return bins, harmonics

    def rednoise_normalize(self, ps_shape, inj_bins, rn_medians, rn_scales):
            """
            A scaled down version of the rednoise_normalize function from utilities.py for
            when we already know all the median information. Currently this is DM-secular
            but can be implemented easily to not be.

            Inputs:
            -------
                ps_len (int):     the length of the power spectrum
                inj_harms (arr):  array containing injection harmonics
                inj_bins (arr):   array containing bins into which to inject
                rn_medians (arr): array containing rednoise medians
                rn_scales (arr):  array containing rednoise scales

            Returns:
            -------
                rednoise-normalized harmonics
            """
            # get rid of 0th harmonic
            ps_shape = (ps_shape[0], ps_shape[1] - 1)
            # recall that the medians set the normalization of the power spectrum
            print(rn_scales)
            # assign the medians to the correct bins
            normalizer = np.zeros(ps_shape)
            
            for day in range(self.ndays):

                rn_medians = rn_medians[day] / rn_medians[day, -1]
                rn_scales = rn_scales[day]
                start = 0
                old_mid_bin = 0
                old_median = 1

                i = 0

                for bins in rn_scales:
                    mid_bin = int(start + bins / 2)
                    new_median = rn_medians[day, :, i]
                    if start == 0:
                        normalizer[:, start : start + mid_bin] += 1/new_median

                    # make sure we don't overflow our array
                    elif start + bins >= medians.shape[2]:
                        median_slope = np.linspace(
                            old_median, new_median, num=medians.shape[2] - old_mid_bin
                        )
                        normalizer[:, old_mid_bin:] += 1/median_slope

                    else:
                        # compute slopes
                        median_slope = np.linspace(
                            old_median, new_median, num=mid_bin - old_mid_bin
                        )
                        normalizer[:, old_mid_bin:mid_bin] += 1/median_slope
                    start += bins
                    old_mid_bin = mid_bin
                    old_median = new_median
                    i += 1

                # normalize:
                return normalizer[:, inj_bins]

    def predict_sigma(self, harms, bins, dm_indices, used_nharm, add_expected_mean):
        """
        This function predicts the sigma of an injection and scales it to a specific
        value if wanted.

        Inputs:
        _______
        Returns:
        ________
                harms (ndarray) : (Potentially rescaled) Fourier-transformed harmonics
                bins (ndarray) : frequency bins of the injection
                dm_indices (ndarray) : dm bins of the injection
                used_nharm (int) : Number of harmonics that will be injected
        """
        # estimate sigma
        true_dm_injection_index = self.true_dm_trial - dm_indices[0]
        bins_reshaped = bins.reshape(used_nharm, -1)
        allowed_harmonics = np.array([1, 2, 4, 8, 16, 32])
        used_search_harmonics = allowed_harmonics[
            allowed_harmonics * self.f <= self.pspec_obj.freq_labels[-1]
        ]
        all_summed_powers = []
        all_summed_powers_no_bg = []
        all_nsums = []
        all_nharms = []
        for search_harmonic in used_search_harmonics:
            used_injection_harmonic = min(search_harmonic, used_nharm)
            last_bins = bins_reshaped[used_injection_harmonic - 1, :]
            for bin in last_bins:
                bins_in_sum = self.full_harm_bins[:used_injection_harmonic, bin]
                _, _, bin_indices = np.intersect1d(
                    bins_in_sum, bins, return_indices=True
                )
                summed_power = harms[true_dm_injection_index, bin_indices].sum()
                nsum = search_harmonic * self.ndays
                # If search nharm is higher than injected nharm
                # Or are there other reasons?
                if search_harmonic > len(bin_indices):
                    bin_difference = len(bins_in_sum) - len(bin_indices)
                    summed_power += bin_difference * self.ndays
                # Add background
                all_summed_powers_no_bg.append(summed_power)
                if add_expected_mean:
                    summed_power += nsum
                all_summed_powers.append(summed_power)
                all_nsums.append(nsum)
                all_nharms.append(search_harmonic)
        all_summed_powers = np.array(all_summed_powers)
        all_summed_powers_no_bg = np.array(all_summed_powers_no_bg)
        all_nsums = np.array(all_nsums)
        all_nharms = np.array(all_nharms)
        possible_sigmas = sigma_sum_powers(all_summed_powers, all_nsums)
        best_sigma_trial = np.nanargmax(possible_sigmas)
        best_nharm = all_nharms[best_sigma_trial]
        best_sigma = possible_sigmas[best_sigma_trial]
        if self.rescale_to_expected_sigma and add_expected_mean:
            expected_power = (
                x_to_chi2(self.sigma, self.ndays * best_nharm * 2)
                - self.ndays * best_nharm
            )
            rescale_factor = expected_power / all_summed_powers_no_bg[best_sigma_trial]
            if np.isfinite(rescale_factor):
                harms *= rescale_factor
            best_sigma = self.sigma
        else:
            rescale_factor = 1
        return harms, best_nharm, best_sigma, rescale_factor

    def injection(self):
        """
        This function creates the fake power spectrum and then interpolates it onto the
        range of the real power spectrum.

        Returns:
        _______
                output_dict (dict): Dictionary containing the fields:
                    injected_powers (arr): 2D power grid of form (trial DM, frequency)
                    freq_indices (arr) : 1D array of bin indices at which the pulse was injected
                    dm_indices (arr): 1D array of the injected dm indices
                    predicted_nharm (int): nharm at which the injection should be detected,
                                            excluding the influence of the power spectrum.
                    predicted_sigma (float): sigma at which the injection should be detected,
                                            excluding the influence of the power spectrum.
                    predicted_nharm (int): nharm at which the injection should be detected,
                                            including the influence of the power spectrum.
                    predicted_sigma (float): sigma at which the injection should be detected,
                                           including the influence of the power spectrum.
                    injected_nharm (int): Nuber of harmonics which have been injected.
        """

        # pull frequency bins from target power spectrum
        freqs = self.pspec_obj.freq_labels
        df = freqs[1] - freqs[0]
        f_nyquist = freqs[-1]
        n_harm = int(np.floor(f_nyquist / self.f))
        scaled_prof_fft, n_harm = self.sigma_to_power(n_harm, df)
        used_nharm = len(scaled_prof_fft)
        log.info(f"Injecting {n_harm} harmonics.")

        dispersed_prof_fft, dm_indices = self.disperse(
            scaled_prof_fft, kernels, kernel_scaling
        )

        harms = []

        for i in range(len(dispersed_prof_fft)):
            bins, harm = self.harmonics(dispersed_prof_fft[i], df)
            harms.append(harm)
        
        harms = np.asarray(harms)
        normalized_harms = harms
        normalized_harms *= self.rednoise_normalize(
                self.pspec_obj.power_spectra.shape,
                bins,
                self.pspec_obj.rn_medians,
                self.pspec_obj.rn_scales,
            )

        # estimate sigma
        (
            harms,
            predicted_nharm,
            predicted_sigma,
            rescale_factor,
        ) = self.predict_sigma(harms, bins, dm_indices, used_nharm, True)

        if self.use_rfi_information:
            # Maybe want to enable buffering this value for faster multiple injection
            bin_fractions = self.pspec_obj.get_bin_weights_fraction()
            # Is linear scaling the way to go?
            harms *= bin_fractions[bins]
        injected_indices = np.ix_(dm_indices, bins)
        _, detection_nharm, detection_sigma, _ = self.predict_sigma(
            harms + self.pspec[injected_indices],
            bins,
            dm_indices,
            used_nharm,
            False,
        )

        log.info(
            f"Expected detection at {detection_sigma:.2f} sigma with"
            f" {detection_nharm} harmonics."
        )
        log.info(
            "Without spectra effects detection would be at"
            f" {predicted_sigma:.2f} sigma with {predicted_nharm} harmonics."
        )
        if self.rescale_to_expected_sigma:
            log.info(
                f"Rescaling injection so that it should be detected at {self.sigma}."
            )
        # prediction_ denotes that this injection would appear at those values without any
        #    effects from the power spectra values
        # detection_ includes the power spectra value for a more accurate prediction
        output_dict = {
            "injected_powers": harms,
            "freq_indices": bins,
            "dm_indices": dm_indices,
            "predicted_nharm": predicted_nharm,
            "predicted_sigma": predicted_sigma,
            "detection_nharm": detection_nharm,
            "detection_sigma": detection_sigma,
            "injected_nharm": n_harm,
        }
        return output_dict


def main(
    pspec,
    full_harm_bins,
    injection_profile="random",
    num_injections=1,
    remove_spectra=False,
    scale_injections=False,
    only_predict=False,
):
    """
    This function runs the injection.

    Inputs:
    ______
            injection_profile (str or dict): either a string specifying the key that references the
                                              profile in the dictionary defaults, or a dict with
                                              custom injection profile keys
                                              (profile, sigma, frequency, DM)
            num_injections (int)            : provided if injection_profile == 'random.' How
                                              many profiles to randomly generate.
            remove_spectra (bool)           : Whether to remove the spectra. Replaced by static value
            scale_injections (bool)         : Whether to scale the injection so that the
                                            detected sigma should be
                                            the same as the input sigma. Default: False
            only_predict (bool)             : Only predict the inection without changing the spectrum.

    Returns:
    --------
            injection_profiles (list(dict)) : List containing dict describing the injection.
    """
    default_freq = rand.choice(
        np.linspace(0.1, 100, 10000), num_injections, replace=False
    )
    default_dm = rand.choice(np.linspace(10, 200, 10000), num_injections, replace=False)
    default_sigma = rand.choice(np.linspace(5, 20, 1000), num_injections, replace=False)

    defaults = {
        "gaussian": {
            "profile": gaussian(0.5, 0.025),
            "sigma": 20,
            "frequency": default_freq[0],
            "DM": 121.4375,
        },
        "subpulse": {
            "profile": gaussian(0.5, 0.025) + 0.5 * gaussian(0.6, 0.015),
            "sigma": 20,
            "frequency": default_freq[0],
            "DM": 121.4375,
        },
        "interpulse": {
            "profile": gaussian(0.5, 0.025) + 0.8 * gaussian(0.1, 0.02),
            "sigma": 20,
            "frequency": default_freq[0],
            "DM": 121.4375,
        },
        "faint": {
            "profile": gaussian(0.5, 0.025),
            "sigma": 10,
            "frequency": default_freq[0],
            "DM": 121.4375,
        },
        "high-DM": {
            "profile": gaussian(0.5, 0.025),
            "sigma": 20,
            "frequency": default_freq[0],
            "DM": 212.3,
        },
        "slow": {
            "profile": gaussian(0.5, 0.025),
            "sigma": 20,
            "frequency": 3.27,
            "DM": 121.4375,
        },
        "fast": {
            "profile": gaussian(0.5, 0.025),
            "sigma": 20,
            "frequency": 70.26,
            "DM": 121.4375,
        },
    }

    injection_profiles = []

    if type(injection_profile) == str and injection_profile != "random":
        injection_profile = defaults[injection_profile]
        injection_profiles.append(injection_profile)
        if injection_profile != "slow" and injection_profile != "fast":
            log.info(f"Your randomly assigned frequency is {default_freq} Hz.")

    elif injection_profile == "random":
        for i in range(num_injections):
            pulse = generate_pulse()
            injection_profiles.append(
                {
                    "profile": pulse,
                    "sigma": default_sigma[i],
                    "frequency": default_freq[i],
                    "DM": default_dm[i],
                }
            )

    else:
        injection_profiles.append(injection_profile)

    if remove_spectra:
        log.info("Replacing spectra with expected mean value.")
        pspec.power_spectra[:] = pspec.num_days
        pspec.bad_freq_indices = [[]]

    # If the power is 0 nothing should be injected
    # Here I just check the zero bins in DM0 at the start and set them all to 0 at the end
    # There are probably easier ways to do this
    zero_bins = pspec.power_spectra[0, :] == 0

    i = 0
    for injection_dict in injection_profiles:
        injection_output_dict = Injection(
            pspec, full_harm_bins, **injection_dict, scale_injections=scale_injections
        ).injection()
        if len(injection_output_dict["injected_powers"]) == 0:
            log.info("Pulsar too weak.")
            continue

        injection_dict["dms"] = injection_output_dict["dm_indices"]
        injection_dict["bins"] = injection_output_dict["freq_indices"]
        injection_dict["predicted_nharm"] = injection_output_dict["predicted_nharm"]
        injection_dict["predicted_sigma"] = injection_output_dict["predicted_sigma"]
        injection_dict["detection_nharm"] = injection_output_dict["detection_nharm"]
        injection_dict["detection_sigma"] = injection_output_dict["detection_sigma"]
        injection_dict["injected_nharm"] = injection_output_dict["injected_nharm"]
        if isinstance(injection_dict["profile"], (np.ndarray, list)):
            injection_dict["profile"] = "custom_profile"

        if not only_predict:
            # Just using pspec.power_spectra[dms_temp,:][:, bins_temp] will return the slice but
            # not change the object
            injected_indices = np.ix_(
                injection_output_dict["dm_indices"],
                injection_output_dict["freq_indices"],
            )

            pspec.power_spectra[injected_indices] += injection_output_dict[
                "injected_powers"
            ].astype(pspec.power_spectra.dtype)

        i += 1
    pspec.power_spectra[:, zero_bins] = 0

    return injection_profiles
