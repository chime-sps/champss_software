import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import os
from astropy.time import Time
from astropy.constants import au, c
from astropy.coordinates import BarycentricTrueEcliptic, SkyCoord, get_body_barycentric, EarthLocation
import astropy.units as u
import psrchive

import glob
from scipy.ndimage import uniform_filter
from folding.archive_utils import *
from beamformer.utilities.common import find_closest_pointing


class Signal:
    def __init__(self, data):
        """
        Return phase vs. signal
        """
        self.data = data
        self.length = len(data)
        self.domain = np.linspace(0, 1, self.length)
        self.function = sp.interpolate.interp1d(self.domain, self.data)
        
    def modified_function(self, phi):
        """
        Shifts phase by phase offset. 
        """
        try:
            mod_x = (self.domain - phi) % 1
            return self.function(mod_x)
        except:
            signals = np.zeros(shape=(*phi.shape, *self.domain.shape))
            phi_it = np.nditer(phi, flags=["multi_index"])
            for phi_element in phi_it:
                mod_x = (self.domain - phi_element) % 1
                signals[phi_it.multi_index] = self.function(mod_x)
            return signals
        
class PhaseOffset:
    """
    Calculate phase offset 
    """
    def __init__(self, time):
        self.obs_time = time
        
    def __call__(self, P0, P1, P0_incoherent):
        phi_change = (self.obs_time / P0) + (0.5 * P1 / (P0**2) * (self.obs_time)**2)
        phi0 = (self.obs_time / P0_incoherent)
        phi = phi_change - phi0
        return phi
    
class Observation:
    """
    Output shifted pulse profile
    """
    def __init__(self, data, time):
        self.signal = Signal(data)
        self.signal_length = self.signal.length
        self.phase_offset = PhaseOffset(time)
        
    def shifted_signal(self, P0, P1, P0_incoherent):
        phi = self.phase_offset(P0, P1, P0_incoherent)
        new_signal = self.signal.modified_function(phi)
        return new_signal

class SNR:
    """
    Calculates SNR of the sum of the phase-shifted profiles
    """
    def __init__(self, shifted_signals):
        total = np.zeros_like(shifted_signals[0])
        shape = total.shape    
        for signals in shifted_signals:
            total += signals
    
    # For each profile in shifted signals, calculate SNR using optimal binning
        # Need output same size  as peak_sums: (P_1, P_0)
        SNRs = np.zeros((shape[0],shape[1]))
        for j in range(shape[0]):
            for k, profile in enumerate(total[j]): 
                ngate = len(profile)
                maxbin = int(np.log2(ngate//2))
                binning = 2**np.arange(maxbin)
                SNprofs = np.zeros((len(binning), len(profile)))
                for i,b in enumerate(binning):
                    prof_filtered = uniform_filter(profile, b)
                    profsort = np.sort(prof_filtered)
                    prof_N = profsort[:3*ngate//4]
                    std = np.std(prof_N)
                    mean = np.mean(prof_N)
                    SNprof = (prof_filtered - mean) / std
                    SNprofs[i] = SNprof
                SNmax = np.max(SNprofs)
                SNRs[j][k] = SNmax 

        self.peak_sums = SNRs
        self.index_of_maximum = np.nonzero(self.peak_sums == np.max(self.peak_sums))

class ExploreGrid:
    def __init__(self, data, P0_lims, P1_lims, P0_points, P1_points):  
        self.P0_lims = P0_lims
        self.P1_lims = P1_lims
        self.data_time_array = data['data_time_array']
        self.F0_incoherent = data['F0']
        self.P0_incoherent = 1/self.F0_incoherent
        self.DM = data['DM']
        self.RA = data['RA']
        self.DEC = data['DEC']
        self.directory = data['directory']
        
        data_length = len(self.data_time_array[0][0])
        
        self.P0_points = P0_points
        self.P1_points = P1_points
        P0_ax = np.linspace(*self.P0_lims, self.P0_points)
        P1_ax = np.geomspace(*self.P1_lims, self.P1_points)
        self.P0s, self.P1s = np.meshgrid(P0_ax, P1_ax)
        
        self.observations = []
        signal_grids = []
        for (data, time) in self.data_time_array:
            observation = Observation(data, time)
            self.observations.append(observation)
            signal_arr = observation.shifted_signal(self.P0s.flatten(), self.P1s.flatten(), self.P0_incoherent)
            signal_grid = np.reshape(signal_arr, newshape=(*self.P0s.shape, observation.signal_length))
            signal_grids.append(signal_grid)
        self.SNRs = SNR(signal_grids)
        self.optimal_index = self.SNRs.index_of_maximum 
        self.optimal_parameters = (self.P0s[self.optimal_index], self.P1s[self.optimal_index])
    def output(self):
        print('P0: ' + str(self.optimal_parameters[0][0]))
        print('P1: ' + str(self.optimal_parameters[1][0]))
        print('SNR: ' + str(np.max(self.SNRs.peak_sums)))
        return self.P0s, self.P1s, self.SNRs.peak_sums, self.optimal_parameters, self.observations 
            
    def plot(self, squeeze=True):
        plt.rcParams.update({'font.size': 18})
        fig, axs = plt.subplots(2, 3, figsize=(20, 20), gridspec_kw={'width_ratios': [1, 2, 2],'height_ratios': [1, 2] })

        # Search grid plot
        axs[0, 0].scatter(self.P0s, self.P1s, c=self.SNRs.peak_sums, alpha=0.4)
        axs[0, 0].scatter(*self.optimal_parameters, color='red', marker='x')
        axs[0, 0].set_xlabel('$P_0$ (s)', fontsize=18)
        axs[0, 0].set_ylabel('$P_1$ (s/s)', fontsize=18)
        axs[0, 0].set_yscale('log')
    
        # Aliasing plot
        axs[1, 0].plot((self.P0s - self.P0_incoherent) / 1e-6, self.SNRs.peak_sums,'b.',alpha=0.1)
        axs[1, 0].set_xlabel('$\Delta P_0  (\mu s)$')
        axs[1, 0].set_ylabel('Signal')

        profiles = []
        for observation in self.observations:
            profiles.append(observation.shifted_signal((*self.optimal_parameters),self.P0_incoherent))

        plt.subplots_adjust(hspace=0.15)
        plt.subplots_adjust(wspace=0.2)

        # Total summed pulse profile plot
        axs[0, 1].plot(np.sum(profiles, 0), color='k', label='Aligned sum')    
        axs[0, 1].label_outer()

        # Concatenated averaged single day observations vs phase plot
        profarray = np.array(profiles)

        axs[1, 2].imshow(profarray, aspect='auto', interpolation='Nearest',extent=[0,1,0,len(profarray)])
        axs[1, 2].set_xlabel("Phase")
        axs[1, 2].set_ylabel("Days")

        # Frequency vs phase plot

        if squeeze:
            files = np.sort(glob.glob(f'{self.directory}/added.T'))
            i = -1  
            fn = files[i]
            data, F, T, psr, tel = readpsrarch(fn)
            fs, flag, mask, bg, bpass = clean_foldspec(data.squeeze()) 
            bpass = np.std(fs.mean(0), axis=-1)
            bpass /= np.median(bpass)
            fs[:,bpass>1.2] = 0
            taxis = ((T - T[0])*u.day).to(u.min)
            T0 = Time(T[0], format='mjd')
            fs_bg = fs - np.median(fs, axis=-1, keepdims=True)
            fs_bg = fs[1:-1]
            ngate = fs.shape[-1]
            nc = 256
            ng = 512
            binf = fs.shape[1]//nc
            bing = fs.shape[-1]//ng
            fs_bin = np.copy(fs_bg)
            maskchan = np.argwhere(fs_bin.mean(0).mean(-1)==0).squeeze()
            fs_bin[:,maskchan] = np.nan
            fs_bin = np.nanmean(fs_bin.reshape(fs_bin.shape[0], fs_bin.shape[1]//binf, binf, -1), 2)
            if fs.shape[-1] > ng:
                fs_bin = np.nanmean(fs_bin.reshape(fs_bin.shape[0], fs_bin.shape[1], -1, bing), -1)
                ngate = np.min([ngate, ng])
            profile = np.nanmean(np.nanmean(fs_bin,0), 0)
            SNsort = np.sort(profile)
            SN = (profile - np.nanmean(SNsort[:3*ngate//4])) / np.nanstd(SNsort[:3*ngate//4])
            vfmin = np.nanmean(fs_bin) - 2*np.nanstd(np.nanmean(fs_bin, 0))
            vfmax = np.nanmean(fs_bin) + 5*np.nanstd(np.nanmean(fs_bin, 0))
            vtmin = np.nanmean(fs_bin) - 2*np.nanstd(np.nanmean(fs_bin, 1))
            vtmax = np.nanmean(fs_bin) + 5*np.nanstd(np.nanmean(fs_bin, 1))
    
            axs[1,1].imshow(np.nanmean(fs_bin, 0), aspect='auto', interpolation='nearest', vmin=vfmin, vmax=vfmax, origin='lower',
              extent=[0,1, F[0], F[-1]])
            axs[1,1].set_ylabel('Frequency (MHz)')
            axs[1,1].set_xlabel('phase')
            # Time vs phase plot, stacked observations
    
            files = np.sort(glob.glob(f'{self.directory}/*.F'))
            if len(files) > 0:
                for i,fn in enumerate(files):
                    data, F, T, psr, tel = readpsrarch(fn)
                    fs = data.squeeze()
                    if i == 0:
                        z = fs.shape[0]
                        fs_comb = np.copy(fs)
                    else:
                        if len(fs) != z:
                            fs = np.resize(fs, (z, len(fs[0])))
                        fs_comb += fs[:fs_comb.shape[0]]
                fs_comb -= np.median(fs_comb, axis=-1, keepdims=True)
                axs[0, 2].imshow(fs_comb, aspect='auto', interpolation='nearest')
                axs[0, 2].set_ylabel('T (subints)')
                axs[0, 2].set_xlabel('phase')
                axs[0, 2].yaxis.tick_right()
                axs[0, 2].yaxis.set_label_position("right")
            

        param_txt1 = (
        f"$P_0$: {self.optimal_parameters[0][0]}\n"
        f"$P_1$: {self.optimal_parameters[1][0]}\n"
        f"SNR: {np.max(self.SNRs.peak_sums)}"
        )

        param_txt2 = (
        f"RA (deg): {self.RA}\n"
        f"DEC (deg): {self.DEC}\n"
        f"DM: {self.DM}"
        )

        gal_coord = SkyCoord(ra=self.RA*u.degree, dec=self.DEC*u.degree, frame='icrs')
        l = gal_coord.galactic.l.deg
        b = gal_coord.galactic.b.deg
        pointing = find_closest_pointing(self.RA, self.DEC)
        max_dm = pointing.maxdm
        
        param_txt3 = (
        f"$\ell$ (deg): {l}\n"
        f"$\it{{b}}$ (deg): {b}\n"
        f"Max DM LOS: {max_dm}"
        )

        axs[0,1].text(0, 1.3, param_txt1, transform=axs[0,0].transAxes, 
                      fontsize=18, va='top', ha='left', backgroundcolor='white')
        axs[0,1].text(1.5, 1.3, param_txt2, transform=axs[0,0].transAxes, 
                      fontsize=18, va='top', ha='left', backgroundcolor='white')
        axs[0,1].text(3, 1.3, param_txt3, transform=axs[0,0].transAxes, 
                      fontsize=18, va='top', ha='left', backgroundcolor='white') 
        directory = '/data/chime/sps/archives/candidates/' + f'{round(self.RA,2)}' + '_' + f'{round(self.DEC,2)}'
        print('Saving diagnostic plot...')
        plt.savefig(directory + "/phase_search_{0}_{1}.png".format(round(self.DM,2), round(self.F0_incoherent,2)))
