# Re-define classes / functions in phase_aligned_search
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import os
from astropy.time import Time
from astropy.constants import au, c
from astropy.coordinates import BarycentricTrueEcliptic, SkyCoord, get_body_barycentric, EarthLocation
import astropy.units as u

from scipy.ndimage import uniform_filter
from beamformer.utilities.common import find_closest_pointing

from numpy import unravel_index
from numba import njit, prange, set_num_threads
from folding.archive_utils import get_SN

@njit(parallel=True)
def phase_loop(profiles, dts, f0s, f1s):
    """
    Calculates SNR of the sum of the phase-shifted profiles

    profiles array has shape [ntime, nphase]
    """
    # average noise, assuming individual pulses are faint
    set_num_threads(8)

    sigma_off = np.std(profiles.sum(0)) 
    npbin = profiles.shape[1]
    Nf1 = len(f1s)
    Nf0 = len(f0s)
    chi2_grid = np.zeros((Nf0, Nf1))

    for i, f0i in enumerate(f0s):
        profsums = np.zeros((Nf1, profiles.shape[1]))
        for j in prange(Nf1):
            f1j = f1s[j]
            dphis = f0i*dts + 0.5*f1j*dts**2
            i_phis = (dphis*npbin).astype('int')
            
            for k, prof in enumerate(profiles):
                profsums[j] += np.roll(prof, -i_phis[k])

            chi2_grid[i,j] = np.sum( (profsums[j]-np.mean(profsums[j]))**2 / sigma_off**2 )
            
    return chi2_grid

def unwrap_profiles(profiles, dts, f0, f1):
    npbin = profiles.shape[1]
    dphis = f0*dts + 0.5*f1*dts**2
    i_phis = (dphis*npbin).astype('int')
    profs_shifted = np.zeros_like(profiles)
    for k, prof in enumerate(profiles):
        profs_shifted[k] = np.roll(prof, -i_phis[k])
    return profs_shifted
    
class ExploreGrid:
    def __init__(self, data, f0_lims, f1_lims, f0_points, f1_points):  
        self.f0_lims = f0_lims
        self.f1_lims = f1_lims
        self.profiles = data['profiles']
        self.ngate = len(data['profiles'][0])
        self.dts = data['times']
        self.f0_incoherent = data['F0']
        self.P0_incoherent = 1/self.f0_incoherent
        self.DM = data['DM']
        self.RA = data['RA']
        self.DEC = data['DEC']
        self.directory = data['directory']
                
        self.f0_points = f0_points
        self.f1_points = f1_points
        f0_ax = np.linspace(*self.f0_lims, self.f0_points) - self.f0_incoherent
        f1_ax = np.linspace(*self.f1_lims, self.f1_points)
        self.f0s, self.f1s = np.meshgrid(f0_ax, f1_ax)
        
        self.chi2_grid = phase_loop(self.profiles, self.dts, f0_ax, f1_ax)
        
        index_of_maximum = unravel_index(self.chi2_grid.argmax(), self.chi2_grid.shape)
        print(index_of_maximum, f0_ax[index_of_maximum[0]], f1_ax[index_of_maximum[1]] )
        
        df0_best = f0_ax[index_of_maximum[0]]
        f0_best = df0_best + self.f0_incoherent
        f1_best = f1_ax[index_of_maximum[1]]
        self.max_indeces = index_of_maximum
        
        self.optimal_parameters = (f0_best, f1_best)
        self.profiles_aligned = unwrap_profiles(self.profiles, self.dts, df0_best, f1_best)
        self.SNmax = get_SN(self.profiles_aligned.sum(0))
        
    def output(self):
        print('f0: ' + str(self.optimal_parameters[0]))
        print('f1: ' + str(self.optimal_parameters[1]))
        print('SNR: ' + str(np.max(self.SNmax)))
        return self.f0s, self.f1s, self.chi2_grid, self.optimal_parameters
            
    def plot(self, squeeze=True):
        plt.rcParams.update({'font.size': 18})
        if squeeze:
            fig, axs = plt.subplots(2, 3, figsize=(20, 20), gridspec_kw={'width_ratios': [1, 2, 2],'height_ratios': [1, 2] })
        else:
            fig, axs = plt.subplots(2, 2, figsize=(20, 15), gridspec_kw={'width_ratios': [1, 2],'height_ratios': [1, 2] })

        # Search grid plot
        f0best_plot = (self.optimal_parameters[0] - self.f0_incoherent)*1e6
        f1best_plot = self.optimal_parameters[1]*1e15
        axs[0, 0].pcolormesh(1e6*self.f0s, 1e15*self.f1s, self.chi2_grid.T)
        axs[0, 0].scatter(f0best_plot, f1best_plot, color='tab:orange', marker='x', s=20)
        axs[0, 0].set_xlabel(r'$\Delta f_0 (\mu Hz)$', fontsize=18)
        axs[0, 0].set_ylabel(r'$f_1$ (1e-15 s/s)', fontsize=18)
    
        # Aliasing plot
        axs[1, 0].plot(np.max(self.chi2_grid, axis=1).T, (self.f0s[0]) / 1e-6,
                       'b',alpha=0.6)
        axs[1, 0].set_ylabel(r'$\Delta f_0  (\mu Hz)$')
        axs[1, 0].set_xlabel(r'$\chi^{2}$')

        plt.subplots_adjust(hspace=0.15)
        plt.subplots_adjust(wspace=0.2)

        # Total summed pulse profile plot
        axs[0, 1].plot(np.linspace(0,1,self.ngate), np.sum(self.profiles_aligned, 0), color='k', label='Aligned sum')    
        axs[0, 1].label_outer()
        axs[0, 1].set_xlim(0, 1)
        
        axs[1, 1].imshow(self.profiles_aligned, aspect='auto', interpolation='Nearest',
                         extent=[0,1,0,len(self.profiles_aligned)])
        axs[1, 1].set_xlabel("Phase")
        axs[1, 1].set_ylabel("Days")

        # Frequency vs phase plot

        param_txt1 = (
        f"$f_0$: {self.optimal_parameters[0]}\n"
        f"$f_1$: {self.optimal_parameters[1]}\n"
        f"SNR: {np.max(self.SNmax)}"
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
        f"Max DM: {max_dm}"
        )

        axs[0,1].text(0, 1.3, param_txt1, transform=axs[0,0].transAxes, 
                      fontsize=18, va='top', ha='left', backgroundcolor='white')
        axs[0,1].text(1.5, 1.3, param_txt2, transform=axs[0,0].transAxes, 
                      fontsize=18, va='top', ha='left', backgroundcolor='white')
        axs[0,1].text(3, 1.3, param_txt3, transform=axs[0,0].transAxes, 
                      fontsize=18, va='top', ha='left', backgroundcolor='white') 
        directory = '/data/chime/sps/archives/candidates/' + f'{round(self.RA,2)}' + '_' + f'{round(self.DEC,2)}'
        print('Saving diagnostic plot...')
        plt.savefig(directory + "/phase_search_{0}_{1}.png".format(round(self.DM,2), round(self.f0_incoherent,2)))
