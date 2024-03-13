import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import click

import os
from astropy.time import Time
from astropy.constants import au, c
from astropy.coordinates import BarycentricTrueEcliptic, SkyCoord, get_body_barycentric, EarthLocation
import astropy.units as u
import psrchive
import importlib

import glob
from scipy.ndimage import uniform_filter
from folding.archive_utils import *
from load_profiles import *
from phase_aligned_search import *

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--dm", type=float, required=True, help="DM")
@click.option("--f0", type=float, required=True, help="F0")
@click.option("--ra", type=float, required=True, help="RA")
@click.option("--dec", type=float, required=True, help="DEC")
@click.option(
    "--load_only",
    is_flag=True,
    help="Load archives without changing their epoch",
)
@click.option(
    "--full_plot",
    is_flag=True,
    help="Apply P0 and P1 from phase search to archives and display diagnostic plot",
)
@click.option(
    "--no_search",
    is_flag=True,
    help="Don't perform P0/P1 search",
)


def main(ra, dec, dm, f0, load_only=False, full_plot=False, no_search=False):
    DM_incoherent = dm
    F0_incoherent = f0
    RA = ra
    DEC = dec
    directory = '/data/chime/sps/archives/candidates/' + f'{round(RA,2)}' + '_' +  f'{round(DEC,2)}'
    print(f"Searching in directory {directory}...")
    P0_incoherent = 1/F0_incoherent

    if not no_search:
        data = load_profiles(directory, F0_incoherent, DM_incoherent, RA, DEC, load_only)
    
        P0_min = P0_incoherent - 15e-6
        P0_max = P0_incoherent + 15e-6
        P1_min = 1e-15
        P1_max = 1e-11
        P0_lims = (P0_min, P0_max)
        P1_lims = (P1_min, P1_max)
        P0_points = 1000
        P1_points = 100
        
        explore_grid = ExploreGrid(data, P0_lims, P1_lims, P0_points, P1_points)
        P0s, P1s, SNRs_peak_sums, optimal_parameters, observations  = explore_grid.output()
        
        explore_grid.plot(squeeze=False)
    
    
    if full_plot:
    
        # Rewrite new ephemeris using new P0 and P1
        
        P0_optimal = optimal_parameters[0][0] 
        P1_optimal = optimal_parameters[1][0] 
        F0_optimal = 1 / P0_optimal
        F1_optimal = -1 / (P0_optimal**2) * P1_optimal
        
        first_obs_par = find_oldest_obs(directory, F0_optimal, DM_incoherent)
        optimal_par_file = directory + "/" + "optimal_par"
        
        with open(first_obs_par, "r") as input:
            with open(optimal_par_file, "w") as output:
                for line in input:
                    if line.strip("\n")[0:2] != "F0":
                        output.write(line)
                output.write("\t".join(["F0", str(F0_optimal)]) + "\n")
                output.write("\t".join(["F1", str(-F1_optimal)]) + "\n")
        
        subprocess.run(["pam", "-E", optimal_par_file, "-m", "cand*.newar"], cwd=directory)
        
        # Create archive scrunched in time
        subprocess.run(["pam", "-T", "-e", ".T", "cand*.newar"], cwd=directory)
        
        # Create archive scrunched in frequency
        subprocess.run(["pam", "-F", "-e", ".F", "cand*.newar"], cwd=directory)
        
        # Create archive scrunched in both
        subprocess.run(["pam", "-FT", "-e", ".FT", "cand*.newar"], cwd=directory)
        
        # Concatenate the modified archives
        subprocess.run(["psradd", "*.T", "-o", "added.T"], cwd=directory)
        
        explore_grid.plot(squeeze=True)

if __name__ == "__main__":
    main()