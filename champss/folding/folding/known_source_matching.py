"""
Known source matching utilities for candidate plotting.

This module provides functions for:
- Computing the EW arc for sidelobe source matching
- Checking frequency aliases (harmonics/sub-harmonics)
- Finding and categorizing nearby known sources
"""

import importlib
import sys

import numpy as np
from astropy.time import Time

from sps_databases.db_api import get_nearby_known_sources
from sps_multi_pointing.known_source_sifter import known_source_filters

# Import beam model for arc calculation
if importlib.util.find_spec("cfbm"):
    import cfbm as beam_model
    bm_config = beam_model.current_config
    beammod = beam_model.current_model_class(bm_config)


def get_arc_for_beam(beam_no, time, delta_x=90, samples=101):
    """
    Get the EW arc in RA and Dec for a given beam at a given time.

    Parameters
    ----------
    beam_no : int
        Beam number for beam of interest.
    time : datetime object
        Time of interest (central transit time).
    delta_x : float
        Number of degrees away from meridian that arc subtends. Default
        is 90 (horizon to horizon).
    samples : int
        Number of points in the arc.

    Returns
    -------
    eq_positions : ndarray
        Array of (RA, Dec) positions along the arc.
    """
    if "cfbm" not in sys.modules:
        return None

    eq_positions = []
    beam_pos = np.squeeze(beammod.get_beam_positions(beam_no, freqs=beammod.clamp_freq))
    y = beam_pos[1]

    xes = np.linspace(
        -delta_x * np.cos(np.deg2rad(y)), delta_x * np.cos(np.deg2rad(y)), samples
    )

    for x in xes:
        eq_positions.append(beammod.get_equatorial_from_position(x, y, time))

    return np.array(eq_positions)


def check_frequency_alias(f0_cand, f0_known, tolerance=0.001):
    """
    Check if a candidate frequency is an alias (harmonic or sub-harmonic) of a known pulsar.

    Parameters
    ----------
    f0_cand : float
        Candidate spin frequency in Hz
    f0_known : float
        Known pulsar spin frequency in Hz
    tolerance : float
        Fractional tolerance for frequency matching (default 0.1%)

    Returns
    -------
    alias_factor : float or None
        The harmonic ratio (e.g., 2 for 2nd harmonic, 0.5 for sub-harmonic),
        or None if no alias match found
    """
    # Integer harmonics and sub-harmonics up to 16
    alias_factors = [1/n for n in range(16, 0, -1)] + [n for n in range(2, 17)]
    # Half-integer harmonics: 3/2, 5/2, 7/2, ... up to 31/2 (covers up to 15.5)
    alias_factors += [n/2 for n in range(3, 33, 2)]
    # Sub-half-integer: 2/3, 2/5, 2/7, etc.
    alias_factors += [2/n for n in range(3, 33, 2)]

    for factor in alias_factors:
        expected_f0 = f0_known * factor
        if abs(f0_cand - expected_f0) / expected_f0 < tolerance:
            return factor

    return None


def find_matching_sources(ra, dec, f0, T, max_beam=None, radius=5, num_ks=5,
                          arc_search_radius=0.5, bright_threshold=50,
                          bright_radius=5.0, other_radius=1.0):
    """
    Find and categorize nearby known sources for a candidate.

    Parameters
    ----------
    ra : float
        Right ascension of candidate in degrees
    dec : float
        Declination of candidate in degrees
    f0 : float
        Spin frequency of candidate in Hz
    T : array
        Array of MJD times from the observation
    max_beam : int, optional
        Beam number for arc calculation
    radius : float
        Search radius for known sources in degrees (default 5)
    num_ks : int
        Maximum number of sources to return (default 5)
    arc_search_radius : float
        Search radius around arc for bright sources (default 0.5 degrees)
    bright_threshold : float
        Sigma threshold for bright sources (default 50)
    bright_radius : float
        Angular threshold for bright sources (default 5.0 degrees)
    other_radius : float
        Angular threshold for other sources (default 1.0 degrees)

    Returns
    -------
    dict : Dictionary containing:
        - 'ks_df_data': List of source entries for table
        - 'column_labels': Column labels for table
        - 'likely_match_count': Number of harmonic matches
        - 'ks_is_psr_scraper': List of booleans for scraper sources
        - 'ks_positions_match': List of (ra, dec) for harmonic matches
        - 'ks_positions_arc': List of (ra, dec) for arc sources
        - 'ks_positions_scraper': List of (ra, dec) for scraper sources
        - 'ks_positions_other': List of (ra, dec) for other sources
        - 'arc_positions': Array of arc positions or None
        - 'num_sources_not_displayed': Number of sources beyond num_ks
    """
    # Get nearby sources and sort by distance
    sources = get_nearby_known_sources(ra, dec, radius)
    pos_diffs = []
    for source in sources:
        pos_diff = known_source_filters.angular_separation(
            ra, dec, source.pos_ra_deg, source.pos_dec_deg
        )[1]
        pos_diffs.append(pos_diff)
    i_order = np.argsort(pos_diffs)
    sources_ordered = [sources[i] for i in i_order]

    # Compute the source arc for sidelobe matching
    arc_positions = None
    arc_sources = []
    if max_beam is not None and "cfbm" in sys.modules:
        # Get central transit time from T array
        T_mid = Time(np.median(T), format="mjd")
        transit_datetime = T_mid.to_datetime()
        arc_positions = get_arc_for_beam(max_beam, transit_datetime, delta_x=90, samples=201)

        if arc_positions is not None:
            # Search for bright sources along the arc
            for source in sources:
                # Check if this is a bright source
                is_bright = False
                if hasattr(source, 'detection_history') and source.detection_history:
                    sigmas = [d.get('sigma', 0) for d in source.detection_history if isinstance(d, dict)]
                    if sigmas and max(sigmas) > bright_threshold:
                        is_bright = True

                if is_bright:
                    # Check distance to arc
                    for arc_ra, arc_dec in arc_positions:
                        arc_dist = known_source_filters.angular_separation(
                            arc_ra, arc_dec, source.pos_ra_deg, source.pos_dec_deg
                        )[1]
                        if arc_dist < arc_search_radius:
                            arc_sources.append(source)
                            break

    # Collect nearby known sources
    ks_likely_matches = []
    ks_other = []
    ks_is_psr_scraper = []
    likely_match_count = 0

    # Track positions for sky scatter plot
    ks_positions_match = []
    ks_positions_other = []
    ks_positions_scraper = []
    ks_positions_arc = []

    # Track total sources that passed filters but weren't displayed
    num_sources_passed_filter = 0
    num_sources_not_displayed = 0

    # Combine sources_ordered with arc_sources (avoid duplicates)
    arc_source_names = set(s.source_name for s in arc_sources)
    all_sources = list(sources_ordered)
    for arc_src in arc_sources:
        if arc_src not in all_sources:
            all_sources.append(arc_src)

    for source in all_sources:
        # Check if this is a bright source (max sigma in detection_history > threshold)
        is_bright = False
        if hasattr(source, 'detection_history') and source.detection_history:
            sigmas = [d.get('sigma', 0) for d in source.detection_history if isinstance(d, dict)]
            if sigmas and max(sigmas) > bright_threshold:
                is_bright = True

        # Check if source is on the arc
        is_on_arc = source.source_name in arc_source_names

        pos_diff = known_source_filters.angular_separation(
            ra, dec, source.pos_ra_deg, source.pos_dec_deg
        )[1]

        # Bright sources: bright_radius threshold or on arc; other sources: other_radius threshold
        if is_on_arc and is_bright:
            # Arc sources are included regardless of angular distance
            pass
        else:
            angular_threshold = bright_radius if is_bright else other_radius
            if pos_diff > angular_threshold:
                continue

        # Source passed angular distance filter
        num_sources_passed_filter += 1

        ks_f0 = 1 / source.spin_period_s
        ks_dm = round(source.dm, 1)

        # Compute f0 ratios for display
        f0_ratio = f0 / ks_f0
        f0_ratio_inv = ks_f0 / f0

        # Check for harmonic match with 0.1% tolerance
        alias_factor = check_frequency_alias(f0, ks_f0, tolerance=0.001)
        is_harmonic_match = alias_factor is not None

        has_psr_scraper = source.survey and "psr_scraper" in source.survey

        ks_entry = [
            source.source_name,
            round(source.pos_ra_deg, 2),
            round(source.pos_dec_deg, 2),
            round(pos_diff, 2),
            round(ks_f0, 4),
            ks_dm,
            f"{f0_ratio:.3f}",
            f"{f0_ratio_inv:.3f}",
        ]

        if is_harmonic_match:
            ks_likely_matches.append(ks_entry)
            ks_is_psr_scraper.insert(likely_match_count, has_psr_scraper)
            ks_positions_match.append((source.pos_ra_deg, source.pos_dec_deg))
            likely_match_count += 1
        else:
            if len(ks_likely_matches) + len(ks_other) < num_ks:
                ks_other.append(ks_entry)
                ks_is_psr_scraper.append(has_psr_scraper)
                if is_on_arc:
                    ks_positions_arc.append((source.pos_ra_deg, source.pos_dec_deg))
                elif has_psr_scraper:
                    ks_positions_scraper.append((source.pos_ra_deg, source.pos_dec_deg))
                else:
                    ks_positions_other.append((source.pos_ra_deg, source.pos_dec_deg))
            else:
                # Source passed filter but not displayed due to num_ks limit
                num_sources_not_displayed += 1

    # Merge with likely matches first
    ks_all = ks_likely_matches + ks_other
    column_labels = ["Name", "RA", "Dec", r"$\Delta$Pos.", "F0", "DM", r"$f_0/f_{psr}$", r"$f_{psr}/f_0$"]

    return {
        'ks_df_data': ks_all,
        'column_labels': column_labels,
        'likely_match_count': likely_match_count,
        'ks_is_psr_scraper': ks_is_psr_scraper,
        'ks_positions_match': ks_positions_match,
        'ks_positions_arc': ks_positions_arc,
        'ks_positions_scraper': ks_positions_scraper,
        'ks_positions_other': ks_positions_other,
        'arc_positions': arc_positions,
        'num_sources_not_displayed': num_sources_not_displayed,
    }
