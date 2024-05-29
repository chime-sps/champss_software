#!/usr/bin/env python3

import argparse
import os

import yaml
from candidate_processor.clustering import PowerSpectraDetectionClusterer
from candidate_processor.feature_generator import Features
from candidate_processor.harmonic_filter import (
    HarmonicallyRelatedClustersCollection,
    HarmonicFilter,
)
from sps_common.interfaces.ps_processes import PowerSpectraDetections

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        help="hdf5 file name containing the power spectra detections",
        type=str,
    )
    parser.add_argument(
        "-g", help="min group size to be considered a cluster", type=int, default=5
    )
    parser.add_argument(
        "--scale",
        help="the scale of the frequency bins relative to the DM bins",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--eps",
        help="the eps value for the DBSCAN clustering process",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--max-harm",
        help="the maximum harmonic to match to a cluster",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--min-harm",
        help="the minimum fractional harmonic to match to a cluster",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--min-freq",
        help="the minimum frequency to consider a detection cluster as a candidate",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--min-dm",
        help="the minimum dm to consider a detection cluster as a candidate",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--plot",
        help="plot the harmonically related clusters",
        action="store_true",
    )
    parser.add_argument(
        "-o",
        help=(
            "output filename to store the single pointing candidate collection npz file"
        ),
        type=str,
        default="./single_pointing_candidate_collection.npz",
    )
    parser.add_argument(
        "-o_hrcs",
        help=(
            "npz filename to store the harmonically related clusters, if wish to write"
            " them"
        ),
        type=str,
        default="",
    )
    parser.add_argument(
        "--config",
        help="features configuration file",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__))
        + "/candidate_processor/config_features.yaml",
    )
    args = parser.parse_args()
    psd = PowerSpectraDetections.read(args.f)
    clusterer = PowerSpectraDetectionClusterer(
        cluster_scale_factor=args.scale, dbscan_eps=args.eps, dbscan_min_samples=args.g
    )
    spdc = clusterer.cluster(psd)
    hf = HarmonicFilter(
        max_harm=args.max_harm,
        min_harm=args.min_harm,
        freq_threshold=args.min_freq,
        dm_threshold=args.min_dm,
    )
    hrc_list = hf.group_harmonics(spdc)
    if args.plot:
        for hrc in hrc_list:
            hrc.plot_cluster(args.o)

    if args.o_hrcs:
        hrcc = HarmonicallyRelatedClustersCollection(clusters=hrc_list)
        hrcc.write(args.o_hrcs)

    with open(args.config) as f:
        features_config = yaml.safe_load(f)
    fg = Features.from_config(features_config)

    spcc = fg.make_single_pointing_candidate_collection(hrc_list)
    spcc.write(args.o)
