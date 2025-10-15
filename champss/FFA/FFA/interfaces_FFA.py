#!/usr/bin/env python

import numpy as np
from attr import ib as attrib
from attr import s as attrs
from attr.setters import validate
from attr.validators import instance_of
from easydict import EasyDict

@attrs
class Cluster_FFA:
    """
    A cluster of detections.

    Attributes:
    ===========
    detections (np.ndaray): detections output from PowerSpectra search. A numpy
        structured array with fields "dm", "freq", "sigma", "width"

    Post-init:
    ==========
    max_sig_det (np.ndarray): highest-sigma detection
    freq (float): frequency of the highest-sigma detection
    dm (float): DM of the highest-sigma detection
    sigma (float): sigma of the highest-sigma detection
    width (int): width of the highest-sigma detection

    Properties
    ==========
    ndetections (int): Number of detections
    unique_freqs (np.ndarray):  Unique frequencies in detections
    unique_dms (np.ndarray):  Unique DMs in detections
    num_unique_freqs (int): Number of unique freqs in detections
    num_unique_dms (int): Number of unique DMs in detections
    log_p_dm (float): log_10 (Period / DM)
    """

    detections: np.ndarray = attrib()
    max_sig_det: np.ndarray = attrib()
    freq: float = attrib()
    dm: float = attrib()
    sigma: float = attrib()
    width: int = attrib()

    @classmethod
    def from_raw_detections(cls, detections):
        max_sig_det = detections[np.argmax(detections["sigma"])]
        init_dict = dict(
            max_sig_det=max_sig_det,
            freq=max_sig_det["freq"],
            dm=max_sig_det["dm"],
            sigma=max_sig_det["sigma"],
            width=max_sig_det["width"],
            detections=detections,
        )
        return cls(**init_dict)

    @classmethod
    def from_saved_psdc(cls, detections, max_sig_det):
        init_dict = dict(
            max_sig_det=max_sig_det,
            freq=max_sig_det["freq"],
            dm=max_sig_det["dm"],
            sigma=max_sig_det["sigma"],
            width=max_sig_det["width"],
            detections=detections,
        )
        return cls(**init_dict)

    def filter_width(self):
        """Filter the cluster, only keeping detections which have the same width as the
        highest-sigma detection.
        """
        self.detections = self.detections[self.detections["width"] == self.width]

    @property
    def ndetections(self):
        """Number of detections."""
        return self.detections.shape[0]

    @property
    def unique_freqs(self):
        """Unique frequencies in detections."""
        return np.unique(self.detections["freq"])

    @property
    def unique_dms(self):
        """Unique DMs in detections."""
        return np.unique(self.detections["dm"])

    @property
    def num_unique_freqs(self):
        """Number of unique frequencies in detections."""
        return len(self.unique_freqs)

    @property
    def num_unique_dms(self):
        """Number of unique DMs in detections."""
        return len(self.unique_dms)

    @property
    def log_p_dm(self):
        """Log (P/DM)"""
        if self.dm == 0 or self.freq == 0:
            return np.NaN
        else:
            return np.log10(1 / (self.freq * self.dm))
