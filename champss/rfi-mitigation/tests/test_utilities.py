#!/usr/bin/env python3

import numpy as np
from rfi_mitigation.utilities.cleaner_utils import combine_cleaner_masks

np.random.seed(2020)


def test_combine_cleaner_masks():
    mask1 = np.zeros((256, 1024), dtype=bool)
    mask2 = np.zeros_like(mask1, dtype=bool)
    mask3 = np.zeros_like(mask1, dtype=bool)

    mask1[np.arange(12), :] = True
    mask2[:-10, 0:13] = True
    mask3[:, [128, 256, 512, 768]] = True

    mask1_masked_frac = np.mean(mask1)
    mask2_masked_frac = np.mean(mask2)
    mask3_masked_frac = np.mean(mask3)
    mask_frac_list = [mask1_masked_frac, mask2_masked_frac, mask3_masked_frac]
    summed_mask_frac = np.sum(mask_frac_list)

    combined_mask, combined_mask_frac = combine_cleaner_masks(
        np.array([mask1, mask2, mask3])
    )

    # the combined mask must be at least as large as the smallest individual mask
    np.testing.assert_(min(mask_frac_list) <= combined_mask_frac)
    # the combined mask must be cannot be larger than the sum of all masks
    # (since the correct operation is logical OR)
    np.testing.assert_(combined_mask_frac <= summed_mask_frac)
