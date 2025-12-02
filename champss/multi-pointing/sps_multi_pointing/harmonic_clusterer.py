import logging
import numpy as np
from attr import attrib, attrs
from multiprocessing import Pool
from sklearn.cluster import DBSCAN
import tqdm
import pandas as pd
from sklearn.neighbors import sort_graph_by_row_values
from sklearn.neighbors import radius_neighbors_graph
from scipy import sparse
from functools import partial

log = logging.getLogger(__name__)


def frac_metric(f0, f1, max_harm=32):
    # Find distance based on rounding fractions
    stacked = np.stack([f0, f1])
    max_val = np.max(stacked, 0)
    min_val = np.min(stacked, 0)
    frac = max_val / min_val
    metric = np.abs(frac - np.round(frac).clip(1, max_harm))
    return metric


def get_frac_distances_for_chunk(
    data, max_dist=0.1, frac_eps=0.001, neighborhood_metric="chebyshev", max_harm=64
):
    # Run fract metric on chunk
    data_pos = data[0]
    data_freq = data[1]
    start_index = data[2]
    if data_pos.shape[0] == 0:
        return ([], [], [])
    neighbors_batch = radius_neighbors_graph(
        data_pos, max_dist, n_jobs=1, metric=neighborhood_metric
    ).tocoo()
    frac_dist_neighbors = frac_metric(
        data_freq[neighbors_batch.coords[0]],
        data_freq[neighbors_batch.coords[1]],
        max_harm=max_harm,
    )
    recorded_neighbors_bool = frac_dist_neighbors < frac_eps * 2
    return (
        neighbors_batch.coords[0][recorded_neighbors_bool] + start_index,
        neighbors_batch.coords[1][recorded_neighbors_bool] + start_index,
        frac_dist_neighbors[recorded_neighbors_bool],
    )


@attrs
class MultiPointingHarmonicClusterer:
    """Groups multi pointing candidates and writes clusters to a dataframe"""

    neighborhood_max_dist = attrib(default=1.1)
    dbscan_min_samples = attrib(default=1)
    freq_spacing = attrib(default=9.70127682e-04)
    dm_spacing = attrib(default=0.10119793310713615)
    dm_scale = attrib(default=1)
    freq_scale = attrib(default=1)
    ra_scale = attrib(default=1.0)
    dec_scale = attrib(default=1.0)
    neighborhood_metric = attrib(default="chebyshev")
    frac_eps = attrib(default=0.001)
    max_harm = attrib(default=64)

    def cluster(self, df: pd.DataFrame, num_threads: int = 16) -> pd.DataFrame:
        """
        Clusters the candidates included in a DataFrame
        """
        # For now, do not cluster along 360 degree edge

        # Sort for easier chunking
        df_sorted = df.sort_values("ra")

        data_pos = df_sorted[:][["ra", "dec", "mean_dm"]].to_numpy()
        data_pos[:, 0] *= self.ra_scale
        data_pos[:, 1] *= self.dec_scale
        data_pos[:, 2] *= self.dm_scale / self.dm_spacing

        data_freq = df_sorted[:]["mean_freq"].to_numpy()

        # Chunk data to save on computations
        # Chunks contain two times the maximum ra distance
        max_val = data_pos[-1, 0]
        max_dist = self.neighborhood_max_dist * self.ra_scale
        edge_values = np.arange(data_pos[0, 0], max_val + max_dist, max_dist)

        chunk_edges = np.searchsorted(data_pos[:, 0], edge_values, side="right")
        chunk_edges[0] = 0

        N = data_pos.shape[0]
        start_stop_indices = [
            (chunk_edges[i], chunk_edges[i + 2]) for i in range(len(chunk_edges) - 2)
        ]
        chunked_data = [
            [data_pos[start:stop, :], data_freq[start:stop], start]
            for (start, stop) in start_stop_indices
        ]

        with Pool(num_threads) as p:
            frac_distances = list(
                tqdm.tqdm(
                    p.imap(
                        partial(
                            get_frac_distances_for_chunk,
                            max_dist=max_dist,
                            frac_eps=self.frac_eps,
                            neighborhood_metric=self.neighborhood_metric,
                            max_harm=self.max_harm,
                        ),
                        chunked_data,
                    ),
                    total=len(chunked_data),
                )
            )
        frac_distances_out = np.concatenate(frac_distances, axis=1)

        # Filter duplicate values
        _, unique_indices = np.unique(frac_distances_out, axis=1, return_index=True)
        frac_distances_out = frac_distances_out[:, unique_indices]

        sparse_distances = sparse.coo_matrix(
            (
                frac_distances_out[2, :],
                (frac_distances_out[0, :], frac_distances_out[1, :]),
            ),
            shape=(N, N),
        ).tocsr()
        log.info(f"Created sparse distance matric {sparse_distances.__repr__()}")
        if sparse_distances.size == 0:
            log.info("No neightbouring points found. Aborting clustering process.")
            return df
        sparse_distances = sort_graph_by_row_values(
            sparse_distances, warn_when_not_sorted=False
        )

        dbres = DBSCAN(
            eps=self.frac_eps,
            min_samples=self.dbscan_min_samples,
            n_jobs=num_threads,
            metric="precomputed",
        ).fit(sparse_distances)

        labels = dbres.labels_
        uniq_labels = set(labels)
        np_unique_labels_output = np.unique(
            dbres.labels_, return_index=True, return_counts=True
        )
        log.info(
            f"Finished Harmonic clustering. Found {len(np_unique_labels_output[0])} clusters in {len(df)} multi-pointing candidates."
        )
        df_sorted["harm_cluster_label"] = dbres.labels_
        df_sorted["strongest_in_cluster"] = 0
        df_sorted["harm_cluster_size"] = 0
        sorted_label = np.argsort(np_unique_labels_output[2])[::-1]
        for label in sorted_label:
            subset = df_sorted[df_sorted["harm_cluster_label"] == label]
            df_sorted.loc[subset.index, "harm_cluster_size"] = len(subset)

            df_sorted.loc[subset["sigma"].idxmax(), "strongest_in_cluster"] = 1

        df_sorted = df_sorted.sort_values("sigma", ascending=False)
        df_sorted = df_sorted.reset_index()

        return df_sorted
