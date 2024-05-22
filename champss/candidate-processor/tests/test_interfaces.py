import numpy as np
from candidate_processor import harmonic_filter
from sps_common.interfaces.single_pointing import SearchAlgorithm


def test_harmonically_related_clusters(tmp_path):
    """Reading a saved HarmonicallyRelatedClusters instance should be equal to the
    original.
    """
    cluster = np.zeros(
        shape=1,
        dtype=[
            ("freq", float),
            ("dm", float),
            ("nharm", int),
            ("sigma", float),
        ],
    )
    harm_cluster = np.zeros(
        shape=1,
        dtype=[
            ("freq", float),
            ("dm", float),
            ("nharm", int),
            ("sigma", float),
        ],
    )
    raw_harmonic_powers0 = np.zeros(
        shape=32,
        dtype=[
            ("power", float),
            ("freq", float),
        ],
    )
    raw_harmonic_powers1 = np.zeros(
        shape=32,
        dtype=[
            ("power", float),
            ("freq", float),
        ],
    )
    cluster_ids = [np.int64(0), np.int64(1)]
    hrc_init_dict = dict(
        cluster_ids=cluster_ids,
        freq=1.0,
        dm=1.0,
        sigma=10.0,
        dc=0.1,
        main_cluster=cluster,
        ra=200.0,
        dec=42.0,
        threshold=5.5,
        freq_spacing=0.01,
        obs_id=["1"],
        detection_statistic=SearchAlgorithm(1),  # otherwise doesn't get saved properly
        harmonics_clusters={1: harm_cluster},
        raw_harmonic_powers={
            cluster_ids[0]: raw_harmonic_powers0,
            cluster_ids[1]: raw_harmonic_powers1,
        },
        rfi=False,
    )
    hrc = harmonic_filter.HarmonicallyRelatedClusters(**hrc_init_dict)
    hrcc = harmonic_filter.HarmonicallyRelatedClustersCollection([hrc])
    file_name = tmp_path / "test_harmonically_related_clusters.npz"
    hrcc.write(file_name)
    hrcc2 = harmonic_filter.HarmonicallyRelatedClustersCollection.read(file_name)
    hrc2 = hrcc2.clusters[0]

    assert hrc.cluster_ids == hrc2.cluster_ids
    assert hrc.freq == hrc2.freq
    assert hrc.dm == hrc2.dm
    assert hrc.sigma == hrc2.sigma
    assert hrc.main_cluster.dtype == hrc2.main_cluster.dtype
    assert (hrc.main_cluster == hrc2.main_cluster).all()
    assert hrc.ra == hrc2.ra
    assert hrc.dec == hrc2.dec
    assert hrc.threshold == hrc2.threshold
    assert hrc.freq_spacing == hrc2.freq_spacing
    assert hrc.obs_id == hrc2.obs_id
    assert hrc.detection_statistic == hrc2.detection_statistic
    assert hrc.harmonics_clusters.keys() == hrc2.harmonics_clusters.keys()
    for key in hrc.harmonics_clusters.keys():
        assert hrc.harmonics_clusters[key].dtype == hrc2.harmonics_clusters[key].dtype
        assert (hrc.harmonics_clusters[key] == hrc2.harmonics_clusters[key]).all()
    assert hrc.raw_harmonic_powers.keys() == hrc2.raw_harmonic_powers.keys()
    for key in hrc.raw_harmonic_powers.keys():
        assert hrc.raw_harmonic_powers[key].dtype == hrc2.raw_harmonic_powers[key].dtype
        assert (hrc.raw_harmonic_powers[key] == hrc2.raw_harmonic_powers[key]).all()
    assert hrc.rfi == hrc2.rfi
