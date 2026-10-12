"""ADAPT 0.4.5 neighbour-count weights without its unused TensorFlow import.

Uses the same sklearn NearestNeighbors defaults and count construction as
adapt.instance_based._nearestneighborsweighting.NearestNeighborsWeighting.fit_weights.
This deterministic operation needs no neural backend. Parity is tested against
the installed ADAPT wrapper before use in the post-hoc extension.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array


def nnw_weights(source, target):
    source, target = check_array(source), check_array(target)
    model = NearestNeighbors(n_neighbors=6, radius=1.0, algorithm="auto", leaf_size=30,
                             metric="minkowski", p=2, metric_params=None, n_jobs=None)
    model.fit(source)
    indices, counts = np.unique(model.kneighbors(target, return_distance=False).ravel(),
                                return_counts=True)
    weights = np.zeros(len(source))
    weights[indices] = counts
    return weights
