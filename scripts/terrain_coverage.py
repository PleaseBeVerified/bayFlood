"""Training-only coverage weights shared by tabular and CNN experiments."""

from collections import Counter
import numpy as np


ELEVATION_EDGES = (0, 5, 20, 50, 150)
HAND_EDGES = (1, 3, 10, 30)
SLOPE_EDGES = (1, 5, 15)
BLOCK_METERS = 2000


def terrain_stratum(values):
    return "e%d_h%d_s%d" % (
        np.searchsorted(ELEVATION_EDGES, values[0], side="right"),
        np.searchsorted(HAND_EDGES, values[9], side="right"),
        np.searchsorted(SLOPE_EDGES, values[5], side="right"),
    )


def coverage_weights(labels, counties, values, xy):
    """Retain all rows, flatten dense terrain/county/block groups mildly.

    Inputs MUST contain only the actual training subset after spatial buffers.
    Each class retains total weight equal to its row count. This does not change
    the class prior. Class balancing, when desired, is a separate loss weight.
    Empty terrain cells cannot receive support through weighting.
    """
    labels = np.asarray(labels)
    strata = [terrain_stratum(v) for v in values]
    blocks = [tuple(np.floor(point / BLOCK_METERS).astype(int)) for point in xy]
    result = np.ones(len(labels), dtype=float)
    for label in np.unique(labels):
        indexes = np.flatnonzero(labels == label)
        terrain_counts = Counter(strata[i] for i in indexes)
        county_counts = Counter(counties[i] for i in indexes)
        spatial_counts = Counter((strata[i], blocks[i]) for i in indexes)
        terrain_blocks = {s: len({blocks[i] for i in indexes if strata[i] == s})
                          for s in terrain_counts}
        raw = []
        for i in indexes:
            s = strata[i]
            terrain_factor = np.sqrt(len(indexes) / len(terrain_counts) / terrain_counts[s])
            county_factor = np.sqrt(len(indexes) / len(county_counts) / county_counts[counties[i]])
            block_factor = np.sqrt(terrain_counts[s] / terrain_blocks[s] /
                                   spatial_counts[(s, blocks[i])])
            raw.append(terrain_factor * county_factor * block_factor)
        raw = np.asarray(raw)
        # Normalize within class while retaining final 0.25–4 bounds.
        low, high = 0.0, 1e6
        for _ in range(70):
            mid = (low + high) / 2
            if np.clip(raw * mid, .25, 4).mean() < 1:
                low = mid
            else:
                high = mid
        result[indexes] = np.clip(raw * ((low + high) / 2), .25, 4)
        assert np.isclose(result[indexes].sum(), len(indexes))
    return result
