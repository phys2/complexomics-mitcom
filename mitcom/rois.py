"""
Find, merge and cluster Regions of Interest from proteins profiles.
"""
from collections.abc import Sequence
from typing import Iterator, Any

import numpy as np
import pandas as pd
import sklearn
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from mitcom.components import component2trace
from mitcom.similarity import pearson


def find(
    df_profiles: pd.DataFrame,
    df_components: pd.DataFrame,
    xx: Sequence[int],
    correlate_with="fit",
    rmin=0.95,
) -> Iterator[dict[str, Any]]:
    """
    Find Regions of Interest (ROIs) in protein profiles.

    A ROI is a protein profile range that positively correlates with
    a specific pattern (seed), which is either a fitted protein component
    or a stretch of a protein profile (defined by a fitted component).

    :param df_profiles: Protein profiles dataframe
    :param df_components: Component dataframe
    :param xx: x-values
    :param correlate_with: 'fit' | 'profile'
    :param rmin: Minimum correlation coefficient
    :return:
    """
    proteins = df_profiles.index
    df_p_max = df_profiles.max(axis=1)

    for row in df_components.itertuples(name="Component"):
        component = row.Index  # noqa
        protein = row.Protein  # noqa
        x0 = row.x0  # noqa
        x1 = row.x1  # noqa

        prot_pos = proteins.get_loc(protein)
        profiles = df_profiles.iloc[:, x0 : x1 + 1].to_numpy()

        if correlate_with == "fit":
            # todo: scale is not relevant for Pearson
            scale = df_p_max.loc[protein]
            trace = scale * component2trace(row, xx[x0 : x1 + 1])
        elif correlate_with == "profile":
            trace = profiles[prot_pos]
        else:
            raise ValueError(f"Invalid parameter value: '{correlate_with=}'")

        v_corr = pearson(trace, profiles)[0, 1:]

        # skip seed if seed trace has no positive correlation with its own profile
        # if v_corr[prot_pos] < rmin:
        #     continue

        for j in np.nonzero(v_corr >= rmin)[0]:
            prot = proteins[j]
            yield {
                "roi": f"{prot}@{component}",
                "prot": prot,
                "seed": component,
                # 'seed_center': row.Center,
                # 'seed_sigma': row.Sigma,
                "abund": df_profiles.iloc[j, x0 : x1 + 1].max(),
                "r": v_corr[j],
            }


def normalize(df_rois: pd.DataFrame):
    """
    Normalize values in ROIs dataframe.

    :param df_rois:
    :return:
    """
    scaler = sklearn.preprocessing.MinMaxScaler()

    # log10
    abund = np.log10(df_rois["abund"].to_numpy())

    # rescale
    # todo: is this first fit actually required?
    abund = scaler.fit_transform(abund.reshape(-1, 1))

    # clip outliers (spread distribution)
    sd = np.std(abund)
    mean = np.mean(abund)
    lo = mean - 2.5 * sd
    hi = mean + 2.5 * sd
    print("sd:", sd, "mean:", mean, "lo:", lo, "hi:", hi)
    df_rois["abund"] = np.clip(abund, lo, hi)

    cols = ["r", "abund"]
    df_rois[cols] = scaler.fit_transform(df_rois[cols].to_numpy())


def range_dist(a, b):
    """Absolute distance of two ranges"""
    return np.abs(a - b).max()


def cluster(df_rois: pd.DataFrame, maxdist=3, verbose=False):
    """
    Cluster ROIs on range distance.

    :param df_rois: ROIs dataframe
    :param maxdist: Maximum distance threshold for flat clusters
    :param verbose: Print some informational data
    :return:
    """
    if len(df_rois) == 1:
        labels = [1]
    else:
        try:
            d = pdist(df_rois.to_numpy(), metric=range_dist)
            z = linkage(d, method="average")
        except ValueError:
            print(df_rois)
            raise
        labels = fcluster(z, t=maxdist, criterion="distance")

    if verbose:
        for i in range(min(labels), max(labels) + 1):
            print(df_rois)
            print("---")
            print("cluster", i)
            mask = labels == i
            print(df_rois.loc[mask])

    return pd.Series(data=labels, index=df_rois.index, name="label")


def merge(df_rois: pd.DataFrame, maxdist=3, verbose=False):
    """
    Cluster and merge ROIs.

    First, cluster ROIs, then merge all ROIs in each cluster
    that belong to the same protein.

    :param df_rois: ROIs dataframe
    :param maxdist: Maximum distance threshold for flat clusters
    :param verbose: Print some informational data
    :return: New dataframe with closely overlapping ROIs merged.
    """
    groups = df_rois[["prot", "x0", "x1"]].groupby("prot")
    df_labels = groups.apply(cluster, maxdist=maxdist, verbose=verbose).droplevel(0)
    df_rois = df_rois.assign(cluster=df_labels)

    df_merged = df_rois.groupby(["prot", "cluster"], as_index=False).agg(
        {
            "prot": "first",
            "seed": lambda x: ",".join(sorted(x)),
            "abund": "first",
            "r": "median",
            "sigma": "median",
            "x0": "min",
            "x1": "max",
        }
    )

    del df_merged["cluster"]

    df_merged["roi"] = "{}@[{}-{}]".format(
        df_merged["prot"], df_merged["x0"].astype(str), df_merged["x1"].astype(str)
    )
    df_merged = df_merged.set_index("roi")

    # set center strip
    df_merged["center"] = (df_merged["x0"] + df_merged["x1"]) / 2

    return df_merged
