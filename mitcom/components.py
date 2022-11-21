import math

import numpy as np
from lmfit.models import GaussianModel

gaussian_model = GaussianModel()
SQRT2PI = math.sqrt(2 * math.pi)


def gaussian(xx, center, sigma, height):
    """
    Return y values for a normal distribution with specified parameters.
    """
    amplitude = height * sigma * SQRT2PI
    return gaussian_model.eval(x=xx, center=center, sigma=sigma, amplitude=amplitude)


def component2trace(c, xx, clip=False):
    """
    Convert a component into a full trace (list of y-values).

    :param c: Component
    :param xx: x-values
    :param clip: If true, set values outside component boundaries to zero.
    :return:
    """
    yy = gaussian(xx, c.Center, c.Sigma, c.Height)
    if clip:
        yy[: c.x0] = 0
        yy[c.x1 :] = 0
    return yy


def component_boundaries(c, scale=1.0):
    """
    Clip component at specific sigma factor and return x-values.

    :param c: Component
    :param scale: Sigma scale factor
    """
    a = math.floor(c.Center - c.Sigma * scale)
    b = math.ceil(c.Center + c.Sigma * scale)

    return a, b


def filter_components(
    df, sigma_scale, n_strips, max_sigma=None, require_fwhm_in_range=True
):
    """Store section boundaries, filter components, etc"""
    # store boundaries for each component
    df[["x0", "x1"]] = df.apply(
        component_boundaries, args=(sigma_scale,), axis=1, result_type="expand"
    )

    # remove components with invalid component start/end values
    mask: np.ndarray = df["x0"] >= df["x1"]  # noqa
    print(f"Ignore {mask.sum()} components with invalid start/end values")
    df = df.loc[~mask]

    # remove components having their fwhm-range not completely inside the strip range
    if require_fwhm_in_range:
        # half width at half maximum
        df["hwhm"] = 0.5 * 2.3548 * df["Sigma"]

        mask = (df["Center"] - df["hwhm"]) < 0
        print(
            f"Ignore {mask.sum()} components having their FWHM-range "
            f"start before strip 0"
        )
        df = df.loc[~mask]

        mask = (df["Center"] + df["hwhm"]) > n_strips  # noqa
        print(
            f"Ignore {mask.sum()} components having their FWHM-range "
            f"end after strip {n_strips}"
        )
        df = df.loc[~mask]

        del df["hwhm"]

    # remove components with sigma being too large
    if max_sigma:
        mask = df["Sigma"] > max_sigma  # noqa
        print(f"Ignore {mask.sum()} components with sigma > {max_sigma}")
        df = df[mask]
        print("Number of components after sigma filter:", len(df))

    # remove protein name from component index
    df = df.reset_index().set_index("ComponentId")

    return df
