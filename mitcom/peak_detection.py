import itertools
import sys
from collections import defaultdict
from collections.abc import Sequence, Mapping, Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional, Union

import joblib
import lmfit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from lmfit.model import ModelResult
from lmfit.models import GaussianModel
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import find_peaks

Vector = Union[Sequence[float | int], np.ndarray]


@dataclass
class ProfileComponent:
    """A protein profile fit component (aka peak)."""

    name: str
    center: float = -1
    height: float = -1
    sigma: float = -1
    fwhm: float = -1
    share: float = -1


@dataclass
class ProfileFitResult:
    """Protein profile fit results."""

    protein: str
    chisq: float = 0.0
    aic: float = 0.0
    bic: float = 0.0
    success: bool = True
    error: Optional[Exception] = None
    components: list[ProfileComponent] = field(default_factory=list)


class InformationCriteria:
    """Store AIC and BIC values."""

    def __init__(self, n, aic, bic):
        self.n = n
        self.aic = aic
        self.bic = bic


def mw2strip(x: float | Vector, xp: Vector, fp: Vector):
    """Convert molecular weight scale to strip scale"""
    # sample points must be monotonically increasing
    _xp = xp
    if _xp[0] > _xp[-1]:
        _xp = _xp[::-1]
    if not np.all(np.diff(xp) > 0):
        raise ValueError("Interpolation x-values must be monotonically increasing")
    return np.interp(x, _xp, fp)


def strip2mw(x: float | Vector, xp: Vector, fp: Vector):
    """Convert strip scale to molecular weight scale"""
    if not np.all(np.diff(xp) > 0):
        raise ValueError("Interpolation x-values must be monotonically increasing")
    return np.interp(x, xp, fp)


def detect_peaks(
    xx: np.ndarray,
    min_rel_prominence=0.5,
    peak_min_height=0.1,
    peak_min_distance=5,
    peak_min_prominence=0.01,
) -> np.ndarray:
    """
    Detect peak_locations in a protein profile and return their center x-values.
    """
    peaks, properties = find_peaks(
        x=xx,
        height=peak_min_height,
        distance=peak_min_distance,
        prominence=peak_min_prominence,
        width=(1, 50),
        wlen=50,
    )

    if len(peaks) == 0:
        # return simple maximum
        return np.array([np.argmax(xx)])

    # filter peak_locations by relative prominence
    mask = np.ones_like(peaks, dtype=bool)
    for i in range(len(peaks)):
        y_peak = properties["peak_heights"][i]
        y_left_base = xx[properties["left_bases"][i]]
        y_right_base = xx[properties["right_bases"][i]]

        prom_left = (y_peak - y_left_base) / y_peak
        prom_right = (y_peak - y_right_base) / y_peak
        prom_max = max(prom_left, prom_right)
        # print(peak_locations[i], prom_left, prom_right, properties['widths'][i])
        mask[i] = min_rel_prominence and prom_max >= min_rel_prominence

    if np.any(mask):
        peaks = peaks[mask]

    # sort by intensity
    peaks = np.array(sorted(peaks, key=lambda k: xx[k], reverse=True))

    return peaks


def build_model(
    xx: np.ndarray, peak_locations, max_sigma=20
) -> tuple[lmfit.Model, lmfit.Parameters]:
    """
    Build composite model for fitting a protein profile.
    """
    # alernative: M = LorentzianModel
    M = GaussianModel  # noqa

    composite_model = M()
    composite_params = lmfit.Parameters()

    # sort peak_locations (and model components) from left to right
    peak_locations = sorted(peak_locations)

    for i, peak in enumerate(peak_locations):
        model = M(prefix=f"P{i + 1}_")

        # let edge peak_locations have their center outside x range
        if peak >= len(xx) - 5:
            center_max = len(xx) + 10
        else:
            center_max = peak + 5
        if peak <= 5:
            center_min = -10
        else:
            center_min = peak - 5

        model.set_param_hint("center", value=peak, min=center_min, max=center_max)
        model.set_param_hint("sigma", value=2, min=1, max=max_sigma)
        model.set_param_hint("height", min=0.05, max=xx[peak])
        model.set_param_hint("amplitude", value=1, min=0)
        params = model.make_params()

        if i == 0:
            composite_model = model
            composite_params = params
        else:
            composite_model += model
            composite_params.update(params)

    return composite_model, composite_params


def create_plot(
    x: Vector,
    profile: np.ndarray,
    fit_out: ModelResult,
    components: Mapping[str, np.ndarray],
    protein: str,
    peaks: np.ndarray,
    profile_smooth: np.ndarray | None = None,
    xtick_locs: list[float] | None = None,
    xtick_labels: list[str] | None = None,
):
    fig, ax = plt.subplots(figsize=(15, 10), dpi=150)
    if profile_smooth is not None:
        ax.plot(
            x,
            profile_smooth,
            color="grey",
            linewidth=1,
            ls="dotted",
            label="Profile (smoothed)",
        )
    ax.plot(x, profile, color="grey", linewidth=1, label="Profile")
    ax.fill_between(x, profile, 0, color="whitesmoke")
    ax.scatter(peaks, profile[peaks], color="darkorange", label="peak", marker="x")

    ax2 = ax.twiny()
    ax2.plot(x, fit_out.best_fit, color="black", label="best fit")
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(20))

    for comp_name, comp_y in components.items():
        shapes = ax.plot(x, comp_y, label=comp_name[:-1], alpha=0.8)
        line_color = shapes[0].get_color()
        ax.fill_between(x, comp_y, 0, color=line_color, alpha=0.3)

        x_max = np.argmax(comp_y)
        y_max = np.max(comp_y)
        text = comp_name[:-1]
        xy = (x_max, 0)
        xytext = (x_max, -0.025)

        ax.annotate(
            text,
            xy=xy,
            xytext=xytext,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor=None,
                linewidth=0.5,
            ),
        )
        ax.annotate(
            text,
            xy=xy,
            xytext=xytext,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=8,
            backgroundcolor="white",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor=line_color,
                edgecolor="k",
                # color=line_color,
                linewidth=0.5,
                alpha=0.3,
            ),
        )
        ax.vlines(
            x_max,
            ymin=-0.05,
            ymax=y_max,
            colors=line_color,
            linestyle="dashed",
            linewidth=1,
            alpha=0.5,
        )

    if xtick_locs:
        ax.set_xticks(xtick_locs, xtick_labels)

    ax2.set_xlabel("strip number")
    ax.set_xlabel("apparent MW")
    ax.legend(loc="best")
    ax.set_title(protein)
    ax.set_ylim(bottom=-0.05)

    return fig


def get_fit_components(fit_result: lmfit.model.ModelResult) -> list[ProfileComponent]:
    """
    Build component objects from model fit result.
    """
    # collect parameters for each component
    component_parameters: dict[str, dict[str, float]] = defaultdict(dict)
    for name, val in fit_result.params.valuesdict().items():
        # name contains component and parameter name
        cname, pname = name.split("_", maxsplit=1)
        component_parameters[cname][pname] = val

    components = [
        ProfileComponent(
            name,
            height=params["height"],
            sigma=params["sigma"],
            fwhm=params["fwhm"],
            center=params["center"],
        )
        for name, params in component_parameters.items()
    ]

    return components


def process_protein(
    protein: str,
    profile: np.ndarray,
    plotdir: Optional[Path] = None,
    tick_locations: Optional[list[float]] = None,
    tick_labels: Optional[list[str]] = None,
    quiet: bool = False,
) -> ProfileFitResult:
    """
    Unmix a protein profile into multiple components.

    Perform initial peak fitting, model building and
    incremental addition of more components.

    :param protein: protein name
    :param profile: protein profile data
    :param plotdir: if set, write plot file to this directory
    :param tick_locations: manual tick label positions (molecular weight scale)
    :param tick_labels: manual tick labels
    :param quiet: if True, disables output of protein names

    :return: ProfileFitResult object"""
    if not quiet:
        print(protein, end="", flush=True)

    try:
        result = _process_protein_impl(
            protein, profile, plotdir, tick_locations, tick_labels
        )
        state = "ok"
    except Exception as e:
        result = ProfileFitResult(protein, success=False, error=e)
        state = f"error"

    if not quiet:
        print(": " + state)

    return result


def _process_protein_impl(
    protein: str,
    profile: np.ndarray,
    plotdir: Optional[Path] = None,
    tick_locations: Optional[list[float]] = None,
    tick_labels: Optional[list[str]] = None,
) -> ProfileFitResult:
    profile_smooth = gaussian_filter1d(profile, 2, mode="reflect")
    peaks = detect_peaks(profile_smooth)

    x = np.arange(len(profile))
    weights = profile
    profile_area = np.sum(profile)

    model, params = build_model(profile, peaks)
    result = model.fit(profile, params, x=x, weights=weights)

    # add more peaks if needed
    while len(peaks) < 12:
        # area above 'best fit' line
        remain = (profile - result.best_fit).clip(min=0)

        # stretches of continuous positive values
        chunks = [
            list(g) for k, g in itertools.groupby(x, lambda k: remain[k] != 0) if k
        ]
        chunks = [ch for ch in chunks if len(ch) > 5]

        if len(chunks) == 0:
            break

        # detect new peak location based on chunk area

        # find largest remaining chunk (area)
        chunk_areas = [np.sum(remain[ch]) for ch in chunks]

        # select chunk with largest area
        best_chunk_index = int(np.argmax(chunk_areas))
        best_chunk = chunks[best_chunk_index]

        best_chunk_area = chunk_areas[best_chunk_index]
        best_chunk_height = np.max(best_chunk)

        i = int(np.argmax(remain[best_chunk]))
        best_new_peak_index = best_chunk[i]

        if np.any(np.abs(peaks - best_new_peak_index) < 5):
            # print(f'there is already a peak at {best_new_peak_index}')
            break

        if not (best_chunk_area > 0.005 * profile_area and best_chunk_height > 0.1):
            # print(f'remain area or height small enough '
            #       f'(a={best_chunk_area / profile_area:.3f}, '
            #       f'h={best_chunk_height:.3f})')
            break

        # detect new peak location based on max chunk height
        # chunk_heights = [np.max(remain[ch]) for ch in chunks]
        # i_max_height = np.argmax(chunk_heights)
        # chunk_max_height = chunk_heights[i_max_height]

        # if chunk_areas[best_chunk_index] < 0.1 * profile_area:
        #     break

        # add additional peak
        # print("Adding additional peak at", best_new_peak_index)
        new_peaks = np.append(peaks, best_new_peak_index)
        model, params = build_model(profile, peaks)
        new_result = model.fit(profile, params, x=x, weights=weights)

        if not new_result.success:
            break

        result = new_result
        peaks = new_peaks

    if not result.success:
        raise ValueError(f"Fit aborted: {result.message}")

    components_yy = result.eval_components(x=x)

    if plotdir:
        fig = create_plot(
            x=x,
            profile=profile,
            fit_out=result,
            components=components_yy,
            protein=protein,
            peaks=peaks,
            profile_smooth=profile_smooth,
            xtick_locs=tick_locations,
            xtick_labels=tick_labels,
        )

        for file_format in ["png"]:  # ['png', 'svg', 'pdf']
            plot_file = plotdir / f"{protein}.{file_format}"
            plt.savefig(plot_file)

        plt.close(fig)

    # noinspection PyUnresolvedReferences
    fitresult = ProfileFitResult(
        protein, chisq=result.chisqr, aic=result.aic, bic=result.bic
    )
    fitresult.components = get_fit_components(result)

    best_fit_area = np.sum(result.best_fit)
    for component in fitresult.components:
        # part of total signal covered/explained by component
        comp_area = np.sum(components_yy[component.name + "_"])
        component.share = comp_area / best_fit_area

    return fitresult


def write_fit_results(
    results, outfile_fit: Path, outfile_components: Path, strip_to_mw: Callable
) -> None:
    with open(outfile_fit, "w") as f1, open(outfile_components, "w") as f2:
        print(
            "Protein",
            "Npeaks",
            # "ChiSqr",
            "AIC",
            "BIC",
            sep="\t",
            file=f1,
        )
        print(
            "Protein",
            "Peak",
            "Share",
            "Center(Strip)",
            "Center(MW)",
            "Height",
            "FWHM",
            "Sigma",
            sep="\t",
            file=f2,
        )
        for res in results:
            if res is None:
                continue
            if res.success:
                print(
                    res.protein,
                    len(res.components),
                    # res.chisq,
                    res.aic,
                    res.bic,
                    sep="\t",
                    file=f1,
                    flush=True,
                )
                for comp in res.components:
                    mw = strip_to_mw(comp.center)
                    print(
                        res.protein,
                        comp.name,
                        f"{comp.share}",
                        f"{comp.center:.1f}",
                        f"{mw:.1f}",
                        f"{comp.height:.5f}",
                        f"{comp.fwhm:.5f}",
                        f"{comp.sigma:.5f}",
                        sep="\t",
                        file=f2,
                        flush=True,
                    )
            elif res.error:
                print(
                    f"{res.protein}: An error ocurred: {res.error}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(f"{res.protein}: An unknown error ocurred (success==false)")


def run(
    profiles_file: Path,
    mw_map_file: Path,
    outfile_fit: Path,
    outfile_components: Path,
    plotdir: Optional[Path] = None,
    n_jobs: int = -2,
    whitelist: Sequence[str] | None = None,
    tick_locations: Sequence[float] | None = None,
    quiet=False,
) -> None:
    """
    Run peak detection.

    :param profiles_file: Protein profiles file
    :param mw_map_file: Strip index to molecular weight mapping file
    :param outfile_fit: Output file for fit values
    :param outfile_components: Output file for profile components
    :param plotdir: Output directory for plots
    :param n_jobs: Number of parallel jobs
    :param whitelist: Protein whitelist
    :param tick_locations: Tick locations (MW scale)
    :param quiet: Don't report progress
    """
    # load profile data
    df = pd.read_table(profiles_file, index_col=0)

    # load MW mapping
    mw_mapping = pd.read_table(mw_map_file, index_col=0).to_dict()["MW"]
    x_mw = [mw_mapping[col] for col in df.columns]
    x_strip = list(range(1, len(df.columns) + 1))

    # set locations and labels of plot ticks
    func_strip2mw = partial(strip2mw, xp=x_mw, fp=x_strip)
    if tick_locations is None:
        # auto mode: 7 ticks
        d = x_strip[-1] // 6
        xtick_locations = [x_strip[i] for i in range(0, x_strip[-1], d)]
        xtick_labels = list(func_strip2mw(xtick_locations))
    else:
        # silently drop invalid MW values
        tick_locations = [x for x in tick_locations if x_mw[0] <= x <= x_mw[-1]]
        # calculate tick x locations from provided MW values
        xtick_labels = [str(x) for x in tick_locations]
        xtick_locations = list(mw2strip(tick_locations, x_mw, x_strip))

    # handle whitelist
    if whitelist:
        s_index = set(df.index)
        s_whitelist = set(whitelist)
        if not_found := s_whitelist - s_index:
            print(
                f"Warning: {len(not_found)} protein(s) from whitelist not found "
                f"in data file: {sorted(not_found)}",
                file=sys.stderr,
            )
        proteins = sorted(s_index & s_whitelist)
    else:
        proteins = sorted(df.index)

    if plotdir:
        plotdir.mkdir(exist_ok=True)

    nprot = len(proteins)
    if nprot == 0:
        raise ValueError("No proteins to process")
    elif nprot == 1:
        n_jobs = 1

    func_process = partial(
        process_protein,
        plotdir=plotdir,
        tick_labels=xtick_labels,
        tick_locations=xtick_locations,
        quiet=True if n_jobs > 1 else quiet,
    )

    # for easier debugging, skip joblib entirely for n_jobs == 1
    if n_jobs == 1:
        results_gen = (
            func_process(protein=p, profile=df.loc[p].to_numpy()) for p in proteins
        )
    else:
        # adjust worker count to joblib style
        if n_jobs < 0:
            n_jobs -= 1
        # in multiprocessing mode, leave progress reporting to joblib
        verbose = 0 if quiet else 10
        results_gen = joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
            joblib.delayed(func_process)(protein=p, profile=df.loc[p].to_numpy())
            for p in proteins
        )

    # write to file in main process
    write_fit_results(
        results_gen, outfile_fit, outfile_components, strip_to_mw=func_strip2mw
    )
