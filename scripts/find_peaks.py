"""
Run peak/component detection, optionally create fit plots.
"""
import argparse
from pathlib import Path

from mitcom import peak_detection


# noinspection PyTypeChecker
def arguments():
    defaults = {
        "plotdir": "peakfit_plots",
        "outfile": "peakfit_results.tsv",
        "componentsfile": "peakfit_components.tsv",
    }

    argp = argparse.ArgumentParser(description=__doc__)
    argp.add_argument(
        "profiles",
        type=Path,
        help="Profiles file",
    )
    argp.add_argument(
        "mapfile",
        type=Path,
        help="Strip-to-weight mapping file",
    )
    argp.add_argument(
        "-p",
        "--proteins",
        metavar="PROT",
        default=[],
        nargs="+",
        help="Consider only these proteins",
    )
    argp.add_argument(
        "-d",
        "--plotdir",
        metavar="DIR",
        type=Path,
        help="Plot output directory",
    )
    argp.add_argument(
        "-t",
        "--ticks",
        nargs='+',
        type=int,
        help="Specify MW tick label positions, eg '-t 150 200 500 1000'",
    )
    argp.add_argument(
        "-o",
        "--outfile",
        metavar="FILE",
        type=Path,
        default=defaults["outfile"],
        help=f"Write profile fit data to this file. "
        f"Defaults to {defaults['outfile']}.",
    )
    argp.add_argument(
        "-c",
        "--components",
        metavar="FILE",
        type=Path,
        default=defaults["componentsfile"],
        help=f"Write profile component data to this file. "
        f"Defaults to {defaults['componentsfile']}.",
    )
    argp.add_argument(
        "-j",
        "--jobs",
        metavar="N",
        type=int,
        default=-2,
        help="Number of concurrent jobs (negative values \n"
        "allowed: -1 uses all cores but one)",
    )
    argp.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Don't print progress to stdout",
    )

    return argp.parse_args()


def main():
    args = arguments()
    peak_detection.run(
        profiles_file=args.profiles,
        mw_map_file=args.mapfile,
        outfile_fit=args.outfile,
        outfile_components=args.components,
        plotdir=args.plotdir,
        n_jobs=args.jobs,
        whitelist=args.proteins,
        tick_locations=args.ticks,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
