import warnings
import zipfile
from collections.abc import Mapping, Callable
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from tables import NaturalNameWarning


def load_column(file, column: str | int = 0, header=None) -> list[Any]:
    """Load single column from text file"""
    if header is None and isinstance(column, str):
        # auto mode
        header = 1
    s = pd.read_table(file, usecols=[column], header=header).squeeze()
    return s.to_list()


def load_abundances(
    file, protein_name_transform: Optional[Callable] = None
) -> pd.DataFrame:
    """Load protein abundance values from text file"""
    df = pd.read_table(file, index_col=0, header=None, skiprows=1)
    if protein_name_transform is not None:
        df.index = df.index.map(protein_name_transform)
    df.index.name = "Protein"
    df.columns.name = "Slice"
    return df


def load_components(
    file, protein_name_transform: Optional[Callable] = None
) -> pd.DataFrame:
    """Load profile components from text file"""
    df = pd.read_table(file)

    if protein_name_transform is not None:
        df["Protein"] = df["Protein"].map(protein_name_transform)

    # rename some columns
    new_names = {
        "Center(Strip)": "Center",
        "Center_slice": "Center",
        "Peak": "Component",
        "PeakWidth": "Sigma",
    }
    df = df.rename(columns=new_names)

    # Create component id by joining protein and component name
    df["ComponentId"] = df["Protein"] + "-" + df["Component"].str.replace("P", "C")
    df = df.set_index(["Protein", "ComponentId"])

    # Remove obsolete columns
    for col in ["PeakName", "Center_mw"]:
        if col in df.columns:
            del df[col]

    return df


def load_complexes(file: Path, all_seeds=None) -> pd.DataFrame:
    """Load sets of pre-defined complexes from a file"""
    df_compl = pd.read_table(file)
    df_compl["component"] = df_compl["component"].str.replace("_P", "-C")

    # add _all_ seeds as one virtual component
    if all_seeds is not None:
        df_compl = pd.concat(
            [df_compl, pd.DataFrame({"component": all_seeds, "complex": "All Seeds"})],
            axis=0,
        )

    df_compl["fig"] = df_compl["complex"].str.split("_", expand=True)[0]
    df_compl["seed"] = df_compl["component"].apply(lambda c: c.split("-")[0] + "@" + c)
    df_compl["prot"] = df_compl["component"].str.split("-", expand=True)[0]

    return df_compl


def store_dataframes_zip(frames: Mapping[str, pd.DataFrame], storefile: Path, **kwargs):
    """
    Store dataframes to ZIP file.

    Notes:
        - Individual frames will be added as CSV files.
        - This method does NOT preserve indices and has issues with NaN values.
        - If possible, use the HDF storage methods instead.

    :param frames: Mapping of names to dataframes
    :param storefile: Path to storage file
    :param kwargs: Will be passed to DataFrame.to_csv()
    """
    storefile.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(storefile, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, frame in frames.items():
            content = frame.to_csv(**kwargs)
            zf.writestr(name, content)


def load_dataframes_zip(storefile: str | Path) -> dict[str, pd.DataFrame]:
    """
    Load dataframes from ZIP file.

    :param storefile: Path to storage file
    :return: Dict mapping name to frame
    """
    frames = {}
    with zipfile.ZipFile(storefile, "r") as zf:
        for f in zipfile.Path(zf).iterdir():
            if f.is_file() and f.name.endswith(".tsv"):
                with f.open() as f_data:
                    df = pd.read_table(f_data)
                frames[f.name] = df
    return frames


def store_dataframes_hdf(
    frames: Mapping[str, pd.DataFrame], storefile: str | Path
) -> None:
    """
    Store dataframes to HDF file.

    :param frames: Mapping of names to dataframes
    :param storefile: Path to storage file
    """

    if isinstance(storefile, Path):
        storefile = str(storefile.resolve())

    hdf = pd.HDFStore(storefile, complevel=9, complib="zlib")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NaturalNameWarning)
        for name, df in frames.items():
            # hdf.put(name, df, format='table', data_columns=True)
            hdf.put(name, df, format="table")
    hdf.close()


def load_dataframes_hdf(storefile: str | Path) -> dict[str, pd.DataFrame]:
    """
    Load dataframes from HDF file.

    :param storefile: Path to HDF file
    :return: Dict mapping key to dataframe
    """
    if isinstance(storefile, Path):
        storefile = str(storefile.resolve())

    frames = {}
    hdf = pd.HDFStore(storefile, complevel=9, complib="zlib")
    for key in hdf.keys():
        obj = hdf.get(key)
        if not isinstance(obj, pd.DataFrame):
            continue
        key = key.lstrip("/")
        frames[key] = obj
    hdf.close()

    return frames
