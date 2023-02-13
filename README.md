# Complexomics: MitCOM

This package contain supplementary code for the 
[MitCOM](https://www.complexomics.org/datasets/mitcom) publication.

[![DOI](https://zenodo.org/badge/540069120.svg)](https://zenodo.org/badge/latestdoi/540069120)

## Content

### Package

The `mitcom` package contains all relevant methods for handling protein profiles
and single profile components: from component detection and correlation to
similarity distance matrix calculation.

### Peak detection script

The `complex_find_peaks` executable for detecting peaks (aka components)
in protein profile data is available after package installation. 

It expects at least two tab-separated (tsv) input files with the following 
structure:

**File 1:**  Protein abundance values

| Protein | Sample-1 | Sample-N-1  | Sample-N    |
|---------|----------|-------------|-------------|
| Prot-1  | ...      | ...         | ...         |
| Prot-2  | ...      | ...         | ...         |
| Prot-3  | ...      | ...         | ...         |

**File 2:** Apparent molecular weight values

The file may contain any number of columns, only the **first** and the **MW**
column will be read. Values in the first column must match headers in File 1
above.

| Sample   | ... | MW     |
|----------|-----|--------|
| Sample-1 | ... | 350.3  |
| Sample-2 | ... | 423.6  |
| ...      | ... | ...    |
| Sample-N | ... | 3244.5 |

**Parameters:**

You can trigger plot creation by specifying a plot output directory, set the 
number of parallel workers, specify the output file names, a protein whitelist 
and more. 
Run `complex_find_peaks --help` to see all options.

### Notebook

The `MitCOM.ipynb` notebook exemplifies the way from data import to t-SNE 
output. It requires the `mitcom` package to be installed.

#### Data files loaded from notebook
  - *protein_abundance_file*: 
    Protein abundances (see above)

  - *components_file*: 
    Component output file from peak detection script
 
  - *protein_whitelist_file*:
    Text file containing whitelisted proteins (optional)

  - *predefined_complexes_file*: 
    Two-column text file (tsv). Can be used to define *known* 
    complexes by mapping component identifiers to arbitrary complex names. 
    This mapping is only used during t-SNE visualization. Data points belonging 
    to the same complex will be plotted as one individual trace that can be 
    shown/hidden by clicking on its associated legend item.
    _Note:_ The file is expected to exist and contain at least the header row.  

    Example:
    ```
    component    complex
    ATPA_P6      ATP#1
    ATPB_P6      ATP#1
    ATPD_P3      ATP#1
    SDH3_P8      SDH
    SDHA_P8      SDH
    SDHB_P9      SDH
    ...          ...
    ```
    

## Installation

### Standard

```bash
# optional: create and activate virtual environment
python -m venv .venv && .venv/Scripts/activate

# optional: install wheels to speed up installations
pip install wheels

pip install git+https://github.com/phys2/complexomics-mitcom.git@main
```

### With Jupyter notebook support

The optional dependency group `notebook` contains additional packages 
for running the Jupyter notebook, namely *jupyter*, *jupyterlab* and *plotly*.
To have these included, run these commands instead:

```bash
pip install "git+https://github.com/phys2/complexomics-mitcom.git@main#egg=complexomics-mitcom[notebook]"

# download MitCOM notebook
curl -L -O https://github.com/phys2/complexomics-mitcom/raw/main/mitcom/notebooks/MitCOM.ipynb
```


