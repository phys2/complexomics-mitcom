# Complexomics: MitCOM

This package contain supplementary code for the 
[MitCOM](https://www.complexomics.org/datasets/mitcom) publication.

## Content

### Package

The `mitcom` package contains all relevant methods for handling protein profiles
and single components: from detection over correlation to distance matrix and
similarity calculation.

### Notebook

The `MitCOM.ipynb` notebook makes use of these methods and shows the
way from data import to t-SNE output. It requires the `mitcom` package to be
installed/available.

### Peak detection script

For detecting peaks in protein profile data, the package makes the 
`complex_find_peaks` script available upon installation. 
It reads two tab-separated input files (tsv) with the following structure:

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

## Installation

### Standard

```bash
# with pip (consider to run this in a virtual environment)
pip install git+https://github.com/phys2/complexomics-mitcom.git

# with poetry
poetry install git+https://github.com/phys2/complexomics-mitcom.git
```

### With Jupyter notebook support

The optional dependency group `notebook` contains additional packages 
for running the Jupyter notebook, namely *jupyter*, *jupyterlab* and *plotly*.
To have these included, run these commands instead:

```bash
# with pip
pip install "git+https://github.com/phys2/complexomics-mitcom.git#egg=complexomics-mitcom[notebook]"

# with poetry
poetry install --extras notebook git+https://github.com/phys2/complexomics-mitcom.git
```


