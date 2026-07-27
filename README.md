# CATAN

**CATAN** (**C**uration and **A**nalysis of **T**racked **A**cross-sessions **N**eurons)
is an open-source, cross-platform toolbox for the analysis, tracking and curation of neuronal populations
in longitudinal calcium imaging experiments.

It provides an interactive GUI together with a Python API for

- visualization of calcium imaging sessions
- ROI inspection
- registration and remapping between sessions
- probabilistic neuron matching
- quality assessment
- export/import of tracking results

CATAN is primarily designed for CaImAn output but aims to remain extensible.

---

## Features

- Interactive Qt/VisPy GUI
- Fast visualization of thousands of ROIs
- Across-session neuron tracking
- Rigid and non-rigid remapping
- Sparse matrix support
- HDF5-based project format
- Scriptable Python API

---

## Installation

```bash
pip install catan-toolbox[gui]
```

or

```bash
git clone https://github.com/goecidbn/CATAN_toolbox.git
cd CATAN
pip install -e ".[gui]"
```


CATAN allows for a GUI-less setup, not requiring graphic libraries such as PySide6 and vispy. For most users, the installation including the GUI will be of interest and can be set up via the commands above. For a graphic-less setup, use

```bash
pip install catan-toolbox
```
---

## Running CATAN

```bash
catan
```

or

```python
from catan.gui.app import main

main()
```

---

## Development

```bash
git clone https://github.com/goecidbn/CATAN_toolbox.git
pip install -e ".[gui,dev]"
```

Run the tests

```bash
pytest
```

---

## Project status

CATAN is currently under active development.
The file format and Python API may change until the first stable release.

---

## Citation

If you use CATAN in scientific work, please cite

> (paper / DOI once available)

---

## License

MIT License