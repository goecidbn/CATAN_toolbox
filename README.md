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
catan-gui
```
or in GUI-less mode, matching a list of sessions into a CATAN project that the GUI opens (File ▸ Open project):
```bash
catan match --root /data/mouse01 --glob "*/neuron_detection_*.hdf5" --pixel-size 1.05
```
How the files are read (load preset, background image, orientation) is detected; whatever stays unclear, and any session that does not align, is asked in the terminal. Without a terminal, `--non-interactive` stops with a message naming the option that answers the question (for example `--on-issue skip|accept|abort`, `--background PATTERN`, `--load-config NAME`). `--recipe FILE.json` repeats a run, or loads a recipe saved in the GUI. See `catan match --help`.

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