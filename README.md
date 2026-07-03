# Calorimetry — Muon/Particle Identification and Tools

Overview
--------
This directory contains code and scripts for calorimeter-based particle identification (MuonID and related studies), clustering and shapes calculation, small utilities for validation, and machine-learning related artifacts (ONNX models and helper scripts).

Contents
--------
- `CalorimeterShapes.cxx` — compute calorimeter shape variables and energy.
- `GreatCluster.cxx` — clustering utilities and cluster reconstruction helpers.
- `TrainingMacro.cxx`, `TestingMacro.cxx` — ROOT macros used for ML training/testing workflows.
- `PyGymXGBOOST.py` — Python script for preparing data and creating XGBoost Tree.
- `CalorimeterCheck/` — utilities and checks (`CalorimeterCheck.cxx`, `CalorimeterValues.cxx`) for verifying calorimeter quantities (specially the left hadronic calorimeter that isnt working properly).
- `FunctionTest/` — Not ready script for normal use in eic.
- `MuonDiffPlots/` — Scripts searching feature-difference in the processes.
- `ONNX/` — exported ONNX models, example inputs, and setup scripts.
- `Plots/` — generated ROOT files and subfolders with plotting output (efficiencies, rejection curves, presentation plots).

Requirements
------------
- ROOT (6.x recommended) for building and running the C++ macros and producing plots.
- A C++ compiler compatible with ROOT (g++/gcc).
- Python 3 for auxiliary scripts; install `xgboost`, `numpy`, and `pandas` if you plan to run training helpers.
- ONNX Runtime (optional) to evaluate exported ONNX models.

Run
-----------
Most analyses are implemented as ROOT macros. You can run a macro directly with ROOT, for example:

```bash
root -l -q 'TrainingMacro.cxx++'
```

Data and Outputs
----------------
- Input datasets and intermediate ROOT files are typically produced by upstream reconstruction steps; example or small datasets may be present under `Plots/` or `ONNX/`.
- Generated plots are stored in `Plots/` (subfolders for efficiency, presentation, rejection, etc.).

Contributing / Notes
--------------------
- Follow the existing macro patterns when adding new analyses.
- Keep machine-learning model exports (ONNX) and their associated input descriptions together in `ONNX/`.

Contact
-------
If you need help running these tools or want to add documentation, open an issue or contact the repository maintainer.

