# Identification

This directory contains code for particle identification in EPIC analyses: feature extraction, classifier training and testing, and validation using reconstructed data.

## Main directories

- [`CalorimetryHits/`](CalorimetryHits/) — the current analysis of calorimeter-hit features. It contains macros for data preparation, classifier training and testing, and exported ONNX models. This code was used in the algorithm presented in Glasgow.
- [`ToF/`](ToF/) — analysis of information from the Time of Flight detector. It contains feature preparation, classifier training and testing, and ONNX models; it is part of the algorithm presented in Glasgow.
- [`MuonID/`](MuonID/) — implementation of the `MuonID(...)` function, which can be called from an analysis for a reconstructed track and event frame. The function returns a muon-identification probability score. It does not loop over events, perform truth selection, apply a threshold, or fill histograms; those steps belong in the analysis code.
- [`FinalClassificator/`](FinalClassificator/) — macro for checking the performance of the complete classification algorithm.
- [`Tracker/`](Tracker/) — auxiliary checks and track-feature analysis; it is not the main classifier.
- [`DRICH/`](DRICH/) — auxiliary checks of dRICH detector features; it is not the main classifier.
- [`CalorimetryClusters/`](CalorimetryClusters/) — an older version of the calorimeter analysis based on clusters. These are legacy materials, not the main current algorithm workflow.

## Requirements

- **EIC Shell**, with ROOT, podio, and the EDM4eic/EDM4hep libraries available in the environment.
- **ONNX Runtime** to run exported classifiers. The repository includes an `onnxruntime/` directory with the required headers and library.
- Python 3 and the required ML packages when running training scripts. Dependencies may vary between scripts; check the imports in the script you intend to use.

## Environment setup

Start EIC Shell, then source the ONNX Runtime environment setup from the repository root:

```bash
source /usr/local/eic/eic-shell
source Identification/onnx_setup.sh
```

`onnx_setup.sh` sets `LD_LIBRARY_PATH` and the header search paths for the ONNX Runtime bundled with this repository. If EIC Shell is installed elsewhere, use the path for your installation.

## Running

Most analyses are implemented as ROOT macros (`.cxx`). Run them from the relevant module directory after setting up EIC Shell and ONNX Runtime, for example:

```bash
cd Identification/CalorimetryHits
root -l -b -q 'TrainingPodioMacro.cxx+'
```

Input files, collection names, model paths, and output directories may be configured directly in the macros. Check and adapt these settings to your data before running. Testing macros evaluate a trained model; training scripts and macros prepare data and train classifiers.

## ONNX models

The models used by `MuonID` are in `CalorimetryHits/ONNX/` and `ToF/ONNX/`. When retraining or replacing a model, check that the feature count and ordering match the implementation in `MuonID/MuonID.cxx` and the model paths configured in that file.