# Identification

This directory contains code for particle identification in EPIC analyses: feature extraction, classifier training and testing, and validation using reconstructed data.

## Main directories

- [`MuonID/`](MuonID/) — reusable implementation of the `MuonID(...)` function, prepared for integration into another analysis, together with [`example.cxx`](MuonID/example.cxx) showing how to call it. The function takes a reconstructed particle and event frame and returns a muon probability score; its optional `hasDetectorInfo` output indicates whether the track had enough detector information to be scored. The caller handles event loops, truth selection, thresholds, and histograms.
- [`FiveParticleClassificator/`](FiveParticleClassificator/) — detailed testing and performance studies of the `MuonID` function for five particle species.
- [`PionMuonClassificator/`](PionMuonClassificator/) — an earlier version of the classifier, limited to separating pions and muons.
- [`CalorimetryHits/`](CalorimetryHits/) — exploratory work on calorimeter hits and clusters, including checks and older algorithms; it is not the current, reusable `MuonID` implementation.
- [`ToF/`](ToF/) — exploratory checks and older algorithms involving Time-of-Flight information, similar in purpose to `CalorimetryHits/`.
- [`Tracker/`](Tracker/) — auxiliary checks and track-feature analysis; it is not the main classifier.
- [`DRICH/`](DRICH/) — an initial investigation into using dRICH information for classification. There are no results yet.
- [`CalorimetryClusters/`](CalorimetryClusters/) — an older version of the calorimeter analysis based on clusters. These are legacy materials, not the main current algorithm workflow. This code was used in the algorithm presented in Glasgow.

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

## Using `MuonID`

Include the implementation and call it for a reconstructed particle and its event frame:

```cpp
#include "MuonID.cxx"

bool hasDetectorInfo = false;
const float muonProbability = MuonID(reconstructedParticle, &event, &hasDetectorInfo);
```

`muonProbability` is the model's muon score. Check `hasDetectorInfo` before using it: when false, the track could not be scored and the returned value should not be treated as a classifier result. See [`MuonID/example.cxx`](MuonID/example.cxx) for a complete analysis example. When integrating the function elsewhere, make the required ONNX model files available at the paths expected by the implementation.

## ONNX models

The ONNX models used by `MuonID` are in [`MuonID/ONNX/`](MuonID/ONNX/). When replacing a model, ensure that its feature count and ordering match the implementation and that its path is configured correctly in `MuonID.cxx`.