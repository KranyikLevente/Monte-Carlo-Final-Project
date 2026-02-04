# Monte Carlo Simulation of NaI Scintillation Detector

![Python](https://img.shields.io/badge/Python-3.x-blue)
![NumPy](https://img.shields.io/badge/NumPy-Vectorization-orange)
![SciPy](https://img.shields.io/badge/SciPy-Interpolation-green)
![Physics](https://img.shields.io/badge/Physics-Gamma__Spectroscopy-purple)

## Project Overview

This project simulates the response of a Sodium Iodide (NaI) scintillation detector using Monte Carlo methods. The objective is to model the transport and detection of photons from monoenergetic gamma point sources, accounting for source geometry, stochastic interaction mechanisms, and detector resolution (Gaussian broadening).

The simulation utilizes cross-section data from the NIST XCOM database to generate realistic gamma spectra and analyze detection efficiency as a function of source position and energy.

## Physical Model and Methods

The simulation models a cylindrical NaI crystal in a vacuum environment. The photon transport logic incorporates the following physical processes:

### 1. Interactions
* **Photoelectric Effect:** Complete energy absorption.
* **Compton Scattering:** Inelastic scattering with energy and direction updates sampled via Kahn's rejection method.
* **Pair Production:** Threshold interactions (> 1.022 MeV) generating positron-electron pairs and subsequent annihilation photons (2x 511 keV).

### 2. Geometry and Tracking
* **Path Length:** Sampled from the exponential attenuation distribution based on the total attenuation coefficient.
* **Ray Tracing:** Analytical calculation of intersection points between photon trajectories and the cylindrical detector volume.

### 3. Detector Response
* **Energy Deposition:** Tracks the total energy deposited by the primary photon and all secondaries (e.g., annihilation photons) within the active volume.
* **Resolution:** Applies Gaussian noise to the deposited energy to simulate the finite energy resolution (FWHM) of a real detector.

## Simulation Tasks

The project performs three primary analyses:

### Task (i): Gamma Spectrometry
Generation of simulated spectra for specific radioisotopes:
* **Cs-137 (661.7 keV):** Reproduction of the photopeak and Compton edge (approx. 477.4 keV).
* **Co-60 (1332.5 keV):** Observation of Single Escape (SE) and Double Escape (DE) peaks due to pair production.

### Task (ii): Efficiency vs. Source Position
Analysis of Total Efficiency and Intrinsic Efficiency as the point source moves linearly relative to the detector. Results demonstrate the inverse relationship between geometric solid angle and path length effects.

### Task (iii): Efficiency vs. Photon Energy
Investigation of detection efficiency across the 0.4 MeV – 4.0 MeV range. The results highlight the decrease in total efficiency at higher energies due to increased photon penetrability.

## Installation and Usage

### Requirements
The simulation requires Python 3 and the following scientific libraries:

```bash
pip install numpy matplotlib scipy pandas
