# Nondegenerate Optical Cavity HG Mode Simulator

This tool simulates the transmission of a **nondegenerate symmetric
Fabry--Perot cavity** including **higher-order Hermite--Gauss (HG)
transverse modes**, with:

- A **photodiode (PD) signal** vs cavity length scan\
- A **camera image** of the transmitted spatial mode\
- An **interactive GUI** with sliders for alignment, scan range, and
    mode order\
- A **separate config file** to keep all tunable parameters in one
    place

It is designed to closely mimic what you see in the lab when
mode-matching and scanning a real cavity with a piezo.

------------------------------------------------------------------------

## File Overview

    .
    ├── cavity_gui.py   # Main simulation + GUI
    ├── config.py       # All tunable parameters live here
    └── README.md       # This file

### `config.py`

Contains: - Cavity geometry - Laser wavelength - Input beam parameters -
Mirror reflectivity - Camera/scan resolution - Initial slider values -
Slider ranges

You **only edit this file** when changing experimental parameters.

### `cavity_gui.py`

- Implements the full:
  - Gaussian + HG mode basis
  - Nondegenerate Gouy-phase mode splitting
  - Cavity field transfer function
- Builds:
  - PD scan plot
  - Camera image
  - Interactive sliders

------------------------------------------------------------------------

## Physics Model Summary

- **Cavity type:** symmetric two-mirror Fabry--Perot
- **Mode basis:** Hermite--Gauss HGₙₘ up to order `Nmax`
- **Nondegeneracy:** enforced using the Gouy phase

    $$\zeta = arccos(g),\; g = 1 − \frac{L}{R}$$

    $$\Delta L_{nm} = \frac{−(n + m) \zeta}{k}$$

- **Photodiode signal:**
    $$P(dL) = \sum |c_{nm}|^2 \; |H_{nm}(dL)|^2$$

- **Camera field:**

    We assume a fast dither so the camera avergages over the entire dither range.

    $$E(x,y) = \sum_{n,m} c_{nm} H_{nm} HG_{nm}(x,y)$$

Where $H_{nm}$ is the cavity transfer function as a function over cavity length for the $HG_{nm}$ mode, and $c_{nm}$ is the overlap coefficient of the input gaussian with the HG_{nm} mode. Misalignment (`x_off`, `y_off`) injects power into odd HG modes naturally.

------------------------------------------------------------------------

## Requirements

You need:

- python 3.9 or newer
- numpy
- matplotlib

Install dependencies with:

    pip install numpy matplotlib

------------------------------------------------------------------------

## How to Run

From the directory containing the files:

    python cavity_gui.py

A single window will open with:

- **Left panel:** Camera image\
- **Right panel:** PD transmission vs cavity scan\
- **Bottom sliders:** Alignment + scan + max HG order

------------------------------------------------------------------------

## Sliders (Live Controls)

| Slider     | Meaning                       | Units       |
|------------|-------------------------------|-------------|
| x_off      | Beam offset at input mirror   | µm          |
| y_off      | Beam offset at input mirror   | µm          |
| scan_range | Piezo scan range              | wavelengths |
| Nmax       | Max HG order included         | integer     |

------------------------------------------------------------------------

## Editing Experimental Parameters

Open `config.py` and modify:

    CONFIG = {
        "R": 0.5,
        "L": 0.05,
        "w0_in": 0.3e-3,
        "Rb_in": float("inf"),
        "lam": 780e-9,
        "Rmirror": 0.995,
        ...
    }

------------------------------------------------------------------------

## Performance Notes

- Runtime scales approximately as:

    $$\mathcal{O}(N_{pix}^2 \cdot N_{max}^2 \cdot N_{scan})$$

    where

  - $N_{pix} \times N_{pix}$ is the size of the camera image
  - $N_{max}$ is the number of simulated HG modes
  - $N_{scan}$ is the number of discrete cavity lengths that are used for the simulation

- Large values slow slider response.

------------------------------------------------------------------------

## Typical Use Cases

- Debugging mode-matching sensitivity
- Understanding odd/even mode injection vs misalignment
- Predicting PD peak structure before alignment
- Teaching nondegenerate cavity transverse mode physics
- Testing piezo scan range vs expected HOM spacing

------------------------------------------------------------------------

## Known Simplifications

- No astigmatism
- No mirror defects
- No angular misalignment
- Same finesse for all modes
- No finite-aperture effects

------------------------------------------------------------------------

## Possible Extensions

- Astigmatic cavities
- Mode-dependent finesse
- Longitudinal mode index
- Animated scan GIFs
- Image-plane propagation to the camera
