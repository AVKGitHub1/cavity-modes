# config.py
# All user-tunable parameters for the cavity HG simulation + GUI.

CONFIG = {
    # ---- Physical cavity / beam parameters ----
    "R": 0.5,          # mirror ROC [m]
    "L": 0.05,         # cavity length [m]
    "w0_in": 0.3e-3,   # input beam radius at mirror [m]
    "Rb_in": float("inf"),  # input beam R_b at mirror [m] (inf = collimated)
    "lam": 780e-9,     # wavelength [m]
    "Rmirror": 0.995,  # mirror intensity reflectivity

    # ---- Numerical / grid settings ----
    "Npix": 128,       # camera grid size (Npix x Npix)
    "FOV_factor": 4.0, # half-size of grid = FOV_factor * max(w_m, w0_in)
    "Nscan": 600,      # number of points in cavity-length scan

    # ---- Initial slider values ----
    "init_x_off_um": 0,    # x offset [µm]
    "init_y_off_um": 0,   # y offset [µm]
    "init_scan_range_lam": 4.0,# scan range in units of λ
    "init_Nmax": 10,            # initial maximum HG order

    # ---- Slider ranges ----
    "slider_ranges": {
        "x_off_um":   {"min": -500.0, "max": 500.0},
        "y_off_um":   {"min": -500.0, "max": 500.0},
        "scan_range": {"min": 1.0,    "max": 10.0},
        "Nmax":       {"min": 5,      "max": 15},  # integer, still treated as slider
    }
}
