import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from numpy.polynomial.hermite import hermval
import math
import logging
#setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import CONFIG  # import all tunable parameters

# Global cache for HG modes & grid
_HG_CACHE = {}

USE_NUMBA = False
###############################################################################
# Physics helpers
###############################################################################

def symmetric_cavity_mode(R, L, lam):
    """
    Symmetric two-mirror cavity: both mirrors have ROC = R, separated by L.
    Returns:
        g    : g-parameter
        z_R  : Rayleigh range
        w0   : waist at cavity center
        z_m  : distance from waist to each mirror (L/2)
        w_m  : spot size at mirrors
        R_m  : radius of curvature at mirrors
    """
    g = 1.0 - L / R

    if not (0 < g < 1):
        logger.warning("Warning: cavity may be unstable (g={:.3f})".format(g))

    z_R = (L / 2.0) * np.sqrt((1 + g) / (1 - g))
    w0  = np.sqrt(lam * z_R / np.pi)

    z_m = L / 2.0
    w_m = w0 * np.sqrt(1 + (z_m / z_R) ** 2)
    R_m = z_m * (1 + (z_R**2 / z_m**2))

    return g, z_R, w0, z_m, w_m, R_m

def gaussian_field_1d(x, w, Rb, lam, x0=0.0):
    """
    1D complex Gaussian field at a given plane.
    Normalized so ∫|E|^2 dx ~ 1 if you integrate over a big enough range.
    """
    k = 2 * np.pi / lam
    X = x - x0

    amp = (1.0 / (np.pi**0.25 * np.sqrt(w)))  # 1D HG00-like normalization
    envelope = np.exp(-(X**2) / (w**2))

    if np.isinf(Rb):
        phase = np.ones_like(envelope, dtype=complex)
    else:
        phase = np.exp(-1j * k * X**2 / (2 * Rb))

    return amp * envelope * phase

def get_grid_and_1d_modes(R, L, lam, w0_in, Npix, FOV_factor, Nmax):
    """
    Return (from cache if possible):

        x, y        : 1D grids [m]
        u_nx        : array [Nn, Nx]  1D HG_n(x)
        u_my        : array [Nm, Ny]  1D HG_m(y)
        dx, dy      : grid spacings
        g, z_R, w0_cav, z_m, w_m, R_m : cavity mode params
    """
    key = (R, L, lam, w0_in, Npix, FOV_factor, Nmax)
    if key in _HG_CACHE:
        return _HG_CACHE[key]

    # Cavity mode at mirrors
    g, z_R, w0_cav, z_m, w_m, R_m = symmetric_cavity_mode(R, L, lam)
    k = 2 * np.pi / lam

    # Grid extent: based on larger of cavity spot and input beam
    w_max = max(w_m, w0_in)
    half_size = FOV_factor * w_max

    x = np.linspace(-half_size, half_size, Npix)
    y = np.linspace(-half_size, half_size, Npix)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Dimensionless coordinates for HG modes
    xi  = np.sqrt(2) * x / w_m
    eta = np.sqrt(2) * y / w_m

    # Common envelopes and phases (1D, separable)
    env_x = np.exp(-(x**2) / (w_m**2))
    env_y = np.exp(-(y**2) / (w_m**2))

    if np.isinf(R_m):
        phase_x = np.ones_like(x, dtype=complex)
        phase_y = np.ones_like(y, dtype=complex)
    else:
        phase_x = np.exp(-1j * k * x**2 / (2 * R_m))
        phase_y = np.exp(-1j * k * y**2 / (2 * R_m))

    Nn = Nmax + 1
    Nm = Nmax + 1
    u_nx = np.zeros((Nn, Npix), dtype=complex)
    u_my = np.zeros((Nm, Npix), dtype=complex)

    # Build orthonormal 1D HG basis along x and y
    for n in range(Nn):
        coeffs_n = [0] * n + [1]
        Hn_x = hermval(xi, coeffs_n)
        pref_n = 1.0 / ( (2.0**n * math.factorial(n))**0.5 * (np.pi**0.25 * np.sqrt(w_m)) )
        u_nx[n, :] = pref_n * Hn_x * env_x * phase_x

    for m in range(Nm):
        coeffs_m = [0] * m + [1]
        Hm_y = hermval(eta, coeffs_m)
        pref_m = 1.0 / ( (2.0**m * math.factorial(m))**0.5 * (np.pi**0.25 * np.sqrt(w_m)) )
        u_my[m, :] = pref_m * Hm_y * env_y * phase_y

    _HG_CACHE[key] = (x, y, u_nx, u_my, dx, dy, g, z_R, w0_cav, z_m, w_m, R_m)
    return _HG_CACHE[key]


def gaussian_field(x, y, w, Rb, lam, x0=0.0, y0=0.0):
    """
    Complex Gaussian HG00 field at a given plane.
    Normalized so that ∫|E|^2 dx dy = 1 (for infinite aperture).
    """
    k = 2 * np.pi / lam
    X = x - x0
    Y = y - y0
    r2 = X**2 + Y**2

    amp = np.sqrt(2 / np.pi) / w
    envelope = np.exp(-r2 / w**2)

    if np.isinf(Rb):
        phase = np.ones_like(envelope, dtype=complex)
    else:
        phase = np.exp(-1j * k * r2 / (2 * Rb))

    return amp * envelope * phase

def hg_mode(x, y, n, m, w, Rb, lam):
    """
    Hermite–Gauss mode HG_nm at mirror plane.
    Normalized: ∫|HG_nm|^2 dx dy = 1.
    """
    k = 2 * np.pi / lam
    X = x
    Y = y
    r2 = X**2 + Y**2

    # Dimensionless coordinates
    xi  = np.sqrt(2) * X / w
    eta = np.sqrt(2) * Y / w

    coeffs_n = [0] * n + [1]
    coeffs_m = [0] * m + [1]
    Hn = hermval(xi,  coeffs_n)
    Hm = hermval(eta, coeffs_m)

    pref = (np.sqrt(2/np.pi) / w) / np.sqrt((2.0**(n+m)) * math.factorial(n) * math.factorial(m))

    envelope = np.exp(-r2 / w**2)

    if np.isinf(Rb):
        phase = np.ones_like(envelope, dtype=complex)
    else:
        phase = np.exp(-1j * k * r2 / (2 * Rb))

    return pref * Hn * Hm * envelope * phase

def mode_overlap(E_mode, E_in, dx, dy):
    """
    Complex overlap <mode|in> = ∫ E_mode*(x,y) E_in(x,y) dx dy
    """
    integrand = np.conjugate(E_mode) * E_in
    return np.sum(integrand) * dx * dy

def cavity_transfer_function(phi, r1, r2, t1, t2):
    """
    Complex field transfer function of a Fabry–Perot cavity:
    E_trans / E_in as a function of round-trip phase detuning phi.
    """
    denom = 1 - r1 * r2 * np.exp(1j * phi)
    return t1 * t2 * np.exp(1j * phi / 2) / denom   # field

###############################################################################
# Core simulation: nondegenerate HG modes
###############################################################################
def simulate_cavity_camera_pd(
    R,
    L,
    w0_in,
    Rb_in,
    x_off,
    y_off,
    lam,
    Rmirror,
    Npix,
    FOV_factor,
    Nscan,
    scan_range_lambda,
    Nmax,
):
    k = 2 * np.pi / lam

    (x, y,
     u_nx, u_my,
     dx, dy,
     g, z_R, w0_cav, z_m, w_m, R_m
    ) = get_grid_and_1d_modes(R, L, lam, w0_in, Npix, FOV_factor, Nmax)

    Nn = Nmax + 1
    Nm = Nmax + 1
    Nx = x.size
    Ny = y.size

    f_x = gaussian_field_1d(x, w0_in, Rb_in, lam, x0=x_off)
    f_y = gaussian_field_1d(y, w0_in, Rb_in, lam, x0=y_off)

    alpha = np.zeros(Nn, dtype=complex)
    beta  = np.zeros(Nm, dtype=complex)

    for n in range(Nn):
        alpha[n] = np.sum(np.conjugate(u_nx[n, :]) * f_x) * dx
    for m in range(Nm):
        beta[m]  = np.sum(np.conjugate(u_my[m, :]) * f_y) * dy

    c_nm_2d = alpha[:, None] * beta[None, :]
    Nmodes = Nn * Nm
    c_nm = c_nm_2d.reshape(Nmodes)

    # Build 2D HG modes
    modes_nm = np.zeros((Nmodes, Ny, Nx), dtype=np.complex128)
    mode_index = 0
    for n in range(Nn):
        for m in range(Nm):
            modes_nm[mode_index, :, :] = np.outer(u_my[m, :], u_nx[n, :])
            mode_index += 1

    zeta = np.arccos(g)
    n_indices = np.repeat(np.arange(Nn), Nm)
    m_indices = np.tile(np.arange(Nm), Nn)
    order_sum = n_indices + m_indices
    deltaL_modes = -(order_sum.astype(float) * zeta / k)

    dL = np.linspace(-scan_range_lambda * lam,
                     +scan_range_lambda * lam, Nscan)

    r1 = np.sqrt(Rmirror)
    r2 = np.sqrt(Rmirror)
    t1 = np.sqrt(1 - Rmirror)
    t2 = np.sqrt(1 - Rmirror)

    dL_matrix = dL[None, :] - deltaL_modes[:, None]
    phi_nm = 2 * k * dL_matrix

    denom = 1 - r1 * r2 * np.exp(1j * phi_nm)
    H_nm_scan = t1 * t2 * np.exp(1j * phi_nm / 2) / denom  # [Nmodes, Nscan]

    # PD signal (already pretty vectorized)
    abs_c2 = (np.abs(c_nm) ** 2)[:, None]   # [Nmodes, 1]
    abs_H2 = np.abs(H_nm_scan) ** 2         # [Nmodes, Nscan]
    pd_signal = np.sum(abs_c2 * abs_H2, axis=0)

    # ----- Camera signal: fully vectorized using BLAS -----
    # Flatten spatial dimensions: modes_nm [Nmodes, Ny, Nx] -> [Nmodes, Np]
    Nmodes = c_nm.size
    Np = Ny * Nx
    modes_flat = modes_nm.reshape(Nmodes, Np)        # [Nmodes, Np]

    # Weights for each mode and scan: W_k,i = c_k * H_k(i)
    weights = c_nm[:, None] * H_nm_scan             # [Nmodes, Nscan]

    # E_flat[p, i] = sum_k modes_flat[k, p] * weights[k, i]
    # This is a single big matmul in C/Fortran:
    E_flat = modes_flat.T @ weights                  # [Np, Nscan]

    # Intensity and average over scans
    I_flat = np.abs(E_flat)**2                       # [Np, Nscan]
    camera_img = I_flat.mean(axis=1).reshape(Ny, Nx) # [Ny, Nx]


    return dL, pd_signal, camera_img, x, y


###############################################################################
# Interactive plotting
###############################################################################

if __name__ == "__main__":
    # Unpack config
    R       = CONFIG["R"]
    L       = CONFIG["L"]
    w0_in   = CONFIG["w0_in"]
    Rb_in   = CONFIG["Rb_in"]
    lam     = CONFIG["lam"]
    Rmirror = CONFIG["Rmirror"]
    Npix    = CONFIG["Npix"]
    FOV_factor = CONFIG["FOV_factor"]
    Nscan   = CONFIG["Nscan"]

    init_x_off_um       = CONFIG["init_x_off_um"]
    init_y_off_um       = CONFIG["init_y_off_um"]
    init_scan_range_lam = CONFIG["init_scan_range_lam"]
    init_Nmax           = CONFIG["init_Nmax"]

    slider_ranges = CONFIG["slider_ranges"]

    # Initial simulation
    dL, pd_signal, camera_img, x, y = simulate_cavity_camera_pd(
        R=R,
        L=L,
        w0_in=w0_in,
        Rb_in=Rb_in,
        x_off=init_x_off_um * 1e-6,
        y_off=init_y_off_um * 1e-6,
        lam=lam,
        Rmirror=Rmirror,
        Npix=Npix,
        FOV_factor=FOV_factor,
        Nscan=Nscan,
        scan_range_lambda=init_scan_range_lam,
        Nmax=init_Nmax
    )


    # Figure with camera + PD
    fig, (ax_cam, ax_pd) = plt.subplots(1, 2, figsize=(10, 4))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.25, wspace=0.35)

    extent = [x[0]*1e3, x[-1]*1e3, y[0]*1e3, y[-1]*1e3]  # mm
    im = ax_cam.imshow(
        camera_img,
        origin="lower",
        extent=extent,
        aspect="equal"
    )
    ax_cam.set_xlabel("x [mm]")
    ax_cam.set_ylabel("y [mm]")
    ax_cam.set_title("Camera intensity")
    cbar = fig.colorbar(im, ax=ax_cam)
    cbar.set_label("Intensity (arb. units)")

    line_pd, = ax_pd.plot(dL * 1e9, pd_signal)
    ax_pd.set_xlabel("Cavity length change dL [nm]")
    ax_pd.set_ylabel("PD signal (arb. units)")
    ax_pd.set_title("Cavity transmission vs scan")
    ax_pd.grid(True)

    # Slider axes
    axcolor = "lightgoldenrodyellow"
    ax_xoff = fig.add_axes([0.15, 0.17, 0.7, 0.03], facecolor=axcolor)
    ax_yoff = fig.add_axes([0.15, 0.12, 0.7, 0.03], facecolor=axcolor)
    ax_scan = fig.add_axes([0.15, 0.07, 0.7, 0.03], facecolor=axcolor)
    ax_Nmax = fig.add_axes([0.15, 0.02, 0.7, 0.03], facecolor=axcolor)

    # Slider objects (ranges from config)
    s_xoff = Slider(
        ax_xoff,
        'x_off [µm]',
        slider_ranges["x_off_um"]["min"],
        slider_ranges["x_off_um"]["max"],
        valinit=init_x_off_um,
    )
    s_yoff = Slider(
        ax_yoff,
        'y_off [µm]',
        slider_ranges["y_off_um"]["min"],
        slider_ranges["y_off_um"]["max"],
        valinit=init_y_off_um,
    )
    s_scan = Slider(
        ax_scan,
        'scan_range [λ]',
        slider_ranges["scan_range"]["min"],
        slider_ranges["scan_range"]["max"],
        valinit=init_scan_range_lam,
    )
    s_Nmax = Slider(
        ax_Nmax,
        'Nmax',
        slider_ranges["Nmax"]["min"],
        slider_ranges["Nmax"]["max"],
        valinit=init_Nmax,
        valstep=1.0,
    )

    def update(val):
        # Get slider values
        x_off_um = s_xoff.val
        y_off_um = s_yoff.val
        scan_range_lambda = s_scan.val
        Nmax_int = int(s_Nmax.val)

        # Re-run simulation with new parameters
        dL_new, pd_new, cam_new, x_new, y_new = simulate_cavity_camera_pd(
            R=R,
            L=L,
            w0_in=w0_in,
            Rb_in=Rb_in,
            x_off=x_off_um * 1e-6,
            y_off=y_off_um * 1e-6,
            lam=lam,
            Rmirror=Rmirror,
            Npix=Npix,
            FOV_factor=FOV_factor,
            Nscan=Nscan,
            scan_range_lambda=scan_range_lambda,
            Nmax=Nmax_int,
        )

        # Update PD plot
        line_pd.set_xdata(dL_new * 1e9)
        line_pd.set_ydata(pd_new)
        ax_pd.relim()
        ax_pd.autoscale_view()

        # Update camera
        im.set_data(cam_new)
        im.set_clim(vmin=cam_new.min(), vmax=cam_new.max())

        fig.canvas.draw_idle()


    s_xoff.on_changed(update)
    s_yoff.on_changed(update)
    s_scan.on_changed(update)
    s_Nmax.on_changed(update)

    plt.show()