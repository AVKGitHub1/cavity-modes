import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from numpy.polynomial.hermite import hermval
import math
import logging

#set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



###############################################################################
# Physics helpers
###############################################################################

def symmetric_cavity_mode(R, L, lam):
    """
    Symmetric two-mirror cavity: both mirrors have ROC = R, separated by L.
    Returns:
        g    : g-parameter (symmetric, so g1 = g2 = g)
        z_R  : Rayleigh range
        w0   : waist at cavity center
        z_m  : distance from waist to each mirror (L/2)
        w_m  : spot size at mirrors
        R_m  : radius of curvature at mirrors
    """
    g = 1.0 - L / R

    if not (0 < g < 1):
        logger.warning("Cavity may be unstable (g={:.3f})".format(g))

    z_R = (L / 2.0) * np.sqrt((1 + g) / (1 - g))
    w0  = np.sqrt(lam * z_R / np.pi)

    z_m = L / 2.0
    w_m = w0 * np.sqrt(1 + (z_m / z_R) ** 2)
    R_m = z_m * (1 + (z_R**2 / z_m**2))

    return g, z_R, w0, z_m, w_m, R_m

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

    # Physicists' Hermite polynomials H_n, H_m
    coeffs_n = [0] * n + [1]
    coeffs_m = [0] * m + [1]
    Hn = hermval(xi,  coeffs_n)
    Hm = hermval(eta, coeffs_m)

    # Normalization prefactor
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
    camera_index="max",
):
    """
    Nondegenerate symmetric cavity with HG_nm modes up to Nmax.

    Returns:
        dL         : cavity length change array [m]
        pd_signal  : PD signal vs dL (arb. units)
        camera_img : 2D intensity at chosen dL
        x, y       : coordinate vectors [m]
    """
    k = 2 * np.pi / lam

    # Cavity mode at mirrors
    g, z_R, w0_cav, z_m, w_m, R_m = symmetric_cavity_mode(R, L, lam)

    # Gouy phase per half round trip
    zeta = np.arccos(g)   # radians

    # Grid for mode expansion and camera
    w_max = max(w_m, w0_in)
    half_size = FOV_factor * w_max
    x = np.linspace(-half_size, half_size, Npix)
    y = np.linspace(-half_size, half_size, Npix)
    X, Y = np.meshgrid(x, y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Input field at mirror
    E_in = gaussian_field(X, Y, w0_in, Rb_in, lam, x0=x_off, y0=y_off)

    # HG expansion
    coeffs = {}
    modes  = {}
    dL_nm  = {}
    eta_total = 0.0

    for n in range(Nmax+1):
        for m in range(Nmax+1):
            E_nm = hg_mode(X, Y, n, m, w_m, R_m, lam)
            c_nm = mode_overlap(E_nm, E_in, dx, dy)
            coeffs[(n, m)] = c_nm
            modes[(n, m)]  = E_nm
            # nondegenerate resonance length shift
            dL_nm[(n, m)]  = -(n + m) * zeta / k
            eta_total += np.abs(c_nm)**2

    # Mirrors
    r1 = np.sqrt(Rmirror)
    r2 = np.sqrt(Rmirror)
    t1 = np.sqrt(1 - Rmirror)
    t2 = np.sqrt(1 - Rmirror)

    # Cavity length scan
    dL = np.linspace(-scan_range_lambda * lam,
                     +scan_range_lambda * lam, Nscan)

    pd_signal = np.zeros_like(dL, dtype=float)
    H_nm_dict = {}

    for (n, m), c_nm in coeffs.items():
        deltaL = dL - dL_nm[(n, m)]
        phi_nm = 2 * k * deltaL
        H_nm = cavity_transfer_function(phi_nm, r1, r2, t1, t2)
        H_nm_dict[(n, m)] = H_nm
        pd_signal += np.abs(c_nm)**2 * np.abs(H_nm)**2

    # Choose index for camera
    if camera_index == "max":
        idx_cam = int(np.argmax(pd_signal))
    else:
        idx_cam = int(camera_index)
        idx_cam = max(0, min(idx_cam, len(dL)-1))

    # Camera field at that dL
    E_cam = np.zeros_like(X, dtype=complex)
    for (n, m), c_nm in coeffs.items():
        h_nm = H_nm_dict[(n, m)][idx_cam]
        E_cam += c_nm * h_nm * modes[(n, m)]

    camera_img = np.abs(E_cam)**2

    return dL, pd_signal, camera_img, x, y

###############################################################################
# Interactive plotting
###############################################################################

if __name__ == "__main__":
    # Fixed physical parameters (tweak if you want)
    R       = 0.5        # ROC [m]
    L       = 0.05       # cavity length [m]
    w0_in   = 0.3e-3     # input beam radius at mirror [m]
    Rb_in   = np.inf     # collimated input beam at mirror
    lam     = 780e-9     # wavelength [m]
    Rmirror = 0.995      # mirror intensity reflectivity
    Npix    = 128        # grid size for camera
    FOV_factor     = 4.0
    Nscan          = 600

    # Initial slider values
    init_x_off_um        = 0   # µm
    init_y_off_um        = 0   # µm
    init_scan_range_lam  = 4.0      # in units of λ
    init_Nmax            = 10

    # Run initial simulation
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
        Nmax=init_Nmax,
        camera_index="max",
    )

    # Set up figure with 2 subplots: left = camera, right = PD
    fig, (ax_cam, ax_pd) = plt.subplots(1, 2, figsize=(10, 4))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.25, wspace=0.35)

    # Camera plot
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

    # PD plot
    line_pd, = ax_pd.plot(dL * 1e9, pd_signal)
    ax_pd.set_xlabel("Cavity length change dL [nm]")
    ax_pd.set_ylabel("PD signal (arb. units)")
    ax_pd.set_title("Cavity transmission vs scan")
    ax_pd.grid(True)

    # Slider axes
    axcolor = "lightgoldenrodyellow"
    ax_xoff   = fig.add_axes([0.15, 0.17, 0.7, 0.03], facecolor=axcolor)
    ax_yoff   = fig.add_axes([0.15, 0.12, 0.7, 0.03], facecolor=axcolor)
    ax_scan   = fig.add_axes([0.15, 0.07, 0.7, 0.03], facecolor=axcolor)
    ax_Nmax   = fig.add_axes([0.15, 0.02, 0.7, 0.03], facecolor=axcolor)

    # Sliders
    s_xoff = Slider(ax_xoff, 'x_off [µm]', -500.0, 500.0, valinit=init_x_off_um)
    s_yoff = Slider(ax_yoff, 'y_off [µm]', -500.0, 500.0, valinit=init_y_off_um)
    s_scan = Slider(ax_scan, 'scan_range [λ]', 1.0, 10.0, valinit=init_scan_range_lam)
    s_Nmax = Slider(ax_Nmax, 'Nmax', 5, 15, valinit=init_Nmax, valstep=1.0)

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
            camera_index="max",
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

    # Connect sliders to update
    s_xoff.on_changed(update)
    s_yoff.on_changed(update)
    s_scan.on_changed(update)
    s_Nmax.on_changed(update)

    plt.show()