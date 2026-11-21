"""Fit response matrices from finished runs and map them to U and J.

chi[i, j]  = d n_i / d v_j   (i, j in up, dn), self-consistent runs
chi0[i, j] = same, non-self-consistent runs (ICHARG = 11)
f = inv(chi) - inv(chi0)     (VASP sign convention)
"""
import os
import numpy as np

from .inputs import read_incar

# ---------------------------------------------------------------- OUTCAR
def table_occupations(outcar, atom=1, l=2):
    """(n_up, n_dn) from the LORBIT = 11 'total charge' / 'magnetization (x)' tables.

    The LDAUPRINT onsite density matrix is NOT usable: in ICHARG = 11 runs it is
    the frozen ground-state matrix, so the non-self-consistent response reads as zero.
    """
    with open(outcar) as fh:
        lines = fh.readlines()
    vals = {}
    for key in ("total charge", "magnetization (x)"):
        idx = [i for i, ln in enumerate(lines) if ln.strip() == key]
        if not idx:
            raise ValueError(f"'{key}' table not found in {outcar}")
        for ln in lines[idx[-1] + 4 :]:
            parts = ln.split()
            if parts and parts[0] == str(atom):
                vals[key] = float(parts[1 + l])
                break
    n, m = vals["total charge"], vals["magnetization (x)"]
    return 0.5 * (n + m), 0.5 * (n - m)


def converged(outcar):
    with open(outcar) as fh:
        return "aborting loop because EDIFF is reached" in fh.read()


# ---------------------------------------------------------------- collect
def collect(root="."):
    """Read every run directory under root. Returns a list of dicts."""
    runs = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        incar, outcar = os.path.join(d, "INCAR"), os.path.join(d, "OUTCAR")
        if not (os.path.isfile(incar) and os.path.isfile(outcar)):
            continue
        tags = read_incar(incar)
        if tags.get("LDAUTYPE") != "3":
            continue
        vup = float(tags["LDAUU"].split()[0])
        vdn = float(tags["LDAUJ"].split()[0])
        if vup and vdn:
            print(f"skipping {name}: both spin channels shifted")
            continue
        l = int(tags["LDAUL"].split()[0])
        nup, ndn = table_occupations(outcar, l=l)
        kind = "gs" if not (vup or vdn) else ("nscf" if tags.get("ICHARG") == "11" else "scf")
        if not converged(outcar):
            print(f"warning: {name} did not reach EDIFF")
        runs.append(dict(name=name, kind=kind, spin="up" if vup else "dn",
                         v=vup or vdn, nup=nup, ndn=ndn))
    return runs


# ---------------------------------------------------------------- fitting
def slope(x, y):
    """Least-squares slope and its standard error."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    dx = x - x.mean()
    sxx = np.sum(dx**2)
    b = np.sum(dx * (y - y.mean())) / sxx
    if len(x) < 3:
        return b, 0.0
    resid = y - y.mean() - b * dx
    return b, np.sqrt(np.sum(resid**2) / (len(x) - 2) / sxx)


def response_matrices(runs):
    gs = [r for r in runs if r["kind"] == "gs"]
    if len(gs) > 1:
        raise ValueError(f"expected one ground-state run, found {len(gs)}")
    if not gs:
        print("note: no ground-state run found (gs/ needs INCAR and OUTCAR); "
              "fitting without the v = 0 point")
    out = {}
    for kind in ("scf", "nscf"):
        chi, err = np.zeros((2, 2)), np.zeros((2, 2))
        for j, spin in enumerate(("up", "dn")):
            pts = [r for r in runs if r["kind"] == kind and r["spin"] == spin]
            pts = pts + gs
            if len(pts) < 2:
                raise ValueError(f"not enough {kind} runs for the {spin} channel")
            v = [r["v"] for r in pts]
            for i, key in enumerate(("nup", "ndn")):
                chi[i, j], err[i, j] = slope(v, [r[key] for r in pts])
        out[kind] = (chi, err)
    return out["scf"], out["nscf"]


# ---------------------------------------------------------------- U, J
def uj_from_chi(chi, chi0):
    """Return dict with f and the two U/J mappings of Linscott et al. (2018).

    scaled ("conventional" in Moore et al. 2024): Eqs. (20)-(25)
    simple ("constrained"):                        Eqs. (19), (23)
    """
    f = np.linalg.inv(chi) - np.linalg.inv(chi0)
    lam_u = (chi[0, 0] + chi[0, 1]) / (chi[1, 0] + chi[1, 1])
    lam_j = (chi[0, 0] - chi[0, 1]) / (chi[1, 0] - chi[1, 1])
    u = 0.5 * (lam_u * (f[0, 0] + f[1, 0]) + f[0, 1] + f[1, 1]) / (lam_u + 1)
    j = -0.5 * (lam_j * (f[0, 0] - f[1, 0]) + f[0, 1] - f[1, 1]) / (lam_j - 1)
    return dict(f=f, U=u, J=j,
                U_simple=0.25 * f.sum(),
                J_simple=-0.25 * (f[0, 0] - f[0, 1] - f[1, 0] + f[1, 1]))


def uj_errors(chi, chi_err, chi0, chi0_err, nsamples=4000, seed=0):
    """Monte Carlo propagation of the slope standard errors."""
    rng = np.random.default_rng(seed)
    keys = ("U", "J", "U_simple", "J_simple")
    samples = {k: [] for k in keys}
    for _ in range(nsamples):
        try:
            res = uj_from_chi(rng.normal(chi, chi_err), rng.normal(chi0, chi0_err))
        except np.linalg.LinAlgError:
            continue
        for k in keys:
            samples[k].append(res[k])
    return {k: float(np.std(samples[k])) for k in keys}


def analyze(root=".", verbose=True):
    runs = collect(root)
    if verbose:
        report_runs(runs)
    (chi, chi_err), (chi0, chi0_err) = response_matrices(runs)
    for name, mat in (("chi (SCF)", chi), ("chi0 (NSCF)", chi0)):
        if abs(np.linalg.det(mat)) < 1e-12:
            raise ValueError(
                f"{name} is singular: the occupations in the table above do not "
                f"respond to the shift.\n{mat}")
    res = uj_from_chi(chi, chi0)
    res.update(chi=chi, chi0=chi0, runs=runs,
               err=uj_errors(chi, chi_err, chi0, chi0_err))
    if verbose:
        report(res)
    return res


def report_runs(runs):
    gs = next((r for r in runs if r["kind"] == "gs"), None)
    print(f"{'run':<22}{'v (eV)':>8}{'n_up':>10}{'n_dn':>10}{'m':>10}")
    for r in runs:
        m = r["nup"] - r["ndn"]
        flag = "  <-- moment changed" if gs and abs(m - (gs["nup"] - gs["ndn"])) > 0.3 else ""
        print(f"{r['name']:<22}{r['v']:>8.3f}{r['nup']:>10.4f}{r['ndn']:>10.4f}{m:>10.4f}{flag}")


def report(res):
    np.set_printoptions(precision=5, suppress=True)
    print("\nchi  (SCF),  rows n_up/n_dn, cols v_up/v_dn [1/eV]\n", res["chi"])
    print("chi0 (NSCF)\n", res["chi0"])
    print("f = inv(chi) - inv(chi0) [eV]\n", res["f"])
    e = res["err"]
    print(f"\nconventional (scaled 2x2):  U = {res['U']:.3f} +/- {e['U']:.3f} eV   "
          f"J = {res['J']:.3f} +/- {e['J']:.3f} eV")
    print(f"constrained  (simple 2x2):  U = {res['U_simple']:.3f} +/- {e['U_simple']:.3f} eV   "
          f"J = {res['J_simple']:.3f} +/- {e['J_simple']:.3f} eV")
