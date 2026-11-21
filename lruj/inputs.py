"""Write VASP inputs for linear-response U/J.

Two stages:
  setup()   -> gs/            ground state, perturbed atom isolated as species 1
  perturb() -> nscf_*/ scf_*/ one pair of runs per (spin channel, shift)

With LDAUTYPE = 3, LDAUU / LDAUJ are potential shifts (eV) on the spin-up /
spin-down l-manifold of a species, not U and J.
"""
import os
import re
import shutil

DEFAULT_SHIFTS = (-0.2, -0.1, 0.1, 0.2)


# ---------------------------------------------------------------- INCAR
def read_incar(path):
    tags = {}
    with open(path) as fh:
        text = re.sub(r"\\\s*\n", " ", fh.read())  # backslash line continuation
    for line in text.splitlines():
        line = line.split("#")[0].split("!")[0]
        for part in line.split(";"):
            if "=" in part:
                key, val = part.split("=", 1)
                tags[key.strip().upper()] = val.strip()
    return tags


def write_incar(path, tags):
    with open(path, "w") as fh:
        for key, val in tags.items():
            fh.write(f"{key} = {val}\n")


def expand_magmom(text):
    """'2*5.0 -5 3*0' -> [5.0, 5.0, -5.0, 0.0, 0.0, 0.0]"""
    out = []
    for tok in text.split():
        if "*" in tok:
            n, val = tok.split("*")
            out += [float(val)] * int(n)
        else:
            out.append(float(tok))
    return out


def lr_tags(nspecies, l, vup=0.0, vdn=0.0):
    """LDAU block: only species 1 (the isolated atom) is active."""
    rest = nspecies - 1
    return {
        "LDAU": ".TRUE.",
        "LDAUTYPE": "3",
        "LDAUPRINT": "2",
        "LDAUL": f"{l}" + " -1" * rest,
        "LDAUU": f"{vup:.6f}" + " 0" * rest,
        "LDAUJ": f"{vdn:.6f}" + " 0" * rest,
    }


# ---------------------------------------------------------------- POSCAR / POTCAR
def isolate_site(poscar_lines, site):
    """Move atom `site` (1-based, VASP numbering) to the front as its own species.

    Returns (new_lines, order, old_species_index, new_species) where `order`
    lists the old 0-based atom indices in their new order.
    """
    lines = poscar_lines
    species = lines[5].split()
    if species[0].lstrip("-").isdigit():
        raise ValueError("POSCAR needs a species-name line (VASP 5 format).")
    counts = [int(x) for x in lines[6].split()]
    natoms = sum(counts)
    if not 1 <= site <= natoms:
        raise ValueError(f"site must be in 1..{natoms}")

    i = 7
    selective = lines[i].strip()[:1].lower() == "s"
    if selective:
        i += 1
    coords = lines[i + 1 : i + 1 + natoms]

    k = site - 1
    owner = [s for s, c in enumerate(counts) for _ in range(c)]
    s = owner[k]
    order = [k] + [a for a in range(natoms) if a != k]

    counts[s] -= 1
    new_species = [species[s]] + [sp for sp, c in zip(species, counts) if c > 0]
    new_counts = [1] + [c for c in counts if c > 0]

    new = lines[:5]
    new.append(" ".join(new_species))
    new.append(" ".join(str(c) for c in new_counts))
    new += lines[7 : i + 1]
    new += [coords[a] for a in order]
    kept = [idx for idx, c in enumerate(counts) if c > 0]
    return new, order, s, kept


def split_potcar(path):
    blocks, cur = [], []
    with open(path) as fh:
        for line in fh:
            cur.append(line)
            if "End of Dataset" in line:
                blocks.append("".join(cur))
                cur = []
    return blocks


# ---------------------------------------------------------------- launchers
_LOOP = """#!/bin/bash
{header}# {what}
cd "$(dirname "$0")"; [ -d {first} ] || cd "$SLURM_SUBMIT_DIR"   # sbatch runs a spooled copy
for d in {dirs}; do
    grep -q "General timing" "$d/OUTCAR" 2>/dev/null && {{ echo "skip $d (finished)"; continue; }}
    {action}
done
"""


def write_launchers(root, dirs, job):
    """Copy the Slurm script into each run dir as job.sh; write submit.sh and run.sh.

    submit.sh: sbatch job.sh in every directory (job name = directory name).
    run.sh:    one job for everything. It has the job script's #SBATCH header and
               runs job.sh with bash in each unfinished directory, in sequence,
               until done or out of walltime. Resubmit to continue.
    """
    names = [os.path.relpath(d, root) for d in dirs]
    for d in dirs:
        dst = os.path.join(d, "job.sh")
        if os.path.abspath(job) != os.path.abspath(dst):
            shutil.copy(job, dst)
    with open(job) as fh:   # run.sh carries the #SBATCH header so it can be submitted itself
        header = "".join(ln.rstrip("\r\n") + "\n" for ln in fh if ln.startswith("#SBATCH"))
    scripts = {
        "submit.sh": ("submit every run as its own batch job", "",
                      '(cd "$d" && sbatch -J "$d" job.sh)'),
        "run.sh": ("run every unfinished run in sequence inside ONE job: `sbatch run.sh`, "
                   "or `./run.sh` on an interactive node", header,
                   'echo "$(date +%T) running $d"; (cd "$d" && bash job.sh > job.out 2>&1)'),
    }
    for name, (what, head, action) in scripts.items():
        path = os.path.join(root, name)
        with open(path, "w", newline="\n") as fh:
            fh.write(_LOOP.format(what=what, header=head, first=names[0],
                                  dirs=" ".join(names), action=action))
        os.chmod(path, 0o755)
    print(f"wrote submit.sh and run.sh for {len(dirs)} run(s)")


# ---------------------------------------------------------------- stage 1
def setup(site, l=2, src=".", root=".", copy=(), job=None):
    """Create <root>/gs from INCAR/POSCAR/KPOINTS/POTCAR in `src`."""
    gs = os.path.join(root, "gs")
    os.makedirs(gs, exist_ok=True)

    with open(os.path.join(src, "POSCAR")) as fh:
        lines = fh.read().splitlines()
    new, order, s, kept = isolate_site(lines, site)
    with open(os.path.join(gs, "POSCAR"), "w") as fh:
        fh.write("\n".join(new) + "\n")

    blocks = split_potcar(os.path.join(src, "POTCAR"))
    nsp_old = len(lines[5].split())
    if len(blocks) != nsp_old:
        raise ValueError(f"POTCAR has {len(blocks)} datasets, POSCAR has {nsp_old} species.")
    with open(os.path.join(gs, "POTCAR"), "w") as fh:
        fh.write(blocks[s] + "".join(blocks[i] for i in kept))

    tags = read_incar(os.path.join(src, "INCAR"))
    if tags.get("LDAU", "").upper().strip(".").startswith("T"):
        print("note: +U settings in the source INCAR are dropped; LDAUTYPE = 3 "
              "replaces them, so the response is taken on the plain-DFT state.")
    for key in [k for k in tags if k.startswith("LDAU")] + ["ICHARG", "ISTART"]:
        tags.pop(key, None)

    if "MAGMOM" in tags:
        mag = expand_magmom(tags["MAGMOM"])
        if len(mag) == len(order):
            tags["MAGMOM"] = " ".join(f"{mag[a]:g}" for a in order)
        else:
            print("warning: MAGMOM length != number of atoms; left unchanged.")

    if "RWIGS" in tags:  # per species: add an entry for the isolated atom
        rw = tags["RWIGS"].split()
        if len(rw) == nsp_old:
            tags["RWIGS"] = " ".join([rw[s]] + [rw[i] for i in kept])

    lmaxmix = max(int(tags.get("LMAXMIX", 2)), 6 if l == 3 else 4)
    ediff = min(float(tags.get("EDIFF", "1E-4").lower().replace("d", "e")), 1e-6)
    tags.update({
        "ISPIN": "2", "ISYM": "-1", "LORBIT": "11", "LMAXMIX": str(lmaxmix),
        "EDIFF": f"{ediff:.0E}", "NSW": "0", "IBRION": "-1",
        "LWAVE": ".TRUE.", "LCHARG": ".TRUE.",
    })
    tags.update(lr_tags(len(kept) + 1, l))
    write_incar(os.path.join(gs, "INCAR"), tags)

    for name in ("KPOINTS",) + tuple(copy):
        shutil.copy(os.path.join(src, name), gs)
    print(f"wrote {gs}: atom {site} ({new[5].split()[0]}, l={l}) is now species 1 / atom 1")
    if job:
        write_launchers(root, [gs], job)
    return gs


# ---------------------------------------------------------------- stage 2
def perturb(shifts=DEFAULT_SHIFTS, root=".", link=False, copy=(), job=None):
    """Create nscf_/scf_ run directories from a finished <root>/gs."""
    gs = os.path.join(root, "gs")
    for name in ("WAVECAR", "CHGCAR"):
        p = os.path.join(gs, name)
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            raise FileNotFoundError(f"{p} missing or empty: run the ground state first.")

    base = read_incar(os.path.join(gs, "INCAR"))
    nspecies = len(base["LDAUL"].split())
    l = int(base["LDAUL"].split()[0])
    dirs = []
    for spin in ("up", "dn"):
        for v in shifts:
            if v == 0:
                continue
            vup, vdn = (v, 0.0) if spin == "up" else (0.0, v)
            for kind in ("nscf", "scf"):
                d = os.path.join(root, f"{kind}_{spin}_{v:+.3f}")
                os.makedirs(d, exist_ok=True)
                tags = dict(base)
                tags.update(lr_tags(nspecies, l, vup, vdn))
                tags.update({"ISTART": "1", "ICHARG": "11" if kind == "nscf" else "0",
                             "LWAVE": ".FALSE.", "LCHARG": ".FALSE."})
                write_incar(os.path.join(d, "INCAR"), tags)
                for name in ("POSCAR", "POTCAR", "KPOINTS") + tuple(copy):
                    shutil.copy(os.path.join(gs, name), d)
                restart = ["WAVECAR"] + (["CHGCAR"] if kind == "nscf" else [])
                for name in restart:
                    dst = os.path.join(d, name)
                    if os.path.lexists(dst):
                        os.remove(dst)
                    if link:
                        os.symlink(os.path.relpath(os.path.join(gs, name), d), dst)
                    else:
                        shutil.copy(os.path.join(gs, name), dst)
                dirs.append(d)
    print(f"wrote {len(dirs)} run directories")
    job = job or os.path.join(gs, "job.sh")   # default: reuse the one given to setup
    if os.path.isfile(job):
        write_launchers(root, dirs, job)
    return dirs
