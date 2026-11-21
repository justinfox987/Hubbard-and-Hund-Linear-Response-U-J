import argparse

from .inputs import setup, perturb, DEFAULT_SHIFTS
from .analysis import analyze


def main():
    p = argparse.ArgumentParser(prog="lruj", description="Linear-response U/J for VASP")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup", help="write gs/ from INCAR POSCAR KPOINTS POTCAR")
    s.add_argument("--site", type=int, required=True, help="atom to perturb (1-based, as in OUTCAR)")
    s.add_argument("--l", type=int, default=2, help="angular momentum of the manifold (default 2)")
    s.add_argument("--src", default=".", help="directory holding the input files")
    s.add_argument("--copy", nargs="*", default=[], help="extra files to copy, e.g. job.sh")

    s = sub.add_parser("perturb", help="write nscf_*/scf_* directories after gs/ has finished")
    s.add_argument("--shifts", type=float, nargs="+", default=list(DEFAULT_SHIFTS), help="potential shifts in eV")
    s.add_argument("--link", action="store_true", help="symlink WAVECAR/CHGCAR instead of copying")
    s.add_argument("--copy", nargs="*", default=[], help="extra files to copy from gs/, e.g. job.sh")

    s = sub.add_parser("analyze", help="fit responses and print U, J")

    for name in ("setup", "perturb"):
        sub.choices[name].add_argument(
            "--job", help="Slurm job script; copied into each run as job.sh, and "
                          "submit.sh / run.sh are written to launch them")

    for s in sub.choices.values():
        s.add_argument("--root", default=".", help="working directory (default .)")

    a = p.parse_args()
    if a.cmd == "setup":
        setup(a.site, a.l, a.src, a.root, a.copy, a.job)
    elif a.cmd == "perturb":
        perturb(a.shifts, a.root, a.link, a.copy, a.job)
    else:
        analyze(a.root)


if __name__ == "__main__":
    main()
