"""End-to-end check on fake data with a known answer. Run: python tests/test_synthetic.py

"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lruj import setup, perturb, analyze
from lruj.inputs import read_incar

U_TRUE, J_TRUE = 5.0, 0.8
CHI0 = np.array([[0.30, -0.05], [-0.04, 0.22]])          # asymmetric on purpose
F = np.array([[U_TRUE - J_TRUE, U_TRUE + J_TRUE],
              [U_TRUE + J_TRUE, U_TRUE - J_TRUE]])
CHI = np.linalg.inv(np.linalg.inv(CHI0) + F)
N0 = np.array([4.6, 1.1])


def fake_outcar(path, nup, ndn, l=2):
    dim = 2 * l + 1
    def block(n):
        return "\n".join(" " + " ".join(f"{(n / dim if a == b else 0.0):8.4f}" for b in range(dim))
                         for a in range(dim))
    text = (f" atom =   1  type =  1  l = {l}\n\n onsite density matrix\n\n"
            f" spin component  1\n\n{block(nup)}\n\n spin component  2\n\n{block(ndn)}\n\n"
            " occupancies and eigenvectors\n\n"
            " aborting loop because EDIFF is reached\n\n"
            " total charge\n\n# of ion       s       p       d       tot\n" + "-" * 42 + "\n"
            f"    1        0.000   0.000 {nup + ndn:7.3f} {nup + ndn:7.3f}\n\n"
            " magnetization (x)\n\n# of ion       s       p       d       tot\n" + "-" * 42 + "\n"
            f"    1        0.000   0.000 {nup - ndn:7.3f} {nup - ndn:7.3f}\n")
    with open(path, "w") as fh:
        fh.write(text)


def main():
    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, "src"); os.makedirs(src)
    open(os.path.join(src, "POSCAR"), "w").write(
        "test\n1.0\n4 0 0\n0 4 0\n0 0 4\nBi Fe O\n1 2 3\nDirect\n"
        + "\n".join(f"0.{i} 0.{i} 0.{i}" for i in range(1, 7)) + "\n")
    open(os.path.join(src, "POTCAR"), "w").write(
        "".join(f" PAW_PBE {s}\n data\n End of Dataset\n" for s in ("Bi", "Fe", "O")))
    open(os.path.join(src, "INCAR"), "w").write(
        "ENCUT = 520 ! comment\nMAGMOM = 0 5 -4 3*0\nLDAU = .TRUE.\nLDAUU = 0 4 0\n")
    open(os.path.join(src, "KPOINTS"), "w").write("k\n0\nG\n2 2 2\n")

    setup(site=3, l=2, src=src, root=tmp)
    gs = os.path.join(tmp, "gs")
    pos = open(os.path.join(gs, "POSCAR")).read().splitlines()
    assert pos[5].split() == ["Fe", "Bi", "Fe", "O"] and pos[6].split() == ["1", "1", "1", "3"]
    assert pos[8].startswith("0.3") and pos[9].startswith("0.1")
    pot = open(os.path.join(gs, "POTCAR")).read()
    assert [ln.split()[1] for ln in pot.splitlines() if "PAW_PBE" in ln] == ["Fe", "Bi", "Fe", "O"]
    tags = read_incar(os.path.join(gs, "INCAR"))
    assert tags["MAGMOM"] == "-4 0 5 0 0 0" and tags["LDAUL"] == "2 -1 -1 -1"
    assert tags["LDAUTYPE"] == "3" and float(tags["LDAUU"].split()[0]) == 0

    for name in ("WAVECAR", "CHGCAR"):
        open(os.path.join(gs, name), "w").write("x")
    fake_outcar(os.path.join(gs, "OUTCAR"), *N0)
    dirs = perturb(root=tmp)
    assert len(dirs) == 16
    for d in dirs:
        t = read_incar(os.path.join(d, "INCAR"))
        v = np.array([float(t["LDAUU"].split()[0]), float(t["LDAUJ"].split()[0])])
        n = N0 + (CHI0 if t["ICHARG"] == "11" else CHI) @ v
        fake_outcar(os.path.join(d, "OUTCAR"), *n)

    res = analyze(tmp)
    assert abs(res["U_simple"] - U_TRUE) < 0.5 and abs(res["J_simple"] - J_TRUE) < 0.5
    print("\nOK: U_simple, J_simple recovered; expected", U_TRUE, J_TRUE)


if __name__ == "__main__":
    main()
