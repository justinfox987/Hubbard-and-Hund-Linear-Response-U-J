# LRUJ

Minimal linear-response Hubbard U / Hund J for VASP (`LDAUTYPE = 3`), collinear only.
Determines UJ for the Liechenstien et al method. 
Needs only numpy. Install with `pip install -e .` or run in place with `python -m lruj`.

## Workflow

Start in a directory holding a converged-settings `INCAR`, `POSCAR`, `KPOINTS`, `POTCAR`.

    lruj setup --site 19 --l 2 --job sample.sh  # writes gs/, submit.sh, run.sh
    ./submit.sh                                 # or ./run.sh on an interactive node
    lruj perturb                                # writes nscf_*/ scf_*/ (16 runs), new submit.sh, run.sh
    ./submit.sh                                 # or ./run.sh
    lruj analyze                                # prints U and J

- `--site` is the atom number as VASP prints it (1-based). It is moved to the
           front of the POSCAR as its own species; POTCAR, MAGMOM and RWIGS are reordered to match.
- `--l` sets the quantum number l to find the interaction for
- `--job` is a Slurm script. It is copied into each run directory as `job.sh`.
  `submit.sh` calls `sbatch -J <dir> job.sh` in each directory (one job per run).
  `run.sh` is a single job for all runs: it carries the `#SBATCH` header of your
  script and runs `bash job.sh` in each directory in sequence. Use `sbatch run.sh`
  (from this directory), or `bash run.sh` on an interactive node. Both scripts skip
  directories whose OUTCAR is finished, so if the walltime runs out, submit
  `run.sh` again and it continues. `perturb` reuses `gs/job.sh` unless given `--job`.
- `--shifts` (perturb) sets the potential shifts in eV; default `-0.2 -0.1 0.1 0.2`.
- `--link` (perturb) symlinks WAVECAR/CHGCAR from `gs/` instead of copying.

Occupations are read from the LORBIT = 11 `total charge` and `magnetization (x)`
tables in OUTCAR (3 decimals). The LDAUPRINT onsite density matrix cannot be used:
in the `ICHARG = 11` runs it stays frozen at the ground-state value.

## What it sets

Ground state: `LDAU=T, LDAUTYPE=3, LDAUPRINT=2, LDAUL = l -1 ..., LDAUU = LDAUJ = 0`,
`ISPIN=2, ISYM=-1, LORBIT=11, LMAXMIX>=4, EDIFF<=1E-6, NSW=0`. Any +U in the source
INCAR is dropped: the response is measured on the plain-DFT state.

Perturbed runs restart from `gs/WAVECAR`. `LDAUU = v` shifts spin up, `LDAUJ = v`
shifts spin down. NSCF runs use `ICHARG = 11` with `gs/CHGCAR`; SCF runs use `ICHARG = 0`.

## Output

`chi[i,j] = dn_i/dv_j` from SCF runs, `chi0` from NSCF runs, `f = inv(chi) - inv(chi0)`.
Two mappings from Linscott et al., PRB 98, 235157 (2018) are printed:
conventional (scaled 2x2, Eqs. 20-25) and constrained (simple 2x2, Eqs. 19, 23).
Uncertainties are a Monte Carlo over the slope standard errors.

`python tests/test_synthetic.py` runs the whole chain on fake data with a known answer.

## CREDIT AND SOURCES

This is directly adapted from Moore et al.'s Atomate implementation of the same process [1].
The mathematics come from Linscott et al.'s work [2].

[1] G. C. Moore, M. K. Horton, E. Linscott, A. M. Ganose, M. Siron, D. D. O’Regan, and K. A. Persson, Physical Review Materials 8, (2024). 

[2] E. B. Linscott, D. J. Cole, M. C. Payne, and D. D. O’Regan, Physical Review B 98, (2018)
