# Running all VMS examples (batch runner)

`run_all.py` runs every example of this folder with `--full`, for both presets where the example has one, in parallel,
and writes one report. It needs only the Python standard library (matplotlib is used by the examples' `plot.py`).

## Install

    python -m pip install -e <path-to-adaptdg>      # on a machine with dolfinx 0.11, PETSc and MUMPS
    cd examples/vms

Set one thread per process before starting, so that the parallel workers do not compete:

    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1

## Run

    python run_all.py --list                                       # the 14 jobs with their estimates
    python run_all.py --smoke --jobs 2 --out /tmp/vms_smoke        # tiny settings, about 1 min: tests the plumbing
    nohup python run_all.py --jobs 6 --out ~/vms_out > ~/vms_out.log 2>&1 &
    python run_all.py --jobs 6 --out ~/vms_out --skip-done         # resume: jobs with done.json and returncode 0 are skipped
    python run_all.py --report ~/vms_out                           # rebuild REPORT.md and summary.csv only

Other options: `--only name1,name2` (job names such as `fichera_3d__legacy` or example names such as `poisson_fem`),
`--timeout-min M` (per job, default 60; the process group is killed and the job marked `timeout`).

Each job runs in its own copy of the example folder, `~/vms_out/jobs/<job>/vms/<example>/` (so the `out/` folders never
collide and the repository is never written), with `log.txt` (stdout and stderr), `done.json` (times, return code, peak
memory, files produced) and, after a successful run, the example's `plot.py` figures. `status.json` is updated after
every job and can be read while the batch runs. A failing job never stops the batch. The final output is
`~/vms_out/REPORT.md` (status, wall time, peak memory, slopes of the errors for both targets, ratio Vh(phi)/Vh(theta),
Newton convergence, cell-size clustering, legacy versus paper, last log lines of failures) and `summary.csv`;
the PNG files are collected in `~/vms_out/figures/<job>/`.

## Expected time and memory per job (`--full`, estimates from the example READMEs)

| job | wall | peak memory |
|---|---|---|
| reentrant_corner | 0.5 min | < 1 GB |
| heterogeneous | 3 min | about 2 GB |
| anisotropic | 2 min | about 1.5 GB |
| fichera_3d (paper, legacy) | 4 min each | 3 GB |
| eriksson_johnson (paper, legacy) | 8 min each | 4 GB |
| poisson_fem (paper, legacy) | 6 min each | 5 GB |
| burgers_single (paper, legacy) | 3 and 3-8 min | 0.9 GB |
| burgers_isotropic (paper, legacy) | 4.5 and 6.5 min | 3.5 GB |
| compare_presets | a few minutes (5 min assumed) | about 2 GB |

The serial total is about 66 min; the longest job is about 8 min, so with 6 or 8 workers the batch should take roughly
12-15 min (these are estimates, not measurements). Six workers need about 25 GB in the worst case (the six
largest jobs at once).
