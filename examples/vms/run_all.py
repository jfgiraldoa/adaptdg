"""Batch runner for all VMS examples (every example x preset), with resume, timeouts and a report.

Usage:
    python run_all.py --jobs 6 --out ~/vms_out          # full runs, 6 workers (start with nohup)
    python run_all.py --jobs 6 --out ~/vms_out --skip-done   # resume: skip jobs with done.json, returncode 0
    python run_all.py --smoke --jobs 2 --out /tmp/vms_smoke  # tiny settings, tests the plumbing (< 4 min)
    python run_all.py --report ~/vms_out                # regenerate REPORT.md and summary.csv only
    python run_all.py --list

Each job runs in its own copy of the example folder (``<out>/jobs/<job>/vms/<example>/``), so the
``out/`` directories never collide and the repository tree is never written. Python standard
library only; matplotlib is optional (only to decide whether the PNGs are copied to the report).
"""

import argparse
import csv
import datetime
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
NA = "n/a"
THREAD_ENV = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")}
SMOKE_ARGS = ["--degrees", "1", "--dofmax", "2500"]


@dataclass
class Job:
    """One example x preset. ``fake_cmd`` replaces the command (used by the tests)."""

    name: str
    example: str
    preset: str | None
    full_args: list[str] = field(default_factory=list)
    smoke_args: list[str] = field(default_factory=list)
    needs: tuple[str, ...] = ()  # other example folders that run.py reads
    est_min: float = 0.0  # README estimate of the --full wall time
    est_gb: float = 0.0  # README estimate of the --full peak memory
    fake_cmd: list[str] | None = None

    def command(self, smoke: bool) -> list[str]:
        if self.fake_cmd is not None:
            return list(self.fake_cmd)
        args = self.smoke_args if smoke else ["--full", *self.full_args]
        preset = ["--preset", self.preset] if self.preset else []
        return [sys.executable, "-u", "run.py", *preset, *args]


def _jobs() -> list[Job]:
    # (example, takes --preset, smoke extra args, needs, (est_min paper, est_min legacy), est_gb)
    table = [
        ("reentrant_corner", False, SMOKE_ARGS, (), (0.5,), 0.5),
        ("heterogeneous", False, SMOKE_ARGS, (), (3.0,), 2.0),
        ("anisotropic", False, SMOKE_ARGS, (), (2.0,), 1.5),
        ("fichera_3d", True, SMOKE_ARGS, (), (4.0, 4.0), 3.0),
        ("eriksson_johnson", True, [*SMOKE_ARGS, "--kappa", "1e-2"], (), (8.0, 8.0), 4.0),
        ("poisson_fem", True, SMOKE_ARGS, (), (6.0, 6.0), 5.0),
        ("burgers_single", True, [*SMOKE_ARGS, "--kappa", "1e-2"], (), (1.0, 5.0), 1.7),
        ("burgers_isotropic", True, SMOKE_ARGS, (), (4.5, 6.5), 3.5),
        ("compare_presets", False, SMOKE_ARGS, ("reentrant_corner", "heterogeneous", "anisotropic"), (5.0,), 2.0),
    ]
    jobs = []
    for example, has_preset, smoke, needs, est, gb in table:
        for k, preset in enumerate(("paper", "legacy") if has_preset else (None,)):
            name = f"{example}__{preset}" if preset else example
            jobs.append(Job(name, example, preset, [], smoke, needs, est[min(k, len(est) - 1)], gb))
    return jobs


JOBS = _jobs()


# ----------------------------------------------------------------------------- running

def _peak_rss_mb(ru) -> float | None:
    """ru_maxrss of a waited child: bytes on macOS, kilobytes on Linux."""
    if ru is None:
        return None
    return ru.ru_maxrss / (1e6 if sys.platform == "darwin" else 1e3)


def _kill_group(pgid: int) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            return
        time.sleep(1.0 if sig == signal.SIGTERM else 0.0)


def _files(folder: Path) -> list[str]:
    out = folder / "vms"
    return sorted(str(p.relative_to(folder)) for p in out.rglob("*") if p.is_file() and "/out/" in p.as_posix()) \
        if out.exists() else []


def run_job(job: Job, root: Path, smoke: bool, timeout_s: float, live: set) -> dict:
    """Run one job in its own directory; always returns the ``done.json`` record (never raises)."""
    jobdir = root / "jobs" / job.name
    if jobdir.exists():
        shutil.rmtree(jobdir)
    jobdir.mkdir(parents=True)
    cwd = jobdir
    rec = {"name": job.name, "example": job.example, "preset": job.preset, "smoke": smoke,
           "returncode": None, "status": "failed", "start": time.time(), "end": None, "wall_s": None,
           "peak_rss_mb": None, "files": [], "command": job.command(smoke)}
    log = jobdir / "log.txt"
    try:
        if job.fake_cmd is None:
            for folder in (job.example, *job.needs):
                shutil.copytree(HERE / folder, jobdir / "vms" / folder,
                                ignore=shutil.ignore_patterns("out", "__pycache__"))
            cwd = jobdir / "vms" / job.example
        env = {**os.environ, "FI_PROVIDER": os.environ.get("FI_PROVIDER", "tcp"), "MPLBACKEND": "Agg",
               "PYTHONUNBUFFERED": "1"}
        for key, value in THREAD_ENV.items():
            env.setdefault(key, value)
        with open(log, "w") as fh:
            proc = subprocess.Popen(rec["command"], cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            live.add(proc.pid)
            rusage = None
            timed_out = False
            while True:
                pid, status, rusage = os.wait4(proc.pid, os.WNOHANG)
                if pid:
                    break
                if time.time() - rec["start"] > timeout_s:
                    timed_out = True
                    _kill_group(proc.pid)
                    pid, status, rusage = os.wait4(proc.pid, 0)
                    break
                time.sleep(0.2)
            live.discard(proc.pid)
            proc.returncode = 0  # reaped by wait4 above; keeps Popen.__del__ quiet
        rec["returncode"] = os.waitstatus_to_exitcode(status)
        rec["peak_rss_mb"] = _peak_rss_mb(rusage)
        rec["status"] = "timeout" if timed_out else ("ok" if rec["returncode"] == 0 else "failed")
        plot = cwd / "plot.py"
        if rec["status"] == "ok" and job.fake_cmd is None and plot.exists():  # figures are best effort
            with open(log, "a") as fh:
                fh.write("\n--- plot.py ---\n")
                fh.flush()
                try:
                    subprocess.run([sys.executable, "plot.py"], cwd=cwd, env=env, stdout=fh,
                                   stderr=subprocess.STDOUT, timeout=300)
                except Exception as exc:
                    fh.write(f"plot.py failed: {exc!r}\n")
    except Exception as exc:  # a failing job never stops the batch
        rec["error"] = repr(exc)
        with open(log, "a") as fh:
            fh.write(f"\nrunner error: {exc!r}\n")
    rec["end"] = time.time()
    rec["wall_s"] = rec["end"] - rec["start"]
    rec["files"] = _files(jobdir)
    (jobdir / "done.json").write_text(json.dumps(rec, indent=1))
    return rec


def _is_done(root: Path, name: str) -> bool:
    try:
        return json.loads((root / "jobs" / name / "done.json").read_text()).get("returncode") == 0
    except (OSError, ValueError):
        return False


def _fmt_s(seconds) -> str:
    return NA if seconds is None else f"{int(seconds // 60)}m{int(seconds % 60):02d}s"


def run_batch(jobs: list[Job], root: Path, workers: int = 4, smoke: bool = False, timeout_s: float = 3600.0,
              skip_done: bool = False, report: bool = True) -> dict[str, dict]:
    """Run ``jobs`` with ``workers`` threads; updates ``status.json`` after every job."""
    root.mkdir(parents=True, exist_ok=True)
    total, t0 = len(jobs), time.time()
    lock, live = threading.Lock(), set()
    state = {"started": datetime.datetime.now().isoformat(timespec="seconds"), "total": total,
             "jobs": {j.name: {"status": "pending"} for j in jobs}}
    records: dict[str, dict] = {}
    counter = [0]

    def write_status() -> None:
        state["elapsed_s"] = round(time.time() - t0)
        (root / "status.json").write_text(json.dumps(state, indent=1))

    def table() -> str:
        return "  ".join(f"{n}={s['status']}" for n, s in state["jobs"].items())

    def work(job: Job) -> None:
        with lock:
            counter[0] += 1
            i = counter[0]
            if skip_done and _is_done(root, job.name):
                state["jobs"][job.name] = {"status": "skipped"}
                print(f"run {i}/{total}: {job.name} skipped (done)  [{_fmt_s(time.time() - t0)}]", flush=True)
                write_status()
                return
            state["jobs"][job.name] = {"status": "running"}
            print(f"run {i}/{total}: {job.name} started  [{_fmt_s(time.time() - t0)}]", flush=True)
            write_status()
        rec = run_job(job, root, smoke, timeout_s, live)
        with lock:
            records[job.name] = rec
            state["jobs"][job.name] = {"status": rec["status"], "wall_s": round(rec["wall_s"], 1),
                                       "peak_rss_mb": rec["peak_rss_mb"], "returncode": rec["returncode"]}
            print(f"run {i}/{total}: {job.name} {rec['status']} wall={_fmt_s(rec['wall_s'])} "
                  f"rss={NA if rec['peak_rss_mb'] is None else format(rec['peak_rss_mb'], '.0f') + 'MB'}  "
                  f"[{_fmt_s(time.time() - t0)}]\n  status: {table()}", flush=True)
            write_status()

    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(work, jobs))
    finally:
        for pid in list(live):
            _kill_group(pid)
    state["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    write_status()
    if report:
        try:
            make_report(root)
        except Exception as exc:  # the report must never crash the batch
            print(f"report failed: {exc!r}", flush=True)
    print(f"all done in {_fmt_s(time.time() - t0)}; report: {root / 'REPORT.md'}", flush=True)
    return records


# ----------------------------------------------------------------------------- report

SLOPE_KEYS = ("E", "L2", "Vh", "L2_theta", "Vh_theta", "L2_phi", "Vh_phi", "L2_u", "Vh_u")
PROJ_KEYS = tuple(f"{k}_proj" for k in SLOPE_KEYS if k != "E")


def _num(value) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _f(value, fmt: str = ".2f") -> str:
    x = _num(value)
    return NA if x is None else format(x, fmt)


def _get(d, *keys):
    for key in keys:
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def _csv_slope(rows: list[dict], key: str, span: float, dim: int = 2) -> float | None:
    """Same fit as ``adaptdg.vms.loop.fitted_slope`` (log error versus log DoF^(1/dim))."""
    try:
        final = float(rows[-1]["ndofs"])
        sel = [r for r in rows if float(r["ndofs"]) * span >= final]
        sel = sel if len(sel) >= 3 else rows[-3:]
        xs = [math.log(float(r["ndofs"]) ** (1.0 / dim)) for r in sel]
        ys = [math.log(float(r[key])) for r in sel]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        den = sum((x - mx) ** 2 for x in xs)
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den > 0 and len(xs) > 1 else None
    except (KeyError, ValueError, ZeroDivisionError, IndexError):
        return None


def _entry(label: str, **kw) -> dict:
    return {"run": label, "ndofs": None, "levels": None, "slopes": {}, "ratio_exact": None, "ratio_proj": None,
            "newton": None, "h_ratio": None, "time_s": None, "peak_rss_mb": None, "cg_over_rm": None, **kw}


def _from_summary_item(item: dict) -> dict:
    parts = [f"{k}{item[k]:g}" if isinstance(item.get(k), float) else f"{k}{item[k]}"
             for k in ("dim", "kappa", "q", "p") if k in item]
    parts += [str(item[k]) for k in ("run", "refinement") if k in item]
    newton = item.get("newton") or {}
    cluster = item.get("clustering") or item.get("layer") or {}
    return _entry(" ".join(parts) or "run", ndofs=item.get("ndofs"), levels=item.get("levels"),
                  slopes=item.get("slopes") or {}, ratio_exact=_get(item, "Vh_phi_over_theta", "exact"),
                  ratio_proj=_get(item, "Vh_phi_over_theta", "proj"),
                  newton=(f"{newton.get('converged', NA)}/{newton.get('levels', NA)} "
                          f"(it {newton.get('iterations_min', NA)}-{newton.get('iterations_max', NA)})")
                  if newton else None,
                  h_ratio=cluster.get("h_far_over_near"), time_s=item.get("time_s"),
                  peak_rss_mb=item.get("peak_rss_mb"),
                  cg_over_rm=_get(item, "ratios_last_mesh", "Vh_cg/Vh_ubar"))


def _from_history_csv(path: Path) -> dict:
    rows = list(csv.DictReader(open(path)))
    stem = path.stem.replace("history_", "")
    label = (path.parent.name + " " if path.parent.name.startswith("r1e") else "") + stem.replace("_", " ")
    span = 20.0 if "uniform" in stem else 4.0
    keys = [k for k in (*SLOPE_KEYS, *PROJ_KEYS) if k in rows[0]]
    slopes = {k: _csv_slope(rows, k, span) for k in keys} if len(rows) > 3 else {}
    last = rows[-1]

    def ratio(sfx):
        a, b = _num(last.get(f"Vh_phi{sfx}")), _num(last.get(f"Vh_theta{sfx}"))
        return a / b if a is not None and b else None

    return _entry(label, ndofs=last.get("ndofs"), levels=len(rows), slopes=slopes, ratio_exact=ratio(""),
                  ratio_proj=ratio("_proj"))


def _from_compare_csv(path: Path) -> list[dict]:
    groups: dict[tuple, dict] = {}
    for r in csv.DictReader(open(path)):
        e = groups.setdefault((r["case"], r["p"], r["preset"]),
                              _entry(f"{r['case']} p{r['p']} {r['preset']}", ndofs=r.get("ndofs")))
        field_ = "" if r["field"] == "ubar" else "_" + r["field"]
        sfx = "_proj" if r["target"] == "projection" else ""
        e["slopes"][f"{r['norm']}{field_}{sfx}"] = r["slope"]
        e["ratio_proj" if sfx else "ratio_exact"] = r["ratio_Vh_phi_over_theta"]
    return list(groups.values())


def collect_entries(jobdir: Path) -> list[dict]:
    """Key numbers of one job from whatever its summary files contain (never raises)."""
    entries: list[dict] = []
    for path in sorted(jobdir.rglob("summary.json")):
        try:
            data = json.loads(path.read_text())
            entries += [_from_summary_item(it) for it in (data if isinstance(data, list) else [data])]
        except Exception:
            pass
    if entries:
        return entries
    for path in sorted(jobdir.rglob("summary.csv")):
        try:
            entries += _from_compare_csv(path)
        except Exception:
            pass
    if entries:
        return entries
    for path in sorted(jobdir.rglob("history_*.csv")):
        try:
            entries.append(_from_history_csv(path))
        except Exception:
            pass
    return entries


def _tail(path: Path, n: int = 15) -> str:
    try:
        return "".join(path.read_text(errors="replace").splitlines(keepends=True)[-n:]).rstrip()
    except OSError:
        return NA


def _md_table(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out + [""]


def make_report(root: Path, jobs: list[Job] | None = None) -> Path:
    """Write ``REPORT.md``, ``summary.csv`` and ``figures/`` from the job folders under ``root``."""
    root = Path(root)
    jobs = jobs or JOBS
    known = {j.name for j in jobs}
    extra = sorted(p.name for p in (root / "jobs").glob("*") if p.is_dir() and p.name not in known) \
        if (root / "jobs").exists() else []
    jobs = [*jobs, *[Job(n, n, None) for n in extra]]
    recs, entries = {}, {}
    for job in jobs:
        jd = root / "jobs" / job.name
        try:
            recs[job.name] = json.loads((jd / "done.json").read_text())
        except (OSError, ValueError):
            recs[job.name] = None
        entries[job.name] = collect_entries(jd) if jd.exists() else []

    def status(job):
        r = recs[job.name]
        return NA if r is None else r.get("status", NA)

    summary_rows = []
    for job in jobs:
        r = recs[job.name] or {}
        for e in entries[job.name] or [_entry(NA)]:
            row = {"job": job.name, "example": job.example, "preset": job.preset or NA, "status": status(job)
                   if recs[job.name] else "not run", "wall_s": _f(r.get("wall_s"), ".1f"),
                   "peak_rss_mb": _f(r.get("peak_rss_mb"), ".0f"), "run": e["run"], "ndofs": e["ndofs"] or NA,
                   "levels": e["levels"] or NA}
            for k in (*SLOPE_KEYS, *PROJ_KEYS):
                row[f"slope_{k}"] = _f(e["slopes"].get(k), ".3f")
            row.update(ratio_Vh_phi_over_theta_exact=_f(e["ratio_exact"], ".3f"),
                       ratio_Vh_phi_over_theta_proj=_f(e["ratio_proj"], ".3f"),
                       newton_converged=e["newton"] or NA, h_far_over_near=_f(e["h_ratio"], ".1f"),
                       cg_over_rm_Vh=_f(e["cg_over_rm"], ".2f"))
            summary_rows.append(row)
    with open(root / "summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    md = ["# VMS batch report", "",
          f"Generated {datetime.datetime.now().isoformat(timespec='seconds')} on {platform.node() or NA} "
          f"({platform.system()} {platform.machine()}). Slopes are against DoF^(1/dim) over the last 2x "
          "window of DoF (adaptive histories without `summary.json` use the last 4x, uniform 20x); "
          "peak memory is the child's ru_maxrss.", "", "## Jobs", ""]
    rows = []
    for job in jobs:
        r = recs[job.name]
        rows.append([job.name, status(job) if r else "not run", _fmt_s(r and r.get("wall_s")),
                     f"{job.est_min:g} min" if job.est_min else NA, _f(r and r.get("peak_rss_mb"), ".0f"),
                     f"{job.est_gb:g}" if job.est_gb else NA, NA if not r else r.get("returncode"),
                     len((r or {}).get("files", []))])
    md += _md_table(["job", "status", "wall", "est. wall", "peak MB", "est. GB", "rc", "files"], rows)

    md += ["## Slopes (exact solution target)", ""]
    rows = [[j.name, e["run"], e["ndofs"] or NA, *[_f(e["slopes"].get(k)) for k in SLOPE_KEYS],
             _f(e["ratio_exact"], ".3f"), e["newton"] or NA, _f(e["h_ratio"], ".1f")]
            for j in jobs for e in entries[j.name]]
    md += _md_table(["job", "run", "DoF", *SLOPE_KEYS, "Vh(phi)/Vh(theta)", "Newton conv.", "h far/near"], rows)
    md += ["## Slopes (L2-projection target) and ratio at the finest level", ""]
    rows = [[j.name, e["run"], *[_f(e["slopes"].get(k)) for k in PROJ_KEYS], _f(e["ratio_proj"], ".3f"),
             _f(e["cg_over_rm"], ".2f")] for j in jobs for e in entries[j.name]]
    md += _md_table(["job", "run", *PROJ_KEYS, "Vh(phi)/Vh(theta)", "Vh CG/RM"], rows)

    md += ["## Legacy vs paper", ""]
    pairs = []
    for job in jobs:
        if job.preset == "paper":
            other = next((j for j in jobs if j.example == job.example and j.preset == "legacy"), None)
            if other:
                pairs.append((job, other))
    rows = []
    for a, b in pairs:
        eb = {e["run"]: e for e in entries[b.name]}
        for e in entries[a.name]:
            o = eb.get(e["run"])
            cells = [_f(e["slopes"].get("Vh")), _f(o and o["slopes"].get("Vh")),
                     _f(e["slopes"].get("Vh_phi")), _f(o and o["slopes"].get("Vh_phi")),
                     _f(e["ratio_exact"], ".3f"), _f(o and o["ratio_exact"], ".3f")]
            rows.append([a.example, e["run"], *cells])
    md += _md_table(["case", "run", "Vh paper", "Vh legacy", "Vh_phi paper", "Vh_phi legacy",
                     "ratio paper", "ratio legacy"], rows) if rows else ["n/a (no paired jobs)", ""]
    md.append("`compare_presets` pairs both presets inside one job (see its rows above).")
    md.append("")

    bad = [j for j in jobs if recs[j.name] and recs[j.name].get("status") != "ok"]
    md += ["## Failures and timeouts", ""]
    if not bad:
        md += ["None.", ""]
    for job in bad:
        md += [f"### {job.name}: {recs[job.name].get('status')} (rc {recs[job.name].get('returncode')})", "",
               "```", _tail(root / "jobs" / job.name / "log.txt"), "```", ""]
    missing = [j.name for j in jobs if recs[j.name] is None]
    if missing:
        md += ["## Not run", "", ", ".join(missing), ""]

    copied = 0
    try:
        import matplotlib  # noqa: F401  (only decides whether figures are collected)

        for job in jobs:
            for png in sorted((root / "jobs" / job.name).rglob("*.png")):
                dest = root / "figures" / job.name / png.relative_to(root / "jobs" / job.name).as_posix().split(
                    "/out/", 1)[-1].replace("/", "_")
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(png, dest)
                copied += 1
    except ImportError:
        pass
    md += ["## Figures", "", f"{copied} PNG files copied to `figures/<job>/`." if copied else
           "none (matplotlib missing or the jobs wrote no PNG; run each example's `plot.py` in its job folder).", ""]
    (root / "REPORT.md").write_text("\n".join(md))
    return root / "REPORT.md"


# ----------------------------------------------------------------------------- CLI

def select(only: str | None) -> list[Job]:
    if not only:
        return list(JOBS)
    wanted = {s.strip() for s in only.split(",") if s.strip()}
    chosen = [j for j in JOBS if j.name in wanted or j.example in wanted]
    unknown = wanted - {j.name for j in JOBS} - {j.example for j in JOBS}
    if unknown:
        raise SystemExit(f"unknown job(s): {', '.join(sorted(unknown))}; see --list")
    return chosen


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="vms_batch_out", help="output directory (default ./vms_batch_out)")
    ap.add_argument("--jobs", type=int, default=4, help="parallel workers (default 4)")
    ap.add_argument("--only", help="comma-separated job or example names")
    ap.add_argument("--skip-done", action="store_true", help="skip jobs whose done.json has returncode 0")
    ap.add_argument("--timeout-min", type=float, default=60.0, help="per-job timeout in minutes (default 60)")
    ap.add_argument("--smoke", action="store_true", help="short tiny runs instead of --full (plumbing test)")
    ap.add_argument("--list", action="store_true", help="list the jobs and exit")
    ap.add_argument("--report", metavar="DIR", help="only (re)generate REPORT.md and summary.csv in DIR")
    args = ap.parse_args(argv)
    if args.report:
        print(make_report(Path(args.report).expanduser()))
        return 0
    jobs = select(args.only)
    if args.list:
        for j in jobs:
            print(f"{j.name:30s} est {j.est_min:4.1f} min {j.est_gb:3.1f} GB  "
                  f"{' '.join(j.command(False)[2:])}")
        print(f"{len(jobs)} jobs, estimated {sum(j.est_min for j in jobs):.0f} min serial")
        return 0
    root = Path(args.out).expanduser().resolve()
    records = run_batch(jobs, root, args.jobs, args.smoke, args.timeout_min * 60.0, args.skip_done)
    return 0 if all(r["status"] == "ok" for r in records.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
