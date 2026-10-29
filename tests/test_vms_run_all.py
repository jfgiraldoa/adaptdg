"""Batch runner ``examples/vms/run_all.py``: registry, resume, failures, timeouts, report (no dolfinx needed)."""

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("vms_run_all", ROOT / "examples" / "vms" / "run_all.py")
run_all = importlib.util.module_from_spec(spec)
sys.modules["vms_run_all"] = run_all
spec.loader.exec_module(run_all)

EXAMPLES = ("reentrant_corner", "heterogeneous", "anisotropic", "fichera_3d", "eriksson_johnson", "poisson_fem",
            "burgers_single", "burgers_isotropic", "compare_presets")
WITH_PRESET = ("fichera_3d", "eriksson_johnson", "poisson_fem", "burgers_single", "burgers_isotropic")


def fake(name, code):
    return run_all.Job(name, name, None, fake_cmd=[sys.executable, "-c", code])


def test_registry_lists_every_example_and_preset():
    names = {j.name for j in run_all.JOBS}
    assert len(names) == len(run_all.JOBS)
    for ex in EXAMPLES:
        assert (ROOT / "examples" / "vms" / ex / "run.py").exists()
        if ex in WITH_PRESET:
            assert {f"{ex}__paper", f"{ex}__legacy"} <= names
        else:
            assert ex in names
    full = next(j for j in run_all.JOBS if j.name == "fichera_3d__legacy").command(False)
    assert full[-3:] == ["--preset", "legacy", "--full"]


def test_list_and_only(capsys):
    assert run_all.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "eriksson_johnson__paper" in out and "compare_presets" in out
    assert [j.name for j in run_all.select("poisson_fem")] == ["poisson_fem__paper", "poisson_fem__legacy"]
    with pytest.raises(SystemExit):
        run_all.select("nope")


def test_failure_does_not_stop_batch_and_resume_skips_done(tmp_path):
    jobs = [fake("bad", "import sys; sys.exit(3)"), fake("good", "print('hello')")]
    rec = run_all.run_batch(jobs, tmp_path, workers=2, timeout_s=30)
    assert rec["bad"]["status"] == "failed" and rec["bad"]["returncode"] == 3
    assert rec["good"]["status"] == "ok"
    assert json.loads((tmp_path / "status.json").read_text())["jobs"]["good"]["status"] == "ok"
    assert (tmp_path / "REPORT.md").exists() and (tmp_path / "summary.csv").exists()
    assert "bad" in (tmp_path / "REPORT.md").read_text()
    marker = tmp_path / "jobs" / "good" / "done.json"
    before = marker.read_text()
    rec2 = run_all.run_batch(jobs, tmp_path, workers=2, timeout_s=30, skip_done=True)
    assert "good" not in rec2 and rec2["bad"]["status"] == "failed"  # only the failed job is re-run
    assert marker.read_text() == before


def test_timeout_kills_sleeping_job(tmp_path):
    t0 = time.time()
    rec = run_all.run_batch([fake("slow", "import time; time.sleep(60)")], tmp_path, timeout_s=1.0, report=False)
    assert rec["slow"]["status"] == "timeout"
    assert time.time() - t0 < 20


def test_report_on_fake_tree_with_missing_fields(tmp_path):
    base = tmp_path / "jobs"
    out = base / "fichera_3d__paper" / "vms" / "fichera_3d" / "out" / "paper"
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps([{"p": 1, "refinement": "adaptive", "ndofs": 100,
                                                   "slopes": {"Vh": -0.66}}, {"weird": True}]))
    other = base / "reentrant_corner" / "vms" / "reentrant_corner" / "out"
    other.mkdir(parents=True)
    (other / "history_p1_adaptive.csv").write_text(
        "level,ndofs,ncells,E,L2,Vh,time\n" + "\n".join(f"{i},{100 * 2 ** i},1,{2.0 ** -i},{4.0 ** -i},"
                                                          f"{2.0 ** -i},0" for i in range(6)) + "\n")
    (base / "burgers_single__paper").mkdir()
    (base / "burgers_single__paper" / "done.json").write_text("{not json")
    (base / "burgers_single__paper" / "log.txt").write_text("\n".join(f"line {i}" for i in range(40)))
    report = run_all.make_report(tmp_path)
    text = report.read_text()
    assert "n/a" in text and "-0.660" in (tmp_path / "summary.csv").read_text()
    assert "not run" in text
    assert (tmp_path / "summary.csv").read_text().count("reentrant_corner") >= 1


@pytest.mark.skipif(importlib.util.find_spec("dolfinx") is None, reason="needs dolfinx")
def test_smoke_reentrant_corner(tmp_path):
    rec = run_all.run_batch(run_all.select("reentrant_corner"), tmp_path, workers=1, smoke=True, timeout_s=120)
    assert rec["reentrant_corner"]["status"] == "ok"
    assert "reentrant_corner" in (tmp_path / "REPORT.md").read_text()
