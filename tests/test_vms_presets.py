"""Presets and error targets of the VMS solver (dolfinx)."""

import numpy as np
import pytest

pytest.importorskip("dolfinx")

import ufl  # noqa: E402
from dolfinx import fem  # noqa: E402

from adaptdg.marking import fixed_fraction_bands  # noqa: E402
from adaptdg.vms import meshes  # noqa: E402
from adaptdg.vms.dgforms import DGForms  # noqa: E402
from adaptdg.vms.loop import AdaptiveStationaryLoop, LoopSettings  # noqa: E402
from adaptdg.vms.norms import error_targets, project_exact  # noqa: E402
from adaptdg.vms.preset import resolve_preset  # noqa: E402
from adaptdg.vms.problems import Problem, ReentrantCorner  # noqa: E402
from adaptdg.vms.reconstruction import reconstruct  # noqa: E402
from adaptdg.vms.refine import refine_marked  # noqa: E402
from adaptdg.vms.saddle import RMSolver  # noqa: E402

BASELINE = {  # values of the release before the presets (paper behaviour), reentrant corner, reconstruct
    1: [
        (1.33012302837354e-01, 2.61109645719458e-02, 2.51222163034537e-01, 1.71029414318187e-02, 2.43733604629722e-01, 8.84636856618351e-03, 3.04628212831275e-01),
        (1.03431414825361e-01, 1.65893140334796e-02, 1.99426983233606e-01, 1.15470538346205e-02, 1.92206286725934e-01, 6.60800228371513e-03, 2.37303695314071e-01),
        (7.54651690877752e-02, 1.01921568932380e-02, 1.54525086864098e-01, 7.56499959070420e-03, 1.50596691338811e-01, 5.07011860918153e-03, 1.82283280234210e-01),
    ],
    2: [
        (5.94668128021078e-02, 5.03118138150710e-03, 1.11976893703485e-01, 3.45081722137976e-03, 1.09578472598341e-01, 1.98500195310475e-03, 1.37937372401348e-01),
        (4.57064057946856e-02, 2.79570988381096e-03, 8.31564402901790e-02, 1.91357242453786e-03, 7.93007780808741e-02, 1.02341757316565e-03, 1.00544916658365e-01),
        (2.89024268968986e-02, 1.05101532187543e-03, 5.33280100029322e-02, 7.59686168984835e-04, 5.09997730291503e-02, 4.48747321633007e-04, 6.43219227933629e-02),
    ],
}
KEYS = ("E", "L2", "Vh", "L2_theta", "Vh_theta", "L2_phi", "Vh_phi")


class Smooth(Problem):
    """Manufactured problem on the unit square (source derived); optional advection."""

    name = "smooth"
    dim = 2

    def __init__(self, beta=None):
        self._beta = beta

    def exact(self, x):
        return ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    def beta(self, x):
        return None if self._beta is None else ufl.as_vector(self._beta)

    def initial_mesh(self, n, comm=None):
        return meshes.crossed_square(n, 0.0, 1.0, comm)


@pytest.mark.parametrize("p", [1, 2])
def test_paper_preset_reproduces_previous_numbers(p):
    """Default preset = behaviour before the presets existed (same rows to 11 digits)."""
    loop = AdaptiveStationaryLoop(ReentrantCorner(), LoopSettings(p=p, max_levels=3, reconstruct=True))
    hist = loop.run(progress=None)
    for row, expected in zip(hist, BASELINE[p], strict=True):
        for key, value in zip(KEYS, expected, strict=True):
            assert row[key] == pytest.approx(value, rel=1e-11)
        assert "L2_proj" in row and "Vh_phi_proj" in row  # both targets are reported by default


@pytest.mark.parametrize("p", [1, 2])
def test_targets_agree_up_to_best_approximation_error(p):
    problem = Smooth()
    domain = problem.initial_mesh(4)
    res = RMSolver(problem, p).solve(domain)
    errs = error_targets(res, problem, p, degree=12)
    exact, proj = errs["exact"]["ubar"], errs["projection"]["ubar"]
    hq = DGForms(domain, problem, p, degree=12)
    best = fem.assemble_scalar(fem.form((problem.exact(hq.x) - project_exact(hq, problem, res.r.function_space)) ** 2 * hq.dx))
    best = float(np.sqrt(best))  # ||u - P u||_0
    assert proj[0] <= exact[0] * (1 + 1e-12)  # projection target error <= exact target error (L2)
    assert exact[0] ** 2 == pytest.approx(proj[0] ** 2 + best**2, rel=1e-8)  # Pythagoras
    assert exact[0] - proj[0] <= best * (1 + 1e-8)


def test_targets_selection_and_unknown_target():
    problem = Smooth()
    res = RMSolver(problem, 1).solve(problem.initial_mesh(2))
    assert list(error_targets(res, problem, 1, targets=("exact",))) == ["exact"]
    with pytest.raises(ValueError):
        error_targets(res, problem, 1, targets=("nope",))
    hist = AdaptiveStationaryLoop(problem, LoopSettings(p=1, max_levels=1, initial_mesh_size=2,
                                                        error_targets="exact")).run(progress=None)
    assert "L2_proj" not in hist[0]


DOMAIN = Smooth().initial_mesh(4)


def _forms(preset, problem=None):
    return DGForms(DOMAIN, problem or Smooth(), 1, preset=preset)


def _dg0_jump_function(forms):
    space = fem.functionspace(forms.domain, ("DG", 0))
    f = fem.Function(space)
    f.x.array[:] = np.arange(f.x.array.size) % 3.0  # jumps across interior facets, zero gradient
    return f


def test_gram_l2_switch_removes_exactly_the_l2_term():
    forms_p, forms_l = _forms("paper"), _forms(resolve_preset("legacy", gamma_factor_gram=1.0))
    space = fem.functionspace(forms_p.domain, ("DG", 1))
    f = fem.Function(space)
    f.interpolate(lambda x: 1 + x[0] + 2 * x[1] ** 2)
    gp = fem.assemble_scalar(fem.form(forms_p.gram(f, f)))
    gl = fem.assemble_scalar(fem.form(forms_l.gram(f, f)))
    l2 = fem.assemble_scalar(fem.form(f * f * forms_p.dx))
    assert gp - gl == pytest.approx(l2, rel=1e-10)
    # with advection the L2 term (legacy beta_inf = 1) is present in both presets
    adv = Smooth((1.0, 0.5))
    ap = fem.assemble_scalar(fem.form(_forms("paper", adv).gram(f, f)))
    al = fem.assemble_scalar(fem.form(_forms(resolve_preset("legacy", gamma_factor_gram=1.0), adv).gram(f, f)))
    assert ap == pytest.approx(al, rel=1e-12)


@pytest.mark.parametrize("which", ["gamma_factor_b", "gamma_factor_gram"])
def test_gamma_factor_scales_the_interior_penalty_only(which):
    def value(factor):
        forms = _forms(resolve_preset("paper", **{which: factor}))
        f = _dg0_jump_function(forms)
        form = forms.b(f, f) if which == "gamma_factor_b" else forms.gram(f, f)
        return fem.assemble_scalar(fem.form(form))

    full, half, none = value(1.0), value(0.5), value(0.0)
    assert full > none
    assert half == pytest.approx(0.5 * (full + none), rel=1e-12)  # linear in the factor


def test_overriding_every_legacy_field_gives_the_paper_numbers():
    undo = dict(gram_l2=True, gamma_factor_b=1.0, gamma_factor_gram=1.0, quadrature_degree=None,
                marking="dorfler", marking_ratio=None, adjoint_ordering=None)
    kw = dict(p=1, max_levels=3)
    paper = AdaptiveStationaryLoop(ReentrantCorner(), LoopSettings(**kw)).run(progress=None)
    legacy = AdaptiveStationaryLoop(ReentrantCorner(), LoopSettings(preset="legacy", **kw)).run(progress=None)
    undone = AdaptiveStationaryLoop(ReentrantCorner(), LoopSettings(preset="legacy", preset_overrides=undo,
                                                                    **kw)).run(progress=None)
    assert [r["E"] for r in undone] == pytest.approx([r["E"] for r in paper], rel=1e-12)
    assert [r["ncells"] for r in undone] == [r["ncells"] for r in paper]
    assert legacy[0]["E"] != pytest.approx(paper[0]["E"], rel=1e-3)  # the legacy Gram changes the estimator


def test_legacy_loop_uses_fixed_fraction_marking():
    problem = ReentrantCorner()
    loop = AdaptiveStationaryLoop(problem, LoopSettings(p=1, max_levels=2, preset="legacy"))
    hist = loop.run(progress=None)
    preset = resolve_preset("legacy")
    domain = problem.initial_mesh(4)
    res = RMSolver(problem, 1, quadrature_degree=8, preset=preset).solve(domain)
    refined = refine_marked(domain, fixed_fraction_bands(res.indicators, 0.25))[0]
    assert hist[1]["ncells"] == refined.topology.index_map(2).size_local


def test_adjoint_ordering_follows_preset_and_matters_with_advection():
    kw = dict(p=1, max_levels=1, initial_mesh_size=4, reconstruct=True, refinement="uniform")
    prob = Smooth((1.0, 0.5))
    paper = AdaptiveStationaryLoop(prob, LoopSettings(**kw)).run(progress=None)[0]
    via_setting = AdaptiveStationaryLoop(prob, LoopSettings(adjoint_ordering="legacy", **kw)).run(progress=None)[0]
    via_preset = AdaptiveStationaryLoop(
        prob, LoopSettings(preset_overrides={"adjoint_ordering": "legacy"}, **kw)).run(progress=None)[0]
    assert via_preset["Vh_phi"] == pytest.approx(via_setting["Vh_phi"], rel=1e-12)
    assert via_setting["Vh_phi"] != pytest.approx(paper["Vh_phi"], rel=1e-6)
    assert via_setting["Vh_theta"] == pytest.approx(paper["Vh_theta"], rel=1e-12)  # theta does not depend on it


def test_error_forms_inherit_the_preset():
    problem = Smooth()
    domain = problem.initial_mesh(4)
    legacy = RMSolver(problem, 1, preset="legacy").solve(domain)
    assert legacy.forms.preset.name == "legacy"
    vms = reconstruct(legacy, adjoint_ordering="legacy")
    paper_errs = error_targets(RMSolver(problem, 1).solve(domain), problem, 1, degree=8)
    legacy_errs = error_targets(legacy, problem, 1, degree=8, vms=vms)
    assert legacy_errs["exact"]["ubar"][1] != pytest.approx(paper_errs["exact"]["ubar"][1], rel=1e-6)
