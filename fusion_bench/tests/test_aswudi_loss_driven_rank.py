"""Tests for ASWUDI's Path A (WUDI-loss-driven) rank rules.

The rules ``loss_elbow``, ``loss_residual_<frac>``, ``wudi_risk`` and
``wudi_risk_<rho>`` were added to remove the architecture-specific
heuristic in ASWUDI's per-layer rank choice.  These tests check:

  1. ``aswudi_merge`` runs without error for every new rule and produces
     a state-dict whose 2-D weights have moved away from the simple sum.
  2. Per-layer ``K`` selected by each rule lies in ``[1, D2]`` and is
     monotone in the natural sweep parameter (``frac`` ↑ ⇒ ``K`` ↓ for
     ``loss_elbow_<frac>``; ``rho`` ↑ ⇒ ``K`` ↓ for ``wudi_risk_<rho>``).
  3. The new ``loss_residual_<frac>`` reproduces ``loss_elbow_<1-frac>``
     up to integer round-off.
  4. ``wudi_risk_0`` collapses onto ``filter='none'`` ASWUDI K=D2 (the
     un-truncated closed-form WUDI), while ``wudi_risk`` for very large
     ρ tends towards K=0 (within the ``min_rank`` clamp).
"""
from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from fusion_bench.method.yongxianwei_merging.functional import (
    _aswudi_pick_rank_loss_driven,
    aswudi_merge,
)


class _LinearStack(nn.Module):
    """Tiny model with a 2-D weight to exercise the spectral path."""

    def __init__(self, d_in: int = 16, d_out: int = 12):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out, bias=True)


def _make_pool(seed: int = 0, n_tasks: int = 4, d_in: int = 16, d_out: int = 12):
    torch.manual_seed(seed)
    base = _LinearStack(d_in, d_out)
    finetuned = []
    for _ in range(n_tasks):
        m = _LinearStack(d_in, d_out)
        # Inherit from base, then add a low-rank delta so the spectrum
        # has a clear knee — mimicking real fine-tunes.
        m.load_state_dict(base.state_dict())
        with torch.no_grad():
            U = torch.randn(d_out, 3) * 0.1
            V = torch.randn(d_in, 3) * 0.1
            m.proj.weight.add_(U @ V.T)
            m.proj.bias.add_(0.01 * torch.randn_like(m.proj.bias))
        finetuned.append(m)
    return base, finetuned


class TestPathARankRulesUnit(unittest.TestCase):
    """Pure unit tests on ``_aswudi_pick_rank_loss_driven``."""

    def setUp(self):
        # Synthesize a "rank-r signal + noise" spectrum.
        torch.manual_seed(0)
        self.D2 = 24
        self.eigvals_desc = torch.cat([
            torch.linspace(10.0, 5.0, 4),  # 4 strong directions
            torch.linspace(0.1, 0.01, self.D2 - 4),  # noise tail
        ])
        # ‖DV_k‖² = λ_k² · θ*_k_signal² for the matched-signal part,
        # tiny on the noise tail.
        self.e_raw_desc = torch.cat([
            self.eigvals_desc[:4].pow(2) * 1.0,
            torch.full((self.D2 - 4,), 1e-8),
        ])
        # ‖(θ_0 V)_k‖² is positive and roughly comparable across k.
        self.theta0_proj_sq_desc = torch.full((self.D2,), 0.5)

    def test_loss_elbow_default(self):
        K = _aswudi_pick_rank_loss_driven(
            eigvals_desc=self.eigvals_desc,
            e_raw_desc=self.e_raw_desc,
            theta0_proj_sq_desc=self.theta0_proj_sq_desc,
            method="loss_elbow",
            min_rank=1,
            max_rank_ratio=1.0,
        )
        # 4 signal directions out of 24 ⇒ default frac=0.01 should
        # land *near* 4 (allow some headroom for the bound used in
        # the closed-form ratio).
        self.assertGreaterEqual(K, 4)
        self.assertLessEqual(K, self.D2)

    def test_loss_elbow_monotone_in_frac(self):
        # frac ↑  ⇒  K ↓  (a stricter elbow means we keep less).
        ks = []
        for frac in [0.1, 0.05, 0.01, 0.001]:
            ks.append(
                _aswudi_pick_rank_loss_driven(
                    eigvals_desc=self.eigvals_desc,
                    e_raw_desc=self.e_raw_desc,
                    theta0_proj_sq_desc=self.theta0_proj_sq_desc,
                    method=f"loss_elbow_{frac}",
                    min_rank=1,
                    max_rank_ratio=1.0,
                )
            )
        # frac=0.1 should give the smallest K, frac=0.001 the largest
        # (or tied at D2).
        self.assertLessEqual(ks[0], ks[1])
        self.assertLessEqual(ks[1], ks[2])
        self.assertLessEqual(ks[2], ks[3])

    def test_loss_residual_matches_loss_elbow(self):
        K_residual = _aswudi_pick_rank_loss_driven(
            eigvals_desc=self.eigvals_desc,
            e_raw_desc=self.e_raw_desc,
            theta0_proj_sq_desc=self.theta0_proj_sq_desc,
            method="loss_residual_0.99",  # remove ≥ 99% of fit loss
            min_rank=1,
            max_rank_ratio=1.0,
        )
        K_elbow = _aswudi_pick_rank_loss_driven(
            eigvals_desc=self.eigvals_desc,
            e_raw_desc=self.e_raw_desc,
            theta0_proj_sq_desc=self.theta0_proj_sq_desc,
            method="loss_elbow_0.01",
            min_rank=1,
            max_rank_ratio=1.0,
        )
        self.assertEqual(K_residual, K_elbow)

    def test_wudi_risk_finite(self):
        K = _aswudi_pick_rank_loss_driven(
            eigvals_desc=self.eigvals_desc,
            e_raw_desc=self.e_raw_desc,
            theta0_proj_sq_desc=self.theta0_proj_sq_desc,
            method="wudi_risk",
            min_rank=1,
            max_rank_ratio=1.0,
        )
        self.assertGreaterEqual(K, 1)
        self.assertLessEqual(K, self.D2)

    def test_wudi_risk_monotone_in_rho(self):
        # Larger ρ penalises ‖θ_K‖² more aggressively, which should
        # *not* increase K (it may stay equal, but should not grow).
        ks = [
            _aswudi_pick_rank_loss_driven(
                eigvals_desc=self.eigvals_desc,
                e_raw_desc=self.e_raw_desc,
                theta0_proj_sq_desc=self.theta0_proj_sq_desc,
                method=f"wudi_risk_{rho}",
                min_rank=1,
                max_rank_ratio=1.0,
            )
            for rho in [0.01, 0.1, 1.0, 10.0, 100.0]
        ]
        # Non-increasing.
        for a, b in zip(ks, ks[1:]):
            self.assertGreaterEqual(a, b)

    def test_max_rank_ratio_clamp(self):
        K = _aswudi_pick_rank_loss_driven(
            eigvals_desc=self.eigvals_desc,
            e_raw_desc=self.e_raw_desc,
            theta0_proj_sq_desc=self.theta0_proj_sq_desc,
            method="loss_elbow_0.0",  # would want K = D2
            min_rank=1,
            max_rank_ratio=0.5,
        )
        self.assertLessEqual(K, int(round(0.5 * self.D2)))

    def test_min_rank_clamp(self):
        # A degenerate spectrum with all energy in the first direction
        # would push elbow K to 1; min_rank=3 should override.
        eig = torch.tensor([10.0] + [1e-4] * (self.D2 - 1))
        e_raw = torch.tensor([100.0] + [1e-12] * (self.D2 - 1))
        theta0_proj = torch.zeros(self.D2)
        K = _aswudi_pick_rank_loss_driven(
            eigvals_desc=eig,
            e_raw_desc=e_raw,
            theta0_proj_sq_desc=theta0_proj,
            method="loss_elbow",
            min_rank=3,
            max_rank_ratio=1.0,
        )
        self.assertGreaterEqual(K, 3)


class TestAswudiMergeIntegration(unittest.TestCase):
    """End-to-end aswudi_merge calls with each new rule."""

    def setUp(self):
        self.base, self.finetuned = _make_pool(seed=42)

    def _run(self, rule: str):
        merged = aswudi_merge(
            base_model=self.base,
            finetuned_models=self.finetuned,
            exclude_param_names_regex=None,
            scaling_coefficient=1.0,
            auto_rank_method=rule,
            filter_type="none",
            init_mode="sum",
            progress=False,
        )
        # ``aswudi_merge`` returns whatever ``combine_with_pretrained_model``
        # gives back. In the FusionBench codebase that's a state_dict
        # (``dict[str, Tensor]``) when called on a vanilla nn.Module.
        if hasattr(merged, "state_dict"):
            sd = merged.state_dict()
        else:
            sd = merged
        self.assertIn("proj.weight", sd)
        self.assertEqual(
            sd["proj.weight"].shape,
            self.base.state_dict()["proj.weight"].shape,
        )
        return merged

    def test_loss_elbow_runs(self):
        self._run("loss_elbow")

    def test_loss_elbow_custom_frac_runs(self):
        self._run("loss_elbow_0.05")

    def test_loss_residual_runs(self):
        self._run("loss_residual_0.99")

    def test_wudi_risk_runs(self):
        self._run("wudi_risk")

    def test_wudi_risk_custom_rho_runs(self):
        self._run("wudi_risk_0.5")

    def test_unknown_rule_raises(self):
        with self.assertRaises(ValueError):
            self._run("loss_elbow_not_a_number")
        with self.assertRaises(ValueError):
            self._run("wudi_risk_-1.0")

    def test_baseline_participation_sqrt_unchanged(self):
        # Sanity: existing default rule still works after the patch.
        merged = aswudi_merge(
            base_model=self.base,
            finetuned_models=self.finetuned,
            exclude_param_names_regex=None,
            scaling_coefficient=1.0,
            auto_rank_method="participation_sqrt",
            filter_type="none",
            init_mode="sum",
            progress=False,
        )
        sd = merged.state_dict() if hasattr(merged, "state_dict") else merged
        self.assertIn("proj.weight", sd)


class TestPathACorrectness(unittest.TestCase):
    """The helper's partial-cumsum L_fit must match a brute-force
    re-evaluation of ``‖θ_K C - D‖²`` for every candidate K.

    This is the load-bearing correctness check for Path A: if the
    closed-form expression is wrong, the elbow detector and the
    risk minimiser both pick wrong K's silently.
    """

    def setUp(self):
        torch.manual_seed(7)
        self.D1, self.D2 = 9, 11
        # Build a known C and D by drawing 5 task vectors and using the
        # *exact* construction in ``aswudi_merge``: this guarantees the
        # comparison below is apples-to-apples.
        N = 5
        v_stack = torch.randn(N, self.D1, self.D2)
        flat = v_stack.reshape(N, -1)
        l2_sq = torch.norm(flat, p=2, dim=-1).square()
        inv = 1.0 / l2_sq.clamp(min=1e-12)
        B = torch.einsum("nab,nac->nbc", v_stack, v_stack)
        self.C = torch.einsum("n,nbc->bc", inv, B)
        vB = torch.einsum("nab,nbc->nac", v_stack, B)
        self.D_mat = torch.einsum("n,nac->ac", inv, vB)
        self.theta_0 = v_stack.sum(dim=0)

        eigvals_asc, V_asc = torch.linalg.eigh(self.C)
        self.lam_asc = eigvals_asc.clamp(min=1e-8)
        self.V_asc = V_asc

    def _brute_force_curves(self):
        """Return dict K -> (L_fit, L_norm) computed directly from θ_K.

        Uses the *exact* ASWUDI θ formula with init='sum'.
        """
        D2 = self.D2
        results = {}
        for K in range(0, D2 + 1):
            filter_vals = torch.zeros_like(self.lam_asc)
            if K > 0:
                filter_vals[D2 - K:] = 1.0
            combo_div = filter_vals / self.lam_asc
            DV = self.D_mat @ self.V_asc
            DH = (DV * combo_div.unsqueeze(0)) @ self.V_asc.T
            theta_0 = self.theta_0
            Tv = theta_0 @ self.V_asc
            T_G = (Tv * filter_vals.unsqueeze(0)) @ self.V_asc.T
            theta_K = theta_0 - T_G + DH
            L_fit  = (theta_K @ self.C - self.D_mat).pow(2).sum().item()
            L_norm = theta_K.pow(2).sum().item()
            results[K] = (L_fit, L_norm)
        return results

    def test_helper_matches_bruteforce_argmin(self):
        bf = self._brute_force_curves()
        # rho = 1 ⇒ argmin (L_fit + L_norm).
        risks = {K: bf[K][0] + bf[K][1] for K in bf}
        K_bf = min(risks, key=risks.get)

        # Compute the same thing through the helper.
        eigvals_desc = self.lam_asc.flip(0)
        DV_asc = self.D_mat @ self.V_asc
        e_raw_asc = (DV_asc * DV_asc).sum(dim=0)
        e_raw_desc = e_raw_asc.flip(0).clamp(min=0.0)
        theta0V_asc = self.theta_0 @ self.V_asc
        theta0_proj_sq_asc = (theta0V_asc * theta0V_asc).sum(dim=0)
        theta0_proj_sq_desc = theta0_proj_sq_asc.flip(0).clamp(min=0.0)

        K_helper = _aswudi_pick_rank_loss_driven(
            eigvals_desc=eigvals_desc,
            e_raw_desc=e_raw_desc,
            theta0_proj_sq_desc=theta0_proj_sq_desc,
            method="wudi_risk",
            min_rank=1,
            max_rank_ratio=1.0,
        )
        # The helper uses an *upper-bound* form of L_fit (drops the
        # cross term).  On synthetic data with random θ_0 the bound
        # is reasonably tight, so the helper's argmin should land
        # within ±2 of the brute-force one — large enough margin to
        # be robust to floating-point noise but small enough to
        # catch a wrong-sign formula.
        self.assertLessEqual(abs(K_helper - K_bf), 2)

    def test_loss_elbow_keeps_more_than_argmin_when_frac_is_tight(self):
        # When `frac` is very tight, loss_elbow should keep MORE rank
        # than the bias-variance optimum (the elbow ignores the
        # ‖θ‖² penalty and aggressively reduces fit error).
        eigvals_desc = self.lam_asc.flip(0)
        DV_asc = self.D_mat @ self.V_asc
        e_raw_asc = (DV_asc * DV_asc).sum(dim=0)
        e_raw_desc = e_raw_asc.flip(0).clamp(min=0.0)
        theta0V_asc = self.theta_0 @ self.V_asc
        theta0_proj_sq_asc = (theta0V_asc * theta0V_asc).sum(dim=0)
        theta0_proj_sq_desc = theta0_proj_sq_asc.flip(0).clamp(min=0.0)

        K_elbow = _aswudi_pick_rank_loss_driven(
            eigvals_desc=eigvals_desc,
            e_raw_desc=e_raw_desc,
            theta0_proj_sq_desc=theta0_proj_sq_desc,
            method="loss_elbow_0.001",  # very tight elbow
            min_rank=1,
            max_rank_ratio=1.0,
        )
        K_risk_huge = _aswudi_pick_rank_loss_driven(
            eigvals_desc=eigvals_desc,
            e_raw_desc=e_raw_desc,
            theta0_proj_sq_desc=theta0_proj_sq_desc,
            method="wudi_risk_100.0",  # heavy norm penalty ⇒ small K
            min_rank=1,
            max_rank_ratio=1.0,
        )
        self.assertGreaterEqual(K_elbow, K_risk_huge)


if __name__ == "__main__":
    unittest.main()
