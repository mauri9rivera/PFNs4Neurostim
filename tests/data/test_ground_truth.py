"""Tests for split-half ground truth (task #7, P0.7)."""
from __future__ import annotations

import os
import textwrap

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from pfns4neurostim.data.channels import ChannelData
from pfns4neurostim.data.ground_truth import (
    build_ground_truth,
    draw_split_masks,
    ground_truth_instances,
    instance_for_rep,
    split_half_instances,
    split_half_reliability,
)


def _channel(n_sites: int = 20, n_trials: int = 7, noise: float = 0.3, seed: int = 0) -> ChannelData:
    """Synthetic channel in raw units with a recorded StandardScaler, some NaN trials."""
    rng = np.random.default_rng(seed)
    coords = np.stack(np.meshgrid(np.arange(5), np.arange(n_sites // 5)), axis=-1).reshape(-1, 2)
    x = coords / coords.max(axis=0)                                           # [N, 2]
    signal = 3.0 + 2.0 * np.exp(-((x[:, 0] - 0.7) ** 2 + (x[:, 1] - 0.3) ** 2) / 0.1)  # [N]
    raw = signal[:, None] + noise * rng.normal(size=(n_sites, n_trials))      # [N, R]
    raw[2, 1] = np.nan                                                        # a flagged trial
    raw[5, :] = np.nan
    raw[5, 0] = 4.0                                                           # a 1-trial site
    scaler = StandardScaler().fit(raw[np.isfinite(raw)].reshape(-1, 1))
    Y = scaler.transform(np.nan_to_num(raw).reshape(-1, 1)).reshape(raw.shape)
    Y[~np.isfinite(raw)] = np.nan
    y_gt = np.nanmean(Y, axis=1)
    return ChannelData(
        "nhp", 1, 0, x, Y, y_gt, coords, (5, n_sites // 5), meta={"scaler_y": scaler}
    )


class TestMasks:
    def test_halves_disjoint_and_cover_valid_trials(self, rng: np.random.Generator) -> None:
        ch = _channel()
        masks = draw_split_masks(ch.Y_trials, rng)
        assert not (masks.half_a & masks.half_b).any()
        valid = np.isfinite(ch.Y_trials)
        kept = masks.kept_sites
        np.testing.assert_array_equal((masks.half_a | masks.half_b)[kept], valid[kept])
        assert not (masks.half_a | masks.half_b)[~kept].any()

    def test_odd_count_extra_trial_goes_to_observation_half(self, rng: np.random.Generator) -> None:
        ch = _channel(n_trials=7)
        masks = draw_split_masks(ch.Y_trials, rng)
        n_valid = np.isfinite(ch.Y_trials).sum(axis=1)
        kept = masks.kept_sites
        np.testing.assert_array_equal(masks.half_a.sum(axis=1)[kept], n_valid[kept] // 2)
        np.testing.assert_array_equal(masks.half_b.sum(axis=1)[kept], n_valid[kept] - n_valid[kept] // 2)
        # 7 valid trials -> B gets 4, A gets 3; the site with a flagged trial has 6 -> 3/3.
        assert masks.half_b[0].sum() == 4 and masks.half_a[0].sum() == 3
        assert masks.half_b[2].sum() == 3 and masks.half_a[2].sum() == 3

    def test_sites_with_fewer_than_two_trials_are_dropped(self, rng: np.random.Generator) -> None:
        masks = draw_split_masks(_channel().Y_trials, rng)
        assert not masks.kept_sites[5]
        assert masks.kept_sites.sum() == 19


class TestArrayLevel:
    def test_full_mean_is_nanmean(self) -> None:
        ch = _channel()
        ((y_gt, Y_obs),) = build_ground_truth(ch.Y_trials, "full_mean")
        np.testing.assert_allclose(y_gt, np.nanmean(ch.Y_trials, axis=1))
        assert Y_obs is ch.Y_trials

    def test_split_half_returns_2r_instances_with_swap_symmetry(self, rng: np.random.Generator) -> None:
        ch = _channel()
        inst = build_ground_truth(ch.Y_trials, "split_half", rng, n_splits=3)
        assert len(inst) == 6
        for k in range(3):
            (gt_a, obs_b), (gt_b, obs_a) = inst[2 * k], inst[2 * k + 1]
            # The swap's observation half is exactly the first instance's GT half.
            np.testing.assert_allclose(np.nanmean(obs_a, axis=1), gt_a)
            np.testing.assert_allclose(np.nanmean(obs_b, axis=1), gt_b)
            # Observation halves are disjoint.
            assert not (np.isfinite(obs_a) & np.isfinite(obs_b)).any()

    def test_split_half_requires_rng(self) -> None:
        with pytest.raises(ValueError, match="rng"):
            build_ground_truth(_channel().Y_trials, "split_half")

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown mode"):
            build_ground_truth(_channel().Y_trials, "bogus")


class TestChannelLevel:
    def test_full_mean_returns_channel_unchanged(self) -> None:
        ch = _channel()
        assert ground_truth_instances(ch, "full_mean") == [ch]

    def test_instances_are_split_half_channels(self, rng: np.random.Generator) -> None:
        ch = _channel()
        inst = split_half_instances(ch, rng, n_splits=2)
        assert [i.split_id for i in inst] == [0, 1, 2, 3]
        for i in inst:
            assert i.gt_mode == "split_half"
            assert i.n_sites == 19                      # the 1-trial site is dropped
            assert i.meta["n_dropped_sites"] == 1
            assert i.X_pool.shape == (19, 2) and i.ch2xy.shape == (19, 2)

    def test_scaler_fitted_on_observation_half_only(self, rng: np.random.Generator) -> None:
        inst = split_half_instances(_channel(), rng, n_splits=1)[0]
        obs = inst.Y_trials[np.isfinite(inst.Y_trials)]
        assert abs(obs.mean()) < 1e-10
        assert abs(obs.std() - 1.0) < 1e-10
        # The GT half is expressed with that same scaler, so it is *not* re-centred.
        assert abs(inst.y_gt.mean()) > 1e-6

    def test_gt_and_observations_disjoint_in_raw_units(self, rng: np.random.Generator) -> None:
        ch = _channel()
        inst = split_half_instances(ch, rng, n_splits=1)
        keep = inst[0].meta["site_mask_split"]
        raw = ch.meta["scaler_y"].inverse_transform(np.nan_to_num(ch.Y_trials).reshape(-1, 1))
        raw = raw.reshape(ch.Y_trials.shape)[keep]
        sc0, sc1 = inst[0].meta["scaler_y"], inst[1].meta["scaler_y"]
        obs0 = np.isfinite(inst[0].Y_trials)
        obs1 = np.isfinite(inst[1].Y_trials)
        assert not (obs0 & obs1).any()
        # GT of instance 0 = raw mean of the trials instance 1 observes.
        gt0_raw = sc0.inverse_transform(inst[0].y_gt.reshape(-1, 1)).ravel()
        np.testing.assert_allclose(gt0_raw, np.where(obs1, raw, 0).sum(1) / obs1.sum(1), rtol=1e-6)
        del sc1

    def test_determinism(self) -> None:
        ch = _channel()
        a = split_half_instances(ch, np.random.default_rng(7), n_splits=2)
        b = split_half_instances(ch, np.random.default_rng(7), n_splits=2)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x.y_gt, y.y_gt)
            np.testing.assert_array_equal(np.isnan(x.Y_trials), np.isnan(y.Y_trials))

    def test_rejects_already_split_channel(self, rng: np.random.Generator) -> None:
        inst = split_half_instances(_channel(), rng, n_splits=1)[0]
        with pytest.raises(ValueError, match="full_mean"):
            split_half_instances(inst, rng)

    def test_instance_for_rep_cycles(self) -> None:
        assert [instance_for_rep(r, 4) for r in range(6)] == [0, 1, 2, 3, 0, 1]


class TestReliability:
    def test_noiseless_trials_give_reliability_one(self, rng: np.random.Generator) -> None:
        ch = _channel(noise=0.0)
        rel = split_half_reliability(ch.Y_trials, rng, n_splits=5)
        assert rel["r_half"] == pytest.approx(1.0)
        assert rel["spearman_brown"] == pytest.approx(1.0)

    def test_noise_lowers_reliability_and_sb_exceeds_half(self, rng: np.random.Generator) -> None:
        rel = split_half_reliability(_channel(noise=2.0).Y_trials, rng, n_splits=10)
        assert rel["r_half"] < 0.99
        assert rel["spearman_brown"] > rel["r_half"]

    def test_affine_invariance(self) -> None:
        Y = _channel(noise=1.0).Y_trials
        a = split_half_reliability(Y, np.random.default_rng(3), n_splits=4)
        b = split_half_reliability(5.0 * Y - 2.0, np.random.default_rng(3), n_splits=4)
        assert a["r_half"] == pytest.approx(b["r_half"])


_DATA = os.path.join("data", "monkeys", "Cebus2_M1_200123.mat")


@pytest.mark.skipif(not os.path.exists(_DATA), reason="raw NHP data not present")
def test_real_channel_full_mean_matches_loader_and_split_runs() -> None:
    """full_mean GT reproduces the loader's respMean; split_half runs on a real channel."""
    from pfns4neurostim.data.channels import load_channel

    ch = load_channel("nhp", 1, 0)
    ((y_gt, _),) = build_ground_truth(ch.Y_trials, "full_mean")
    # The loader mean is float64 over raw trials; Y_trials came through float32.
    np.testing.assert_allclose(y_gt, ch.y_gt, atol=1e-5)
    inst = split_half_instances(ch, np.random.default_rng(0), n_splits=1)
    assert len(inst) == 2 and all(np.isfinite(i.y_gt).all() for i in inst)
    rel = split_half_reliability(ch.Y_trials, np.random.default_rng(0), n_splits=3)
    assert -1.0 < rel["r_half"] <= 1.0


def test_bo_benchmark_split_half_end_to_end(
    tmp_path: "os.PathLike[str]", monkeypatch: pytest.MonkeyPatch, tiny_channel: ChannelData
) -> None:
    """bo_benchmark expands a channel into split instances: rep i -> instance i mod 2R."""
    from pfns4neurostim.experiments import bo_benchmark

    monkeypatch.setattr(bo_benchmark, "iter_channels", lambda *a, **k: iter([tiny_channel]))
    cfg = tmp_path / "exp.yaml"
    cfg.write_text(
        textwrap.dedent(
            f"""
            defaults:
              dataset: nhp
              model: [gp_mll]
              acquisition: ei
            experiment: bo_benchmark
            family: gt-test
            tag: tiny
            budget: 6
            n_init: 3
            n_reps: 3
            gt_mode: split_half
            gt_n_splits: 1
            device: cpu
            output_root: {tmp_path.as_posix()}
            """
        )
    )
    run_dir = bo_benchmark.run_bo_benchmark(str(cfg), use_cache=False)
    df = pd.read_csv(os.path.join(run_dir, "tidy.csv"))
    assert (df["gt_mode"] == "split_half").all()
    assert df.sort_values("rep")["split_id"].tolist() == [0, 1, 0]
    assert df["gt_r_half"].notna().all() and df["gt_reliability"].notna().all()
    assert (df["gt_reliability"] >= df["gt_r_half"]).all()


def test_pair_gt_modes_pairs_channel_means() -> None:
    """The sensitivity pairing averages reps per channel and differences the modes."""
    from pfns4neurostim.experiments.gt_sensitivity import pair_gt_modes

    base = {"dataset": "nhp", "subject": 1, "emg": 0, "model": "gp_mll", "acq_label": "ei"}
    full = pd.DataFrame([{**base, "recommended_regret": v, "r2": 0.9} for v in (0.1, 0.3)])
    split = pd.DataFrame(
        [{**base, "recommended_regret": v, "r2": 0.5, "gt_r_half": 0.6, "gt_reliability": 0.75}
         for v in (0.2, 0.4)]
    )
    out = pair_gt_modes(full, split)
    assert len(out) == 1
    assert out.loc[0, "recommended_regret_full_mean"] == pytest.approx(0.2)
    assert out.loc[0, "recommended_regret_delta"] == pytest.approx(0.1)
    assert out.loc[0, "r2_delta"] == pytest.approx(-0.4)
    assert out.loc[0, "gt_r_half"] == pytest.approx(0.6)
