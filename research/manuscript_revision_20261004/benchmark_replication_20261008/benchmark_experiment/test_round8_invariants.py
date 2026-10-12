"""Regression checks for the corrected comparator and retained-prediction analysis."""
import unittest

import numpy as np
import pandas as pd

from make_tables import AFF, TARGET_ONLY, with_best
from nnw_cpu_weights import nnw_weights
from summarise_window_sensitivity import best_eight_bootstrap, crossed_draws, self_test


class ComparatorTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame([
            {"task": "yrd", "protocol": "M3", "k": 5, "split": 0,
             "model": name, "rmse": 2.0 if name == "GBR (target-only)" else 9.0}
            for name in TARGET_ONLY
        ] + [{"task": "yrd", "protocol": "M3", "k": 5, "split": 0,
              "model": "GBR[LDF] (target-only)", "rmse": 1.0}])

    def test_all_eight_and_no_source_features(self):
        self.assertEqual(len(set(TARGET_ONLY)), 8)
        self.assertIn("RF (target-only)", TARGET_ONLY)
        self.assertIn("GBR (target-only)", TARGET_ONLY)
        self.assertNotIn("GBR[LDF] (target-only)", TARGET_ONLY)

    def test_benchmark_gbr_can_be_posthoc_best(self):
        result = with_best(self.fixture())
        self.assertEqual(result.loc[result.model.eq("Best local"), "rmse"].item(), 2.0)

    def test_missing_comparator_fails_instead_of_weakening_reference(self):
        rows = self.fixture()
        with self.assertRaises(ValueError):
            with_best(rows[rows.model.ne("GBR (target-only)")])

    def test_selection_is_per_split(self):
        first = self.fixture()
        second = first.copy()
        second["split"] = 1
        second.loc[second.model.eq("Local ridge"), "rmse"] = .5
        result = with_best(pd.concat([first, second], ignore_index=True))
        np.testing.assert_array_equal(
            result[result.model.eq("Best local")].sort_values("split").rmse, [2., .5])


class PredictionAnalysisTests(unittest.TestCase):
    def test_neighbour_counts_and_weight_total(self):
        source = np.arange(10, dtype=float)[:, None]
        target = np.array([[0.], [9.]])
        np.testing.assert_array_equal(nnw_weights(source, target), [1, 1, 1, 1, 2, 2, 1, 1, 1, 1])

    def test_shared_bootstrap_matches_direct_rows(self):
        self_test()

    def test_best_eight_is_reselected_within_each_draw(self):
        base = pd.DataFrame({"eFUA_ID": [1, 1, 2, 2], "quarter_id": [1, 2, 1, 2],
                             "observed_pm25": [1., 2., 3., 4.]})
        base[AFF] = base.observed_pm25 + 1.15
        for i, model in enumerate(TARGET_ONLY):
            error = ([.1, 3., .1, 3.] if i == 0 else
                     [3., .1, 3., .1] if i == 1 else [2.5] * 4)
            base[model] = base.observed_pm25 + error
        frames = [base, base.iloc[::-1].copy()]
        got = best_eight_bootstrap(frames, n=100, seed=402)
        cities, quarters, cw, qw = crossed_draws(frames, n=100, seed=402)
        direct, choices = [], set()
        for draw in range(100):
            gains = []
            for d in frames:
                weights = np.array([cw[draw, cities.index(c)] * qw[draw, quarters.index(q)]
                                    for c, q in zip(d.eFUA_ID, d.quarter_id)])
                sse = {m: np.sum(weights * (d[m] - d.observed_pm25) ** 2)
                       for m in TARGET_ONLY + [AFF]}
                choice = min(TARGET_ONLY, key=sse.get)
                choices.add(choice)
                gains.append(100 * (1 - np.sqrt(sse[AFF] / sse[choice])))
            direct.append(np.mean(gains))
        self.assertGreater(len(choices), 1)
        np.testing.assert_allclose([got["ci_low_pct"], got["ci_high_pct"]],
                                   np.quantile(direct, [.025, .975]), atol=1e-10)


if __name__ == "__main__":
    unittest.main()
