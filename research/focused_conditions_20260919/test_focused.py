import unittest
import numpy as np
import pandas as pd
from run_focused import rulemask, scope_mask, support, score, REFS, TRANSFER, Y


class FocusedTests(unittest.TestCase):
    def test_boundary_and_missing(self):
        d = pd.DataFrame({'x': [0.5, 1, np.nan, np.inf]})
        self.assertEqual(rulemask(d, [{'feature': 'x', 'op': '<=', 'threshold': 1}]).tolist(), [True, True, False, False])

    def test_scope_does_not_use_outcome(self):
        d = pd.DataFrame({'country_iso': ['CHN', 'AUS', 'AUS'], 'distance_to_coast_km': [20, 20, 151],
                          'fua_centroid_lat': [30, -30, -30], 'fua_centroid_lon': [120, 150, 150], Y: [999, 0, 3]})
        self.assertEqual(scope_mask(d, 'east_window').tolist(), [True, True, False])
        self.assertEqual(scope_mask(d, 'latitude22_35_coastal150').tolist(), [True, True, False])
        d[Y] = -100
        self.assertEqual(scope_mask(d, 'east_window').tolist(), [True, True, False])

    def test_each_year_support(self):
        d = pd.DataFrame([(y, y * 12 + m, c) for y in (2021, 2022) for m in range(1, 13) for c in range(3)],
                         columns=['year', 't', 'eFUA_ID'])
        self.assertTrue(support(d, np.ones(len(d), bool), .5))
        self.assertFalse(support(d, d.year.eq(2021).to_numpy(), .05))
        self.assertFalse(support(d, d.eFUA_ID.eq(0).to_numpy(), .05))

    def test_comparators_use_same_accepted_rows(self):
        d = pd.DataFrame({Y: [1., 2., 3.]})
        for r in REFS:
            d[r] = [2., 3., 1000.]
        d[TRANSFER[0]] = [1.5, 2.5, -1000.]
        z = score(d, TRANSFER[0], np.array([True, True, False]))
        self.assertAlmostEqual(z['gain'], .5)


if __name__ == '__main__':
    unittest.main()
