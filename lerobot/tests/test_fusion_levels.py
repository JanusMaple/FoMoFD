from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
from benchmark_fusion_levels import aligned, combine_peak


class FusionLevelTests(unittest.TestCase):
    def test_and_allows_different_endpoints_only_at_episode_level(self):
        a,b=[2,-1],[-1,3]
        self.assertEqual(combine_peak(a,b,'and','endpoint'),-1)
        self.assertEqual(combine_peak(a,b,'and','episode'),2)

    def test_pre_onset_component_alarm_does_not_validate_post_onset_and(self):
        a,b=np.array([2,-1]),np.array([-1,3])
        self.assertGreater(combine_peak(a,b,'and','episode'),0)
        self.assertLess(combine_peak(a[1:],b[1:],'and','episode'),0)

    def test_or_equivalence_and_and_dominance(self):
        rng=np.random.default_rng(17)
        for _ in range(100):
            a,b=rng.normal(size=(2,20))
            self.assertEqual(combine_peak(a,b,'or','episode'),combine_peak(a,b,'or','endpoint'))
            self.assertGreaterEqual(combine_peak(a,b,'and','episode'),combine_peak(a,b,'and','endpoint'))

    def test_strict_zero_threshold_and_empty_post_onset_support(self):
        self.assertEqual(combine_peak([0],[1],'and','episode'),0)
        self.assertFalse(combine_peak([0],[1],'and','episode')>0)
        self.assertEqual(combine_peak([],[],'or','episode'),-np.inf)

    def test_alignment_joins_raw_frames_and_checks_wall_time(self):
        a={'raw':np.array([1,2,4]),'wall':np.array([.1,.2,.4]),'score':np.array([10,1,2])}
        b={'raw':np.array([2,3,4]),'wall':np.array([.2,.3,.4]),'score':np.array([3,20,4])}
        raw,x,y=aligned(a,b)
        np.testing.assert_array_equal(raw,[2,4])
        self.assertEqual(combine_peak(x,y,'or','episode'),4)
        b['wall'][0]=.3
        with self.assertRaises(ValueError):aligned(a,b)


if __name__=='__main__':unittest.main()
