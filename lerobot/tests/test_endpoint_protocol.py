"""Raw-frame truncation, conformal splits and report option isolation (CPU)."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'utils'))
from benchmark_endpoint_protocol import protocol
from unified_benchmark import curves, validate_protocol, threshold_for
from calibration_sensitivity import evaluate_splits


class EndpointProtocolTests(unittest.TestCase):
    def arrays(self):
        episodes=np.repeat([5,6],10)
        frames=np.tile(np.arange(10),2)
        return dict(episode_index=episodes,frame_index=frames,raw_frame_index=frames+115,
                    wall_timestamp=frames.astype(float),scores=np.column_stack([frames,100+frames]))

    def job(self,task='needle_pickup',wm=True,k=4):
        return SimpleNamespace(task=task,mode=wm,merged={'data':{'history_len':1,'pred_horizon':k}})

    def test_all_endpoints_and_exclusive_raw_cutoff_both_views(self):
        a=self.arrays()
        for view in (0,1):
            full=curves(a,self.job(),view,protocol())
            cut=curves(a,self.job(),view,protocol(True))
            self.assertEqual(full[5]['raw'][-1],124)
            self.assertEqual(cut[5]['raw'][-1],121)
            self.assertEqual(len(cut[5]['score']),7)
            np.testing.assert_array_equal(full[6]['score'],cut[6]['score'])

    def test_cutoff_also_applies_to_observation_detectors_not_other_tasks(self):
        a=self.arrays()
        self.assertEqual(curves(a,self.job(wm=False),0,protocol(True))[5]['raw'][-1],121)
        self.assertEqual(curves(a,self.job(task='ring_over_post'),0,protocol(True))[5]['raw'][-1],124)

    def test_legacy_cutoff_is_relative_to_shortened_episode(self):
        p=protocol(True);p['wm_frame_policy']='replay_legacy'
        self.assertEqual(curves(self.arrays(),self.job(),0,p)[5]['raw'][-1],118)

    def test_invalid_or_empty_cutoffs_fail(self):
        p=protocol(True);p['frame_limits']*=2
        with self.assertRaises(ValueError):validate_protocol(p)
        p=protocol(True);p['frame_limits'][0]['raw_frame_stop']=1
        with self.assertRaises(ValueError):curves(self.arrays(),self.job(),0,p)

    def test_cutoff_affects_episode_when_resampled_into_test(self):
        q={i:float(i) for i in range(39)};q[5]=100.;q[39]=50.
        corrected={**q,5:5.}
        # Hold out the suspect episode; correction must affect its test score too.
        cal=[i for i in range(20) if i!=5]
        split={'split_id':0,'calibration_ids':cal,'test_success_ids':[i for i in range(39) if i not in cal]}
        a=evaluate_splits(q,list(range(39)),[39],[split])[0]
        b=evaluate_splits(corrected,list(range(39)),[39],[split])[0]
        self.assertEqual(a['threshold'],b['threshold'])
        self.assertEqual(a['false_alarms'],b['false_alarms']+1)
        self.assertEqual(a['detected_failures'],b['detected_failures'])

    def test_options_are_independent(self):
        a=protocol(True);a['frame_limits'][0]['raw_frame_stop']=100
        self.assertEqual(protocol(True)['frame_limits'][0]['raw_frame_stop'],122)
        self.assertNotIn('frame_limits',protocol())


if __name__=='__main__':unittest.main()
