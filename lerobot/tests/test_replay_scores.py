"""Saved score/threshold replay must not refit calibration or shift raw indices."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
from replay_inspector import CachedScores

class ReplayScoresTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root/'scores.npz'
        np.savez(self.cache, episode_index=np.zeros(6,dtype=int), frame_index=np.arange(7,13),
                 raw_frame_index=np.arange(8,14), wall_timestamp=np.arange(6,dtype=float),
                 scores=np.array([[100,1],[100,7],[100,3],[100,50],[100,50],[100,50]],dtype=float))
        self.report = {'protocols': {'paper_protocol': {'window':1,'smoothing':'centered','wm_frame_policy':'replay_legacy'}},
            'score_caches': [{'task':'task','method':'wm_nc','views':['fix','wrist'],'path':str(self.cache),
                'sha256':hashlib.sha256(self.cache.read_bytes()).hexdigest(),
                'scoring_settings':{'K':4,'L':4,'checkpoint':'/runs/model/checkpoints/best.pt'}}],
            'calibration':[{'task':'task','method':'wm_nc','view':'wrist','calibration':'episode','protocol':'paper',
                            'threshold':5.,'n_calibration':19}]}
        self.save()

    def save(self):
        (self.root/'report.json').write_text(json.dumps(self.report))

    def test_same_raw_frames_view_and_frozen_threshold(self):
        result = CachedScores(self.root).episode('task',0)
        self.assertEqual(result['raw_frames'],[8,9,10])
        self.assertEqual(result['values'],[1.,7.,3.])
        self.assertEqual(result['threshold'],5.)
        self.assertEqual(result['n_calibration'],19)
        self.assertIsNone(CachedScores(self.root).episode('unscored_task',0))

    def test_changed_cache_rejected(self):
        self.cache.write_bytes(self.cache.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'SHA-256'):
            CachedScores(self.root).episode('task',0)

    def test_missing_threshold_not_recalibrated(self):
        self.report['calibration']=[];self.save()
        with self.assertRaisesRegex(ValueError,'threshold'):
            CachedScores(self.root).episode('task',0)

    def test_smoothed_or_joint_report_not_silently_reinterpreted(self):
        self.report['protocols']['paper_protocol']['window']=7;self.save()
        with self.assertRaisesRegex(ValueError,'unsmoothed'):CachedScores(self.root)
        self.report['protocols']['paper_protocol']['window']=1
        self.report['wm_score_aggregation']='joint';self.save()
        with self.assertRaisesRegex(ValueError,'per-view'):CachedScores(self.root)

if __name__=='__main__':unittest.main()
