import hashlib
import json
from pathlib import Path
import sys
import unittest

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'utils'))
from benchmark_report_protocol import verified_protocol
from benchmark_frozen_protocol import load_protocol


class ReportHandoffTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2]
        self.config = yaml.safe_load((root/'lerobot/configs/benchmark/act_ensemble.yaml').read_text())
        self.spec = load_protocol()
        self.labels = (root/self.spec['frozen_inputs']['onsets']['path']).read_bytes()
        self.report = {'labels_sha256': hashlib.sha256(self.labels).hexdigest(), 'alpha': .05,
                       'protocols': {k: self.config[k] for k in ('paper_protocol','latency_protocol')}}
        self.episodes = [{'outcome':'failure', 'detection_definition':'at_or_after_onset',
                          'primary_annotator':self.spec['detection']['primary_annotator']}]
        self.latencies = [{'onset_raw_frame':'10', 'evaluation_protocol':self.spec['protocol_id']}]

    def verify(self):
        return verified_protocol(self.report,self.config,self.episodes,self.latencies,self.labels)

    def test_missing_top_level_metadata_can_be_recovered_from_verified_rows(self):
        before=json.dumps(self.report,sort_keys=True)
        self.assertEqual(self.verify(),self.spec)
        self.assertEqual(json.dumps(self.report,sort_keys=True),before)

    def test_historical_anytime_results_are_rejected(self):
        self.episodes[0].pop('detection_definition')
        with self.assertRaises(ValueError): self.verify()

    def test_changed_labels_are_rejected(self):
        self.labels += b'\n'
        with self.assertRaises(ValueError): self.verify()

    def test_historical_latency_is_rejected(self):
        self.latencies[0].pop('evaluation_protocol')
        with self.assertRaises(ValueError): self.verify()

    def test_different_protocol_is_not_silently_replaced(self):
        self.report['evaluation_protocol']={'protocol_id':'other'}
        with self.assertRaises(ValueError): self.verify()

    def test_inconsistent_report_configuration_is_rejected(self):
        self.report['alpha']=.1
        with self.assertRaises(ValueError): self.verify()


if __name__=='__main__': unittest.main()
