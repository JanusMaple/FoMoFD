"""Importing replay results must not replace the frozen offline experiment."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'utils'))
from results_inspector import refresh_online_records


class OnlineRefreshTests(unittest.TestCase):
    def test_changed_offline_variants_require_full_build(self):
        data = {'runs':[{'id':'main'}], 'online':['preserved']}
        original = deepcopy(data)
        with patch('results_inspector.build_data', return_value={'runs':[{'id':'new'}]}):
            with self.assertRaisesRegex(ValueError, 'Offline variants changed'):
                refresh_online_records(data, None, None, None)
        self.assertEqual(data, original)

    def test_online_refresh_preserves_displayed_offline_scores_and_studies(self):
        protected = dict(episodes={'current':'post_onset'}, latencies={'current':'mean_onset'},
                         studies=[{'saved_splits':[1,2,3]}], settings=['current'],
                         runs=[{'id':'main'}], vlm_runs=[{'historical_status':'preserved'}])
        data = dict(deepcopy(protected), annotators={}, onset_analysis={}, archives=[], online=['old'])
        fresh = dict(runs=[{'id':'main'}], episodes={'historical':'wrong_for_current_report'},
                     online=[], online_runs=[], plans=[], online_coverage={'status':'complete'},
                     historical_online_runs=[], archives=[])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            labels = root/'labels.yaml'; labels.write_text('annotators: {}\n')
            spec = {'frozen_inputs':{'onsets':{'path':str(labels),
                     'sha256':hashlib.sha256(labels.read_bytes()).hexdigest()}}}
            with patch('results_inspector.build_data', return_value=fresh), \
                 patch('benchmark_consensus_onsets.derive_labels', return_value={'annotators':{}}), \
                 patch('benchmark_frozen_protocol.update_recorded_online') as update, \
                 patch('benchmark_frozen_protocol.export_interpreted', return_value={'new':'online'}):
                refresh_online_records(data, None, root, spec)
                self.assertEqual(update.call_args.args[0]['vlm_runs'], [])
        for key, value in protected.items():
            self.assertEqual(data[key], value)
        self.assertEqual(data['online'], [])
        self.assertEqual(data['interpreted_results'], {'new':'online'})


if __name__ == '__main__':
    unittest.main()
