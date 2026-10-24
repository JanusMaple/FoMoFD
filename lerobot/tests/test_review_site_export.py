"""Clock fidelity and public export boundaries; no models or GPU."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
from export_review_site import encode_video, public_rows, scan_public, recent_summary_tables, style_study_svg, PAIR_FIELDS
from replay_inspector import Episode
from public_results import current_result, select_public_data


class ReviewSiteExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def test_vfr_preserves_irregular_wall_clock_and_paired_views(self):
        import av
        fix = np.zeros((4, 16, 16, 3), dtype=np.uint8)
        wrist = np.full_like(fix, 200)
        times = np.array([0, .107123, .399234, 1.234567])
        episode = Episode('needle_pickup', 5, 'private_source.npz', fix, wrist, times, None)
        target = self.path / 'episode.mp4'
        encode_video(episode, 3, target)
        with av.open(str(target)) as container:
            frames = list(container.decode(video=0))
        np.testing.assert_allclose([f.time for f in frames], times[:3], atol=2e-6, rtol=0)
        rgb = frames[0].to_ndarray(format='rgb24')
        self.assertEqual(rgb.shape, (16, 32, 3))
        self.assertLess(rgb[:, :16].mean(), 10)
        self.assertGreater(rgb[:, 16:].mean(), 180)

    def test_invalid_clocks_are_not_silently_replaced(self):
        rgb = np.zeros((2, 16, 16, 3), dtype=np.uint8)
        e = Episode('needle_pickup', 0, 'private.npz', rgb, rgb, np.array([0, 0]), None)
        with self.assertRaisesRegex(ValueError, 'timestamps'):
            encode_video(e, 2, self.path / 'video.mp4')

    def test_public_columns_are_an_allowlist(self):
        p = self.path / 'input.csv'
        with p.open('w') as f:
            w = csv.DictWriter(f, fieldnames=['FDR', 'alarm_n', 'main_comparison', 'checkpoint', 'annotator'])
            w.writeheader()
            w.writerow(dict(FDR=.75, alarm_n=15, main_comparison=True,
                            checkpoint='/home/person/private.pt', annotator='Collector'))
        self.assertEqual(public_rows(p, ['FDR', 'alarm_n', 'main_comparison']), [
            {'FDR': .75, 'alarm_n': 15, 'main_comparison': True}])

    def test_identity_scan_blocks_known_local_paths(self):
        (self.path / 'data.json').write_text('{"path":"/home/person/checkpoint.pt"}')
        with self.assertRaisesRegex(ValueError, 'Identity/path leak'):
            scan_public(self.path)

    def test_vector_relabel_preserves_plot_geometry_and_shared_glyph_definitions(self):
        # The W glyph is defined inside the replaced label but reused elsewhere.
        # Long replacement ticks need more canvas without moving scientific paths.
        source = '''<svg xmlns="http://www.w3.org/2000/svg"
            xmlns:xlink="http://www.w3.org/1999/xlink" width="200pt" height="100pt"
            viewBox="0 0 200 100"><g id="ytick_1"><g id="text_1"><!-- WM-NC -->
            <g transform="translate(7.2 30) scale(0.1 -0.1)">
            <defs><path id="DejaVuSans-57" d="M0 0 L1 2"/></defs>
            <use xlink:href="#DejaVuSans-57"/></g></g></g>
            <g id="science"><path id="signal" d="M50 70 L80 20 L170 60"/>
            <use xlink:href="#DejaVuSans-57" x="100" y="80"/></g></svg>'''
        root = ET.fromstring(style_study_svg(source))
        ns = {'s': 'http://www.w3.org/2000/svg'}
        self.assertEqual(root.find('.//s:path[@id="signal"]', ns).get('d'),
                         'M50 70 L80 20 L170 60')
        self.assertIsNotNone(root.find('.//s:path[@id="DejaVuSans-57"]', ns))
        self.assertIsNotNone(root.find('.//s:g[@id="science"]/s:use', ns))
        self.assertEqual(root.find('.//s:g[@data-method-label]', ns).get('aria-label'),
                         'WM-NC (ours)')
        left, _, width, _ = map(float, root.get('viewBox').split())
        self.assertLess(left, 0)
        self.assertAlmostEqual(left + width, 200)

    def test_pair_export_preserves_late_inclusive_distinction(self):
        p = self.path/'pairs.csv'
        with p.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=PAIR_FIELDS)
            writer.writeheader()
            for on_time in (True, False):
                writer.writerow(dict(reference='main', comparator='K1', on_time_only=on_time, shared_n=10))
        rows = public_rows(p, PAIR_FIELDS)
        self.assertEqual([r['on_time_only'] for r in rows], [True, False])

    def test_development_tables_are_not_downloaded(self):
        section = {'id':'local_seed23_training','columns':['Task','Seed'],
                   'rows':[['Needle',2]],'records':[{'checkpoint':'/home/person/private.pt'}]}
        summary = {'reviewer_evidence':{'sections':[section, {**section,'id':'unrelated'}]}}
        self.assertEqual(recent_summary_tables(summary), {})

    def test_public_selection_excludes_superseded_models(self):
        for variant in ('wm_k4_l4_seed1_pace_archived', 'wm_k4_l4_seed2',
                        'wm_k4_l4_seed3', 'wm_k4_l4_seed7', 'wm_k4_l4_action_adaln'):
            self.assertFalse(current_result(variant, 'wm_nc'))
        self.assertFalse(current_result('main', 'ace'))
        self.assertFalse(current_result('act_ensemble', 'act_ensemble_rot'))
        for variant, method in (('ace_normal_bins', 'ace'), ('act_ensemble', 'act_ensemble_pos'),
                                ('wm_k4_l4_seed2_local_20261018', 'wm_nc'),
                                ('wm_k4_l4_action_none', 'wm_nc'), ('main', 'logpzo_native')):
            self.assertTrue(current_result(variant, method))

    def test_public_selection_preserves_values_and_excludes_unmeasured_online_rows(self):
        current = dict(variant='main', method='wm_nc', alarm_n=306, population_n=320, FDR=.95625)
        old = {**current, 'variant': 'wm_k4_l4_seed1_pace_archived'}
        pair = dict(comparator='main|wm_pe|wrist', on_time_only=True, status='available', family='baseline', shared_n=62)
        data = dict(offline=[current, old], online=[{**current, 'recorded_n': 0}],
                    pairs=[pair, {**pair, 'on_time_only': False}],
                    auc_rows=[['main', 'needle_pickup', 'wm_nc'], ['wm_k4_l4_seed7', 'needle_pickup', 'wm_nc']],
                    tables={'local_seed23_training': {}, 'policy_success_rates': {}})
        select_public_data(data)
        self.assertEqual(data['offline'], [current])
        self.assertEqual(data['online'], [])
        self.assertEqual(data['pairs'], [pair])
        self.assertEqual(len(data['auc_rows']), 1)
        self.assertEqual(list(data['tables']), ['policy_success_rates'])


if __name__ == '__main__':
    unittest.main()
