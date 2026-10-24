"""CPU-only causal selection, real timestamps, and native video processing."""
from pathlib import Path
import copy
import sys
import unittest

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
import vlm_online_benchmark as vlm
from vlm_video import select_video_frames, timestamp_metadata, prepare_video_inputs


class VideoTests(unittest.TestCase):
    def sentinel_plan(self):
        base = vlm.DEFAULT_CONFIG.parent
        config = yaml.safe_load((base / 'protocol_video_v10_qwen_sentinel.yaml').read_text())
        return {'config': config,
                'tasks': yaml.safe_load((vlm.REPO / config['task_definitions']).read_text()),
                'prompt_template': (vlm.REPO / config['prompt_file']).read_text()}

    def test_sentinel_uses_editable_task_deadlines_and_observed_wall_time_only(self):
        plan = self.sentinel_plan()
        vlm.validate_protocol(plan['config'])
        vlm.validate_prompt_inputs(plan['config'], plan['tasks'], plan['prompt_template'])
        plan['labels'] = {'outcome': 'SECRET_OUTCOME', 'onset': 'SECRET_ONSET'}
        for task, node in plan['tasks']['tasks'].items():
            node['task_description'] = 'Custom description with literal {braces}.'
            prompt = vlm.render_prompt(plan, task, 'qwen3_vl_8b_instruct', np.array([0., 1., 12.375]))
            self.assertIn(node['task_description'], prompt)
            self.assertIn('up to 30 seconds', prompt)
            self.assertIn('Current elapsed time: 12.375 seconds', prompt)
            self.assertNotIn('SECRET_', prompt)
            self.assertNotIn('{TIME_LIMIT}', prompt)
            self.assertNotIn('{ELAPSED_TIME}', prompt)
        node['time_limit_seconds'] = 45
        prompt = vlm.render_prompt(plan, task, 'qwen3_vl_8b_instruct', np.array([0., 31.25]))
        self.assertIn('up to 45 seconds', prompt)
        self.assertIn('31.250 seconds', prompt)

    def test_sentinel_rejects_missing_deadlines_and_contradictory_history(self):
        for invalid in (None, 0, -1, True, '30', float('nan'), float('inf')):
            plan = self.sentinel_plan()
            plan['tasks']['tasks']['needle_pickup']['time_limit_seconds'] = invalid
            with self.subTest(limit=invalid), self.assertRaisesRegex(ValueError, 'time_limit_seconds'):
                vlm.validate_prompt_inputs(plan['config'], plan['tasks'], plan['prompt_template'])
        for key, value in (('video_fps', 4), ('history_seconds', 20)):
            plan = self.sentinel_plan()
            plan['config']['models']['qwen3_vl_8b_instruct'][key] = value
            with self.assertRaisesRegex(ValueError, 'full-history video at 1 FPS'):
                vlm.validate_prompt_inputs(plan['config'], plan['tasks'], plan['prompt_template'])
        plan = self.sentinel_plan()
        plan['tasks']['tasks']['needle_pickup']['task_description'] = '  '
        with self.assertRaisesRegex(ValueError, 'task_description'):
            vlm.validate_prompt_inputs(plan['config'], plan['tasks'], plan['prompt_template'])

    def test_sentinel_future_duration_cannot_change_prompt_or_sampling(self):
        plan = self.sentinel_plan()
        times = np.arange(801) * .05 + 9000
        ids, seconds = select_video_frames(times, 627, 1, None)
        before = vlm.render_prompt(plan, 'needle_pickup', 'qwen3_vl_8b_instruct', seconds)
        times[628:] = np.nan
        later_ids, later_seconds = select_video_frames(times, 627, 1, None)
        after = vlm.render_prompt(plan, 'needle_pickup', 'qwen3_vl_8b_instruct', later_seconds)
        self.assertEqual(ids, later_ids)
        self.assertEqual(before, after)
        self.assertIn('31.350 seconds', after)  # Do not clamp elapsed time at the 30s allowance.
        self.assertIn('up to 30 seconds', after)

    def test_sentinel_preserves_v9_dataset_scheduler_and_decoding_controls(self):
        old = yaml.safe_load(vlm.DEFAULT_CONFIG.with_name('protocol_video_v9_qwen.yaml').read_text())
        new = self.sentinel_plan()['config']
        for key in ('inputs', 'preprocessing', 'inference', 'generation', 'streaming',
                    'evaluation', 'development', 'outputs', 'labels_for_selection_and_reporting_only'):
            self.assertEqual(old[key], new[key], key)
        model = copy.deepcopy(new['models']['qwen3_vl_8b_instruct'])
        self.assertEqual(model.pop('max_new_tokens'), 1024)
        model.pop('response_instructions')
        prior = copy.deepcopy(old['models']['qwen3_vl_8b_instruct'])
        self.assertEqual(prior.pop('max_new_tokens'), 256)
        prior.pop('response_instructions')
        self.assertEqual(model, prior)
        self.assertNotEqual(old['protocol_id'], new['protocol_id'])

    def test_v6_only_changes_prompt_and_task_stages(self):
        base = vlm.DEFAULT_CONFIG.parent
        old = yaml.safe_load((base / 'protocol_video_v5.yaml').read_text())
        new = yaml.safe_load((base / 'protocol_video_v6.yaml').read_text())
        vlm.validate_protocol(new)
        for key in old:
            if key not in {'protocol_id', 'description', 'prompt_file', 'task_definitions'}:
                self.assertEqual(old[key], new[key], key)
        old_tasks = yaml.safe_load((vlm.REPO / old['task_definitions']).read_text())
        tasks = yaml.safe_load((vlm.REPO / new['task_definitions']).read_text())
        plan = {'config': new, 'tasks': tasks,
                'prompt_template': (vlm.REPO / new['prompt_file']).read_text()}
        for task, node in tasks['tasks'].items():
            self.assertEqual({k: v for k, v in node.items() if k != 'normal_task_stages'},
                             old_tasks['tasks'][task])
            for model in new['models']:
                prompt = vlm.render_prompt(plan, task, model, np.array([0., 1., 2.]))
                self.assertIn(node['normal_task_stages'], prompt)
                self.assertNotIn('{NORMAL_TASK_STAGES}', prompt)
                self.assertIn('Do not infer an unsuccessful grasp merely because successful', prompt)
                self.assertIn(new['models'][model]['response_instructions'], prompt)

    def test_full_history_retains_initial_frame_and_history_older_than_twenty_seconds(self):
        times = np.arange(602) * .05 + 1000
        ids, selected = select_video_frames(times, 546, 1, None)
        self.assertEqual(ids, list(range(0, 541, 20)) + [546])
        self.assertEqual(selected[0], 0.)
        self.assertAlmostEqual(selected[-1], 27.3)
        later, _ = select_video_frames(times, 569, 1, None)
        self.assertTrue(set(ids[:-1]).issubset(later))
        modified = times.copy(); modified[547:] = np.nan
        self.assertEqual(select_video_frames(modified, 546, 1, None)[0], ids)
        self.assertEqual(select_video_frames(times, 7, 1, None)[0], [0, 7])

    def test_v5_model_specific_history_and_cosmos_prompt_preservation(self):
        base = vlm.DEFAULT_CONFIG.parent
        old = yaml.safe_load((base / 'protocol_video_v4.yaml').read_text())
        new = yaml.safe_load((base / 'protocol_video_v5.yaml').read_text())
        vlm.validate_protocol(new)
        self.assertIsNone(vlm.video_history_seconds(new, 'qwen3_vl_8b_instruct'))
        self.assertEqual(vlm.video_history_seconds(new, 'cosmos_reason2_8b'), 20.)
        def plan(c):
            return {'config': c, 'tasks': yaml.safe_load((vlm.REPO / c['task_definitions']).read_text()),
                    'prompt_template': (vlm.REPO / c['prompt_file']).read_text()}
        seconds = np.array([3.25, 3.49, 3.72])
        self.assertEqual(vlm.render_prompt(plan(old), 'needle_pickup', 'cosmos_reason2_8b', seconds),
                         vlm.render_prompt(plan(new), 'needle_pickup', 'cosmos_reason2_8b', seconds))
        qwen = vlm.render_prompt(plan(new), 'needle_pickup', 'qwen3_vl_8b_instruct', np.array([0, 1, 27.3]))
        self.assertIn('all available history from recording time zero', qwen)
        self.assertNotIn('most recent 20', qwen)

    def test_equal_history_different_rates_and_no_future(self):
        times = np.arange(601) * .05 + 1000
        for fps, count in [(1, 21), (4, 81)]:
            ids, selected = select_video_frames(times, 500, fps, 20)
            self.assertEqual(len(ids), count)
            self.assertEqual(ids[-1], 500)
            self.assertAlmostEqual(selected[0], 5)
            self.assertAlmostEqual(selected[-1], 25)
            modified = times.copy()
            modified[501:] = np.nan  # Future contents cannot affect this input.
            self.assertEqual(select_video_frames(modified, 500, fps, 20)[0], ids)

    def test_short_history_irregular_captures_and_deduplication(self):
        times = np.array([0., .13, .36, .8, 1.02, 1.7])
        ids, selected = select_video_frames(times, 4, 4, 20)
        self.assertEqual(ids, [0, 1, 2, 4])
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(np.all(selected <= times[4]))
        self.assertEqual(select_video_frames(times, 0, 1, 20)[0], [0])
        with self.assertRaises(ValueError):
            select_video_frames([0, 2, 1], 2, 1, 20)

    def test_actual_microsecond_timebase(self):
        seconds = np.array([12.17342, 12.4287, 12.71123])
        metadata = timestamp_metadata(seconds)
        np.testing.assert_allclose(np.array(metadata['frames_indices']) / metadata['fps'], seconds, atol=1e-6)

    def test_native_processor_both_models_no_resampling_real_pair_times(self):
        from transformers import AutoProcessor
        config = yaml.safe_load(vlm.DEFAULT_CONFIG.with_name('protocol_video_v4.yaml').read_text())
        vlm.validate_protocol(config)
        seconds = np.array([3.173, 3.412, 3.761])
        for name, spec in config['models'].items():
            with self.subTest(model=name):
                processor = AutoProcessor.from_pretrained(vlm.REPO / spec['path'], local_files_only=True)
                messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'Describe.'}, {'type': 'video'}]}]
                arrays = [np.full((96, 128, 3), i * 40, dtype=np.uint8) for i in range(3)]
                inputs, record = prepare_video_inputs(processor, messages, arrays, seconds, config['preprocessing'])
                self.assertEqual(record['video_grid_thw'], [[2, 6, 8]])
                self.assertEqual(record['video_padding_repeated_last_frames'], 1)
                np.testing.assert_allclose(record['video_temporal_patch_timestamps_s'], [3.2925, 3.761])
                self.assertIn('pixel_values_videos', inputs)
                self.assertNotIn('pixel_values', inputs)


if __name__ == '__main__':
    unittest.main()
