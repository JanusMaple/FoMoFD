"""No GPU/model loading: final-answer semantics, replay timing, and denominators."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
from vlm_answers import FinalAnswerStop, classify_completion, parse_final_answer
import vlm_online_benchmark as vlm


class Tokenizer:
    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids)


class AnswerTests(unittest.TestCase):
    def test_qwen_stops_at_closing_tag_and_256_token_truncation_is_unknown(self):
        response = 'No visible drop; FAIL is not supported.\n<answer>NORMAL</answer>'
        stop = FinalAnswerStop(Tokenizer(), 0, require_think=False, started=0, clock=lambda: 1)
        for n in range(1, len(response)):
            self.assertFalse(stop(np.array([list(map(ord, response[:n]))])))
        self.assertTrue(stop(np.array([list(map(ord, response))])))
        self.assertEqual(stop.reason, 'final_answer')
        result = classify_completion('Repeated assessment without an answer.', [1] * 256,
                                    require_think=False, max_new_tokens=256,
                                    eos_token_ids=[99], stopping_reason=None)
        self.assertEqual(result['status'], 'truncated')
        self.assertIsNone(result['final_label'])

    def test_reasoning_fail_is_not_alarm(self):
        text = '<think>FAIL is possible, but the images do not support it.</think><answer>NORMAL</answer>'
        self.assertEqual(parse_final_answer(text, require_think=True), 'NORMAL')
        self.assertIsNone(parse_final_answer('<think>FAIL</think>'))

    def test_answer_tags_inside_reasoning_are_ignored(self):
        text = '<think>Hypothesis: <answer>FAIL</answer>. Rejected.</think>\n<answer>NORMAL</answer>'
        self.assertEqual(parse_final_answer(text, require_think=True), 'NORMAL')
        self.assertIsNone(parse_final_answer('<think><answer>FAIL</answer>'))

    def test_qwen_assessment_and_cosmos_format(self):
        self.assertEqual(parse_final_answer('The needle visibly drops.\n<answer>FAIL</answer>'), 'FAIL')
        self.assertEqual(parse_final_answer('  <answer>NORMAL</answer>\n'), 'NORMAL')
        self.assertIsNone(parse_final_answer('<answer>NORMAL</answer>', require_think=True))

    def test_malformed_and_ambiguous_never_map_to_normal(self):
        for text in ['FAIL', 'NORMAL', '<answer>fail</answer>', '<answer> NORMAL </answer>',
                     '<answer>FAIL</answer> extra', '<answer>FAIL</answer>\n<answer>NORMAL</answer>',
                     '<answer>FAIL</answer><think>later reasoning</think>',
                     '<think>unfinished <answer>FAIL</answer>', '</think>\n<answer>FAIL</answer>',
                     '<think>a<think>b</think></think><answer>NORMAL</answer>',
                     'An example is <answer>FAIL</answer>', '<answer>FAIL</ans',
                     '<answer>broken\n<answer>NORMAL</answer>']:
            with self.subTest(text=text):
                self.assertIsNone(parse_final_answer(text))

    def test_stopper_only_checks_generated_suffix_and_complete_answer(self):
        prompt = list(map(ord, '<answer>FAIL</answer>'))
        stop = FinalAnswerStop(Tokenizer(), len(prompt), require_think=True, started=0, clock=lambda: 1)
        response = '<think>FAIL was considered.</think><answer>NORMAL</answer>'
        for n in range(len(response)):
            self.assertFalse(stop(np.array([prompt + list(map(ord, response[:n]))])))
        self.assertTrue(stop(np.array([prompt + list(map(ord, response))])))
        self.assertEqual(stop.reason, 'final_answer')
        self.assertEqual(stop.text, response)

    def test_stopper_budget_timeout_and_batch_guard(self):
        stop = FinalAnswerStop(Tokenizer(), 0, require_think=False, started=0, timeout_s=5, clock=lambda: 6)
        self.assertTrue(stop(np.array([list(map(ord, 'still reasoning'))])))
        self.assertEqual(stop.reason, 'timeout')
        with self.assertRaises(ValueError):
            stop(np.array([[1], [2]]))

    def test_completion_failure_types_and_valid_at_budget(self):
        common = dict(require_think=False, max_new_tokens=3, eos_token_ids=[99], stopping_reason=None)
        self.assertEqual(classify_completion('FAIL?', [1, 2, 3], **common)['status'], 'truncated')
        self.assertEqual(classify_completion('FAIL?', [1, 99], **common)['status'], 'malformed')
        self.assertEqual(classify_completion('<answer>NORMAL</answer>', [1, 2, 3], **common)['final_label'], 'NORMAL')
        timed = classify_completion('<answer>FAIL</answer>', [1, 2], **dict(common, stopping_reason='timeout'))
        self.assertEqual(timed['status'], 'timed_out')
        self.assertIsNone(timed['final_label'])


class ProtocolTests(unittest.TestCase):
    def test_mean_onset_uses_exact_time_and_first_post_onset_decision(self):
        config = {'evaluation_protocol': 'synthetic',
                  'evaluation': {'primary_annotator_id': 'consensus_mean'}}
        plan = {'config': config, 'labels': {'tasks': {'shunt_insertion': {
            'failure_onsets': {39: {'consensus_mean': {'views': {'wrist': {
                'frame_index': 9, 'elapsed_seconds': .85}}}}}}}}}
        trial = {'task': 'shunt_insertion', 'episode': 39, 'view': 'wrist', 'outcome': 'failure'}
        payload = {'raw_wall': 100 + np.arange(12)*.1, 'elapsed': np.arange(12)*.1, 'preload_s': 0.}
        events = [{'raw_frame': i, 'endpoint': i, 'source_time_s': i*.1,
                   'alarm_ready_s': t, 'scores': {'wrist': 1.}, 'thresholds': {'wrist': .5}}
                  for i,t in [(7,.75),(9,1.05),(11,1.3)]]
        first = {'raw_frame': 7, 'endpoint': 7, 'source_time_s': .7, 'alarm_time_s': .75}
        replay = {'events': events, 'alarms': {'wrist': first}, 'stop_reason': 'source_exhausted'}
        spec = {'protocol_id': 'test_v3', 'latency': {
            'detection_start': 'first_positive_decision_with_source_endpoint_at_or_after_onset'}}
        with patch('benchmark_frozen_protocol.load_protocol', return_value=spec), \
             patch.object(vlm, 'episode_labels', return_value={39: {'failure_origin': 'natural'}}):
            result = vlm.summarize_trial(plan, trial, payload, replay, [])
        self.assertEqual(result['onset_s'], .85)
        self.assertEqual(result['alarm']['raw_frame'], 9)
        self.assertAlmostEqual(result['post_onset_evidence_delay_s'], .2)
        self.assertTrue(result['frozen_onset_result']['sustained_pre_onset_alarm'])
        self.assertFalse(result['frozen_onset_result']['early_alarm'])
        self.assertEqual(result['any_time_alarm']['raw_frame'], 7)
        self.assertFalse(result['late_alarm'])

    def test_v7_qwen_budget_preserves_inputs_and_cosmos_history(self):
        old = yaml.safe_load(vlm.DEFAULT_CONFIG.with_name('protocol_video_v6.yaml').read_text())
        new = yaml.safe_load(vlm.DEFAULT_CONFIG.with_name('protocol_video_v7_qwen.yaml').read_text())
        vlm.validate_protocol(new)
        self.assertEqual(list(new['models']), ['qwen3_vl_8b_instruct'])
        spec = new['models']['qwen3_vl_8b_instruct']
        self.assertEqual(spec['max_new_tokens'], 256)
        self.assertFalse(spec['require_think'])
        self.assertEqual(old['models']['cosmos_reason2_8b']['max_new_tokens'], 4096)
        for key in ['inputs', 'preprocessing', 'streaming', 'generation', 'inference', 'prompt_file', 'task_definitions']:
            self.assertEqual(old[key], new[key])

    def setUp(self):
        self.config = yaml.safe_load(vlm.DEFAULT_CONFIG.read_text())

    def test_config_preserves_controls_and_per_model_budgets(self):
        vlm.validate_protocol(self.config)
        self.assertTrue(all(m['max_new_tokens'] == 4096 for m in self.config['models'].values()))
        changed = copy.deepcopy(self.config)
        changed['models']['qwen3_vl_8b_instruct']['max_new_tokens'] = 2048
        vlm.validate_protocol(changed)
        changed['inference']['dtype'] = 'float16'
        with self.assertRaises(ValueError):
            vlm.validate_protocol(changed)
        old = yaml.safe_load(vlm.DEFAULT_CONFIG.with_name('direct_label_v1.yaml').read_text())
        self.assertEqual(old['generation']['max_new_tokens'], 8)
        self.assertNotEqual(old['protocol_id'], self.config['protocol_id'])

    def test_transformers_does_not_restore_checkpoint_sampling_or_constraints(self):
        from transformers import GenerationConfig
        from transformers.generation.utils import GenerationMixin
        model = GenerationMixin()
        model.generation_config = GenerationConfig(do_sample=True, num_beams=4, min_new_tokens=80,
                                                   bad_words_ids=[[42]], eos_token_id=99, pad_token_id=0)
        options = vlm.generation_options({'max_new_tokens': 4096}, model.generation_config)
        effective, _ = model._prepare_generation_config(**options)
        self.assertFalse(effective.do_sample)
        self.assertEqual(effective.num_beams, 1)
        self.assertEqual(effective.min_new_tokens, 0)
        self.assertEqual(effective.max_new_tokens, 4096)
        self.assertIsNone(effective.bad_words_ids)
        self.assertIsNone(effective.force_words_ids)
        self.assertIsNone(effective.forced_eos_token_id)

    def test_prompt_uses_all_definitions_and_actual_gap_without_labels(self):
        tasks = yaml.safe_load((vlm.REPO / self.config['task_definitions']).read_text())
        plan = {'config': self.config, 'tasks': tasks,
                'prompt_template': (vlm.REPO / self.config['prompt_file']).read_text(),
                'labels': {'SECRET_OUTCOME': 'SECRET_ONSET'}}
        prompt = vlm.render_prompt(plan, 'needle_pickup', 'qwen3_vl_8b_instruct', np.array([1, 1.2, 1.4, 1.6, 2.5]))
        self.assertIn('0.900 seconds after Frame 4', prompt)
        for name in tasks['tasks']['needle_pickup']['failure_definitions']:
            self.assertIn(name, prompt)
        self.assertNotIn('SECRET_', prompt)
        self.assertNotIn('review_notes', prompt)

    def test_unchanged_scheduler_latest_window_and_late_completion(self):
        class Clock:
            t = 0
            def __call__(self): return self.t
            def sleep(self, dt): self.t += dt
        clock = Clock()
        class Adapter:
            def __init__(self): self.windows = []
            def reset_episode(self): pass
            def score(self, payload, window):
                self.windows.append(window)
                clock.t += .55  # All preprocessing/reasoning/generation/parsing.
                return {'fix': float(len(self.windows) == 2)}
        adapter = Adapter()
        payload = {'raw': np.arange(16), 'elapsed': np.arange(16) * .1}
        replay = vlm.replay_episode(payload, adapter, lambda v, t: .5, ['fix'],
                                    'wm_vlm_alignment', 4, 4, clock=clock, sleep=clock.sleep)
        self.assertEqual([w.endpoint for w in adapter.windows], [7, 12])
        self.assertEqual(adapter.windows[0].history, (0, 1, 2, 3))
        self.assertAlmostEqual(replay['alarms']['fix']['alarm_time_s'], 1.8)
        self.assertGreater(replay['alarms']['fix']['alarm_time_s'], payload['elapsed'][-1])
        self.assertEqual(replay['skipped_endpoints'], 4)

    def test_error_denominators_not_silently_normal_and_late_not_on_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            config = {'models': {'m': {}}, 'protocol_id': 'test', 'inputs': {'views': ['fix']},
                      'outputs': {'per_task_root': str(output / 'task_results')},
                      'evaluation': {'deadline_seconds': [1, 2], 'primary_annotator_id': 'a'}}
            trials = [{'id': str(ep), 'model': 'm', 'task': 't', 'view': 'fix', 'episode': ep,
                       'outcome': outcome} for ep, outcome in [(19, 'success'), (20, 'success'), (39, 'failure'), (40, 'failure')]]
            plan = {'fingerprint': 'test', 'config': config, 'tasks': {'tasks': {'t': {}}}, 'trials': trials,
                    'labels': {'tasks': {'t': {'failure_onsets': {ep: {'a': {'views': {'fix': {'frame_index': 1}}}} for ep in [39, 40]}}}}}
            for trial in trials[:3]:
                row = dict(trial, fingerprint='test', status='indeterminate' if trial['episode'] == 20 else 'complete',
                           alarm={'alarm_time_s': 3} if trial['episode'] == 39 else None,
                           late_alarm=trial['episode'] == 39, requests=[])
                vlm.write_json(vlm.result_path(plan, output, trial), row)
            vlm.generate_report(plan, output)
            row = json.loads((output / 'report.json').read_text())['summary'][0]
            self.assertEqual((row['FAR_denominator'], row['FDR_denominator']), (2, 2))
            self.assertIsNone(row['FAR_any'])
            self.assertEqual((row['FAR_any_lower_bound'], row['FAR_any_upper_bound']), (0, .5))
            self.assertEqual((row['FDR_any_count'], row['FDR_on_time_count']), (1, 0))
            self.assertEqual((row['annotated_failures'], row['annotated_unknown']), (2, 1))
            self.assertEqual(row['detected_by_1s_denominator'], 2)


if __name__ == '__main__':
    unittest.main()
