"""CPU-only causal scheduling, paired thresholds, and full-test denominators."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'utils'))
import online_benchmark as online


class Clock:
    def __init__(self): self.t = 0.
    def __call__(self): return self.t
    def sleep(self, seconds): self.t += seconds


class Adapter:
    def __init__(self, clock, cost=.04, constant=None):
        self.clock, self.cost, self.constant, self.calls = clock, cost, constant, []
    def reset_episode(self): pass
    def score(self, payload, window):
        self.calls.append(window.endpoint)
        self.clock.t += self.cost
        return {'fix': float(window.endpoint) if self.constant is None else self.constant}


def bindings(episode=1.5, functional=3.5):
    return {'episode': {'fix': {'threshold': episode}},
            'functional_band': {'fix': {'h': functional, 'template': {
                'grid_s': [0., 10.], 'mu': [0., 0.], 'g': [1., 1.],
                'extrapolation': 'hold_boundary_values'}}}}


class StreamingTests(unittest.TestCase):
    def test_shared_stream_scores_once_and_keeps_independent_first_alarms(self):
        payload = {'raw': np.arange(6), 'elapsed': np.arange(6)*.1}
        clock = Clock(); adapter = Adapter(clock)
        result = online.replay_paired(payload, adapter, bindings(), ['fix'], 'rnd', clock=clock, sleep=clock.sleep)
        self.assertEqual(adapter.calls, [0, 1, 2, 3, 4])
        self.assertEqual(result['alarms']['episode/fix']['endpoint'], 2)
        self.assertEqual(result['alarms']['functional_band/fix']['endpoint'], 4)
        self.assertEqual(len(result['events']), len(adapter.calls))
        self.assertEqual(result['stop_reason'], 'all_selected_views_alarmed')
        separate_clock = Clock(); separate = Adapter(separate_clock)
        single = online.replay_episode(payload, separate, lambda v,t: 1.5, ['fix'], 'rnd',
                                       clock=separate_clock, sleep=separate_clock.sleep)
        self.assertEqual(single['alarms']['fix'], result['alarms']['episode/fix'])

    def test_latest_available_window_skips_work_without_looking_ahead(self):
        payload = {'raw': np.arange(6), 'elapsed': np.arange(6)*.1}
        clock = Clock(); adapter = Adapter(clock, cost=.25)
        result = online.replay_paired(payload, adapter, bindings(100, 100), ['fix'], 'rnd', clock=clock, sleep=clock.sleep)
        self.assertEqual(adapter.calls, [0, 2, 5])
        self.assertEqual(result['skipped_endpoints'], 3)
        self.assertEqual(result['stop_reason'], 'source_exhausted')
        self.assertEqual(len(result['missed_views']), 2)
        self.assertTrue(all(e['source_time_s'] <= e['started_s'] for e in result['events']))

    def test_functional_time_origin_is_first_prepared_observation(self):
        spec = bindings()
        spec['functional_band']['fix'].update(h=0.)
        spec['functional_band']['fix']['template'].update(grid_s=[0.,1.], mu=[0.,10.])
        threshold = online.paired_thresholds(spec, .2)
        self.assertAlmostEqual(threshold('functional_band/fix', .5), 3.)
        self.assertEqual(threshold('functional_band/fix', 20.), 10.)

    def test_strict_ties_do_not_alarm(self):
        clock = Clock(); adapter = Adapter(clock, constant=1.)
        result = online.replay_paired({'raw':np.arange(3),'elapsed':np.arange(3)*.1}, adapter,
                                     bindings(1.,1.), ['fix'], 'rnd', clock=clock, sleep=clock.sleep)
        self.assertEqual(result['alarms'], {})

    def test_success_has_no_failure_onset_or_latency_row(self):
        trial = {'variant':'main','task':'t','method':'rnd','episode_index':19,'outcome':'success',
                 'failure_mode':'','failure_origin':'not_applicable','requested_detector_views':['fix']}
        # A false alarm must still enter FAR, never a failure-delay denominator.
        result = {'alarms': {'episode/fix': {'alarm_time_s': .1}}}
        outcomes, latencies = online.full_trial_records(trial, result, {}, np.arange(3), ['episode','functional_band'])
        self.assertEqual([r['alarm'] for r in outcomes], [True, False])
        self.assertEqual(latencies, [])

    def test_source_early_alarm_is_separate_from_completion_before_onset(self):
        trial = {'variant':'main','task':'t','method':'rnd','episode_index':39,'outcome':'failure',
                 'failure_mode':'Drop','failure_origin':'natural','requested_detector_views':['fix']}
        labels = {'annotators':{'a':{'name':'A'}}, 'tasks':{'t':{'failure_onsets':{39:{'a':{'views':{'fix':{'frame_index':2}}}}}}}}
        result = {'alarms':{'episode/fix':{'raw_frame':1,'source_time_s':1.,'alarm_time_s':3.}}}
        outcomes, rows = online.full_trial_records(trial,result,labels,np.arange(4),['episode'])
        self.assertTrue(outcomes[0]['alarm'])
        self.assertEqual(rows[0]['status'],'pre_onset_evidence')
        self.assertFalse(rows[0]['alarm_before_onset'])
        self.assertIsNone(rows[0]['delay_s'])

    def fixture(self):
        common = {'variant':'main','task':'t','method':'rnd','view':'fix'}
        trials = [{**common,'episode_index':ep,'outcome':outcome,'requested_detector_views':['fix']}
                  for ep,outcome in [(19,'success'),(20,'success'),(39,'failure'),(40,'failure'),(41,'failure'),(42,'failure')]]
        plan = {'trials':trials,'calibrations':['episode','functional_band'],'deadline_seconds':[.2,1,5]}
        outcomes, latencies = [], []
        for scheme in plan['calibrations']:
            for t in trials:
                r = {**t,'calibration':scheme,'alarm':t['episode_index'] in [19,39,40,42]}
                outcomes.append(r)
                if t['outcome']=='success':continue
                status = {39:'detected',40:'pre_onset_evidence',41:'missed',42:'unannotated'}[t['episode_index']]
                latencies.append({**r,'annotator_id':'a','annotator':'A','onset_view':'fix','status':status,
                                  'delay_s':.5 if status=='detected' else None,'alarm_before_onset':status=='pre_onset_evidence'})
                latencies.append({**r,'annotator_id':'b','annotator':'B','onset_view':'fix','status':'unannotated',
                                  'delay_s':None,'alarm_before_onset':False})
        return plan,outcomes,latencies

    def test_full_test_denominators_do_not_depend_on_annotations(self):
        plan,outcomes,latencies = self.fixture()
        report = online.full_streaming_metrics(plan,outcomes,latencies)
        for r in report['metrics']:
            self.assertEqual((r['successes_tested'],r['failures_tested']), (2,4))
            self.assertEqual((r['FAR'],r['FDR_any_alarm'],r['missed_failures']),(.5,.75,1))
            self.assertEqual(r['status'],'complete')
        for r in report['onset_metrics']:
            if r['annotator_id']=='b':
                self.assertEqual(r['missing_onsets'],4);self.assertIsNone(r['post_onset_detection_rate']);continue
            self.assertEqual((r['annotated_failures'],r['missing_onsets']),(3,1))
            self.assertEqual(r['post_onset_detection_rate'],1/3)
            self.assertEqual(r['no_post_onset_detection'],2)
            self.assertEqual(r['pre_onset_alarms'],1)
            self.assertEqual(r['delay_mean_s'],.5);self.assertIsNone(r['delay_sd_s'])
        for r in report['deadline_rates']:
            if r['annotator_id']=='a':
                self.assertEqual(r['annotated_failures'],3)
                self.assertEqual(r['rate'],0 if r['deadline_s']==.2 else 1/3)

    def test_partial_coverage_never_looks_like_complete_zero_far(self):
        plan,rows,latencies = self.fixture()
        report = online.full_streaming_metrics(plan,[],[])
        for r in report['metrics']:
            self.assertEqual(r['status'],'partial');self.assertIsNone(r['FAR']);self.assertIsNone(r['FDR_any_alarm'])
            self.assertEqual((r['successes_planned'],r['failures_planned']),(2,4))
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            online.full_streaming_metrics(plan,rows+[copy.deepcopy(rows[0])],latencies)


if __name__ == '__main__':
    unittest.main()
