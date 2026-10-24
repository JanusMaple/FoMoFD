"""Category reporting must pool counts and retain unknown/missing outcomes."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'utils'))
from benchmark_failure_categories import CONFIG, load_categories, pool_categories


class FailureCategoryTests(unittest.TestCase):
    def setUp(self):
        self.taxonomy = {'categories': [{'id':'grasp', 'title':'Grasp', 'total':30, 'modes':[
            {'task':'sim','failure_mode':'Drop','failure_origin':'natural','episode_range':[39,58],'episodes':20},
            {'task':'real','failure_mode':'Loss','failure_origin':'induced','episode_range':[39,48],'episodes':10}]}]}
        self.rows = [dict(variant='main',method='wm_nc',view='wrist',calibration='episode',
            task=m['task'],episode_index=ep,split='test',outcome='failure',failure_mode=m['failure_mode'],
            failure_origin=m['failure_origin'],alarm=True,
            onset_assessments={'a':{'post_alarm':m['task']=='sim'}})
            for m in self.taxonomy['categories'][0]['modes']
            for ep in range(m['episode_range'][0],m['episode_range'][1]+1)]

    def overall(self, rows=None):
        return next(r for r in pool_categories(self.rows if rows is None else rows,self.taxonomy,'a') if r['task']=='Overall')

    def test_real_mapping_counts_and_drift(self):
        data = load_categories()
        self.assertEqual([c['total'] for c in data['categories']],[90,50,80,100])
        modes = {(m['task'],m['failure_mode']):c['id'] for c in data['categories'] for m in c['modes']}
        self.assertEqual(len(modes),20)
        for task in ('ring_over_post','shunt_insertion'):
            self.assertEqual(modes[task,'Lateral Grasp Drift'],'sensing_actuation')

    def test_counts_are_pooled_not_mean_of_task_percentages(self):
        r = self.overall()
        self.assertEqual((r['detected'],r['total']),(20,30))
        self.assertAlmostEqual(r['FDR'],2/3)
        self.assertEqual((r['natural'],r['induced']),(20,10))

    def test_early_only_anytime_alarm_does_not_count(self):
        self.assertEqual(self.overall()['misses'],10)  # all input anytime alarms are true

    def test_unknown_and_missing_are_not_silently_misses(self):
        self.rows[0]['onset_assessments']['a']['post_alarm']=None
        r=self.overall(self.rows[:-1])
        self.assertEqual((r['detected'],r['unknown'],r['missing'],r['total']),(19,1,1,30))
        self.assertIsNone(r['FDR'])
        self.assertAlmostEqual(r['FDR_lower'],19/30)
        self.assertAlmostEqual(r['FDR_upper'],21/30)

    def test_views_checkpoints_and_calibrations_remain_separate(self):
        extra=[]
        for field,value in [('view','fix'),('variant','seed3'),('calibration','functional_band')]:
            rows=deepcopy(self.rows)
            for r in rows:r[field]=value
            extra+=rows
        overall=[r for r in pool_categories(self.rows+extra,self.taxonomy,'a') if r['task']=='Overall']
        self.assertEqual(len(overall),4)
        self.assertTrue(all(r['total']==30 and r['detected']==20 for r in overall))

    def test_duplicate_or_unmapped_episode_and_wrong_origin_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.overall(self.rows+[self.rows[0]])
        self.rows[0]['failure_origin']='induced'
        with self.assertRaisesRegex(ValueError,'label mismatch'):self.overall()

    def test_missing_post_assessment_rejected_not_replaced_by_anytime(self):
        self.rows[0].pop('onset_assessments')
        with self.assertRaisesRegex(ValueError,'post-onset'):self.overall()

    def test_successes_and_calibration_excluded(self):
        excluded=[{**self.rows[0],'split':'calibration'}, {**self.rows[0],'outcome':'success'}]
        self.assertEqual(self.overall(),self.overall(self.rows+excluded))

    def test_unassigned_mapping_rejected(self):
        config=yaml.safe_load(CONFIG.read_text())
        config['categories']['sensing_actuation']['modes']['ring_over_post'].remove('Lateral Grasp Drift')
        config['categories']['sensing_actuation']['expected_failure_episodes']=80
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'mapping.yaml';p.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError,'Unassigned'):load_categories(p)


if __name__=='__main__':unittest.main()
