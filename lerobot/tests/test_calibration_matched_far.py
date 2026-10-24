import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'utils'))
from calibration_matched_far import evaluate_split, validate_splits


class MatchedFarSplitTests(unittest.TestCase):
    def fixture(self):
        rows = {i:dict(outcome='success',score_max=i) for i in range(39)}
        for i,whole,post in [(39,100,10),(40,100,38),(41,37,37)]:
            rows[i]=dict(outcome='failure',score_max=whole,post_score_max=post,
                         detection_definition='at_or_after_onset')
        split=dict(split_id=0,calibration_ids=list(range(19)),test_success_ids=list(range(19,39)))
        return rows,dict(failure_ids=[39,40,41]),split

    def test_strict_second_highest_and_post_onset(self):
        rows,task,split=self.fixture()
        result=evaluate_split(rows,task,split,.05)
        self.assertEqual(result['threshold'],37)
        self.assertEqual(result['FAR'],.05)
        self.assertEqual(result['detected_failures'],1)
        self.assertEqual(result['FDR'],1/3)

    def test_ties_do_not_randomize_to_force_far(self):
        rows,task,split=self.fixture()
        rows[38]['score_max']=37
        self.assertEqual(evaluate_split(rows,task,split,.05)['FAR'],0)

    def test_failures_and_calibration_cannot_choose_threshold(self):
        rows,task,split=self.fixture()
        initial=evaluate_split(rows,task,split,.05)
        for i in split['calibration_ids']:
            rows[i]['score_max']=10000
        for i in task['failure_ids']:
            rows[i]['post_score_max']=10000
        result=evaluate_split(rows,task,split,.05)
        self.assertEqual(initial['threshold'],result['threshold'])
        self.assertEqual(result['FDR'],1)

    def test_reproducibility_and_score_coordinates(self):
        rows,task,split=self.fixture()
        initial=evaluate_split(rows,task,split,.05)
        self.assertEqual(initial,evaluate_split(rows,task,split,.05))
        for row in rows.values():
            for key in ('score_max','post_score_max'):
                if key in row:row[key]=2*row[key]+100
        result=evaluate_split(rows,task,split,.05)
        for key in ('FAR','FDR','episode_AUROC'):
            self.assertEqual(initial[key],result[key])

    def test_missing_post_onset_is_an_error(self):
        rows,task,split=self.fixture()
        rows[39]['detection_definition']='any_time'
        with self.assertRaises(ValueError):evaluate_split(rows,task,split,.05)

    def test_saved_partition_validation(self):
        _,_,split=self.fixture()
        task=dict(success_ids=list(range(39)),failure_ids=[39],original=split,random_splits=[split])
        manifest=dict(n_splits=1,tasks={str(i):copy.deepcopy(task) for i in range(4)})
        validate_splits(manifest,1)
        self.assertEqual(manifest,json.loads(json.dumps(manifest)))
        manifest['tasks']['0']['random_splits'][0]['test_success_ids'][0]=0
        with self.assertRaises(ValueError):validate_splits(manifest,1)


if __name__=='__main__':unittest.main()
