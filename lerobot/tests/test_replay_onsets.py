"""CPU tests for shared/per-camera onset annotation and legacy preservation."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import yaml

spec=importlib.util.spec_from_file_location('replay_onsets_tested',Path(__file__).resolve().parents[1]/'utils/replay_inspector.py')
r=importlib.util.module_from_spec(spec);sys.modules[spec.name]=r;spec.loader.exec_module(r)


def legacy():
    return {'schema_version':2,'onset_frame_index_base':0,'onset_frame_id_space':'raw_recording_frame_index',
            'annotators':{'a':{'name':'Alice'},'b':{'name':'Bob'}},'tasks':{'toy':{
                'groups':[{'episode_range':[0,0],'outcome':'success'},{'episode_range':[1,2],'outcome':'failure'}],
                'failure_onsets':{1:{'a':{'frame_index':4,'updated_at':'original-time'}}}}}}


def copy_fixture():
    data=legacy();r.upgrade_onset_schema(data)
    source=deepcopy(data['tasks']['toy']['failure_onsets'][1]['a'])
    node={'groups':[
        {'episode_range':[0,0],'outcome':'success','failure_origin':'not_applicable'},
        {'episode_range':[1,1],'outcome':'failure','failure_origin':'natural'},
        {'episode_range':[2,4],'outcome':'failure','failure_origin':'induced'}],
        'failure_onsets':{i:{'a':deepcopy(source)} for i in (1,2,3)}}
    # Include a missing camera, a valid zero onset, and a different second label.
    node['failure_onsets'][3]['a']={'mode':'separate','views':{
        'wrist':{'frame_index':0,'updated_at':'wrist-original-time'}}}
    node['failure_onsets'][2]['b']={'mode':'shared','views':{
        v:{'frame_index':9,'updated_at':'bob-time'} for v in ('fix','wrist')}}
    data['tasks']={task:deepcopy(node) for task in
                   ('needle_pickup','ring_over_post','tissue_retraction','shunt_insertion')}
    return data


class OnsetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'labels.yaml'
        self.original=legacy();self.path.write_text(yaml.safe_dump(self.original,sort_keys=False))
        self.store=r.OnsetLabels(self.path)

    def test_migration_preserves_existing_frames_timestamps_groups(self):
        self.store.mutate(lambda manifest:None)
        data=self.store.read()
        self.assertEqual(data['schema_version'],3)
        self.assertEqual(data['tasks']['toy']['groups'],self.original['tasks']['toy']['groups'])
        entry=data['tasks']['toy']['failure_onsets'][1]['a']
        self.assertEqual(entry['mode'],'shared')
        self.assertEqual(entry['views']['fix'],self.original['tasks']['toy']['failure_onsets'][1]['a'])
        self.assertEqual(entry['views']['fix'],entry['views']['wrist'])
        self.assertNotIn('&id',self.path.read_text())

    def test_default_save_and_clear_apply_to_both_views(self):
        saved=self.store.set_onset('toy',2,'a',0)['onset_annotations']['a']
        self.assertEqual(saved['mode'],'shared')
        self.assertEqual([saved['views'][v]['frame_index'] for v in ['fix','wrist']],[0,0])
        self.store.set_onset('toy',2,'a',None)
        self.assertNotIn(2,self.store.read()['tasks']['toy']['failure_onsets'])

    def test_separate_edit_clear_and_return_to_shared(self):
        self.store.set_onset('toy',1,'a',8,'fix')
        result=self.store.read()['tasks']['toy']['failure_onsets'][1]['a']
        self.assertEqual(result['views']['wrist'],{'frame_index':4,'updated_at':'original-time'})
        self.assertEqual(result['views']['fix']['frame_index'],8)
        self.assertEqual(result['mode'],'separate')
        self.store.set_onset('toy',1,'a',None,'fix')
        self.assertEqual(set(self.store.read()['tasks']['toy']['failure_onsets'][1]['a']['views']),{'wrist'})
        saved=self.store.set_onset('toy',1,'a',6)['onset_annotations']['a']
        self.assertEqual(saved['mode'],'shared')
        self.assertEqual([saved['views'][v]['frame_index'] for v in ['fix','wrist']],[6,6])

    def test_wrist_only_then_fixed_preserves_wrist_and_other_annotator(self):
        self.store.set_onset('toy',2,'a',5,'wrist')
        second=r.OnsetLabels(self.path)
        second.set_onset('toy',2,'b',7)
        self.store.set_onset('toy',2,'a',9,'fix')
        annotations=self.store.read()['tasks']['toy']['failure_onsets'][2]
        self.assertEqual(annotations['a']['views']['wrist']['frame_index'],5)
        self.assertEqual(annotations['a']['views']['fix']['frame_index'],9)
        self.assertEqual(annotations['b']['views']['fix']['frame_index'],7)

    def test_invalid_or_success_update_keeps_file_unchanged(self):
        before=self.path.read_bytes()
        for episode,annotator,frame,view in [(0,'a',2,'both'),(1,'unknown',2,'both'),(1,'a',-1,'fix'),
                                           (1,'a',True,'both'),(1,'a',2,'other')]:
            with self.assertRaises(ValueError):self.store.set_onset('toy',episode,annotator,frame,view)
            self.assertEqual(self.path.read_bytes(),before)

    def test_shared_cannot_hide_missing_or_distinct_camera_labels(self):
        data=legacy();r.upgrade_onset_schema(data)
        entry=data['tasks']['toy']['failure_onsets'][1]['a']
        entry['views']['fix']['frame_index']=10
        with self.assertRaises(ValueError):r.OnsetLabels.validate(data)
        entry['mode']='separate';r.OnsetLabels.validate(data)
        entry['views']['fix']=None
        with self.assertRaises(ValueError):r.OnsetLabels.validate(data)

    def test_recording_bounds_are_checked_per_view(self):
        recordings=r.Recordings.__new__(r.Recordings)
        recordings.labels=self.store
        recordings.outcomes={('toy',1):'failure'}
        recordings.episode=lambda *args:SimpleNamespace(times=[0,1,2])
        with self.assertRaises(ValueError):recordings.set_onset('toy',1,'a',3,'wrist')
        self.assertEqual(self.store.read()['schema_version'],2)

    def test_new_annotator_copies_only_first_annotator_induced_simulation_onsets(self):
        original=copy_fixture();self.path.write_text(yaml.safe_dump(original,sort_keys=False))
        result=self.store.add_annotator('Carol');target=result['annotator_id']
        data=self.store.read()
        self.assertEqual(result['copied_onsets'],{'source_annotator_id':'a','episodes':4,'views':6})
        for task,node in data['tasks'].items():
            for episode,annotations in node['failure_onsets'].items():
                expected=task in ('needle_pickup','ring_over_post') and episode in (2,3)
                self.assertEqual(target in annotations,expected)
                if expected:
                    copy=annotations.pop(target);source=annotations['a']
                    self.assertEqual(copy['mode'],source['mode'])
                    self.assertEqual(set(copy['views']),set(source['views']))
                    for view,label in copy['views'].items():
                        self.assertEqual(label.pop('copied_from_annotator'),'a')
                        self.assertTrue(label.pop('copied_at'))
                        self.assertEqual(label,source['views'][view])
            self.assertNotIn(4,node['failure_onsets'])  # Missing source stays unannotated.
        data['annotators'].pop(target)
        self.assertEqual(data,original)

    def test_copy_preserves_existing_target_unless_overwrite_requested(self):
        data=copy_fixture();before=deepcopy(data)
        result=r.copy_simulated_induced_onsets(data,'b')
        self.assertEqual((result['episodes'],result['views']),(2,2))
        for task in ('needle_pickup','ring_over_post'):
            self.assertEqual(data['tasks'][task]['failure_onsets'][2]['b'],
                             before['tasks'][task]['failure_onsets'][2]['b'])
        result=r.copy_simulated_induced_onsets(data,'b',overwrite=True)
        self.assertEqual((result['episodes'],result['views']),(4,6))
        entries=data['tasks']['needle_pickup']['failure_onsets'][2]
        self.assertEqual(entries['b']['views']['fix']['frame_index'],4)
        entries['b']['views']['fix']['frame_index']=20
        self.assertEqual(entries['a']['views']['fix']['frame_index'],4)
        self.assertEqual(entries['b']['views']['wrist']['frame_index'],4)
        for task in ('tissue_retraction','shunt_insertion'):
            self.assertEqual(data['tasks'][task],before['tasks'][task])

    def test_manual_edit_removes_only_edited_view_copy_provenance(self):
        self.path.write_text(yaml.safe_dump(copy_fixture(),sort_keys=False))
        result=self.store.add_annotator('Carol');target=result['annotator_id']
        before=self.store.read()['tasks']['needle_pickup']['failure_onsets'][2]
        self.store.set_onset('needle_pickup',2,target,7,'fix')
        after=self.store.read()['tasks']['needle_pickup']['failure_onsets'][2]
        self.assertEqual(after['a'],before['a'])
        self.assertEqual(after[target]['views']['wrist'],before[target]['views']['wrist'])
        self.assertNotIn('copied_from_annotator',after[target]['views']['fix'])
        self.assertNotIn('copied_at',after[target]['views']['fix'])
        self.assertEqual(after[target]['views']['fix']['frame_index'],7)

    def test_first_annotator_starts_empty_and_legacy_addition_upgrades_before_copy(self):
        data=legacy();data['annotators']={};data['tasks']['toy']['failure_onsets']={}
        self.path.write_text(yaml.safe_dump(data,sort_keys=False))
        result=self.store.add_annotator('First')
        self.assertEqual(result['copied_onsets']['episodes'],0)
        data=legacy();data['tasks']['needle_pickup']=data['tasks'].pop('toy')
        data['tasks']['needle_pickup']['groups'][1]['failure_origin']='induced'
        self.path.write_text(yaml.safe_dump(data,sort_keys=False))
        result=self.store.add_annotator('Carol')
        self.assertEqual(result['copied_onsets']['episodes'],1)
        entry=self.store.read()['tasks']['needle_pickup']['failure_onsets'][1][result['annotator_id']]
        self.assertEqual(entry['views']['fix']['frame_index'],4)
        self.assertEqual(entry['views']['wrist']['copied_from_annotator'],'a')


if __name__=='__main__':unittest.main()
