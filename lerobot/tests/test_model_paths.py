"""Local relocation preserves exact-file overrides and model identity checks."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'utils'))
import model_paths


class ModelPathTests(unittest.TestCase):
    def test_directory_and_exact_file_bindings(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest=root/'locations.json'
            old='fomofd/tasks/benchmark_results/imported/run'
            new='fomofd/tasks/needle_pickup/model/fm_wm/run'
            manifest.write_text(json.dumps({'schema_version':1,'repo_root':str(root),
                'relocations':{old:new,old+'/config.json':new+'/config.local.json'}}))
            resolve=lambda value:model_paths.resolve_model_path(value,root,manifest=manifest)
            self.assertEqual(resolve(old+'/checkpoints/best.pt'),root/new/'checkpoints/best.pt')
            self.assertEqual(resolve(old+'/config.json'),root/new/'config.local.json')
            self.assertEqual(resolve(old+'_other/config.json'),root/(old+'_other/config.json'))
            with patch.object(model_paths,'MANIFEST',manifest),patch.object(model_paths,'REPO',root):
                identity={'checkpoint':str(root/old/'checkpoints/best.pt'),
                          'sha256':{str(root/old/'config.json'):'original-digest'},'seed':2}
                normalized=model_paths.relocate_identity(identity)
                self.assertEqual(normalized['sha256'],{str(root/new/'config.local.json'):'original-digest'})
                self.assertEqual(normalized['seed'],2)
                self.assertNotEqual(normalized,model_paths.relocate_identity(dict(identity,seed=1)))

    def test_target_outside_tasks_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest=root/'locations.json'
            manifest.write_text(json.dumps({'schema_version':1,'repo_root':str(root),'relocations':{'old':'../../other'}}))
            with self.assertRaisesRegex(ValueError,'under fomofd/tasks'):
                model_paths.resolve_model_path('old',root,manifest=manifest)


if __name__=='__main__':unittest.main()
