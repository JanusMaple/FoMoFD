"""CPU tests: checkpoint retention, failed writes and strict best selection."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('act_training_runner', Path(__file__).resolve().parents[1]/'train_act_ensemble.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RetentionTests(unittest.TestCase):
    def test_only_best_last_preserve_best_and_do_not_duplicate_current_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'checkpoints'
            store = runner.BestLast(root)
            def save(step, loss):
                def writer(path):
                    path.mkdir()
                    (path/'weights').write_text(str(step))
                    (path/'optimizer').write_text(str(step))
                store.save(step, loss, writer)
            save(5, 2.)
            self.assertEqual((root/'best/weights').stat().st_ino, (root/'last/weights').stat().st_ino)
            save(10, 3.)
            self.assertEqual((root/'best/weights').read_text(), '5')
            self.assertEqual((root/'last/optimizer').read_text(), '10')
            save(15, 2.) # Strict improvement, ties do not replace best.
            self.assertEqual(store.best_step, 5)
            save(20, 1.)
            self.assertEqual(store.best_step, 20)
            self.assertEqual({p.name for p in root.iterdir()}, {'best', 'last'})
            self.assertEqual(json.loads((root/'best/selection.json').read_text())['step'], 20)

    def test_failed_save_preserves_previous_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'checkpoints'
            store = runner.BestLast(root)
            def writer(path):
                path.mkdir()
                (path/'weights').write_text('original')
            store.save(5, 2., writer)
            def fail(path):
                writer(path)
                raise OSError('disk full')
            with self.assertRaises(OSError):
                store.save(10, 1., fail)
            for kind in ['best', 'last']:
                self.assertEqual((root/kind/'weights').read_text(), 'original')
            self.assertEqual(store.best_step, 5)
            with self.assertRaises(ValueError):
                store.save(10, float('nan'), writer)


if __name__ == '__main__': unittest.main()
