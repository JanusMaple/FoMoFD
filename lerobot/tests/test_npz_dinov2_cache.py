"""Cached DINO extraction must work without a GitHub branch-discovery request."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "lerobot/policies/act_dinov2/src"))
sys.path.insert(0, str(REPO / "lerobot"))
spec = importlib.util.spec_from_file_location("npz_converter", REPO / "lerobot/utils/npz_to_le_v21.py")
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class DinoCacheTests(unittest.TestCase):
    def test_cached_hub_module_loads_when_network_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            cached = Path(tmp) / "facebookresearch_dinov2_main"
            cached.mkdir()
            (cached / "hubconf.py").write_text(
                "import torch\ndef dinov2_vitb14_reg():\n    return torch.nn.Identity()\n")
            with patch.object(torch.hub, "get_dir", return_value=tmp), \
                 patch.object(torch.hub, "urlopen", side_effect=AssertionError("Unexpected network request")):
                model = converter._load_dinov2_model("cpu")
            x = torch.ones(2, 3)
            self.assertTrue(torch.equal(model(x), x))
            self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
