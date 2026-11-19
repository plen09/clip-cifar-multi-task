"""No network/CLIP weights required. Synthetic archives and tensor structures."""
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from src.offline_assets import ensure_offline_cifar100, inspect_vit_b32_safetensors


class OfflineAssetsTests(unittest.TestCase):
    def test_cifar100_archive_extraction_and_reuse(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            archive = td / 'cifar-100-python.tar.gz'
            with tarfile.open(archive, 'w:gz') as tar:
                for name in ('train', 'test', 'meta'):
                    contents = name.encode()
                    info = tarfile.TarInfo(f'cifar-100-python/{name}')
                    info.size = len(contents)
                    tar.addfile(info, io.BytesIO(contents))
            root = td / 'data'
            result = ensure_offline_cifar100(root, archive, verify_md5=False)
            self.assertEqual((result / 'train').read_bytes(), b'train')
            # Second call does not modify the existing extracted files.
            self.assertEqual(result, ensure_offline_cifar100(root, archive, verify_md5=False))

    def test_reject_archive_path_traversal(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            archive = td / 'cifar-100-python.tar.gz'
            with tarfile.open(archive, 'w:gz') as tar:
                contents = b'bad'
                info = tarfile.TarInfo('cifar-100-python/../../unsafe.txt')
                info.size = len(contents)
                tar.addfile(info, io.BytesIO(contents))
            with self.assertRaises(ValueError):
                ensure_offline_cifar100(td / 'data', archive, verify_md5=False)
            self.assertFalse((td / 'unsafe.txt').exists())

    def test_safetensors_valid_visual_shape(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / 'open_clip_model.safetensors'
            d = {
                'visual.conv1.weight': torch.zeros(768, 3, 32, 32),
                'visual.positional_embedding': torch.zeros(50, 768),
                'visual.proj': torch.zeros(768, 512),
            }
            # Dummy extras give the expected minimum visual key count.
            d.update({f'visual.dummy_tensor_{i}': torch.tensor([0.]) for i in range(98)})
            save_file(d, str(f))
            info = inspect_vit_b32_safetensors(f)
            self.assertEqual(info['visual_conv1_shape'], [768, 3, 32, 32])

    def test_safetensors_reject_other_patch_size(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / 'open_clip_model.safetensors'
            save_file({
                'visual.conv1.weight': torch.zeros(768, 3, 16, 16),
                'visual.positional_embedding': torch.zeros(50, 768),
                'visual.proj': torch.zeros(768, 512),
            }, str(f))
            with self.assertRaisesRegex(ValueError, 'Wrong visual architecture'):
                inspect_vit_b32_safetensors(f)


if __name__ == '__main__':
    unittest.main()
