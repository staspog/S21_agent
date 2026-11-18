import os
import pathlib
import subprocess
import sys
import tempfile
import unittest


class RuntimeCacheTests(unittest.TestCase):
    def test_default_and_explicit_cache(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        for configured in (None, '/tmp/custom-torch-cache'):
            env = os.environ.copy()
            env.pop('TORCHINDUCTOR_CACHE_DIR', None)
            if configured:
                env['TORCHINDUCTOR_CACHE_DIR'] = configured
            result = subprocess.run(
                [sys.executable, '-c', 'import src, os; print(os.environ["TORCHINDUCTOR_CACHE_DIR"])'],
                cwd=root, env=env, text=True, capture_output=True, check=True,
            )
            expected = configured or os.path.join(tempfile.gettempdir(), f's21-torch-inductor-{os.getuid()}')
            self.assertEqual(result.stdout.strip(), expected)
