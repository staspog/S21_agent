"""Пакет приложения S21 Agent."""

import os
import tempfile

# PyTorch's default cache lookup requires a passwd entry, which minimal runtime
# images may not have for their configured numeric UID. Respect explicit config.
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), f"s21-torch-inductor-{os.getuid()}"),
)
