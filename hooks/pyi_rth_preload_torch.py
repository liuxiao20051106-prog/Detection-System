"""Load Torch before PyInstaller's PyQt runtime hook initializes Qt DLLs."""

import torch  # noqa: F401
