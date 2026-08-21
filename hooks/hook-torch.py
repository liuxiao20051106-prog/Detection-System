"""PyInstaller hook for the self-contained Windows PyPI torch wheel.

The upstream hook classifies every Torch DLL as a binary, so PyInstaller recursively
inspects their dependencies. With the supported Windows wheel this analysis can hang.
The wheel is self-contained, so copy the same DLLs as package data instead.
"""

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

datas = collect_data_files(
    "torch",
    excludes=["**/*.h", "**/*.hpp", "**/*.cuh", "**/*.lib", "**/*.cpp", "**/*.pyi"],
)
datas += collect_dynamic_libs("torch")
hiddenimports = collect_submodules("torch")
module_collection_mode = "pyz+py"
warn_on_missing_hiddenimports = False
