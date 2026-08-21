# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path

import PyQt5
from PyInstaller.utils.hooks import collect_data_files
import PyInstaller.building.build_main as build_main


datas = collect_data_files("ultralytics")
datas += [
    ("models/best.pt", "models"),
    ("UIProgram/style.css", "UIProgram"),
]

python_root = Path(sys.base_prefix)
python_dll_directory = python_root / "DLLs"
python_binaries = [
    (str(path), ".")
    for path in python_root.glob("*.dll")
    if not path.name.lower().startswith("vcruntime140")
]
python_binaries.extend((str(path), ".") for path in python_dll_directory.glob("*.dll"))
system_directory = Path(os.environ["SystemRoot"]) / "System32"
for pattern in ("msvcp140*.dll", "vcruntime140*.dll", "concrt140.dll"):
    python_binaries.extend((str(path), ".") for path in system_directory.glob(pattern))
qt_bin_directory = Path(PyQt5.__file__).parent / "Qt5" / "bin"
python_binaries.extend(
    (str(path), "PyQt5/Qt5/bin")
    for path in qt_bin_directory.glob("*.dll")
    if not path.name.lower().startswith(("msvcp140", "vcruntime140", "concrt140"))
)

# The supported Windows wheels already bundle their runtime DLLs. Importing every
# collected Torch subpackage for a second dependency pass can deadlock in an
# isolated PyInstaller process, so retain the DLLs supplied by package hooks and
# skip only that recursive pass. The release smoke test must open the executable.
build_main.find_binary_dependencies = lambda binaries, packages, patterns: []

analysis = Analysis(
    ["MainProgram.py"],
    binaries=python_binaries,
    datas=datas,
    hiddenimports=[],
    hookspath=["hooks"],
    runtime_hooks=["hooks/pyi_rth_preload_torch.py"],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="Detection-System",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=os.getenv("DETECTION_BUILD_CONSOLE") == "1",
)
