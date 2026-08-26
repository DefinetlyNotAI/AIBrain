from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
gcc = shutil.which("gcc") or r"C:\Program Files\msys64\mingw64\bin\gcc.exe"
source = root / "src" / "native" / "connectome_native.c"
output = root / "src" / "native" / "aibrain_connectome.dll"
subprocess.run([gcc, "-O3", "-shared", "-std=c11", "-o", str(output), str(source)], check=True)
print(output)
