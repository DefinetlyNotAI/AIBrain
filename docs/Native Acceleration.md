# Native acceleration

## Purpose

The activity-decay, edge-activity, and regional-aggregation hot paths live in `src/native/c/connectome_native.c`. They
compile into `dll/aibrain_connectome.dll` and are loaded through `src/native/wrapper/connectome.py`. If the DLL is
missing or cannot be loaded, AIBrain falls back to NumPy implementations so the application remains usable.

The DLL accelerates visual activity bookkeeping. It does not implement the Qt window, OpenGL renderer, or language-model
inference.

## Build tool

After changing the C source, run from the project root:

```powershell
py cli\build_native.py
```

The tool is color-coded in an interactive terminal and:

1. discovers an explicitly supplied compiler, `$env:CC`, GCC, Clang, MSVC, or a common MSYS2 MinGW-w64 GCC location;
2. compiles the x64 Windows DLL with warnings enabled and optimized release flags;
3. checks that the output is a valid-sized `MZ`/PE file;
4. loads it through `ctypes.WinDLL`; and
5. verifies `decay_and_count` and `edge_activity` are exported.

Useful options:

```powershell
py cli\build_native.py --debug
py cli\build_native.py --clean
py cli\build_native.py --compiler "C:\path\to\gcc.exe"
```

`--debug` requests an unoptimized debug build. `--clean` deletes the existing DLL before compiling. `--compiler` is
useful when more than one supported toolchain is installed.

## Toolchain failures

Install one of the following when discovery cannot find a compiler: Visual Studio Build Tools with the C++ build tools,
LLVM/Clang, or MSYS2 MinGW-w64 GCC. Open a developer command prompt for MSVC so `cl.exe` is on `PATH`, or pass the
compiler's full executable path with `--compiler`.
