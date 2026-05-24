@echo off
setlocal enabledelayedexpansion

REM Get LLVM static libs from llvm-config (Windows MSVC build emits .lib paths).
for /f "delims=" %%i in ('"%CONDA_PREFIX%\Library\bin\llvm-config.exe" --link-static --libs support core analysis transformutils frontendopenmp webassemblycodegen webassemblyasmparser webassemblydesc webassemblydisassembler webassemblyinfo webassemblyutils target mc mcparser asmprinter codegen selectiondag globalisel bitwriter') do set LLVM_LIBS=%%i
if errorlevel 1 exit /b 1

REM Link all MLIR static libraries shipped by conda-forge. This pulls in
REM every dialect (Func, Arith, MemRef, SCF, CF, ...) the upstream backend
REM registers, and avoids enumerating transitive deps by hand. Use a
REM linker response file because the full list exceeds cmd.exe's line
REM length limit.
if exist mlir_libs.rsp del mlir_libs.rsp
for %%f in ("%CONDA_PREFIX%\Library\lib\MLIR*.lib") do echo %%~nxf>>mlir_libs.rsp

set COREC_C=corec\base\io.c corec\base\buddy.c corec\base\arena.c corec\base\scratch.c corec\base\format.c corec\base\math.c corec\base\string.c corec\base\strbuf.c corec\base\mem.c corec\base\numconv.c corec\base\assert.c corec\base\exit.c
set PROJ_C=tests\cross\driver.c mlir_generic_printer.c mlir_op_names.c mlir_lower_to_llvm.c mlir_translate_to_llvm_ir.c mlir_wasm_to_wat.c mlir_llvm_to_wasmssa.c mlir_wasmssa_to_wasmstack.c mlir_wasmstack_to_bin.c mlir_lift_cf_to_scf.c

cl /nologo /std:c11 /Zc:preprocessor /MD /I corec /I . /c %COREC_C% %PROJ_C% tests\upstream_main.c
if errorlevel 1 exit /b 1

cl /nologo /std:c11 /Zc:preprocessor /MD /DPLATFORM_SKIP_ENTRY /I corec /I . /c corec\platform\platform_windows.c
if errorlevel 1 exit /b 1

cl /nologo /std:c++17 /MD /EHsc /GR- /I corec /I . /I "%CONDA_PREFIX%\Library\include" /c mlir_api_impl_upstream.cpp
if errorlevel 1 exit /b 1

link /nologo /out:cross_upstream.exe ^
    upstream_main.obj driver.obj mlir_generic_printer.obj mlir_op_names.obj mlir_lower_to_llvm.obj mlir_translate_to_llvm_ir.obj mlir_wasm_to_wat.obj mlir_llvm_to_wasmssa.obj mlir_wasmssa_to_wasmstack.obj mlir_wasmstack_to_bin.obj mlir_lift_cf_to_scf.obj mlir_api_impl_upstream.obj ^
    io.obj buddy.obj arena.obj scratch.obj format.obj math.obj string.obj strbuf.obj mem.obj numconv.obj assert.obj exit.obj platform_windows.obj ^
    /LIBPATH:"%CONDA_PREFIX%\Library\lib" @mlir_libs.rsp %LLVM_LIBS% ntdll.lib zlib.lib zstd.lib
if errorlevel 1 exit /b 1

endlocal
