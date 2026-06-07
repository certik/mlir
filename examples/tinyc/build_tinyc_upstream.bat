@echo off
setlocal enabledelayedexpansion

REM Build the tinyC example compiler against the upstream MLIR backend on
REM Windows. Mirrors tests\build_parser_upstream.bat -- same link line, just
REM substituting tinyC sources + driver for the parser sources.

for /f "delims=" %%i in ('"%CONDA_PREFIX%\Library\bin\llvm-config.exe" --link-static --libs support core analysis transformutils frontendopenmp webassemblycodegen webassemblyasmparser webassemblydesc webassemblydisassembler webassemblyinfo webassemblyutils target mc mcparser asmprinter codegen selectiondag globalisel bitwriter') do set LLVM_LIBS=%%i
if errorlevel 1 exit /b 1

if exist mlir_libs.rsp del mlir_libs.rsp
for %%f in ("%CONDA_PREFIX%\Library\lib\MLIR*.lib") do echo %%~nxf>>mlir_libs.rsp

set COREC_C=corec\base\io.c corec\base\buddy.c corec\base\arena.c corec\base\scratch.c corec\base\format.c corec\base\math.c corec\base\string.c corec\base\strbuf.c corec\base\mem.c corec\base\numconv.c corec\base\assert.c corec\base\exit.c
set TINYC_C=examples\tinyc\lex.c examples\tinyc\preprocess.c examples\tinyc\parse.c examples\tinyc\emit.c examples\tinyc\driver.c mlir_op_names.c mlir_lower_to_llvm.c mlir_translate_to_llvm_ir.c mlir_translate_to_wasm.c mlir_wasm_to_wat.c mlir_wasm_to_macho.c mlir_llvm_to_wasmssa.c mlir_wasmssa_to_wasmstack.c mlir_wasmstack_to_bin.c mlir_wasm_link.c mlir_wasm_to_wasmstack.c mlir_wasmstack_to_wasmssa.c mlir_wasmssa_to_llvm.c mlir_llvm_mem2reg.c mlir_llvm_load_cse.c mlir_llvm_arith_gvn.c mlir_llvm_dce.c mlir_llvm_to_aarch64.c mlir_aarch64_to_macho.c mlir_generic_printer.c mlir_lift_cf_to_scf.c

cl /nologo /std:c11 /Zc:preprocessor /MD /DTINYC_HAS_UPSTREAM /I corec /I . /c %COREC_C% %TINYC_C% tests\upstream_main.c
if errorlevel 1 exit /b 1

cl /nologo /std:c11 /Zc:preprocessor /MD /DPLATFORM_SKIP_ENTRY /I corec /I . /c corec\platform\platform_windows.c
if errorlevel 1 exit /b 1

cl /nologo /std:c++17 /MD /EHsc /GR- /I corec /I . /I "%CONDA_PREFIX%\Library\include" /c mlir_api_impl_upstream.cpp
if errorlevel 1 exit /b 1

link /nologo /out:tinyc.exe ^
    upstream_main.obj lex.obj preprocess.obj parse.obj emit.obj driver.obj ^
    mlir_op_names.obj mlir_lower_to_llvm.obj mlir_translate_to_llvm_ir.obj mlir_translate_to_wasm.obj mlir_wasm_to_wat.obj mlir_wasm_to_macho.obj mlir_llvm_to_wasmssa.obj mlir_wasmssa_to_wasmstack.obj mlir_wasmstack_to_bin.obj mlir_wasm_link.obj mlir_wasm_to_wasmstack.obj mlir_wasmstack_to_wasmssa.obj mlir_wasmssa_to_llvm.obj mlir_llvm_mem2reg.obj mlir_llvm_load_cse.obj mlir_llvm_arith_gvn.obj mlir_llvm_dce.obj mlir_llvm_to_aarch64.obj mlir_aarch64_to_macho.obj mlir_generic_printer.obj mlir_lift_cf_to_scf.obj mlir_api_impl_upstream.obj ^
    io.obj buddy.obj arena.obj scratch.obj format.obj math.obj string.obj strbuf.obj mem.obj numconv.obj assert.obj exit.obj platform_windows.obj ^
    /LIBPATH:"%CONDA_PREFIX%\Library\lib" @mlir_libs.rsp %LLVM_LIBS% ntdll.lib zlib.lib zstd.lib
if errorlevel 1 exit /b 1

endlocal
