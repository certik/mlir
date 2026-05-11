@echo off
setlocal enabledelayedexpansion

REM Build the tinyC example compiler against the NATIVE MLIR backend
REM (mlir_api_impl.c) on Windows. Mirrors examples\tinyc\build_tinyc_native.sh
REM but uses MSVC. No upstream LLVM/MLIR libraries are linked.
REM
REM Output binary: tinyc_native.exe.

set COREC_C=corec\base\io.c corec\base\buddy.c corec\base\arena.c corec\base\scratch.c corec\base\format.c corec\base\math.c corec\base\string.c corec\base\mem.c corec\base\numconv.c corec\base\assert.c corec\base\exit.c
set TINYC_C=examples\tinyc\lex.c examples\tinyc\preprocess.c examples\tinyc\parse.c examples\tinyc\emit.c examples\tinyc\driver.c
set NATIVE_C=mlir_api_impl.c mlir_op_names.c mlir_lower_to_llvm.c mlir_translate_to_llvm_ir.c mlir_translate_to_wasm.c mlir_wasm_pipeline_internal.c mlir_wasm_print.c mlir_lower_llvm_to_wasmssa.c mlir_stackify_wasmssa.c mlir_translate_wasmstack_to_binary.c tokenizer.c mlir_parser.c op_parsers.c mlir_classic_printer.c mlir_generic_printer.c

cl /nologo /std:c11 /Zc:preprocessor /MD /DMLIR_HAS_NATIVE_LOWERING /I corec /I . /c %COREC_C% %TINYC_C% %NATIVE_C% tests\upstream_main.c
if errorlevel 1 exit /b 1

cl /nologo /std:c11 /Zc:preprocessor /MD /DPLATFORM_SKIP_ENTRY /I corec /I . /c corec\platform\platform_windows.c
if errorlevel 1 exit /b 1

link /nologo /out:tinyc_native.exe ^
    upstream_main.obj lex.obj preprocess.obj parse.obj emit.obj driver.obj ^
    mlir_api_impl.obj mlir_op_names.obj mlir_lower_to_llvm.obj mlir_translate_to_llvm_ir.obj mlir_translate_to_wasm.obj mlir_wasm_pipeline_internal.obj mlir_wasm_print.obj mlir_lower_llvm_to_wasmssa.obj mlir_stackify_wasmssa.obj mlir_translate_wasmstack_to_binary.obj ^
    tokenizer.obj mlir_parser.obj op_parsers.obj mlir_classic_printer.obj mlir_generic_printer.obj ^
    io.obj buddy.obj arena.obj scratch.obj format.obj math.obj string.obj mem.obj numconv.obj assert.obj exit.obj platform_windows.obj ^
    ntdll.lib shell32.lib
if errorlevel 1 exit /b 1

endlocal
