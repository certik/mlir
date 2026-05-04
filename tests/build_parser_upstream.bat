@echo off
setlocal enabledelayedexpansion

for /f "delims=" %%i in ('"%CONDA_PREFIX%\Library\bin\llvm-config.exe" --link-static --libs support core') do set LLVM_LIBS=%%i
if errorlevel 1 exit /b 1

REM Use all available MLIR static libraries (mirrors the umbrella libMLIR
REM approach used on Linux/macOS). Use a linker response file because the
REM full list exceeds cmd.exe line length limit.
if exist mlir_libs.rsp del mlir_libs.rsp
for %%f in ("%CONDA_PREFIX%\Library\lib\MLIR*.lib") do echo %%~nxf>>mlir_libs.rsp

set COREC_C=corec\base\io.c corec\base\buddy.c corec\base\arena.c corec\base\scratch.c corec\base\format.c corec\base\math.c corec\base\string.c corec\base\mem.c corec\base\numconv.c corec\base\assert.c corec\base\exit.c
set PROJ_C=parser.c tokenizer.c mlir_parser.c mlir_classic_printer.c mlir_generic_printer.c op_parsers.c mlir_op_names.c

cl /nologo /std:c11 /Zc:preprocessor /MD /I corec /I . /c %COREC_C% %PROJ_C% tests\upstream_main.c
if errorlevel 1 exit /b 1

cl /nologo /std:c11 /Zc:preprocessor /MD /DPLATFORM_SKIP_ENTRY /I corec /I . /c corec\platform\platform_windows.c
if errorlevel 1 exit /b 1

cl /nologo /std:c++17 /MD /EHsc /GR- /I corec /I . /I "%CONDA_PREFIX%\Library\include" /c mlir_api_impl_upstream.cpp
if errorlevel 1 exit /b 1

link /nologo /out:parser_upstream.exe ^
    upstream_main.obj parser.obj tokenizer.obj mlir_parser.obj mlir_classic_printer.obj mlir_generic_printer.obj ^
    op_parsers.obj mlir_op_names.obj mlir_api_impl_upstream.obj ^
    io.obj buddy.obj arena.obj scratch.obj format.obj math.obj string.obj mem.obj numconv.obj assert.obj exit.obj platform_windows.obj ^
    /LIBPATH:"%CONDA_PREFIX%\Library\lib" @mlir_libs.rsp %LLVM_LIBS% ntdll.lib
if errorlevel 1 exit /b 1

endlocal
