@echo off
setlocal enabledelayedexpansion

REM Build the tinyC example compiler against the upstream MLIR backend on
REM Windows. Mirrors tests\build_parser_upstream.bat -- same link line, just
REM substituting tinyC sources + driver for the parser sources.

for /f "delims=" %%i in ('"%CONDA_PREFIX%\Library\bin\llvm-config.exe" --link-static --libs support core analysis transformutils frontendopenmp') do set LLVM_LIBS=%%i
if errorlevel 1 exit /b 1

if exist mlir_libs.rsp del mlir_libs.rsp
for %%f in ("%CONDA_PREFIX%\Library\lib\MLIR*.lib") do echo %%~nxf>>mlir_libs.rsp

set COREC_C=corec\base\io.c corec\base\buddy.c corec\base\arena.c corec\base\scratch.c corec\base\format.c corec\base\math.c corec\base\string.c corec\base\mem.c corec\base\numconv.c corec\base\assert.c corec\base\exit.c
set TINYC_C=examples\tinyc\lex.c examples\tinyc\preprocess.c examples\tinyc\parse.c examples\tinyc\emit.c examples\tinyc\driver.c mlir_op_names.c

cl /nologo /std:c11 /Zc:preprocessor /MD /I corec /I . /c %COREC_C% %TINYC_C% tests\upstream_main.c
if errorlevel 1 exit /b 1

cl /nologo /std:c11 /Zc:preprocessor /MD /DPLATFORM_SKIP_ENTRY /I corec /I . /c corec\platform\platform_windows.c
if errorlevel 1 exit /b 1

cl /nologo /std:c++17 /MD /EHsc /GR- /I corec /I . /I "%CONDA_PREFIX%\Library\include" /c mlir_api_impl_upstream.cpp
if errorlevel 1 exit /b 1

link /nologo /out:tinyc.exe ^
    upstream_main.obj lex.obj preprocess.obj parse.obj emit.obj driver.obj ^
    mlir_op_names.obj mlir_api_impl_upstream.obj ^
    io.obj buddy.obj arena.obj scratch.obj format.obj math.obj string.obj mem.obj numconv.obj assert.obj exit.obj platform_windows.obj ^
    /LIBPATH:"%CONDA_PREFIX%\Library\lib" @mlir_libs.rsp %LLVM_LIBS% ntdll.lib zlib.lib zstd.lib
if errorlevel 1 exit /b 1

endlocal
