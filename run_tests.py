#!/usr/bin/env python

import os
import subprocess as sp
import sys
from typing import Dict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(ROOT_DIR))

from compiler_tester.tester import color, fg, log, run_test, style, tester_main

def run_cmd(cmd, cwd=None):
    print(f"+ {cmd}")
    process = sp.run(cmd, shell=True, cwd=cwd)
    if process.returncode != 0:
        print("Command failed.")
        exit(1)

def single_test(test: Dict, verbose: bool, no_llvm: bool, skip_run_with_dbg: bool,
                update_reference: bool, verify_hash: bool,
                no_color: bool, specific_backends=None,
                excluded_backends=None) -> None:
    def is_included(backend):
        return test.get(backend, False) \
            and (specific_backends is None or backend in specific_backends) \
            and (excluded_backends is None or backend not in excluded_backends)

    filename = test["filename"]
    show_verbose = "" if not verbose else "-v"
    tokens = is_included("tokens")
    ast = is_included("ast")
    ast_indent = is_included("ast_indent")
    ast_disable_style_suggestion = is_included("ast_disable_style_suggestion")
    ast_json = is_included("ast_json")
    ast_no_prescan = is_included("ast_no_prescan")
    ast_f90 = is_included("ast_f90")
    ast_cpp = is_included("ast_cpp")
    ast_cpp_hip = is_included("ast_cpp_hip")
    lookup_name = is_included("lookup_name")
    rename_symbol = is_included("rename_symbol")
    line = "-1"
    if is_included("line"):
        line = str(test["line"])
    column = "-1"
    if is_included("column"):
        column = str(test["column"])
    asr = is_included("asr")
    asr_ignore_pragma = is_included("asr_ignore_pragma")
    asr_implicit_typing = is_included("asr_implicit_typing")
    asr_disable_implicit_typing = is_included("asr_disable_implicit_typing")
    enable_and_disable_implicit_typing = is_included("enable_and_disable_implicit_typing")
    asr_implicit_interface = is_included("asr_implicit_interface")
    asr_implicit_interface_and_typing = is_included("asr_implicit_interface_and_typing")
    asr_implicit_argument_casting = is_included("asr_implicit_argument_casting")
    enable_disable_implicit_argument_casting = is_included("enable_disable_implicit_argument_casting")
    asr_implicit_interface_and_typing_with_llvm = is_included("asr_implicit_interface_and_typing_with_llvm")
    asr_disable_warnings = is_included("asr_disable_warnings")
    asr_disable_style_suggestion_and_warnings = is_included("asr_disable_style_suggestion_and_warnings")
    asr_enable_style_suggestion = is_included("asr_enable_style_suggestion")
    continue_compilation = is_included("continue_compilation")
    fixed_form_cc_asr = is_included("fixed_form_cc_asr")
    semantics_only_cc = is_included("semantics_only_cc")
    show_errors = is_included("show_errors")
    document_symbols = is_included("document_symbols")
    syntax_only_cc = is_included("syntax_only_cc")
    show_asr_with_cc = is_included("show_asr_with_cc")
    asr_use_loop_variable_after_loop = is_included("asr_use_loop_variable_after_loop")
    asr_preprocess = is_included("asr_preprocess")
    asr_indent = is_included("asr_indent")
    asr_json = is_included("asr_json")
    asr_clojure = is_included("asr_clojure")
    asr_openmp = is_included("asr_openmp")
    c_target_omp = is_included("c_target_omp")
    c_target_cuda = is_included("c_target_cuda")
    asr_logical_casting = is_included("asr_logical_casting")
    mod_to_asr = is_included("mod_to_asr")
    llvm = is_included("llvm")
    llvm_new_classes = is_included("llvm_new_classes")
    cpp = is_included("cpp")
    cpp_infer = is_included("cpp_infer")
    c = is_included("c")
    is_cumulative_pass = is_included("cumulative")
    julia = is_included("julia")
    wat = is_included("wat")
    obj = is_included("obj")
    x86 = is_included("x86")
    fortran = is_included("fortran")
    bin_ = is_included("bin")
    fast = is_included("fast")
    print_generic = is_included("print_generic")
    print_classic = is_included("print_classic")
    check_classic = is_included("check_classic")
    print_leading_space = is_included("print_leading_space")
    interactive = is_included("interactive")
    options = test.get("options", "")
    pass_ = test.get("pass", None)
    extrafiles = test.get("extrafiles", "").split(",")
    run = test.get("run")
    run_with_dbg = test.get("run_with_dbg")
    optimization_passes = ["flip_sign", "div_to_mul", "fma", "sign_from_value",
                           "inline_function_calls", "loop_unroll",
                           "dead_code_removal"]

    if pass_ is not None:
        pass_list = pass_.split(",")

        for _pass in pass_list:
            _pass = _pass.rstrip(" ").lstrip(" ")
            if (_pass not in ["do_loops", "global_stmts",
                        "transform_optional_argument_functions",
                        "array_op", "select_case",
                        "class_constructor", "implied_do_loops",
                        "pass_array_by_data", "init_expr", "where",
                        "nested_vars", "insert_deallocate", "openmp",
                        "array_struct_temporary"] and
                _pass not in optimization_passes):
                raise Exception(f"Unknown pass: {_pass}")
    if update_reference:
        log.debug(f"{color(style.bold)} UPDATE TEST: {color(style.reset)} {filename}")
    elif verify_hash:
        log.debug(f"{color(style.bold)} VERIFY HASH: {color(style.reset)} {filename}")
    else:
        log.debug(f"{color(style.bold)} START TEST: {color(style.reset)} {filename}")

    extra_args = ""

    if print_generic:
        run_test(
            filename,
            "print_generic",
            "./parser {infile} > {outfile}",
            filename,
            update_reference,
            verify_hash,
            extra_args)

    if print_classic:
        run_test(
            filename,
            "print_classic",
            "./parser {infile} --classic > {outfile}",
            filename,
            update_reference,
            verify_hash,
            extra_args)

    if check_classic:
        run_test(
            filename,
            "check_classic",
            "./parser {infile} --classic > {infile}2 && diff {infile} {infile}2",
            filename,
            update_reference,
            verify_hash,
            extra_args)

if __name__ == "__main__":
    tester_main("MLIR", single_test)
