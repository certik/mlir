// tests/acceptance/test_acceptance.c
//
// Acceptance test driver for #357: runs the C ports of LFortran's
// `llvm_rsqrt` and `llvm_reductions` passes via MLIR_RunPasses on both the
// native (mlir_api_impl.c) and upstream (mlir_api_impl_upstream.cpp) backends.
// Both binaries must produce byte-identical output and pass MLIR_VerifyOp.
//
// Pass `--bench` to run timing benchmarks of both passes.

#include <stdio.h>
#include <string.h>
#include <base/arena.h>
#include <base/string.h>
#include <platform/platform.h>
#include "mlir_api.h"
#include "mlir_passes.h"
#include "pass_llvm_rsqrt.h"
#include "pass_llvm_reductions.h"

static MLIR_OpHandle make_module(MLIR_Context *ctx, MLIR_BlockHandle *out_body) {
    MLIR_OpSpec ms = { OP_TYPE_UNREGISTERED };
    ms.type = OP_TYPE_MODULE;
    ms.n_regions = 1;
    MLIR_OpHandle mod = MLIR_CreateOpFromSpec(ctx, &ms);
    MLIR_BlockHandle mb = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, MLIR_GetOpRegion(ctx, mod, 0), mb);
    *out_body = mb;
    return mod;
}

static MLIR_OpHandle make_llvm_func(MLIR_Context *ctx, MLIR_BlockHandle mod_body,
                                    string name, MLIR_TypeHandle ret_ty,
                                    const MLIR_TypeHandle *arg_tys, size_t n_args,
                                    MLIR_BlockHandle *out_entry) {
    MLIR_TypeHandle fty = MLIR_CreateTypeLLVMFunction(ctx, ret_ty, arg_tys, n_args, false);
    MLIR_NamedAttr attrs[2];
    attrs[0] = MLIR_Named(MLIR_KEY_SYM_NAME, MLIR_CreateAttributeString(ctx, name));
    attrs[1] = MLIR_Named(MLIR_KEY_FUNCTION_TYPE, MLIR_CreateAttributeType(ctx, fty));
    MLIR_OpSpec fs = { OP_TYPE_UNREGISTERED };
    fs.type = OP_TYPE_LLVM_FUNC;
    fs.attrs = attrs;
    fs.n_attrs = 2;
    fs.n_regions = 1;
    MLIR_OpHandle fn = MLIR_CreateOpFromSpec(ctx, &fs);
    MLIR_AppendBlockOp(ctx, mod_body, fn);
    MLIR_BlockHandle entry = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, MLIR_GetOpRegion(ctx, fn, 0), entry);
    for (size_t i = 0; i < n_args; i++)
        MLIR_AddBlockArgument(ctx, entry, arg_tys[i], MLIR_INVALID_HANDLE);
    *out_entry = entry;
    return fn;
}

static size_t count_ops_of_type(MLIR_Context *ctx, MLIR_OpHandle fn, MLIR_OpType t) {
    size_t cnt = 0;
    MLIR_RegionHandle r = MLIR_GetOpRegion(ctx, fn, 0);
    for (MLIR_BlockHandle b = MLIR_GetRegionFirstBlock(ctx, r); b; b = MLIR_GetBlockNext(ctx, b))
        for (MLIR_OpHandle op = MLIR_GetBlockFirstOp(ctx, b); op; op = MLIR_GetOpNext(ctx, op))
            if (MLIR_GetOpType(ctx, op) == t) cnt++;
    return cnt;
}

static size_t count_intrinsic_calls(MLIR_Context *ctx, MLIR_OpHandle fn, string name) {
    size_t cnt = 0;
    MLIR_RegionHandle r = MLIR_GetOpRegion(ctx, fn, 0);
    for (MLIR_BlockHandle b = MLIR_GetRegionFirstBlock(ctx, r); b; b = MLIR_GetBlockNext(ctx, b))
        for (MLIR_OpHandle op = MLIR_GetBlockFirstOp(ctx, b); op; op = MLIR_GetOpNext(ctx, op))
            if (MLIR_GetOpType(ctx, op) == OP_TYPE_LLVM_CALL_INTRINSIC &&
                str_eq(MLIR_GetCallee(ctx, op), name))
                cnt++;
    return cnt;
}

// Build a float loop reduction:
//   entry: br loop(0.0, 0)
//   loop(%ax, %i):
//     %inv_r = 1.0 / sqrt(%r2)
//     %s = %m * (%inv_r * %inv_r * %inv_r)
//     %ax_next = %ax + %dx * %s
//     cond_br %cond, loop(%ax_next, %i1), exit(%ax_next)
//   exit(%res): return %res
static MLIR_OpHandle build_rsqrt_f32_loop(MLIR_Context *ctx, MLIR_OpHandle *out_fn) {
    MLIR_BlockHandle mb;
    MLIR_OpHandle mod = make_module(ctx, &mb);
    MLIR_TypeHandle f32 = MLIR_CreateTypeFloat(ctx, 32, false);
    MLIR_TypeHandle i1 = MLIR_CreateTypeInteger(ctx, 1, false);
    MLIR_TypeHandle args[4] = { f32, f32, f32, i1 };
    MLIR_BlockHandle entry;
    MLIR_OpHandle fn = make_llvm_func(ctx, mb, str_lit("nbody_f32"), f32, args, 4, &entry);
    MLIR_RegionHandle r = MLIR_GetOpRegion(ctx, fn, 0);
    MLIR_BlockHandle loop = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle exit_b = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, r, loop);
    MLIR_AppendRegionBlock(ctx, r, exit_b);

    MLIR_ValueHandle r2 = MLIR_GetBlockArg(ctx, entry, 0);
    MLIR_ValueHandle m = MLIR_GetBlockArg(ctx, entry, 1);
    MLIR_ValueHandle dx = MLIR_GetBlockArg(ctx, entry, 2);
    MLIR_ValueHandle cond = MLIR_GetBlockArg(ctx, entry, 3);

    MLIR_Builder eb = MLIR_BuilderAtEnd(ctx, entry);
    MLIR_ValueHandle zero = MLIR_BuildConstantFloat(&eb, OP_TYPE_LLVM_MLIR_CONSTANT, f32, 0.0);
    MLIR_BuildBr(&eb, OP_TYPE_LLVM_BR, loop, &zero, 1);

    MLIR_ValueHandle ax = MLIR_AddBlockArgument(ctx, loop, f32, MLIR_INVALID_HANDLE);
    MLIR_Builder lb = MLIR_BuilderAtEnd(ctx, loop);
    MLIR_ValueHandle one = MLIR_BuildConstantFloat(&lb, OP_TYPE_LLVM_MLIR_CONSTANT, f32, 1.0);
    MLIR_ValueHandle sq = MLIR_BuildFUnary(&lb, OP_TYPE_LLVM_INTR_SQRT, f32, r2, MLIR_FMF_FAST);
    MLIR_ValueHandle inv_r = MLIR_BuildFBinary(&lb, OP_TYPE_LLVM_FDIV, f32, one, sq, MLIR_FMF_FAST);
    MLIR_ValueHandle inv_r2 = MLIR_BuildFBinary(&lb, OP_TYPE_LLVM_FMUL, f32, inv_r, inv_r, MLIR_FMF_FAST);
    MLIR_ValueHandle inv_r3 = MLIR_BuildFBinary(&lb, OP_TYPE_LLVM_FMUL, f32, inv_r2, inv_r, MLIR_FMF_FAST);
    MLIR_ValueHandle s = MLIR_BuildFBinary(&lb, OP_TYPE_LLVM_FMUL, f32, m, inv_r3, MLIR_FMF_FAST);
    MLIR_ValueHandle term = MLIR_BuildFBinary(&lb, OP_TYPE_LLVM_FMUL, f32, dx, s, MLIR_FMF_FAST);
    MLIR_ValueHandle ax_next = MLIR_BuildFBinary(&lb, OP_TYPE_LLVM_FADD, f32, ax, term, MLIR_FMF_FAST);
    MLIR_BuildCondBr(&lb, OP_TYPE_LLVM_COND_BR, cond, loop, &ax_next, 1, exit_b, &ax_next, 1);

    MLIR_ValueHandle res = MLIR_AddBlockArgument(ctx, exit_b, f32, MLIR_INVALID_HANDLE);
    MLIR_Builder xb = MLIR_BuilderAtEnd(ctx, exit_b);
    MLIR_OpSpec rs = { OP_TYPE_UNREGISTERED };
    rs.type = OP_TYPE_LLVM_RETURN;
    rs.operands = &res;
    rs.n_operands = 1;
    MLIR_BuildOp(&xb, &rs);

    *out_fn = fn;
    return mod;
}

static MLIR_ValueHandle build_shuf4(MLIR_Builder *b, MLIR_TypeHandle vty,
                                    MLIR_ValueHandle a, MLIR_ValueHandle c,
                                    int32_t m0, int32_t m1, int32_t m2, int32_t m3) {
    int32_t mask[4] = { m0, m1, m2, m3 };
    MLIR_NamedAttr attr = MLIR_Named(MLIR_KEY_MASK,
        MLIR_CreateAttributeDenseI32Array(b->ctx, mask, 4));
    MLIR_ValueHandle ops[2] = { a, c };
    MLIR_OpSpec s = { OP_TYPE_UNREGISTERED };
    s.type = OP_TYPE_LLVM_SHUFFLEVECTOR;
    s.operands = ops;
    s.n_operands = 2;
    s.result_types = &vty;
    s.n_results = 1;
    s.attrs = &attr;
    s.n_attrs = 1;
    return MLIR_GetOpResult(b->ctx, MLIR_BuildOp(b, &s), 0);
}

// Build the SLP 3-component sum-of-squares pattern over four <4 x double> vectors.
static MLIR_OpHandle build_reduction_func(MLIR_Context *ctx, bool force_lane3_zero,
                                          MLIR_OpHandle *out_fn) {
    MLIR_BlockHandle mb;
    MLIR_OpHandle mod = make_module(ctx, &mb);
    MLIR_TypeHandle f64 = MLIR_CreateTypeFloat(ctx, 64, false);
    MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
    int64_t d4[1] = { 4 };
    MLIR_TypeHandle v4f64 = MLIR_CreateTypeVector(ctx, d4, 1, f64);
    MLIR_TypeHandle args[4] = { v4f64, v4f64, v4f64, v4f64 };
    MLIR_BlockHandle entry;
    MLIR_OpHandle fn = make_llvm_func(ctx, mb,
        force_lane3_zero ? str_lit("red_zero_lane3") : str_lit("red_unknown_lane3"),
        v4f64, args, 4, &entry);

    MLIR_Builder b = MLIR_BuilderAtEnd(ctx, entry);
    MLIR_ValueHandle v[4];
    for (int i = 0; i < 4; i++) v[i] = MLIR_GetBlockArg(ctx, entry, (size_t)i);

    if (force_lane3_zero) {
        MLIR_ValueHandle z = MLIR_BuildConstantFloat(&b, OP_TYPE_LLVM_MLIR_CONSTANT, f64, 0.0);
        MLIR_ValueHandle idx3 = MLIR_BuildConstantInt(&b, OP_TYPE_LLVM_MLIR_CONSTANT, i32, 3);
        for (int i = 0; i < 4; i++) {
            MLIR_ValueHandle iops[3] = { v[i], z, idx3 };
            MLIR_OpSpec is = { OP_TYPE_UNREGISTERED };
            is.type = OP_TYPE_LLVM_INSERTELEMENT;
            is.operands = iops;
            is.n_operands = 3;
            is.result_types = &v4f64;
            is.n_results = 1;
            v[i] = MLIR_GetOpResult(ctx, MLIR_BuildOp(&b, &is), 0);
        }
    }

    MLIR_ValueHandle sq[3];
    for (int lane = 0; lane < 3; lane++) {
        MLIR_ValueHandle g1 = build_shuf4(&b, v4f64, v[0], v[1], lane, lane + 4, -1, -1);
        MLIR_ValueHandle g2 = build_shuf4(&b, v4f64, g1, v[2], 0, 1, lane + 4, -1);
        MLIR_ValueHandle g3 = build_shuf4(&b, v4f64, g2, v[3], 0, 1, 2, lane + 4);
        sq[lane] = MLIR_BuildFBinary(&b, OP_TYPE_LLVM_FMUL, v4f64, g3, g3, MLIR_FMF_FAST);
    }
    MLIR_ValueHandle s01 = MLIR_BuildFBinary(&b, OP_TYPE_LLVM_FADD, v4f64, sq[0], sq[1], MLIR_FMF_FAST);
    MLIR_ValueHandle sum = MLIR_BuildFBinary(&b, OP_TYPE_LLVM_FADD, v4f64, s01, sq[2], MLIR_FMF_FAST);

    MLIR_OpSpec rs = { OP_TYPE_UNREGISTERED };
    rs.type = OP_TYPE_LLVM_RETURN;
    rs.operands = &sum;
    rs.n_operands = 1;
    MLIR_BuildOp(&b, &rs);

    *out_fn = fn;
    return mod;
}

static void run_benchmarks(void) {
    const int N = 500;
    MLIR_TargetInfo x86_avx = MLIR_TargetFromTriple(
        str_lit("x86_64-unknown-linux-gnu"),
        MLIR_FEATURE_SSE2 | MLIR_FEATURE_AVX | MLIR_FEATURE_AVX512F);

    Arena *arena = arena_create(32 * 1024 * 1024);
    MLIR_ContextOptions copts = {0};
    copts.arena = arena;
    MLIR_Context *ctx = MLIR_CreateContext(&copts);

    // Benchmark llvm_rsqrt on N functions.
    MLIR_OpHandle *rsqrt_mods = arena_new_array(arena, MLIR_OpHandle, N);
    for (int i = 0; i < N; i++) {
        MLIR_OpHandle fn;
        rsqrt_mods[i] = build_rsqrt_f32_loop(ctx, &fn);
    }
    MLIR_Pass rsqrt_p = MLIR_PassLLVMRsqrt();
    MLIR_PipelineOptions opts = {0};
    opts.target = &x86_avx;
    uint64_t t0 = platform_clock_ns();
    for (int i = 0; i < N; i++)
        MLIR_RunPasses(ctx, rsqrt_mods[i], &rsqrt_p, 1, &opts);
    double rsqrt_ms = (double)(platform_clock_ns() - t0) * 1e-6;

    // Benchmark llvm_reductions on N functions.
    MLIR_OpHandle *red_mods = arena_new_array(arena, MLIR_OpHandle, N);
    for (int i = 0; i < N; i++) {
        MLIR_OpHandle fn;
        red_mods[i] = build_reduction_func(ctx, (i & 1) != 0, &fn);
    }
    MLIR_Pass red_p = MLIR_PassLLVMReductions();
    t0 = platform_clock_ns();
    for (int i = 0; i < N; i++)
        MLIR_RunPasses(ctx, red_mods[i], &red_p, 1, &opts);
    double red_ms = (double)(platform_clock_ns() - t0) * 1e-6;

    printf("bench llvm_rsqrt (%d funcs):      %.3f ms (%.3f us/func)\n",
           N, rsqrt_ms, rsqrt_ms * 1000.0 / N);
    printf("bench llvm_reductions (%d funcs): %.3f ms (%.3f us/func)\n",
           N, red_ms, red_ms * 1000.0 / N);

    MLIR_DestroyContext(ctx);
    arena_destroy(arena);
}

int main(int argc, char **argv) {
    platform_init(argc, argv, NULL);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bench") == 0) {
            run_benchmarks();
            return 0;
        }
    }

    Arena *arena = arena_create(8 * 1024 * 1024);
    MLIR_ContextOptions copts = {0};
    copts.arena = arena;
    MLIR_Context *ctx = MLIR_CreateContext(&copts);

    MLIR_TargetInfo x86_avx512 = MLIR_TargetFromTriple(
        str_lit("x86_64-unknown-linux-gnu"),
        MLIR_FEATURE_SSE2 | MLIR_FEATURE_AVX | MLIR_FEATURE_AVX2 | MLIR_FEATURE_AVX512F);
    MLIR_TargetInfo x86_sse2 = MLIR_TargetFromTriple(
        str_lit("x86_64-unknown-linux-gnu"), MLIR_FEATURE_SSE2);
    MLIR_TargetInfo arm64 = MLIR_TargetFromTriple(
        str_lit("aarch64-apple-darwin"), MLIR_FEATURE_NEON);

    MLIR_Pass rsqrt_p = MLIR_PassLLVMRsqrt();
    MLIR_Pass red_p = MLIR_PassLLVMReductions();

    // 1. llvm_rsqrt on f32 loop reduction: factor hoisted out of loop.
    {
        MLIR_OpHandle fn;
        MLIR_OpHandle mod = build_rsqrt_f32_loop(ctx, &fn);
        MLIR_PipelineOptions opts = {0};
        opts.target = &x86_avx512;
        opts.verify_each = true;
        bool ok = MLIR_RunPasses(ctx, mod, &rsqrt_p, 1, &opts);
        string pt = MLIR_GetFunctionPassthrough(ctx, fn, str_lit("reciprocal-estimates"));
        // Exit block must contain the single hoisted materialize fmul (* -0.125).
        MLIR_RegionHandle r = MLIR_GetOpRegion(ctx, fn, 0);
        MLIR_BlockHandle exit_b = MLIR_GetRegionLastBlock(ctx, r);
        MLIR_OpHandle ret = MLIR_GetBlockLastOp(ctx, exit_b);
        MLIR_ValueHandle ret_v = MLIR_GetOpOperand(ctx, ret, 0);
        bool hoisted_factor = false;
        if (MLIR_GetValueKind(ctx, ret_v) == MLIR_VALUE_OP_RESULT) {
            MLIR_OpHandle def = MLIR_GetValueDefiningOp(ctx, ret_v);
            double k = 0.0;
            if (MLIR_GetOpType(ctx, def) == OP_TYPE_LLVM_FMUL &&
                MLIR_MatchConstantFloat(ctx, MLIR_GetOpOperand(ctx, def, 1), -1, &k) &&
                k == -0.125)
                hoisted_factor = true;
        }
        printf("1. rsqrt f32 loop reduction: %s\n",
               (ok && str_eq(pt, str_lit("sqrtf:0,vec-sqrtf:0")) && hoisted_factor) ? "yes" : "no");
    }

    // 2. llvm_rsqrt on AVX-512 <8 x double>: emits x86.avx512.rsqrt14.pd.512.
    {
        MLIR_BlockHandle mb;
        MLIR_OpHandle mod = make_module(ctx, &mb);
        MLIR_TypeHandle f64 = MLIR_CreateTypeFloat(ctx, 64, false);
        int64_t d8[1] = { 8 };
        MLIR_TypeHandle v8f64 = MLIR_CreateTypeVector(ctx, d8, 1, f64);
        MLIR_BlockHandle entry;
        MLIR_OpHandle fn = make_llvm_func(ctx, mb, str_lit("rsqrt_f64x8"), v8f64, &v8f64, 1, &entry);
        MLIR_Builder b = MLIR_BuilderAtEnd(ctx, entry);
        MLIR_ValueHandle x = MLIR_GetBlockArg(ctx, entry, 0);
        MLIR_ValueHandle one = MLIR_BuildConstantFloat(&b, OP_TYPE_LLVM_MLIR_CONSTANT, v8f64, 1.0);
        MLIR_ValueHandle sq = MLIR_BuildFUnary(&b, OP_TYPE_LLVM_INTR_SQRT, v8f64, x, MLIR_FMF_FAST);
        MLIR_ValueHandle inv = MLIR_BuildFBinary(&b, OP_TYPE_LLVM_FDIV, v8f64, one, sq, MLIR_FMF_FAST);
        MLIR_OpSpec rs = { OP_TYPE_UNREGISTERED };
        rs.type = OP_TYPE_LLVM_RETURN;
        rs.operands = &inv;
        rs.n_operands = 1;
        MLIR_BuildOp(&b, &rs);

        MLIR_PipelineOptions opts = {0};
        opts.target = &x86_avx512;
        opts.verify_each = true;
        bool ok = MLIR_RunPasses(ctx, mod, &rsqrt_p, 1, &opts);
        size_t n_intrin = count_intrinsic_calls(ctx, fn, str_lit("llvm.x86.avx512.rsqrt14.pd.512"));
        bool no_f32_attr = !MLIR_HasFunctionPassthrough(ctx, fn, str_lit("reciprocal-estimates"));
        printf("2. rsqrt f64x8 AVX-512: %s\n", (ok && n_intrin == 1 && no_f32_attr) ? "yes" : "no");
    }

    // 3. llvm_rsqrt skipped on AArch64 and when llvm.intr.pow is present.
    {
        MLIR_OpHandle fn_arm, fn_pow;
        MLIR_OpHandle mod_arm = build_rsqrt_f32_loop(ctx, &fn_arm);
        MLIR_PipelineOptions opts_arm = {0};
        opts_arm.target = &arm64;
        opts_arm.verify_each = true;
        bool ok1 = MLIR_RunPasses(ctx, mod_arm, &rsqrt_p, 1, &opts_arm) &&
                   !MLIR_HasFunctionPassthrough(ctx, fn_arm, str_lit("reciprocal-estimates"));

        MLIR_BlockHandle mb;
        MLIR_OpHandle mod_pow = make_module(ctx, &mb);
        MLIR_TypeHandle f32 = MLIR_CreateTypeFloat(ctx, 32, false);
        MLIR_BlockHandle entry;
        fn_pow = make_llvm_func(ctx, mb, str_lit("has_pow"), f32, &f32, 1, &entry);
        MLIR_Builder b = MLIR_BuilderAtEnd(ctx, entry);
        MLIR_ValueHandle x = MLIR_GetBlockArg(ctx, entry, 0);
        MLIR_ValueHandle one = MLIR_BuildConstantFloat(&b, OP_TYPE_LLVM_MLIR_CONSTANT, f32, 1.0);
        MLIR_ValueHandle sq = MLIR_BuildFUnary(&b, OP_TYPE_LLVM_INTR_SQRT, f32, x, MLIR_FMF_FAST);
        MLIR_ValueHandle inv = MLIR_BuildFBinary(&b, OP_TYPE_LLVM_FDIV, f32, one, sq, MLIR_FMF_FAST);
        MLIR_ValueHandle pw = MLIR_BuildFBinary(&b, OP_TYPE_LLVM_INTR_POW, f32, inv, x, MLIR_FMF_FAST);
        MLIR_OpSpec rs = { OP_TYPE_UNREGISTERED };
        rs.type = OP_TYPE_LLVM_RETURN;
        rs.operands = &pw;
        rs.n_operands = 1;
        MLIR_BuildOp(&b, &rs);
        MLIR_PipelineOptions opts_x86 = {0};
        opts_x86.target = &x86_avx512;
        opts_x86.verify_each = true;
        bool ok2 = MLIR_RunPasses(ctx, mod_pow, &rsqrt_p, 1, &opts_x86) &&
                   !MLIR_HasFunctionPassthrough(ctx, fn_pow, str_lit("reciprocal-estimates"));
        printf("3. rsqrt skipped on arm64/pow: %s\n", (ok1 && ok2) ? "yes" : "no");
    }

    // 4. llvm_reductions with unknown lane 3: emits 2x hadd + 4 zero-mask shuffles.
    {
        MLIR_OpHandle fn;
        MLIR_OpHandle mod = build_reduction_func(ctx, false, &fn);
        size_t shuf_before = count_ops_of_type(ctx, fn, OP_TYPE_LLVM_SHUFFLEVECTOR);
        MLIR_PipelineOptions opts = {0};
        opts.target = &x86_avx512;
        opts.verify_each = true;
        bool ok = MLIR_RunPasses(ctx, mod, &red_p, 1, &opts);
        size_t n_hadd = count_intrinsic_calls(ctx, fn, str_lit("llvm.x86.avx.hadd.pd.256"));
        size_t shuf_after = count_ops_of_type(ctx, fn, OP_TYPE_LLVM_SHUFFLEVECTOR);
        // 4 zero-mask shuffles + 2 combine shuffles = +6 shuffles.
        printf("4. reductions unknown lane3: %s\n",
               (ok && n_hadd == 2 && shuf_after == shuf_before + 6) ? "yes" : "no");
    }

    // 5. llvm_reductions with provably-zero lane 3: omits the 4 zero-mask shuffles.
    {
        MLIR_OpHandle fn;
        MLIR_OpHandle mod = build_reduction_func(ctx, true, &fn);
        size_t shuf_before = count_ops_of_type(ctx, fn, OP_TYPE_LLVM_SHUFFLEVECTOR);
        MLIR_PipelineOptions opts = {0};
        opts.target = &x86_avx512;
        opts.verify_each = true;
        bool ok = MLIR_RunPasses(ctx, mod, &red_p, 1, &opts);
        size_t n_hadd = count_intrinsic_calls(ctx, fn, str_lit("llvm.x86.avx.hadd.pd.256"));
        size_t shuf_after = count_ops_of_type(ctx, fn, OP_TYPE_LLVM_SHUFFLEVECTOR);
        // Only the 2 combine shuffles = +2 shuffles.
        printf("5. reductions zero lane3: %s\n",
               (ok && n_hadd == 2 && shuf_after == shuf_before + 2) ? "yes" : "no");
    }

    // 6. llvm_reductions skipped when target lacks AVX or fn overrides -avx.
    {
        MLIR_OpHandle fn1, fn2;
        MLIR_OpHandle mod1 = build_reduction_func(ctx, false, &fn1);
        MLIR_PipelineOptions opts1 = {0};
        opts1.target = &x86_sse2;
        opts1.verify_each = true;
        bool ok1 = MLIR_RunPasses(ctx, mod1, &red_p, 1, &opts1) &&
                   count_intrinsic_calls(ctx, fn1, str_lit("llvm.x86.avx.hadd.pd.256")) == 0;

        MLIR_OpHandle mod2 = build_reduction_func(ctx, false, &fn2);
        MLIR_SetFunctionPassthrough(ctx, fn2, str_lit("target-features"), str_lit("-avx"));
        MLIR_PipelineOptions opts2 = {0};
        opts2.target = &x86_avx512;
        opts2.verify_each = true;
        bool ok2 = MLIR_RunPasses(ctx, mod2, &red_p, 1, &opts2) &&
                   count_intrinsic_calls(ctx, fn2, str_lit("llvm.x86.avx.hadd.pd.256")) == 0;
        printf("6. reductions skipped without AVX: %s\n", (ok1 && ok2) ? "yes" : "no");
    }

    MLIR_DestroyContext(ctx);
    arena_destroy(arena);
    return 0;
}
