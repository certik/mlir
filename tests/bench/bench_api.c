// tests/bench/bench_api.c
//
// Microbenchmarks and IR memory accounting for the public MLIR C API
// (#343, #358). Compiled twice at -O2:
//   - bench_native   (mlir_api_impl.c)
//   - bench_upstream (mlir_api_impl_upstream.cpp)
// Compare against bench_mlir_cpp.cpp (direct MLIR 19 C++ API).

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sys/resource.h>
#include <base/arena.h>
#include <base/string.h>
#include <platform/platform.h>
#include "mlir_api.h"

static double elapsed_ms(uint64_t t0) {
    return (double)(platform_clock_ns() - t0) * 1e-6;
}

static long current_rss_kb(void) {
    FILE *f = fopen("/proc/self/statm", "r");
    if (!f) return 0;
    long pages = 0, rss = 0;
    if (fscanf(f, "%ld %ld", &pages, &rss) != 2) rss = 0;
    fclose(f);
    return rss * 4; // 4 KiB pages on Linux x86_64
}

static volatile uintptr_t g_sink = 0;

int main(int argc, char **argv) {
    platform_init(argc, argv, NULL);
    const char *label = (argc > 1) ? argv[1] : "api";

    // 1. Op creation (50,000 ops: 1 const + 49,999 llvm.add) + RSS accounting.
    {
        const int N = 50000;
        long rss0 = current_rss_kb();
        Arena *arena = arena_create(64 * 1024 * 1024);
        MLIR_ContextOptions opts = {0};
        opts.arena = arena;
        opts.expected_ops = (size_t)N + 16;
        MLIR_Context *ctx = MLIR_CreateContext(&opts);
        MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
        MLIR_BlockHandle blk = MLIR_CreateBlock(ctx);
        MLIR_Builder b = MLIR_BuilderAtEnd(ctx, blk);

        uint64_t t0 = platform_clock_ns();
        MLIR_ValueHandle v = MLIR_BuildConstantInt(&b, OP_TYPE_LLVM_MLIR_CONSTANT, i32, 1);
        for (int i = 1; i < N; i++) {
            v = MLIR_BuildBinary(&b, OP_TYPE_LLVM_ADD, i32, v, v);
        }
        double create_ms = elapsed_ms(t0);
        long rss1 = current_rss_kb();
        long delta_kb = (rss1 > rss0) ? (rss1 - rss0) : 0;

        // 2. Walk + GetOpType + GetOpOperand (10 reps over 50,000 ops).
        t0 = platform_clock_ns();
        uintptr_t acc = 0;
        for (int rep = 0; rep < 10; rep++) {
            MLIR_BlockOpIter it = MLIR_BlockOpsBegin(ctx, blk);
            for (MLIR_OpHandle op; (op = MLIR_BlockOpIterNext(ctx, &it)) != MLIR_INVALID_HANDLE; ) {
                acc ^= (uintptr_t)MLIR_GetOpType(ctx, op);
                if (MLIR_GetOpNumOperands(ctx, op) == 2) {
                    acc ^= MLIR_GetOpOperand(ctx, op, 0);
                    acc ^= MLIR_GetOpOperand(ctx, op, 1);
                }
            }
        }
        double walk_ms = elapsed_ms(t0) / 10.0;
        g_sink ^= acc;

        // 3. Attribute lookup (10 reps over 50,000 ops).
        t0 = platform_clock_ns();
        uintptr_t aacc = 0;
        for (int rep = 0; rep < 10; rep++) {
            MLIR_BlockOpIter it = MLIR_BlockOpsBegin(ctx, blk);
            for (MLIR_OpHandle op; (op = MLIR_BlockOpIterNext(ctx, &it)) != MLIR_INVALID_HANDLE; ) {
                aacc ^= MLIR_GetOpAttr(ctx, op, MLIR_KEY_VALUE);
            }
        }
        double attr_ms = elapsed_ms(t0) / 10.0;
        g_sink ^= aacc;

        printf("%-10s | create(50k): %7.3f ms (%5.1f ns/op, ~%ld B/op) | walk(50k): %6.3f ms | attr(50k): %6.3f ms\n",
               label, create_ms, create_ms * 1e6 / N, (delta_kb * 1024L) / N, walk_ms, attr_ms);

        MLIR_DestroyContext(ctx);
        arena_destroy(arena);
    }

    // 4. Use iteration (5,000 uses, 50 reps) + RAUW (5,000 uses).
    {
        const int N = 5000;
        Arena *arena = arena_create(16 * 1024 * 1024);
        MLIR_ContextOptions opts = {0};
        opts.arena = arena;
        MLIR_Context *ctx = MLIR_CreateContext(&opts);
        MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
        MLIR_TypeHandle i64 = MLIR_CreateTypeInteger(ctx, 64, true);
        MLIR_BlockHandle blk = MLIR_CreateBlock(ctx);
        MLIR_Builder b = MLIR_BuilderAtEnd(ctx, blk);
        MLIR_ValueHandle c1 = MLIR_BuildConstantInt(&b, OP_TYPE_LLVM_MLIR_CONSTANT, i32, 1);
        MLIR_ValueHandle c2 = MLIR_BuildConstantInt(&b, OP_TYPE_LLVM_MLIR_CONSTANT, i32, 2);
        for (int i = 0; i < N; i++)
            MLIR_BuildUnary(&b, OP_TYPE_LLVM_SEXT, i64, c1);

        uint64_t t0 = platform_clock_ns();
        uintptr_t uacc = 0;
        for (int rep = 0; rep < 50; rep++) {
            for (MLIR_UseHandle u = MLIR_GetValueFirstUse(ctx, c1); u;
                 u = MLIR_GetUseNext(ctx, u)) {
                uacc ^= MLIR_GetUseOwner(ctx, u) ^ MLIR_GetUseOperandIndex(ctx, u);
            }
        }
        double uses_ms = elapsed_ms(t0) / 50.0;
        g_sink ^= uacc;

        t0 = platform_clock_ns();
        MLIR_ReplaceAllUsesOfValue(ctx, c1, c2);
        double rauw_ms = elapsed_ms(t0);

        // 5. Erase 5,000 ops in reverse order.
        t0 = platform_clock_ns();
        for (MLIR_OpHandle op = MLIR_GetBlockLastOp(ctx, blk); op; ) {
            MLIR_OpHandle prev = MLIR_GetOpPrev(ctx, op);
            MLIR_EraseOp(ctx, op);
            op = prev;
        }
        double erase_ms = elapsed_ms(t0);

        printf("%-10s | uses(5k):    %7.4f ms | rauw(5k): %7.4f ms | erase(5k): %7.4f ms\n",
               label, uses_ms, rauw_ms, erase_ms);

        MLIR_DestroyContext(ctx);
        arena_destroy(arena);
    }

    // 6. CFG + dominators (1,000-block chain + back-edges).
    {
        const int N = 1000;
        Arena *arena = arena_create(16 * 1024 * 1024);
        Arena *scratch = arena_create(4 * 1024 * 1024);
        MLIR_ContextOptions opts = {0};
        opts.arena = arena;
        MLIR_Context *ctx = MLIR_CreateContext(&opts);
        MLIR_TypeHandle i1 = MLIR_CreateTypeInteger(ctx, 1, false);
        MLIR_OpSpec ms = { OP_TYPE_UNREGISTERED };
        ms.type = OP_TYPE_MODULE;
        ms.n_regions = 1;
        MLIR_OpHandle mod = MLIR_CreateOpFromSpec(ctx, &ms);
        MLIR_RegionHandle r = MLIR_GetOpRegion(ctx, mod, 0);
        MLIR_BlockHandle *blks = arena_new_array(arena, MLIR_BlockHandle, N);
        for (int i = 0; i < N; i++) {
            blks[i] = MLIR_CreateBlock(ctx);
            MLIR_AppendRegionBlock(ctx, r, blks[i]);
        }
        MLIR_Builder eb = MLIR_BuilderAtEnd(ctx, blks[0]);
        MLIR_ValueHandle cond = MLIR_BuildConstantInt(&eb, OP_TYPE_LLVM_MLIR_CONSTANT, i1, 1);
        for (int i = 0; i + 1 < N; i++) {
            MLIR_Builder b = MLIR_BuilderAtEnd(ctx, blks[i]);
            if (i > 0 && (i % 8) == 0) {
                MLIR_BuildCondBr(&b, OP_TYPE_LLVM_COND_BR, cond,
                                 blks[i + 1], NULL, 0, blks[i - 4], NULL, 0);
            } else {
                MLIR_BuildBr(&b, OP_TYPE_LLVM_BR, blks[i + 1], NULL, 0);
            }
        }
        {
            MLIR_Builder b = MLIR_BuilderAtEnd(ctx, blks[N - 1]);
            MLIR_OpSpec rs = { OP_TYPE_UNREGISTERED };
            rs.type = OP_TYPE_LLVM_RETURN;
            MLIR_BuildOp(&b, &rs);
        }

        uint64_t t0 = platform_clock_ns();
        MLIR_CFGInfo *cfg = MLIR_ComputeCFGInfo(ctx, r, scratch);
        double cfg_ms = elapsed_ms(t0);
        printf("%-10s | cfg+dom(1k): %7.4f ms (reachable=%zu)\n", label, cfg_ms, cfg->n_blocks);

        MLIR_DestroyContext(ctx);
        arena_destroy(scratch);
        arena_destroy(arena);
    }
    return 0;
}
