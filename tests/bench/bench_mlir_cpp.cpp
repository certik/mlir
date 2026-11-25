// tests/bench/bench_mlir_cpp.cpp
//
// Direct MLIR 19 C++ API baseline for #358: runs the exact same workloads as
// tests/bench/bench_api.c using mlir::OpBuilder / mlir::Operation / mlir::Value
// directly so we can measure the exact wrapper overhead of mlir_api_impl_upstream.cpp.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

static double ms_since(std::chrono::steady_clock::time_point t0) {
    auto dt = std::chrono::steady_clock::now() - t0;
    return std::chrono::duration<double, std::milli>(dt).count();
}

static long current_rss_kb() {
    FILE *f = std::fopen("/proc/self/statm", "r");
    if (!f) return 0;
    long pages = 0, rss = 0;
    if (std::fscanf(f, "%ld %ld", &pages, &rss) != 2) rss = 0;
    std::fclose(f);
    return rss * 4;
}

static volatile uintptr_t g_sink = 0;

int main() {
    // 1. Op creation (50,000 ops) + walk + attr lookup.
    {
        const int N = 50000;
        long rss0 = current_rss_kb();
        mlir::MLIRContext ctx;
        ctx.loadDialect<mlir::LLVM::LLVMDialect>();
        auto i32 = mlir::IntegerType::get(&ctx, 32);
        auto loc = mlir::UnknownLoc::get(&ctx);
        mlir::Block blk;
        mlir::OpBuilder b(&ctx);
        b.setInsertionPointToEnd(&blk);
        auto valKey = mlir::StringAttr::get(&ctx, "value");

        auto t0 = std::chrono::steady_clock::now();
        mlir::Value v = b.create<mlir::LLVM::ConstantOp>(loc, i32, b.getI32IntegerAttr(1));
        for (int i = 1; i < N; i++) {
            v = b.create<mlir::LLVM::AddOp>(loc, i32, v, v);
        }
        double create_ms = ms_since(t0);
        long rss1 = current_rss_kb();
        long delta_kb = (rss1 > rss0) ? (rss1 - rss0) : 0;

        t0 = std::chrono::steady_clock::now();
        uintptr_t acc = 0;
        for (int rep = 0; rep < 10; rep++) {
            for (mlir::Operation &op : blk) {
                acc ^= reinterpret_cast<uintptr_t>(op.getName().getAsOpaquePointer());
                if (op.getNumOperands() == 2) {
                    acc ^= reinterpret_cast<uintptr_t>(op.getOperand(0).getAsOpaquePointer());
                    acc ^= reinterpret_cast<uintptr_t>(op.getOperand(1).getAsOpaquePointer());
                }
            }
        }
        double walk_ms = ms_since(t0) / 10.0;
        g_sink ^= acc;

        t0 = std::chrono::steady_clock::now();
        uintptr_t aacc = 0;
        for (int rep = 0; rep < 10; rep++) {
            for (mlir::Operation &op : blk) {
                aacc ^= reinterpret_cast<uintptr_t>(op.getAttr(valKey).getAsOpaquePointer());
            }
        }
        double attr_ms = ms_since(t0) / 10.0;
        g_sink ^= aacc;

        std::printf("%-10s | create(50k): %7.3f ms (%5.1f ns/op, ~%ld B/op) | walk(50k): %6.3f ms | attr(50k): %6.3f ms\n",
                    "mlir_cpp", create_ms, create_ms * 1e6 / N, (delta_kb * 1024L) / N, walk_ms, attr_ms);
        while (!blk.empty()) blk.back().erase();
    }

    // 2. Use iteration (5,000 uses, 50 reps) + RAUW + erase.
    {
        const int N = 5000;
        mlir::MLIRContext ctx;
        ctx.loadDialect<mlir::LLVM::LLVMDialect>();
        auto i32 = mlir::IntegerType::get(&ctx, 32);
        auto i64 = mlir::IntegerType::get(&ctx, 64);
        auto loc = mlir::UnknownLoc::get(&ctx);
        mlir::Block blk;
        mlir::OpBuilder b(&ctx);
        b.setInsertionPointToEnd(&blk);
        mlir::Value c1 = b.create<mlir::LLVM::ConstantOp>(loc, i32, b.getI32IntegerAttr(1));
        mlir::Value c2 = b.create<mlir::LLVM::ConstantOp>(loc, i32, b.getI32IntegerAttr(2));
        for (int i = 0; i < N; i++)
            b.create<mlir::LLVM::SExtOp>(loc, i64, c1);

        auto t0 = std::chrono::steady_clock::now();
        uintptr_t uacc = 0;
        for (int rep = 0; rep < 50; rep++) {
            for (mlir::OpOperand &u : c1.getUses()) {
                uacc ^= reinterpret_cast<uintptr_t>(u.getOwner()) ^ u.getOperandNumber();
            }
        }
        double uses_ms = ms_since(t0) / 50.0;
        g_sink ^= uacc;

        t0 = std::chrono::steady_clock::now();
        c1.replaceAllUsesWith(c2);
        double rauw_ms = ms_since(t0);

        t0 = std::chrono::steady_clock::now();
        while (!blk.empty()) blk.back().erase();
        double erase_ms = ms_since(t0);

        std::printf("%-10s | uses(5k):    %7.4f ms | rauw(5k): %7.4f ms | erase(5k): %7.4f ms\n",
                    "mlir_cpp", uses_ms, rauw_ms, erase_ms);
    }

    // 3. CFG + dominators (1,000 blocks).
    {
        const int N = 1000;
        mlir::MLIRContext ctx;
        ctx.loadDialect<mlir::LLVM::LLVMDialect>();
        auto i1 = mlir::IntegerType::get(&ctx, 1);
        auto loc = mlir::UnknownLoc::get(&ctx);
        auto mod = mlir::ModuleOp::create(loc);
        auto &r = mod.getBodyRegion();
        r.front().erase();
        std::vector<mlir::Block *> blks(N);
        for (int i = 0; i < N; i++) {
            blks[i] = new mlir::Block();
            r.push_back(blks[i]);
        }
        mlir::OpBuilder eb(&ctx);
        eb.setInsertionPointToEnd(blks[0]);
        mlir::Value cond = eb.create<mlir::LLVM::ConstantOp>(loc, i1, eb.getBoolAttr(true));
        for (int i = 0; i + 1 < N; i++) {
            mlir::OpBuilder b(&ctx);
            b.setInsertionPointToEnd(blks[i]);
            if (i > 0 && (i % 8) == 0) {
                b.create<mlir::LLVM::CondBrOp>(loc, cond, blks[i + 1], blks[i - 4]);
            } else {
                b.create<mlir::LLVM::BrOp>(loc, mlir::ValueRange{}, blks[i + 1]);
            }
        }
        {
            mlir::OpBuilder b(&ctx);
            b.setInsertionPointToEnd(blks[N - 1]);
            b.create<mlir::LLVM::ReturnOp>(loc, mlir::ValueRange{});
        }

        auto t0 = std::chrono::steady_clock::now();
        mlir::DominanceInfo dom(mod);
        size_t reach = 0;
        for (int i = 0; i < N; i++)
            if (dom.isReachableFromEntry(blks[i])) reach++;
        double cfg_ms = ms_since(t0);
        std::printf("%-10s | cfg+dom(1k): %7.4f ms (reachable=%zu)\n", "mlir_cpp", cfg_ms, reach);
        mod.erase();
    }
    return 0;
}
