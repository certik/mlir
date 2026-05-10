// Op-type → string mapping. Split out of mlir_parser.c so consumers like
// mlir_api_impl.c that need only the table can link against it without
// pulling in the parser/tokenizer/op_parsers tree.

#include <base/string.h>
#include "mlir_api.h"
#include "mlir_op_names.h"

string op_type_to_string(MLIR_OpType type) {
    switch (type) {
        case OP_TYPE_UNREGISTERED: return str_lit("unregistered");
        case OP_TYPE_MODULE: return str_lit("module");
        case OP_TYPE_ARITH_ADDI: return str_lit("arith.addi");
        case OP_TYPE_ARITH_SUBI: return str_lit("arith.subi");
        case OP_TYPE_ARITH_MULI: return str_lit("arith.muli");
        case OP_TYPE_ARITH_DIVI: return str_lit("arith.divi");
        case OP_TYPE_ARITH_ADDF: return str_lit("arith.addf");
        case OP_TYPE_ARITH_SUBF: return str_lit("arith.subf");
        case OP_TYPE_ARITH_MULF: return str_lit("arith.mulf");
        case OP_TYPE_ARITH_DIVF: return str_lit("arith.divf");
        case OP_TYPE_ARITH_CONSTANT: return str_lit("arith.constant");
        case OP_TYPE_ARITH_CMPI: return str_lit("arith.cmpi");
        case OP_TYPE_ARITH_CMPF: return str_lit("arith.cmpf");
        case OP_TYPE_MEMREF_LOAD: return str_lit("memref.load");
        case OP_TYPE_MEMREF_STORE: return str_lit("memref.store");
        case OP_TYPE_MEMREF_ALLOC: return str_lit("memref.alloc");
        case OP_TYPE_MEMREF_DEALLOC: return str_lit("memref.dealloc");
        case OP_TYPE_CF_BR: return str_lit("cf.br");
        case OP_TYPE_CF_COND_BR: return str_lit("cf.cond_br");
        case OP_TYPE_CF_SWITCH: return str_lit("cf.switch");
        case OP_TYPE_FUNC_FUNC: return str_lit("func.func");
        case OP_TYPE_FUNC_RETURN: return str_lit("func.return");
        case OP_TYPE_FUNC_CALL: return str_lit("func.call");
        case OP_TYPE_FUNC_CALL_INDIRECT: return str_lit("func.call_indirect");
        case OP_TYPE_FUNC_CONSTANT: return str_lit("func.constant");
        case OP_TYPE_UNREALIZED_CONVERSION_CAST: return str_lit("builtin.unrealized_conversion_cast");
        case OP_TYPE_SCF_FOR: return str_lit("scf.for");
        case OP_TYPE_SCF_WHILE: return str_lit("scf.while");
        case OP_TYPE_SCF_IF: return str_lit("scf.if");
        case OP_TYPE_SCF_YIELD: return str_lit("scf.yield");
        case OP_TYPE_TT_GET_PROGRAM_ID: return str_lit("tt.get_program_id");
        case OP_TYPE_TT_LOAD: return str_lit("tt.load");
        case OP_TYPE_TT_STORE: return str_lit("tt.store");
        case OP_TYPE_TT_MAKE_RANGE: return str_lit("tt.make_range");
        case OP_TYPE_TT_SPLAT: return str_lit("tt.splat");
        case OP_TYPE_TT_ADDPTR: return str_lit("tt.addptr");
        case OP_TYPE_TT_RETURN: return str_lit("tt.return");
        case OP_TYPE_ARITH_SELECT: return str_lit("arith.select");
        case OP_TYPE_ARITH_BITCAST: return str_lit("arith.bitcast");
        case OP_TYPE_ARITH_SITOFP: return str_lit("arith.sitofp");
        case OP_TYPE_ARITH_FPTOSI: return str_lit("arith.fptosi");
        case OP_TYPE_ARITH_INDEX_CAST: return str_lit("arith.index_cast");
        case OP_TYPE_ARITH_EXTSI: return str_lit("arith.extsi");
        case OP_TYPE_ARITH_TRUNCI: return str_lit("arith.trunci");
        case OP_TYPE_ARITH_EXTF: return str_lit("arith.extf");
        case OP_TYPE_ARITH_TRUNCF: return str_lit("arith.truncf");
        case OP_TYPE_ARITH_EXTUI: return str_lit("arith.extui");
        case OP_TYPE_ARITH_MAXF: return str_lit("arith.maxf");
        case OP_TYPE_ARITH_DIVSI: return str_lit("arith.divsi");
        case OP_TYPE_ARITH_REMSI: return str_lit("arith.remsi");
        case OP_TYPE_ARITH_DIVUI: return str_lit("arith.divui");
        case OP_TYPE_ARITH_REMUI: return str_lit("arith.remui");
        case OP_TYPE_ARITH_SHRUI: return str_lit("arith.shrui");
        case OP_TYPE_ARITH_ORI: return str_lit("arith.ori");
        case OP_TYPE_ARITH_MINSI: return str_lit("arith.minsi");
        case OP_TYPE_ARITH_ANDI: return str_lit("arith.andi");
        case OP_TYPE_MATH_EXP: return str_lit("math.exp");
        case OP_TYPE_MATH_LOG: return str_lit("math.log");
        case OP_TYPE_TT_FUNC: return str_lit("tt.func");
        case OP_TYPE_TT_CALL: return str_lit("tt.call");
        case OP_TYPE_TT_REDUCE: return str_lit("tt.reduce");
        case OP_TYPE_TT_BROADCAST: return str_lit("tt.broadcast");
        case OP_TYPE_TT_EXPAND_DIMS: return str_lit("tt.expand_dims");
        case OP_TYPE_TT_DOT: return str_lit("tt.dot");
        case OP_TYPE_TT_PURE_EXTERN_ELEMENTWISE: return str_lit("tt.pure_extern_elementwise");
        case OP_TYPE_GPU_LAUNCH: return str_lit("gpu.launch");
        case OP_TYPE_AFFINE_FOR: return str_lit("affine.for");
        case OP_TYPE_AFFINE_LOAD: return str_lit("affine.load");
        case OP_TYPE_VECTOR_PRINT: return str_lit("vector.print");
        case OP_TYPE_STD_CONSTANT: return str_lit("std.constant");
        case OP_TYPE_STD_RETURN: return str_lit("std.return");
        case OP_TYPE_TENSOR_EXTRACT: return str_lit("tensor.extract");
        case OP_TYPE_TENSOR_SPLAT: return str_lit("tensor.splat");
        case OP_TYPE_TENSOR_COLLAPSE_SHAPE: return str_lit("tensor.collapse_shape");
        case OP_TYPE_LINALG_FILL: return str_lit("linalg.fill");
        case OP_TYPE_LINALG_COPY: return str_lit("linalg.copy");
        case OP_TYPE_INDEX_CONSTANT: return str_lit("index.constant");
        case OP_TYPE_LLVM_MLIR_UNDEF: return str_lit("llvm.mlir.undef");
        case OP_TYPE_LLVM_ALLOCA: return str_lit("llvm.alloca");
        case OP_TYPE_LLVM_LOAD: return str_lit("llvm.load");
        case OP_TYPE_LLVM_STORE: return str_lit("llvm.store");
        case OP_TYPE_LLVM_GEP: return str_lit("llvm.getelementptr");
        case OP_TYPE_LLVM_MLIR_ZERO: return str_lit("llvm.mlir.zero");
        case OP_TYPE_LLVM_MLIR_CONSTANT: return str_lit("llvm.mlir.constant");
        case OP_TYPE_LLVM_ICMP: return str_lit("llvm.icmp");
        case OP_TYPE_LLVM_MLIR_ADDRESSOF: return str_lit("llvm.mlir.addressof");
        case OP_TYPE_LLVM_MLIR_GLOBAL: return str_lit("llvm.mlir.global");
        case OP_TYPE_LLVM_RETURN: return str_lit("llvm.return");
        case OP_TYPE_LLVM_PTRTOINT: return str_lit("llvm.ptrtoint");
        case OP_TYPE_ARITH_XORI: return str_lit("arith.xori");
        case OP_TYPE_ARITH_SHLI: return str_lit("arith.shli");
        case OP_TYPE_ARITH_SHRSI: return str_lit("arith.shrsi");
        case OP_TYPE_RETURN: return str_lit("return");
        case OP_TYPE_TT_REDUCE_RETURN: return str_lit("tt.reduce.return");
        default: return str_lit("unknown");
    }
}
