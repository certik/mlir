#include "mlir_classic_printer.h"
#include "mlir_generic_printer.h"

// Minimal classic printer wrapper that delegates to the generic printer.
// This keeps the build green while we migrate to the public API only.

string print_operation_classic(Arena *arena, int indent_level, MlirOperation *op) {
    return print_operation_generic(arena, indent_level, (Operation*)op);
}

string print_region_classic(Arena *arena, int indent_level, MlirRegion *region) {
    return print_region_generic(arena, indent_level, (Region*)region);
}

string print_block_classic(Arena *arena, int bb_index, int indent_level, MlirBlock *block) {
    return print_block_generic(arena, bb_index, indent_level, (Block*)block);
}

string print_module_classic(Arena *arena, MlirOperation *module, LocationMap *location_map) {
    (void)location_map; // TODO: add API-based location map printing
    return print_operation_classic(arena, 0, module);
}

