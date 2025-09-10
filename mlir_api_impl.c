#include <base/arena.h>
#include <base/string.h>
#include <base/hashtable.h>
#include <base/vector.h>
#include "tokenizer.h"
#include "mlir_api.h"
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// Hash function for strings
size_t string_hash(string str) {
    size_t hash = 5381;
    for (size_t i = 0; i < str.size; i++) {
        hash = ((hash << 5) + hash) + str.str[i];
    }
    return hash;
}

// Equality function for strings
bool string_equal(string a, string b) {
    return str_eq(a, b);
}

// Forward declare ValueRef for hashtable
typedef struct ValueRef ValueRef;

// Define hashtable for string -> ValueRef* mapping
#define SymbolTable_HASH string_hash
#define SymbolTable_EQUAL string_equal
DEFINE_HASHTABLE_FOR_TYPES(string, ValueRef*, SymbolTable)

// Scoped symbol table for SSA values
typedef struct ScopedSymbolTable {
    SymbolTable *scopes;
    size_t num_scopes;
    size_t scope_capacity;
} ScopedSymbolTable;

// Forward declare Location for hashtable
typedef struct Location Location;

// Location map for named location references
#define LocationMap_HASH string_hash
#define LocationMap_EQUAL string_equal
DEFINE_HASHTABLE_FOR_TYPES(string, Location*, LocationMap)

typedef struct {
    Arena *arena;
    unsigned char *input;
    TokenType sym;
    uint64_t cur;
    uint64_t first, last;
    ScopedSymbolTable symbol_table;
    LocationMap location_map;  // For #locN -> Location mapping
    int next_loc_id;          // Counter for generating #locN IDs
    Location *unnumbered_loc_def; // Optional: definition of unnumbered '#loc' at file start
    // Parsing mode flag to enable robust trailing comment capture in special contexts
    bool capture_trailing_comments;
} Parser;

typedef struct Region Region;

// Type kinds for MLIR type system
typedef enum {
    TYPE_KIND_UNKNOWN,
    TYPE_KIND_OPAQUE,
    TYPE_KIND_INTEGER,
    TYPE_KIND_FLOAT,
    TYPE_KIND_MEMREF,
    TYPE_KIND_TENSOR,
    TYPE_KIND_FUNCTION,
    TYPE_KIND_INDEX,
    TYPE_KIND_POINTER
} TypeKind;

// MLIR Type representation
typedef struct Type {
    TypeKind kind;
    union {
        struct {
            uint32_t width;     // Bit width for integers
            bool is_signed;
        } integer;
        struct {
            uint32_t width;     // 16, 32, 64, etc.
            bool is_bfloat;     // true for bf16
        } floating;
        struct {
            struct Type *element_type;
            int64_t *shape;     // NULL-terminated or use rank
            uint32_t rank;
        } shaped;  // For memref and tensor
        struct {
            struct Type *element_type;
            uint32_t address_space;  // For !tt.ptr<type, address_space>
            bool has_address_space;
        } pointer;
    } data;
} Type;

// Attribute representation
typedef struct Attribute {
    enum {
        ATTR_KIND_INTEGER,
        ATTR_KIND_FLOAT,
        ATTR_KIND_STRING,
        ATTR_KIND_BOOL,
        ATTR_KIND_ARRAY,
        ATTR_KIND_DICT
    } kind;
    union {
        int64_t integer_value;
        double float_value;
        string string_value;
        bool bool_value;
        struct {
            struct Attribute **elements;
            size_t count;
        } array;
    } data;
    string name;
} Attribute;

// Named attribute for dictionaries
typedef struct NamedAttribute {
    const string *name;
    Attribute *value;
} NamedAttribute;

// Location information for MLIR constructs
typedef enum {
    LOC_KIND_UNKNOWN,
    LOC_KIND_FILE,      // loc("file.py":line:col)
    LOC_KIND_NAME,      // loc("name")
    LOC_KIND_CALLSITE,  // loc(callsite(...))
    LOC_KIND_FUSED,     // loc(fused[...])
    LOC_KIND_REF        // loc(#locN) - reference to named location
} LocationKind;

typedef struct Location {
    LocationKind kind;
    union {
        struct {
            string filename;
            int line;
            int column;
        } file;
        struct {
            string name;
        } name;
        struct {
            int ref_id;  // For #locN references
        } ref;
    } data;
    
    // For storing the original location string for printing
    string original_text;
} Location;

typedef enum ValueKind {
    BLOCK_ARG,
    OP_RESULT
} ValueKind;

struct ValueRef {
    ValueKind kind;
    // TODO: Use an index
    void* def; // Block* or Operation* that produced it
    uint32_t result_index;   // Which result of the operation
    // TODO: use Type by value
    Type *type;              // Type of this value

    // For parsed register names like %0, %c16_i32. These names are not unique
    // in an MLIR module. Two different Values in different regions can have the
    // same name. If this is used for printing, then extra care must be taken
    // that the printed Value name is unique.
    string register_name;

    // Optional per-argument metadata for classic printing
    Location *location;           // e.g., arg loc("file":line:col)
    bool has_divisibility;        // tt.divisibility attribute present
    int64_t divisibility_value;   // value for tt.divisibility
    Type *divisibility_type;      // type for tt.divisibility value (e.g., i32)
    
    bool has_max_divisibility;    // tt.max_divisibility attribute present
    int64_t max_divisibility_value; // value for tt.max_divisibility
    Type *max_divisibility_type;  // type for tt.max_divisibility value (e.g., i32)

    // Maybe later:
    //Operation **users;
    //uint64_t n_users;
};

// Note: we use ** instead of *, because Value has a pointer to Operation or
// Block, so we can't easily move them later. When parsing we do not know how
// many items we will need, so we use pointers, which we can grow by copying as
// needed.

typedef struct Operation {
    OpType op_type; // Enum for registered ops
    // Use indices here
    ValueRef **operands;
    uint64_t n_operands;
    // TODO: use Type by value
    Type **result_types;
    uint64_t n_result_types;
    Attribute **attributes;
    uint64_t n_attributes;
    Region **regions;
    uint64_t n_regions;
    string opname; // Only used for unregistered ops

    // Result values produced by this operation
    ValueRef **results;
    uint64_t n_results;
    
    // Location information
    Location *location;
    // Optional: definition for unnumbered '#loc' header captured pre-module
    Location *unnumbered_loc_def;

    // Optional trailing comment captured from source line (e.g., " // note")
    string trailing_comment;

    // Source line tracking (for accurate trailing comment capture)
    // Byte offset in the original buffer of the first character of the line
    // on which this operation starts.
    int64_t source_line_start;
} Operation;
DEFINE_VECTOR_FOR_TYPE(Operation*, VecOperation)
DEFINE_VECTOR_FOR_TYPE(ValueRef*, VecValueRef)

typedef struct Block {
    Operation **operations;
    uint64_t n_operations;
    ValueRef **arguments;
    uint64_t n_arguments;
} Block;
DEFINE_VECTOR_FOR_TYPE(Block*, VecBlock)

struct Region {
    Block **blocks;
    uint64_t n_blocks;
};


// API wrapper functions that cast between API types and concrete types
void mlir_api_init(MlirOperation *root) {
    // Cast API type to concrete type
    Operation *concrete_root = (Operation*)root;
    (void)concrete_root;
    // No initialization required for the native C implementation.
}

MlirOperation *mlir_op_create(struct Arena *arena, OpType type) {
    // Cast Arena* to Arena*
    Arena *concrete_arena = (Arena*)arena;
    Operation *op = arena_alloc(concrete_arena, Operation);
    *op = (Operation){0};
    op->op_type = type;
    // Cast concrete type to API type
    return (MlirOperation*)op;
}

void mlir_op_add_region(struct Arena *arena, MlirOperation *op, MlirRegion *region) {
    Arena *concrete_arena = (Arena*)arena;
    Operation *concrete_op = (Operation*)op;
    Region *concrete_region = (Region*)region;
    Region **new_regions = arena_alloc_array(concrete_arena, Region*, concrete_op->n_regions + 1);
    if (concrete_op->regions) memcpy(new_regions, concrete_op->regions, concrete_op->n_regions * sizeof(Region*));
    new_regions[concrete_op->n_regions] = concrete_region;
    concrete_op->regions = new_regions;
    concrete_op->n_regions++;
}

void mlir_op_add_operand(struct Arena *arena, MlirOperation *op, MlirValue *operand) {
    Arena *concrete_arena = (Arena*)arena;
    Operation *concrete_op = (Operation*)op;
    ValueRef *concrete_operand = (ValueRef*)operand;
    ValueRef **new_operands = arena_alloc_array(concrete_arena, ValueRef*, concrete_op->n_operands + 1);
    if (concrete_op->operands) memcpy(new_operands, concrete_op->operands, concrete_op->n_operands * sizeof(ValueRef*));
    new_operands[concrete_op->n_operands] = concrete_operand;
    concrete_op->operands = new_operands;
    concrete_op->n_operands++;
}

void mlir_op_add_result(struct Arena *arena, MlirOperation *op, MlirValue *result) {
    Arena *concrete_arena = (Arena*)arena;
    Operation *concrete_op = (Operation*)op;
    ValueRef *concrete_result = (ValueRef*)result;
    ValueRef **new_results = arena_alloc_array(concrete_arena, ValueRef*, concrete_op->n_results + 1);
    if (concrete_op->results) memcpy(new_results, concrete_op->results, concrete_op->n_results * sizeof(ValueRef*));
    new_results[concrete_op->n_results] = concrete_result;
    concrete_op->results = new_results;
    concrete_op->n_results++;
}

void mlir_block_add_operation(struct Arena *arena, MlirBlock *block, MlirOperation *op) {
    Arena *concrete_arena = (Arena*)arena;
    Block *concrete_block = (Block*)block;
    Operation *concrete_op = (Operation*)op;
    Operation **new_ops = arena_alloc_array(concrete_arena, Operation*, concrete_block->n_operations + 1);
    if (concrete_block->operations) memcpy(new_ops, concrete_block->operations, concrete_block->n_operations * sizeof(Operation*));
    new_ops[concrete_block->n_operations] = concrete_op;
    concrete_block->operations = new_ops;
    concrete_block->n_operations++;
}

void mlir_block_add_argument(struct Arena *arena, MlirBlock *block, MlirValue *arg) {
    Arena *concrete_arena = (Arena*)arena;
    Block *concrete_block = (Block*)block;
    ValueRef *concrete_arg = (ValueRef*)arg;
    ValueRef **new_args = arena_alloc_array(concrete_arena, ValueRef*, concrete_block->n_arguments + 1);
    if (concrete_block->arguments) memcpy(new_args, concrete_block->arguments, concrete_block->n_arguments * sizeof(ValueRef*));
    new_args[concrete_block->n_arguments] = concrete_arg;
    concrete_block->arguments = new_args;
    concrete_block->n_arguments++;
}

void mlir_region_add_block(struct Arena *arena, MlirRegion *region, MlirBlock *block) {
    Arena *concrete_arena = (Arena*)arena;
    Region *concrete_region = (Region*)region;
    Block *concrete_block = (Block*)block;
    Block **new_blocks = arena_alloc_array(concrete_arena, Block*, concrete_region->n_blocks + 1);
    if (concrete_region->blocks) memcpy(new_blocks, concrete_region->blocks, concrete_region->n_blocks * sizeof(Block*));
    new_blocks[concrete_region->n_blocks] = concrete_block;
    concrete_region->blocks = new_blocks;
    concrete_region->n_blocks++;
}

size_t mlir_region_num_blocks(const MlirRegion *region) {
    const Region *concrete_region = (const Region*)region;
    return concrete_region->n_blocks;
}

MlirBlock *mlir_region_get_block(const MlirRegion *region, size_t idx) {
    const Region *concrete_region = (const Region*)region;
    return (MlirBlock*)concrete_region->blocks[idx];
}

size_t mlir_block_num_operations(const MlirBlock *block) {
    const Block *concrete_block = (const Block*)block;
    return concrete_block->n_operations;
}

MlirOperation *mlir_block_get_operation(const MlirBlock *block, size_t idx) {
    const Block *concrete_block = (const Block*)block;
    return (MlirOperation*)concrete_block->operations[idx];
}

OpType mlir_operation_get_type(const MlirOperation *op) {
    const Operation *concrete_op = (const Operation*)op;
    return concrete_op->op_type;
}

size_t mlir_operation_num_regions(const MlirOperation *op) {
    const Operation *concrete_op = (const Operation*)op;
    return concrete_op->n_regions;
}

MlirRegion *mlir_operation_get_region(const MlirOperation *op, size_t idx) {
    const Operation *concrete_op = (const Operation*)op;
    return (MlirRegion*)concrete_op->regions[idx];
}

// Additional accessors for printer support
size_t mlir_operation_num_attributes(const MlirOperation *op) {
    const Operation *concrete_op = (const Operation*)op;
    return concrete_op->n_attributes;
}

MlirAttribute *mlir_operation_get_attribute(const MlirOperation *op, size_t idx) {
    const Operation *concrete_op = (const Operation*)op;
    return (MlirAttribute*)concrete_op->attributes[idx];
}

size_t mlir_operation_num_operands(const MlirOperation *op) {
    const Operation *concrete_op = (const Operation*)op;
    return concrete_op->n_operands;
}

MlirValue *mlir_operation_get_operand(const MlirOperation *op, size_t idx) {
    const Operation *concrete_op = (const Operation*)op;
    return (MlirValue*)concrete_op->operands[idx];
}

size_t mlir_operation_num_results(const MlirOperation *op) {
    const Operation *concrete_op = (const Operation*)op;
    return concrete_op->n_results;
}

MlirValue *mlir_operation_get_result(const MlirOperation *op, size_t idx) {
    const Operation *concrete_op = (const Operation*)op;
    return (MlirValue*)concrete_op->results[idx];
}

MlirLocation *mlir_operation_get_location(const MlirOperation *op) {
    const Operation *concrete_op = (const Operation*)op;
    return (MlirLocation*)concrete_op->location;
}

const char *mlir_operation_get_name(const MlirOperation *op) {
    const Operation *concrete_op = (const Operation*)op;
    return concrete_op->opname.str;
}

size_t mlir_block_num_arguments(const MlirBlock *block) {
    const Block *concrete_block = (const Block*)block;
    return concrete_block->n_arguments;
}

MlirValue *mlir_block_get_argument(const MlirBlock *block, size_t idx) {
    const Block *concrete_block = (const Block*)block;
    return (MlirValue*)concrete_block->arguments[idx];
}

#ifdef __cplusplus
}
#endif