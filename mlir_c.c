// clang -g mlir_c.c && ./a.out
//
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

// ============================================================================
// Type System - Using enums for maximum performance
// ============================================================================

// All registered operation types as enum for fast dispatch
typedef enum {
    // Core ops
    OP_TYPE_UNREGISTERED = 0,  // For dynamic/unregistered operations
    OP_TYPE_MODULE,            // Module operation

    // Arithmetic dialect
    OP_TYPE_ARITH_ADDI,
    OP_TYPE_ARITH_SUBI,
    OP_TYPE_ARITH_MULI,
    OP_TYPE_ARITH_DIVI,
    OP_TYPE_ARITH_ADDF,
    OP_TYPE_ARITH_SUBF,
    OP_TYPE_ARITH_MULF,
    OP_TYPE_ARITH_DIVF,
    OP_TYPE_ARITH_CONSTANT,
    OP_TYPE_ARITH_CMPI,
    OP_TYPE_ARITH_CMPF,

    // Memory dialect
    OP_TYPE_MEMREF_LOAD,
    OP_TYPE_MEMREF_STORE,
    OP_TYPE_MEMREF_ALLOC,
    OP_TYPE_MEMREF_DEALLOC,

    // Control flow
    OP_TYPE_CF_BR,
    OP_TYPE_CF_COND_BR,
    OP_TYPE_CF_SWITCH,

    // Function dialect
    OP_TYPE_FUNC_FUNC,
    OP_TYPE_FUNC_RETURN,
    OP_TYPE_FUNC_CALL,

    // SCF dialect
    OP_TYPE_SCF_FOR,
    OP_TYPE_SCF_WHILE,
    OP_TYPE_SCF_IF,

    OP_TYPE_COUNT  // Total number of operation types
} MLIR_OpType;

// Type kinds for MLIR type system
typedef enum {
    TYPE_KIND_INTEGER,
    TYPE_KIND_FLOAT,
    TYPE_KIND_MEMREF,
    TYPE_KIND_TENSOR,
    TYPE_KIND_FUNCTION,
    TYPE_KIND_INDEX
} TypeKind;

// ============================================================================
// Core Data Structures
// ============================================================================

// Forward declarations
typedef struct Value Value;
typedef struct Operation Operation;
typedef struct Block Block;
typedef struct Region Region;

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
        } floating;
        struct {
            struct Type *element_type;
            int64_t *shape;     // NULL-terminated or use rank
            uint32_t rank;
        } shaped;  // For memref and tensor
    } data;
} Type;

// Attribute representation (simplified)
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
        const char *string_value;
        bool bool_value;
        struct {
            struct Attribute **elements;
            size_t count;
        } array;
    } data;
} Attribute;

// Named attribute for dictionaries
typedef struct NamedAttribute {
    const char *name;
    Attribute *value;
} NamedAttribute;

// Location information (simplified)
typedef struct Location {
    const char *filename;
    uint32_t line;
    uint32_t column;
} Location;

// Value representation - SSA values
struct Value {
    Operation *defining_op;  // Operation that produces this value
    uint32_t result_index;   // Which result of the operation
    Type *type;              // Type of this value
    uint32_t ssa_number;     // Unique SSA number for printing

    // Use-def chain (simplified - in practice, would be more sophisticated)
    Operation **users;       // Operations using this value
    size_t num_users;
    size_t users_capacity;
};

// Block - contains a list of operations
struct Block {
    Operation *first_op;
    Operation *last_op;

    // Block arguments
    Value **arguments;
    size_t num_arguments;

    // Parent region
    Region *parent;

    // Linked list of blocks in region
    Block *next_block;
    Block *prev_block;
};

// Region - contains blocks
struct Region {
    Block *first_block;
    Block *last_block;
    size_t num_blocks;

    Operation *parent_op;  // Operation that contains this region
};

// The core Operation structure - designed for cache efficiency
struct Operation {
    // Type identification - FIRST for fastest access
    MLIR_OpType op_type;              // 4 bytes - enum for registered ops

    // For unregistered ops, store the string name
    const char *unregistered_name;  // 8 bytes - NULL for registered ops

    // Location information
    Location location;           // 16 bytes (pointer + 2 ints)

    // Operands and results - using flexible array members for cache locality
    uint16_t num_operands;       // 2 bytes
    uint16_t num_results;        // 2 bytes
    uint16_t num_attributes;     // 2 bytes
    uint16_t num_regions;        // 2 bytes
    uint16_t num_successors;     // 2 bytes
    // 6 bytes padding here for alignment

    // Linked list pointers for block
    Operation *next_op;          // 8 bytes
    Operation *prev_op;          // 8 bytes
    Block *parent_block;         // 8 bytes

    // These are allocated right after the Operation structure
    // for better cache locality (using flexible array member pattern)
    // Order: operands, results, attributes, regions, successors

    // Flexible array member for trailing data
    void *trailing_data[];  // Contains Value*, Value*, NamedAttribute*, Region*, Block**
};

// ============================================================================
// Memory Layout Helpers
// ============================================================================

// Calculate size needed for an operation
static inline size_t operation_size(uint16_t num_operands, uint16_t num_results,
                                   uint16_t num_attributes, uint16_t num_regions,
                                   uint16_t num_successors) {
    return sizeof(Operation) +
           num_operands * sizeof(Value*) +      // Operand values
           num_results * sizeof(Value) +        // Result values (embedded)
           num_attributes * sizeof(NamedAttribute) +
           num_regions * sizeof(Region) +       // Regions (embedded)
           num_successors * sizeof(Block*);     // Successor blocks
}

// Accessors for trailing data
static inline Value** operation_get_operands(Operation *op) {
    return (Value**)op->trailing_data;
}

static inline Value* operation_get_results(Operation *op) {
    return (Value*)((char*)op->trailing_data + op->num_operands * sizeof(Value*));
}

static inline NamedAttribute* operation_get_attributes(Operation *op) {
    return (NamedAttribute*)((char*)op->trailing_data +
                            op->num_operands * sizeof(Value*) +
                            op->num_results * sizeof(Value));
}

static inline Region* operation_get_regions(Operation *op) {
    return (Region*)((char*)op->trailing_data +
                     op->num_operands * sizeof(Value*) +
                     op->num_results * sizeof(Value) +
                     op->num_attributes * sizeof(NamedAttribute));
}

static inline Block** operation_get_successors(Operation *op) {
    return (Block**)((char*)op->trailing_data +
                     op->num_operands * sizeof(Value*) +
                     op->num_results * sizeof(Value) +
                     op->num_attributes * sizeof(NamedAttribute) +
                     op->num_regions * sizeof(Region));
}

// ============================================================================
// Fast Dispatch Implementation
// ============================================================================

// Function pointer type for operation handlers
typedef void (*OpHandler)(Operation *op, void *context);

// Static dispatch table - initialized once at startup
static OpHandler dispatch_table[OP_TYPE_COUNT];

// Register a handler for an operation type
static inline void register_handler(MLIR_OpType type, OpHandler handler) {
    if (type > 0 && type < OP_TYPE_COUNT) {
        dispatch_table[type] = handler;
    }
}

// FASTEST: Direct dispatch using enum
static inline void dispatch_operation_direct(Operation *op, void *context) {
    // Single array lookup - can't get faster than this!
    if (op->op_type < OP_TYPE_COUNT && dispatch_table[op->op_type]) {
        dispatch_table[op->op_type](op, context);
    } else if (op->op_type == OP_TYPE_UNREGISTERED) {
        // Slower path for unregistered ops
        // Would need string-based dispatch here
        printf("Unregistered op: %s\n", op->unregistered_name);
    }
}

// FAST: Switch-based dispatch (compiler can optimize to jump table)
static inline void dispatch_operation_switch(Operation *op) {
    switch (op->op_type) {
        case OP_TYPE_ARITH_ADDI: {
            Value **operands = operation_get_operands(op);
            Value *result = operation_get_results(op);
            printf("  %p = arith.addi %p, %p\n", result, operands[0], operands[1]);
            break;
        }
        case OP_TYPE_ARITH_MULI: {
            Value **operands = operation_get_operands(op);
            Value *result = operation_get_results(op);
            printf("  %p = arith.muli %p, %p\n", result, operands[0], operands[1]);
            break;
        }
        case OP_TYPE_ARITH_CONSTANT: {
            Value *result = operation_get_results(op);
            NamedAttribute *attrs = operation_get_attributes(op);
            // Find "value" attribute
            for (int i = 0; i < op->num_attributes; i++) {
                if (strcmp(attrs[i].name, "value") == 0) {
                    printf("  %p = arith.constant %lld\n", result,
                           (long long)attrs[i].value->data.integer_value);
                    break;
                }
            }
            break;
        }
        case OP_TYPE_FUNC_RETURN: {
            Value **operands = operation_get_operands(op);
            printf("  func.return");
            if (op->num_operands > 0) {
                printf(" %p", operands[0]);
                for (int i = 1; i < op->num_operands; i++) {
                    printf(", %p", operands[i]);
                }
            }
            printf("\n");
            break;
        }
        case OP_TYPE_UNREGISTERED:
            printf("  %s (unregistered)\n", op->unregistered_name);
            break;
        default:
            printf("  Unknown operation type: %d\n", op->op_type);
            break;
    }
}

// Alternative: Batch dispatch for multiple operations (cache-friendly)
static inline void dispatch_operations_batch(Operation **ops, size_t count, void *context) {
    // Prefetch next operations for better cache performance
    for (size_t i = 0; i < count; i++) {
        if (i + 1 < count) {
            __builtin_prefetch(ops[i + 1], 0, 1);  // Prefetch next op for read
        }

        OpHandler handler = dispatch_table[ops[i]->op_type];
        if (handler) {
            handler(ops[i], context);
        }
    }
}

// ============================================================================
// Operation Type to String Conversion
// ============================================================================

static const char* op_type_to_string(MLIR_OpType type) {
    switch (type) {
        case OP_TYPE_UNREGISTERED: return "unregistered";
        case OP_TYPE_MODULE: return "module";
        case OP_TYPE_ARITH_ADDI: return "arith.addi";
        case OP_TYPE_ARITH_SUBI: return "arith.subi";
        case OP_TYPE_ARITH_MULI: return "arith.muli";
        case OP_TYPE_ARITH_DIVI: return "arith.divi";
        case OP_TYPE_ARITH_ADDF: return "arith.addf";
        case OP_TYPE_ARITH_SUBF: return "arith.subf";
        case OP_TYPE_ARITH_MULF: return "arith.mulf";
        case OP_TYPE_ARITH_DIVF: return "arith.divf";
        case OP_TYPE_ARITH_CONSTANT: return "arith.constant";
        case OP_TYPE_ARITH_CMPI: return "arith.cmpi";
        case OP_TYPE_ARITH_CMPF: return "arith.cmpf";
        case OP_TYPE_MEMREF_LOAD: return "memref.load";
        case OP_TYPE_MEMREF_STORE: return "memref.store";
        case OP_TYPE_MEMREF_ALLOC: return "memref.alloc";
        case OP_TYPE_MEMREF_DEALLOC: return "memref.dealloc";
        case OP_TYPE_CF_BR: return "cf.br";
        case OP_TYPE_CF_COND_BR: return "cf.cond_br";
        case OP_TYPE_CF_SWITCH: return "cf.switch";
        case OP_TYPE_FUNC_FUNC: return "func.func";
        case OP_TYPE_FUNC_RETURN: return "func.return";
        case OP_TYPE_FUNC_CALL: return "func.call";
        case OP_TYPE_SCF_FOR: return "scf.for";
        case OP_TYPE_SCF_WHILE: return "scf.while";
        case OP_TYPE_SCF_IF: return "scf.if";
        default: return "unknown";
    }
}

// ============================================================================
// SSA Value Numbering
// ============================================================================

// Global counter for SSA value numbering
static uint32_t next_ssa_number = 0;

// Assign SSA numbers to values
static void assign_ssa_number(Value *val) {
    val->ssa_number = next_ssa_number++;
}

// ============================================================================
// Operation Creation
// ============================================================================

Operation* create_operation(MLIR_OpType type,
                           uint16_t num_operands,
                           uint16_t num_results,
                           uint16_t num_attributes,
                           uint16_t num_regions,
                           uint16_t num_successors) {
    size_t size = operation_size(num_operands, num_results,
                                num_attributes, num_regions, num_successors);

    // Round size up to multiple of 64 for aligned_alloc
    size_t aligned_size = ((size + 63) / 64) * 64;

    // Allocate with proper alignment
    Operation *op = (Operation*)aligned_alloc(64, aligned_size);  // 64-byte aligned for cache
    if (!op) {
        return NULL;  // Handle allocation failure
    }
    memset(op, 0, aligned_size);

    op->op_type = type;
    op->num_operands = num_operands;
    op->num_results = num_results;
    op->num_attributes = num_attributes;
    op->num_regions = num_regions;
    op->num_successors = num_successors;

    // Initialize results as embedded Values
    Value *results = operation_get_results(op);
    for (int i = 0; i < num_results; i++) {
        results[i].defining_op = op;
        results[i].result_index = i;
        assign_ssa_number(&results[i]);
    }

    // Initialize regions
    Region *regions = operation_get_regions(op);
    for (int i = 0; i < num_regions; i++) {
        regions[i].parent_op = op;
    }

    return op;
}

// Create unregistered operation
Operation* create_unregistered_operation(const char *name,
                                        uint16_t num_operands,
                                        uint16_t num_results) {
    Operation *op = create_operation(OP_TYPE_UNREGISTERED,
                                    num_operands, num_results, 0, 0, 0);
    op->unregistered_name = strdup(name);  // Would need to manage this memory
    return op;
}


// ============================================================================
// String Buffer for Printer
// ============================================================================

typedef struct {
    char *buffer;
    size_t size;
    size_t capacity;
} StringBuffer;

static void buffer_init(StringBuffer *buf, size_t initial_capacity) {
    buf->buffer = malloc(initial_capacity);
    buf->buffer[0] = '\0';
    buf->size = 0;
    buf->capacity = initial_capacity;
}

static void buffer_append(StringBuffer *buf, const char *str) {
    size_t len = strlen(str);
    if (buf->size + len + 1 > buf->capacity) {
        buf->capacity = (buf->size + len + 1) * 2;
        buf->buffer = realloc(buf->buffer, buf->capacity);
    }
    strcpy(buf->buffer + buf->size, str);
    buf->size += len;
}

static void buffer_printf(StringBuffer *buf, const char *format, ...) {
    va_list args;
    va_start(args, format);

    // Try with current capacity
    int needed = vsnprintf(buf->buffer + buf->size, buf->capacity - buf->size, format, args);
    va_end(args);

    if (needed >= (int)(buf->capacity - buf->size)) {
        // Need more space
        buf->capacity = buf->size + needed + 1;
        buf->buffer = realloc(buf->buffer, buf->capacity);

        va_start(args, format);
        vsnprintf(buf->buffer + buf->size, buf->capacity - buf->size, format, args);
        va_end(args);
    }

    buf->size += needed;
}

static void buffer_free(StringBuffer *buf) {
    free(buf->buffer);
    buf->buffer = NULL;
    buf->size = 0;
    buf->capacity = 0;
}

// ============================================================================
// Example Usage - Operation Printer
// ============================================================================

// Helper to print value names to stdout
static void print_value(Value *val) {
    if (val->defining_op) {
        printf("%%%d", val->ssa_number);
    } else {
        printf("%%arg%d", val->ssa_number);
    }
}

// Helper to print value names to string buffer
static void buffer_print_value(StringBuffer *buf, Value *val) {
    if (val->defining_op) {
        buffer_printf(buf, "%%%d", val->ssa_number);
    } else {
        buffer_printf(buf, "%%arg%d", val->ssa_number);
    }
}

// Helper to print types
static void print_type(Type *type) {
    if (!type) {
        printf("<null>");
        return;
    }

    switch (type->kind) {
        case TYPE_KIND_INTEGER:
            printf("i%d", type->data.integer.width);
            break;
        case TYPE_KIND_FLOAT:
            printf("f%d", type->data.floating.width);
            break;
        case TYPE_KIND_INDEX:
            printf("index");
            break;
        case TYPE_KIND_MEMREF:
            printf("memref<");
            for (uint32_t i = 0; i < type->data.shaped.rank; i++) {
                if (type->data.shaped.shape[i] < 0) {
                    printf("?");
                } else {
                    printf("%lld", (long long)type->data.shaped.shape[i]);
                }
                if (i < type->data.shaped.rank - 1) printf("x");
            }
            printf("x");
            print_type(type->data.shaped.element_type);
            printf(">");
            break;
        default:
            printf("<unknown>");
    }
}

// Helper to print types to string buffer
static void buffer_print_type(StringBuffer *buf, Type *type) {
    if (!type) {
        buffer_append(buf, "<null>");
        return;
    }

    switch (type->kind) {
        case TYPE_KIND_INTEGER:
            buffer_printf(buf, "i%d", type->data.integer.width);
            break;
        case TYPE_KIND_FLOAT:
            buffer_printf(buf, "f%d", type->data.floating.width);
            break;
        case TYPE_KIND_INDEX:
            buffer_append(buf, "index");
            break;
        case TYPE_KIND_MEMREF:
            buffer_append(buf, "memref<");
            for (uint32_t i = 0; i < type->data.shaped.rank; i++) {
                if (type->data.shaped.shape[i] < 0) {
                    buffer_append(buf, "?");
                } else {
                    buffer_printf(buf, "%lld", (long long)type->data.shaped.shape[i]);
                }
                if (i < type->data.shaped.rank - 1) buffer_append(buf, "x");
            }
            buffer_append(buf, "x");
            buffer_print_type(buf, type->data.shaped.element_type);
            buffer_append(buf, ">");
            break;
        default:
            buffer_append(buf, "<unknown>");
    }
}

// Individual operation printers
static void print_arith_addi(Operation *op, void *ctx) {
    Value **operands = operation_get_operands(op);
    Value *result = operation_get_results(op);

    printf("  ");
    print_value(result);
    printf(" = arith.addi ");
    print_value(operands[0]);
    printf(", ");
    print_value(operands[1]);
    printf(" : ");
    print_type(result->type);
    printf("\n");
}

static void print_arith_muli(Operation *op, void *ctx) {
    Value **operands = operation_get_operands(op);
    Value *result = operation_get_results(op);

    printf("  ");
    print_value(result);
    printf(" = arith.muli ");
    print_value(operands[0]);
    printf(", ");
    print_value(operands[1]);
    printf(" : ");
    print_type(result->type);
    printf("\n");
}

static void print_arith_constant(Operation *op, void *ctx) {
    Value *result = operation_get_results(op);
    NamedAttribute *attrs = operation_get_attributes(op);

    printf("  ");
    print_value(result);
    printf(" = arith.constant ");

    // Find and print the value attribute
    for (int i = 0; i < op->num_attributes; i++) {
        if (strcmp(attrs[i].name, "value") == 0) {
            Attribute *attr = attrs[i].value;
            switch (attr->kind) {
                case ATTR_KIND_INTEGER:
                    printf("%lld", (long long)attr->data.integer_value);
                    break;
                case ATTR_KIND_FLOAT:
                    printf("%f", attr->data.float_value);
                    break;
                default:
                    printf("<unknown>");
            }
            break;
        }
    }

    printf(" : ");
    print_type(result->type);
    printf("\n");
}

static void print_memref_load(Operation *op, void *ctx) {
    Value **operands = operation_get_operands(op);
    Value *result = operation_get_results(op);

    printf("  ");
    print_value(result);
    printf(" = memref.load ");
    print_value(operands[0]);  // memref
    printf("[");
    for (int i = 1; i < op->num_operands; i++) {
        if (i > 1) printf(", ");
        print_value(operands[i]);  // indices
    }
    printf("] : ");
    print_type(operands[0]->type);
    printf("\n");
}

static void print_memref_store(Operation *op, void *ctx) {
    Value **operands = operation_get_operands(op);

    printf("  memref.store ");
    print_value(operands[0]);  // value to store
    printf(", ");
    print_value(operands[1]);  // memref
    printf("[");
    for (int i = 2; i < op->num_operands; i++) {
        if (i > 2) printf(", ");
        print_value(operands[i]);  // indices
    }
    printf("] : ");
    print_type(operands[1]->type);
    printf("\n");
}

static void print_func_return(Operation *op, void *ctx) {
    Value **operands = operation_get_operands(op);

    printf("  func.return");
    if (op->num_operands > 0) {
        printf(" ");
        for (int i = 0; i < op->num_operands; i++) {
            if (i > 0) printf(", ");
            print_value(operands[i]);
        }
        printf(" : ");
        for (int i = 0; i < op->num_operands; i++) {
            if (i > 0) printf(", ");
            print_type(operands[i]->type);
        }
    }
    printf("\n");
}

static void print_unregistered(Operation *op, void *ctx) {
    printf("  \"%s\"", op->unregistered_name);

    // Print operands
    if (op->num_operands > 0) {
        printf("(");
        Value **operands = operation_get_operands(op);
        for (int i = 0; i < op->num_operands; i++) {
            if (i > 0) printf(", ");
            print_value(operands[i]);
        }
        printf(")");
    }

    // Print results
    if (op->num_results > 0) {
        printf(" -> (");
        Value *results = operation_get_results(op);
        for (int i = 0; i < op->num_results; i++) {
            if (i > 0) printf(", ");
            print_value(&results[i]);
        }
        printf(")");
    }

    // Print attributes
    if (op->num_attributes > 0) {
        printf(" {");
        NamedAttribute *attrs = operation_get_attributes(op);
        for (int i = 0; i < op->num_attributes; i++) {
            if (i > 0) printf(", ");
            printf("%s = ", attrs[i].name);
            // Simplified attribute printing
            Attribute *attr = attrs[i].value;
            switch (attr->kind) {
                case ATTR_KIND_INTEGER:
                    printf("%lld", (long long)attr->data.integer_value);
                    break;
                case ATTR_KIND_STRING:
                    printf("\"%s\"", attr->data.string_value);
                    break;
                default:
                    printf("...");
            }
        }
        printf("}");
    }

    printf("\n");
}

// ============================================================================
// Walking Operations in a Block
// ============================================================================

static inline void walk_block_operations(Block *block, OpHandler handler, void *context) {
    Operation *op = block->first_op;
    while (op) {
        Operation *next = op->next_op;  // Save next before handler (might delete op)

        // Prefetch next operation for cache performance
        if (next) {
            __builtin_prefetch(next, 0, 1);
        }

        if (handler) {
            handler(op, context);
        } else {
            dispatch_operation_direct(op, context);
        }

        op = next;
    }
}

// Register all printers
static void register_all_printers(void) {
    register_handler(OP_TYPE_ARITH_ADDI, print_arith_addi);
    register_handler(OP_TYPE_ARITH_MULI, print_arith_muli);
    register_handler(OP_TYPE_ARITH_CONSTANT, print_arith_constant);
    register_handler(OP_TYPE_MEMREF_LOAD, print_memref_load);
    register_handler(OP_TYPE_MEMREF_STORE, print_memref_store);
    register_handler(OP_TYPE_FUNC_RETURN, print_func_return);
    register_handler(OP_TYPE_UNREGISTERED, print_unregistered);

    // Register more as needed...
}

// Forward declaration
static void buffer_print_operation_generic_internal(StringBuffer *buf, Operation *op, int indent_level);

// Generic operation printer - returns string for any operation
static char* print_operation_generic(Operation *op) {
    StringBuffer buf;
    buffer_init(&buf, 1024);
    buffer_print_operation_generic_internal(&buf, op, 0);
    return buf.buffer;  // Caller must free
}

// Internal generic operation printer - same format for all operations
static void buffer_print_operation_generic_internal(StringBuffer *buf, Operation *op, int indent_level) {
    // Add indentation
    for (int i = 0; i < indent_level; i++) {
        buffer_append(buf, "  ");
    }
    
    // Print results if any
    Value *results = operation_get_results(op);
    if (op->num_results > 0) {
        for (int i = 0; i < op->num_results; i++) {
            if (i > 0) buffer_append(buf, ", ");
            buffer_print_value(buf, &results[i]);
        }
        buffer_append(buf, " = ");
    }
    
    // Print operation name
    if (op->op_type == OP_TYPE_UNREGISTERED) {
        buffer_printf(buf, "\"%s\"", op->unregistered_name);
    } else {
        buffer_append(buf, op_type_to_string(op->op_type));
    }
    
    // Print operands with types (always include parentheses)
    Value **operands = operation_get_operands(op);
    buffer_append(buf, "(");
    for (int i = 0; i < op->num_operands; i++) {
        if (i > 0) buffer_append(buf, ", ");
        buffer_print_value(buf, operands[i]);
        buffer_append(buf, ": ");
        buffer_print_type(buf, operands[i]->type);
    }
    buffer_append(buf, ")");
    
    // Print attributes if any
    NamedAttribute *attrs = operation_get_attributes(op);
    if (op->num_attributes > 0) {
        buffer_append(buf, " {");
        for (int i = 0; i < op->num_attributes; i++) {
            if (i > 0) buffer_append(buf, ", ");
            buffer_printf(buf, "%s = ", attrs[i].name);
            Attribute *attr = attrs[i].value;
            switch (attr->kind) {
                case ATTR_KIND_INTEGER:
                    buffer_printf(buf, "%lld", (long long)attr->data.integer_value);
                    break;
                case ATTR_KIND_STRING:
                    buffer_printf(buf, "\"%s\"", attr->data.string_value);
                    break;
                default:
                    buffer_append(buf, "...");
            }
        }
        buffer_append(buf, "}");
    }
    
    // Print result types if any
    if (op->num_results > 0) {
        buffer_append(buf, " -> ");
        for (int i = 0; i < op->num_results; i++) {
            if (i > 0) buffer_append(buf, ", ");
            buffer_print_type(buf, results[i].type);
        }
    }
    
    // Print regions if any
    Region *regions = operation_get_regions(op);
    if (op->num_regions > 0) {
        buffer_append(buf, " {\n");
        for (int i = 0; i < op->num_regions; i++) {
            Block *block = regions[i].first_block;
            while (block) {
                // Always print block header
                for (int j = 0; j < indent_level + 1; j++) {
                    buffer_append(buf, "  ");
                }
                buffer_append(buf, "^bb0(");
                for (size_t j = 0; j < block->num_arguments; j++) {
                    if (j > 0) buffer_append(buf, ", ");
                    buffer_print_value(buf, block->arguments[j]);
                    buffer_append(buf, ": ");
                    buffer_print_type(buf, block->arguments[j]->type);
                }
                buffer_append(buf, "):\n");
                
                // Print operations in block
                Operation *block_op = block->first_op;
                while (block_op) {
                    buffer_print_operation_generic_internal(buf, block_op, indent_level + 1);
                    block_op = block_op->next_op;
                }
                
                block = block->next_block;
            }
        }
        for (int i = 0; i < indent_level; i++) {
            buffer_append(buf, "  ");
        }
        buffer_append(buf, "}");
    }
    
    buffer_append(buf, "\n");
}

// String-based operation printer - specific formatting
static void buffer_print_operation(StringBuffer *buf, Operation *op) {
    switch (op->op_type) {
        case OP_TYPE_ARITH_ADDI: {
            Value **operands = operation_get_operands(op);
            Value *result = operation_get_results(op);
            buffer_append(buf, "    ");
            buffer_print_value(buf, result);
            buffer_append(buf, " = arith.addi ");
            buffer_print_value(buf, operands[0]);
            buffer_append(buf, ", ");
            buffer_print_value(buf, operands[1]);
            buffer_append(buf, " : ");
            buffer_print_type(buf, result->type);
            buffer_append(buf, "\n");
            break;
        }
        case OP_TYPE_ARITH_MULI: {
            Value **operands = operation_get_operands(op);
            Value *result = operation_get_results(op);
            buffer_append(buf, "    ");
            buffer_print_value(buf, result);
            buffer_append(buf, " = arith.muli ");
            buffer_print_value(buf, operands[0]);
            buffer_append(buf, ", ");
            buffer_print_value(buf, operands[1]);
            buffer_append(buf, " : ");
            buffer_print_type(buf, result->type);
            buffer_append(buf, "\n");
            break;
        }
        case OP_TYPE_ARITH_CONSTANT: {
            Value *result = operation_get_results(op);
            NamedAttribute *attrs = operation_get_attributes(op);
            buffer_append(buf, "    ");
            buffer_print_value(buf, result);
            buffer_append(buf, " = arith.constant ");

            // Find and print the value attribute
            for (int i = 0; i < op->num_attributes; i++) {
                if (strcmp(attrs[i].name, "value") == 0) {
                    Attribute *attr = attrs[i].value;
                    switch (attr->kind) {
                        case ATTR_KIND_INTEGER:
                            buffer_printf(buf, "%lld", (long long)attr->data.integer_value);
                            break;
                        case ATTR_KIND_FLOAT:
                            buffer_printf(buf, "%f", attr->data.float_value);
                            break;
                        default:
                            buffer_append(buf, "<unknown>");
                    }
                    break;
                }
            }

            buffer_append(buf, " : ");
            buffer_print_type(buf, result->type);
            buffer_append(buf, "\n");
            break;
        }
        case OP_TYPE_FUNC_RETURN: {
            Value **operands = operation_get_operands(op);
            buffer_append(buf, "    func.return");
            if (op->num_operands > 0) {
                buffer_append(buf, " ");
                for (int i = 0; i < op->num_operands; i++) {
                    if (i > 0) buffer_append(buf, ", ");
                    buffer_print_value(buf, operands[i]);
                }
                buffer_append(buf, " : ");
                for (int i = 0; i < op->num_operands; i++) {
                    if (i > 0) buffer_append(buf, ", ");
                    buffer_print_type(buf, operands[i]->type);
                }
            }
            buffer_append(buf, "\n");
            break;
        }
        case OP_TYPE_FUNC_FUNC: {
            NamedAttribute *attrs = operation_get_attributes(op);
            Region *regions = operation_get_regions(op);

            // Find function name
            const char *func_name = "unknown";
            for (int i = 0; i < op->num_attributes; i++) {
                if (strcmp(attrs[i].name, "sym_name") == 0) {
                    func_name = attrs[i].value->data.string_value;
                    break;
                }
            }

            Block *body = regions[0].first_block;
            buffer_printf(buf, "  func.func @%s(", func_name);

            // Print block arguments
            for (size_t i = 0; i < body->num_arguments; i++) {
                if (i > 0) buffer_append(buf, ", ");
                buffer_print_value(buf, body->arguments[i]);
                buffer_append(buf, ": ");
                buffer_print_type(buf, body->arguments[i]->type);
            }

            buffer_append(buf, ") {\n");

            // Print function body
            Operation *body_op = body->first_op;
            while (body_op) {
                buffer_print_operation(buf, body_op);
                body_op = body_op->next_op;
            }

            buffer_append(buf, "  }\n");
            break;
        }
        case OP_TYPE_MODULE: {
            buffer_append(buf, "module {\n");

            // Print module body
            Region *body_region = operation_get_regions(op);
            Block *body_block = body_region->first_block;
            Operation *body_op = body_block->first_op;
            while (body_op) {
                buffer_print_operation(buf, body_op);
                body_op = body_op->next_op;
            }

            buffer_append(buf, "}\n");
            break;
        }
        case OP_TYPE_UNREGISTERED:
            buffer_print_operation_generic_internal(buf, op, 2);
            break;
        default:
            buffer_printf(buf, "    Unknown operation type: %d\n", op->op_type);
            break;
    }
}

// Print a complete function to string buffer
static char* print_function_to_string(Block *entry_block, const char *name) {
    StringBuffer buf;
    buffer_init(&buf, 1024);

    buffer_printf(&buf, "func.func @%s(", name);

    // Print block arguments
    for (size_t i = 0; i < entry_block->num_arguments; i++) {
        if (i > 0) buffer_append(&buf, ", ");
        buffer_print_value(&buf, entry_block->arguments[i]);
        buffer_append(&buf, ": ");
        buffer_print_type(&buf, entry_block->arguments[i]->type);
    }

    buffer_append(&buf, ") {\n");

    // Walk and print all operations in the block
    Operation *op = entry_block->first_op;
    while (op) {
        buffer_print_operation(&buf, op);
        op = op->next_op;
    }

    buffer_append(&buf, "}\n");

    return buf.buffer;  // Caller must free
}

// ============================================================================
// Module Creation and Management
// ============================================================================

static Operation* create_module(void) {
    Operation *module = create_operation(OP_TYPE_MODULE, 0, 0, 0, 1, 0);

    // Initialize the module's body region
    Region *body_region = operation_get_regions(module);
    body_region->first_block = calloc(1, sizeof(Block));
    body_region->last_block = body_region->first_block;
    body_region->num_blocks = 1;
    body_region->first_block->parent = body_region;

    return module;
}

static void module_add_operation(Operation *module, Operation *op) {
    Region *body_region = operation_get_regions(module);
    Block *body_block = body_region->first_block;

    if (!body_block->first_op) {
        body_block->first_op = op;
        body_block->last_op = op;
    } else {
        body_block->last_op->next_op = op;
        op->prev_op = body_block->last_op;
        body_block->last_op = op;
    }
    op->parent_block = body_block;
}

static Operation* create_func_operation(const char *name, Block *body) {
    Operation *func_op = create_operation(OP_TYPE_FUNC_FUNC, 0, 0, 1, 1, 0);

    // Set function name as attribute
    NamedAttribute *attrs = operation_get_attributes(func_op);
    attrs[0].name = "sym_name";
    attrs[0].value = malloc(sizeof(Attribute));
    attrs[0].value->kind = ATTR_KIND_STRING;
    attrs[0].value->data.string_value = strdup(name);

    // Set body region
    Region *regions = operation_get_regions(func_op);
    regions[0].first_block = body;
    regions[0].last_block = body;
    regions[0].num_blocks = 1;
    body->parent = &regions[0];

    return func_op;
}

// Print a complete module to string buffer
static char* print_module_to_string(Operation *module) {
    StringBuffer buf;
    buffer_init(&buf, 2048);

    buffer_append(&buf, "module {\n");

    // Walk through all operations in module's body
    Region *body_region = operation_get_regions(module);
    Block *body_block = body_region->first_block;
    Operation *op = body_block->first_op;
    while (op) {
        buffer_print_operation(&buf, op);
        op = op->next_op;
    }

    buffer_append(&buf, "}\n");

    return buf.buffer;  // Caller must free
}

// Print a complete function to stdout
static void print_function(Block *entry_block, const char *name) {
    printf("func.func @%s(", name);

    // Print block arguments
    for (size_t i = 0; i < entry_block->num_arguments; i++) {
        if (i > 0) printf(", ");
        print_value(entry_block->arguments[i]);
        printf(": ");
        print_type(entry_block->arguments[i]->type);
    }

    printf(") {\n");

    // Walk and print all operations in the block
    walk_block_operations(entry_block, NULL, NULL);  // Uses dispatch_operation_direct

    printf("}\n");
}

// Example: Create and print a simple function
int main() {
    printf("=== C MLIR Printer Example ===\n\n");

    // Register all printers
    register_all_printers();

    // Create types
    Type *i32_type = malloc(sizeof(Type));
    i32_type->kind = TYPE_KIND_INTEGER;
    i32_type->data.integer.width = 32;
    i32_type->data.integer.is_signed = true;

    Type *i64_type = malloc(sizeof(Type));
    i64_type->kind = TYPE_KIND_INTEGER;
    i64_type->data.integer.width = 64;
    i64_type->data.integer.is_signed = true;

    // Create a block
    Block *block = calloc(1, sizeof(Block));

    // Create block arguments
    block->num_arguments = 2;
    block->arguments = malloc(2 * sizeof(Value*));
    block->arguments[0] = calloc(1, sizeof(Value));
    block->arguments[0]->type = i32_type;
    block->arguments[0]->ssa_number = 0;  // %arg0
    block->arguments[1] = calloc(1, sizeof(Value));
    block->arguments[1]->type = i32_type;
    block->arguments[1]->ssa_number = 1;  // %arg1

    // Start SSA numbering after block arguments
    next_ssa_number = 0;

    // Create operations

    // %0 = arith.constant 5 : i32
    Operation *const_op = create_operation(OP_TYPE_ARITH_CONSTANT, 0, 1, 1, 0, 0);
    Value *const_result = operation_get_results(const_op);
    const_result->type = i32_type;
    NamedAttribute *const_attrs = operation_get_attributes(const_op);
    const_attrs[0].name = "value";
    const_attrs[0].value = malloc(sizeof(Attribute));
    const_attrs[0].value->kind = ATTR_KIND_INTEGER;
    const_attrs[0].value->data.integer_value = 5;

    // %1 = arith.addi %arg0, %arg1 : i32
    Operation *add_op = create_operation(OP_TYPE_ARITH_ADDI, 2, 1, 0, 0, 0);
    Value **add_operands = operation_get_operands(add_op);
    add_operands[0] = block->arguments[0];
    add_operands[1] = block->arguments[1];
    Value *add_result = operation_get_results(add_op);
    add_result->type = i32_type;

    // %2 = arith.muli %1, %0 : i32
    Operation *mul_op = create_operation(OP_TYPE_ARITH_MULI, 2, 1, 0, 0, 0);
    Value **mul_operands = operation_get_operands(mul_op);
    mul_operands[0] = add_result;
    mul_operands[1] = const_result;
    Value *mul_result = operation_get_results(mul_op);
    mul_result->type = i32_type;

    // func.return %2 : i32
    Operation *ret_op = create_operation(OP_TYPE_FUNC_RETURN, 1, 0, 0, 0, 0);
    Value **ret_operands = operation_get_operands(ret_op);
    ret_operands[0] = mul_result;

    // Create an unregistered operation for demonstration
    Operation *custom_op = create_unregistered_operation("custom.my_op", 1, 1);
    Value **custom_operands = operation_get_operands(custom_op);
    custom_operands[0] = mul_result;
    Value *custom_result = operation_get_results(custom_op);
    custom_result->type = i64_type;

    // Link operations in block
    block->first_op = const_op;
    const_op->next_op = add_op;
    add_op->prev_op = const_op;
    add_op->next_op = mul_op;
    mul_op->prev_op = add_op;
    mul_op->next_op = custom_op;
    custom_op->prev_op = mul_op;
    custom_op->next_op = ret_op;
    ret_op->prev_op = custom_op;
    block->last_op = ret_op;

    // Set parent block for all operations
    const_op->parent_block = block;
    add_op->parent_block = block;
    mul_op->parent_block = block;
    custom_op->parent_block = block;
    ret_op->parent_block = block;

    // Create module and add function
    Operation *module = create_module();
    Operation *func_op = create_func_operation("example_func", block);
    module_add_operation(module, func_op);

    // Test 1: Specific printer mode
    printf("=== Test 1: Specific Printer Mode ===\n");
    char *result1 = print_module_to_string(module);
    printf("%s", result1);

    // Reference expected output for specific mode
    const char *expected1 =
        "module {\n"
        "  func.func @example_func(%arg0: i32, %arg1: i32) {\n"
        "    %0 = arith.constant 5 : i32\n"
        "    %1 = arith.addi %arg0, %arg1 : i32\n"
        "    %2 = arith.muli %1, %0 : i32\n"
        "    %3 = \"custom.my_op\"(%2: i32) -> i64\n"
        "    func.return %2 : i32\n"
        "  }\n"
        "}\n";

    // Test comparison for specific mode
    int exit_code = 0;
    if (strcmp(result1, expected1) == 0) {
        printf("✅ Test 1 PASSED: Specific mode output matches expected result\n\n");
    } else {
        printf("❌ Test 1 FAILED: Specific mode output does not match expected result\n");
        printf("\nExpected:\n%s\n", expected1);
        printf("Actual:\n%s\n", result1);
        exit_code = 1;
    }

    // Test 2: Generic printer mode
    printf("=== Test 2: Generic Printer Mode ===\n");
    char *result2 = print_operation_generic(module);
    printf("%s", result2);

    // Reference expected output for generic mode
    const char *expected2 =
        "module() {\n"
        "  ^bb0():\n"
        "  func.func() {sym_name = \"example_func\"} {\n"
        "    ^bb0(%arg0: i32, %arg1: i32):\n"
        "    %0 = arith.constant() {value = 5} -> i32\n"
        "    %1 = arith.addi(%arg0: i32, %arg1: i32) -> i32\n"
        "    %2 = arith.muli(%1: i32, %0: i32) -> i32\n"
        "    %3 = \"custom.my_op\"(%2: i32) -> i64\n"
        "    func.return(%2: i32)\n"
        "  }\n"
        "}\n";

    // Test comparison for generic mode
    if (strcmp(result2, expected2) == 0) {
        printf("✅ Test 2 PASSED: Generic mode output matches expected result\n");
    } else {
        printf("❌ Test 2 FAILED: Generic mode output does not match expected result\n");
        printf("\nExpected:\n%s\n", expected2);
        printf("Actual:\n%s\n", result2);
        exit_code = 1;
    }

    free(result1);
    free(result2);
    return exit_code;
}
