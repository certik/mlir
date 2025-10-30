# Plan to Replace Pointers with Handles in MLIR API

## Overview
The current API uses direct pointer access to `MLIR_Op`, `MLIR_Region`, `MLIR_Block`, `MLIR_Value`, `MLIR_Type`, `MLIR_Attribute`, and `MLIR_Location` structs. This plan replaces these pointers with opaque handles (similar to Win32 HANDLE or OpenGL object IDs) while maintaining the same functionality.

## 1. Core Concept: Handle-Based Architecture

### Handle Types
Replace the current pointer-based opaque types with integer handles:

```c
// Replace these pointer types:
typedef struct MLIR_Op MLIR_Op;
typedef struct MLIR_Region MLIR_Region;
// ... etc

// With these handle types:
typedef uint32_t MLIR_OpHandle;
typedef uint32_t MLIR_RegionHandle;
typedef uint32_t MLIR_BlockHandle;
typedef uint32_t MLIR_ValueHandle;
typedef uint32_t MLIR_TypeHandle;
typedef uint32_t MLIR_AttributeHandle;
typedef uint32_t MLIR_LocationHandle;

#define MLIR_INVALID_HANDLE 0
```

### Handle Manager
Create a central handle management system using existing Vector infrastructure:

```c
// Define vectors for each object type using existing DEFINE_VECTOR_FOR_TYPE
DEFINE_VECTOR_FOR_TYPE(MLIR_Op, OpPool)
DEFINE_VECTOR_FOR_TYPE(MLIR_Region, RegionPool)
DEFINE_VECTOR_FOR_TYPE(MLIR_Block, BlockPool)
DEFINE_VECTOR_FOR_TYPE(MLIR_Value, ValuePool)
DEFINE_VECTOR_FOR_TYPE(MLIR_Type, TypePool)
DEFINE_VECTOR_FOR_TYPE(MLIR_Attribute, AttributePool)
DEFINE_VECTOR_FOR_TYPE(MLIR_Location, LocationPool)

// Define vectors for free handle lists (optional optimization)
DEFINE_VECTOR_FOR_TYPE(uint32_t, HandleFreeList)

typedef struct MLIR_HandleManager {
    Arena *arena;
    
    // Object pools using existing Vector system
    OpPool ops;
    RegionPool regions;
    BlockPool blocks;
    ValuePool values;
    TypePool types;
    AttributePool attributes;
    LocationPool locations;
    
    // Free lists for handle reuse (optional optimization)
    HandleFreeList free_op_handles;
    HandleFreeList free_region_handles;
    HandleFreeList free_block_handles;
    HandleFreeList free_value_handles;
    HandleFreeList free_type_handles;
    HandleFreeList free_attribute_handles;
    HandleFreeList free_location_handles;
} MLIR_HandleManager;
```

## 2. Structural Changes Required

### A. Data Structure Modifications

#### Current Problem: Circular References
The current structures contain pointer arrays that would become handle arrays:

```c
// Current (in mlir_api_impl.c):
struct MLIR_Op {
    MLIR_Value **operands;           // Array of pointers
    MLIR_Type **result_types;        // Array of pointers  
    MLIR_Attribute **attributes;     // Array of pointers
    MLIR_Region **regions;           // Array of pointers
    MLIR_Value **results;            // Array of pointers
    MLIR_Location *location;         // Single pointer
    MLIR_Location *unnumbered_loc_def; // Single pointer
    // ...
};
```

#### New Handle-Based Structure:
```c
// New handle-based version:
struct MLIR_Op {
    MLIR_OpType op_type;
    
    // Replace pointer arrays with handle arrays
    MLIR_ValueHandle *operands;      // Array of handles
    uint64_t n_operands;
    
    MLIR_TypeHandle *result_types;   // Array of handles
    uint64_t n_result_types;
    
    MLIR_AttributeHandle *attributes; // Array of handles
    uint64_t n_attributes;
    
    MLIR_RegionHandle *regions;      // Array of handles
    uint64_t n_regions;
    
    MLIR_ValueHandle *results;       // Array of handles
    uint64_t n_results;
    
    // Single handle references
    MLIR_LocationHandle location;
    MLIR_LocationHandle unnumbered_loc_def;
    
    string opname;
    string trailing_comment;
    int64_t source_line_start;
};
```

### B. Similar Changes for Other Structures

```c
struct MLIR_Block {
    MLIR_OpHandle *operations;       // Array of handles
    uint64_t n_operations;
    MLIR_ValueHandle *arguments;     // Array of handles
    uint64_t n_arguments;
};

struct MLIR_Region {
    MLIR_BlockHandle *blocks;        // Array of handles
    uint64_t n_blocks;
};

struct MLIR_Value {
    MLIR_ValueKind kind;
    uint32_t def_handle;             // Handle to defining op/block
    uint32_t result_index;
    MLIR_TypeHandle type;            // Handle instead of pointer
    string register_name;
    MLIR_LocationHandle location;    // Handle instead of pointer
};
```

## 3. API Function Signature Changes

### Creation Functions
```c
// Current:
MLIR_Op *MLIR_CreateOp(MLIR_OpType type, string opname, 
                       MLIR_Attribute **attributes, size_t n_attributes,
                       MLIR_Type **result_types, size_t n_result_types,
                       // ... more pointer arrays);

// New:
MLIR_OpHandle MLIR_CreateOp(MLIR_OpType type, string opname,
                            MLIR_AttributeHandle *attributes, size_t n_attributes,
                            MLIR_TypeHandle *result_types, size_t n_result_types,
                            // ... more handle arrays);
```

### Accessor Functions
```c
// Current:
MLIR_Value *MLIR_GetOpOperand(const MLIR_Op *op, size_t idx);

// New:
MLIR_ValueHandle MLIR_GetOpOperand(MLIR_OpHandle op, size_t idx);
```

### Mutation Functions
```c
// Current:
void MLIR_AppendBlockOp(MLIR_Block *block, MLIR_Op *op);

// New:
void MLIR_AppendBlockOp(MLIR_BlockHandle block, MLIR_OpHandle op);
```

## 4. Implementation Strategy

### Phase 1: Handle Manager Implementation
1. **Create `mlir_handle_manager.h/c`** with:
   - Handle manager structure using existing Vector types
   - Vector pool definitions (`DEFINE_VECTOR_FOR_TYPE` for each object type)
   - Handle-to-object resolution functions (simple array indexing)
   - Object creation using `Vector_push_back()` functions
   - Optional handle recycling using free lists

### Phase 2: Internal Structure Updates
1. **Update `mlir_api_impl.c`**:
   - Replace all pointer fields with handle fields in internal structs
   - Update allocation patterns to use handle manager
   - Convert between handles and objects in accessor functions

### Phase 3: API Surface Changes
1. **Update `mlir_api.h`**:
   - Replace all pointer parameters with handle parameters
   - Update return types from pointers to handles
   - Keep function names the same for easier migration

### Phase 4: Vector System Updates
1. **Update existing vector usage**:
   - Change `VecOp`, `VecValue`, etc. to store handles instead of pointers:
     ```c
     // Current:
     DEFINE_VECTOR_FOR_TYPE(MLIR_Op*, VecOp)
     DEFINE_VECTOR_FOR_TYPE(MLIR_Value*, VecValue)
     
     // New:
     DEFINE_VECTOR_FOR_TYPE(MLIR_OpHandle, VecOpHandle)
     DEFINE_VECTOR_FOR_TYPE(MLIR_ValueHandle, VecValueHandle)
     ```
   - Leverage existing Vector API (`reserve`, `push_back`) for both handle manager and user code

## 5. Key Implementation Details

### Handle Resolution Functions
```c
// Internal functions for converting handles to objects using Vector access
static inline MLIR_Op* resolve_op_handle(MLIR_OpHandle handle) {
    if (handle == MLIR_INVALID_HANDLE || handle > global_handle_manager->ops.size) {
        return NULL;
    }
    return &global_handle_manager->ops.data[handle - 1]; // handles are 1-based
}

static inline MLIR_Region* resolve_region_handle(MLIR_RegionHandle handle) {
    if (handle == MLIR_INVALID_HANDLE || handle > global_handle_manager->regions.size) {
        return NULL;
    }
    return &global_handle_manager->regions.data[handle - 1];
}

// Handle creation functions using Vector push_back
static inline MLIR_OpHandle create_op_handle(void) {
    // Check free list first (optional optimization)
    if (global_handle_manager->free_op_handles.size > 0) {
        uint32_t handle = global_handle_manager->free_op_handles.data[
            --global_handle_manager->free_op_handles.size];
        return handle;
    }
    
    // Create new handle - allocate new slot in vector
    MLIR_Op new_op = {0}; // Zero-initialize
    OpPool_push_back(global_handle_manager->arena, &global_handle_manager->ops, new_op);
    return (MLIR_OpHandle)global_handle_manager->ops.size; // 1-based handles
}

static inline MLIR_RegionHandle create_region_handle(void) {
    if (global_handle_manager->free_region_handles.size > 0) {
        uint32_t handle = global_handle_manager->free_region_handles.data[
            --global_handle_manager->free_region_handles.size];
        return handle;
    }
    
    MLIR_Region new_region = {0};
    RegionPool_push_back(global_handle_manager->arena, &global_handle_manager->regions, new_region);
    return (MLIR_RegionHandle)global_handle_manager->regions.size;
}
// ... similar for other types
```

### Global State Management
```c
// Global handle manager (or context-based)
static MLIR_HandleManager *global_handle_manager = NULL;

static void init_handle_manager(Arena *arena, MLIR_HandleManager *manager) {
    manager->arena = arena;
    
    // Initialize all vector pools using existing Vector API
    OpPool_reserve(arena, &manager->ops, 64);           // Initial capacity
    RegionPool_reserve(arena, &manager->regions, 32);
    BlockPool_reserve(arena, &manager->blocks, 64);
    ValuePool_reserve(arena, &manager->values, 256);
    TypePool_reserve(arena, &manager->types, 64);
    AttributePool_reserve(arena, &manager->attributes, 128);
    LocationPool_reserve(arena, &manager->locations, 64);
    
    // Initialize free lists
    HandleFreeList_reserve(arena, &manager->free_op_handles, 16);
    HandleFreeList_reserve(arena, &manager->free_region_handles, 16);
    HandleFreeList_reserve(arena, &manager->free_block_handles, 16);
    HandleFreeList_reserve(arena, &manager->free_value_handles, 16);
    HandleFreeList_reserve(arena, &manager->free_type_handles, 16);
    HandleFreeList_reserve(arena, &manager->free_attribute_handles, 16);
    HandleFreeList_reserve(arena, &manager->free_location_handles, 16);
}

void MLIR_SetArena(Arena *arena) {
    if (!global_handle_manager) {
        global_handle_manager = arena_alloc(arena, MLIR_HandleManager);
        init_handle_manager(arena, global_handle_manager);
    }
}
```

## 6. Migration Considerations

### Benefits of Handle-Based Approach:
1. **Memory Safety**: Invalid handles can be detected (vs. dangling pointers)
2. **Serialization**: Handles are naturally serializable as integers
3. **Debugging**: Handle values are more readable in debuggers
4. **Validation**: Can validate handle ranges and detect corruption
5. **Pool Management**: Easier to implement object pools and reuse
6. **ABI Stability**: Handle types don't expose internal structure layouts

### Challenges to Address:
1. **Performance**: Handle resolution adds indirection (mitigated by inlining)
2. **Vector Reallocation**: Handle manager pools need to support growth
3. **Memory Usage**: Additional handle manager overhead
4. **Debugging**: Need tools to inspect handle->object mappings

## 7. Files to Modify

### Core Files:
1. **`mlir_api.h`** - Update all API signatures
2. **`mlir_api_impl.c`** - Reimplement with handle manager
3. **New: `mlir_handle_manager.h/c`** - Handle management system

### Supporting Files:
4. **`mlir_parser.h`** - Update parser to use handles
5. **`mlir_parser.c`** - Update parsing logic
6. **`parser.c`** - Update test construction code  
7. **`mlir_classic_printer.c`** - Update printer to resolve handles
8. **`mlir_generic_printer.c`** - Update printer to resolve handles

### Vector Usage:
9. Update all files using `VecOp`, `VecValue`, etc. to use handle-based vectors

## 8. Testing Strategy

1. **Maintain API Compatibility**: All existing tests should work with new API
2. **Handle Validation Tests**: Add tests for invalid handle detection
3. **Performance Benchmarks**: Ensure handle resolution doesn't significantly impact performance
4. **Memory Usage Tests**: Verify handle manager doesn't cause excessive memory overhead

## 9. Optional Enhancements

### Handle Validation:
```c
typedef struct {
    uint32_t index : 24;    // Index into pool
    uint32_t version : 8;   // Version for ABA problem protection
} MLIR_Handle;
```

### Context-Based Handles:
```c
typedef struct MLIR_Context MLIR_Context;
MLIR_Context* MLIR_CreateContext(Arena *arena);
MLIR_OpHandle MLIR_CreateOp(MLIR_Context *ctx, /* ... */);
```

## 10. Implementation Notes

### Handle Index Mapping
- Handle value 0 = MLIR_INVALID_HANDLE (reserved)
- Handle values 1-N map to array indices 0-(N-1)
- This allows simple validation: `if (handle == 0 || handle > pool_size) return error;`

### Memory Layout Benefits
- All objects of same type stored contiguously via Vector (better cache locality)
- Existing Vector system automatically handles growth and reallocation
- Handles remain stable even when Vector data arrays are reallocated
- Reuses battle-tested Vector implementation (no new memory management bugs)
- Vector's `push_back` and `reserve` provide optimal growth strategies

### API Migration Path
1. Keep both pointer and handle APIs temporarily during transition
2. Add `_Handle` suffix to new functions initially
3. Gradually migrate callers from pointer to handle versions
4. Remove pointer APIs in final step

## 11. Vector System Integration Advantages

### Reusing Existing Infrastructure
The plan leverages the existing `base/vector.h` system which provides:

1. **Proven Implementation**: The Vector system is already tested and used throughout the codebase
2. **Automatic Growth**: `push_back()` handles reallocation automatically with 2x growth strategy
3. **Memory Efficiency**: Contiguous storage with minimal overhead
4. **Debug Support**: Built-in assertions when `WITH_BASE_ASSERT` is enabled
5. **Arena Integration**: Vectors already work seamlessly with the Arena allocator

### Simplified Implementation
Using Vectors eliminates the need to:
- Implement custom dynamic array logic
- Handle memory reallocation edge cases  
- Debug memory management issues
- Optimize growth strategies

### Code Example: Vector-Based Handle Manager
```c
// mlir_handle_manager.h
#include <base/vector.h>

// Define pools using existing Vector macros
DEFINE_VECTOR_FOR_TYPE(MLIR_Op, OpPool)
DEFINE_VECTOR_FOR_TYPE(MLIR_Value, ValuePool)
// ... etc

// Simple handle resolution using Vector direct access
#define RESOLVE_OP_HANDLE(handle) \
    ((handle) == 0 || (handle) > global_handle_manager->ops.size ? NULL : \
     &global_handle_manager->ops.data[(handle) - 1])

// Object creation using Vector push_back
MLIR_OpHandle create_op_handle(MLIR_Op op) {
    OpPool_push_back(global_handle_manager->arena, &global_handle_manager->ops, op);
    return global_handle_manager->ops.size; // 1-based handles
}
```

This approach maximizes code reuse while providing all the benefits of handle-based architecture.

---

This plan provides a complete pathway to replace the pointer-based API with a robust handle-based system while maintaining functionality and improving safety and debuggability.