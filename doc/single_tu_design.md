# C Subset Compiler Design: Single-TU with Clang Compatibility

## Goals

- Compile an entire project as **one translation unit** (like Zig) for better optimization and simpler semantics.
- Keep **full source compatibility** with Clang and GCC.
- Support both:
  - Single-TU builds (your compiler)
  - Traditional multi-TU builds (Clang/GCC)
- Provide Rust/Zig-style **privacy and isolation** using macros and compiler enforcement.
- Allow clean public APIs with documentation (using `.h` files where needed).
- Enforce API consistency (declarations vs implementations) in your compiler.

## Core Design

### 1. Build Modes

Controlled by the `SINGLE_TU_BUILD` preprocessor define:

| Mode                    | How it is built                          | Visibility behavior          | Best for                  |
|-------------------------|------------------------------------------|------------------------------|---------------------------|
| **Single-TU**           | One root file includes all `.c` files   | `pub` items visible across modules | Your compiler            |
| **Traditional (Multi-TU)** | Each `.c` compiled separately          | Standard C rules             | Clang / GCC              |

### 2. Visibility Macros (`config.h`)

```c
// config.h
#if defined(SINGLE_TU_BUILD)
    #define pub
    #define internal   static
#else
    #define pub        extern
    #define internal   static
#endif
```

- `pub` → Makes a symbol visible to other modules.
- `internal` → Makes a symbol private to the current `.c` file (`static`).

### 3. Role of Header Files

- `.h` files are used primarily for **shared types** (structs, enums) and public API declarations.
- They are **optional** for purely internal modules.
- All headers use `#pragma once` (or traditional include guards) for Clang compatibility.
- Your compiler can perform extra checks (e.g., "every declaration in `foo.h` must have a matching `pub` definition in `foo.c`").

### 4. Single-TU Build Process

Create one root file (e.g. `build_single.c`):

```c
#define SINGLE_TU_BUILD
#include "config.h"

#include "vector.c"
#include "physics.c"
// ... other modules in dependency order

int main(void) {
    // ...
}
```

Your compiler treats this as **one translation unit**, enabling cross-module optimizations while still respecting `pub` / `internal` visibility.

## Example Project

### Project Structure

```
src/
├── config.h
├── vector.h
├── vector.c
├── physics.h
├── physics.c
└── build_single.c     # Used only for single-TU builds
```

### `config.h`

```c
#pragma once

#if defined(SINGLE_TU_BUILD)
    #define pub
    #define internal   static
#else
    #define pub        extern
    #define internal   static
#endif
```

### `vector.h` (Public API + Types)

```c
#pragma once

typedef struct {
    float x, y;
} Vec2;

Vec2 vec2_add(Vec2 a, Vec2 b);
Vec2 vec2_scale(Vec2 v, float s);
```

### `vector.c`

```c
#include "config.h"
#include "vector.h"

pub Vec2 vec2_add(Vec2 a, Vec2 b) {
    return (Vec2){a.x + b.x, a.y + b.y};
}

pub Vec2 vec2_scale(Vec2 v, float s) {
    return (Vec2){v.x * s, v.y * s};
}

internal float dot(Vec2 a, Vec2 b) {   // private to this file
    return a.x * b.x + a.y * b.y;
}
```

### `physics.h`

```c
#pragma once

#include "vector.h"

typedef struct {
    Vec2 position;
    Vec2 velocity;
} Entity;

void entity_update(Entity* e);
```

### `physics.c`

```c
#include "config.h"
#include "physics.h"
#include "vector.h"

pub void entity_update(Entity* e) {
    e->position = vec2_add(e->position, e->velocity);
}
```

### Single-TU Root File (`build_single.c`)

```c
#define SINGLE_TU_BUILD
#include "config.h"

#include "vector.c"
#include "physics.c"

int main(void) {
    Entity e = {
        .position = {0, 0},
        .velocity = {1, 2}
    };

    entity_update(&e);
    return 0;
}
```

## How Clang/GCC Builds Work

### Traditional Multi-TU Build (Normal Clang/GCC)

```bash
clang -c vector.c -o vector.o
clang -c physics.c -o physics.o
clang vector.o physics.o -o program
```

- `pub` expands to `extern`
- `internal` expands to `static`
- Standard C rules apply

### Single-TU Build with Clang (Unity Build)

You can also do a unity build with Clang:

```bash
clang -DSINGLE_TU_BUILD -o program build_single.c
```

This works because the code is valid C in both modes.

## What Your Custom Compiler Can Do Extra

Your compiler can add value that plain Clang cannot:

- **API Enforcement**: Verify that every function/type declared in `foo.h` has a matching `pub` definition in `foo.c`.
- **Privacy Analysis**: Warn or error if code accesses a symbol that is only defined inside a `.c` file (should have been marked `internal`).
- **Single-TU Optimizations**: Perform whole-program analysis and inlining across modules.
- **Optional stricter mode**: Require that all `pub` symbols are declared in the corresponding header.

## Benefits

| Benefit                        | How it is achieved                              |
|--------------------------------|-------------------------------------------------|
| Better optimization            | Single-TU mode enables cross-module inlining    |
| Privacy / Isolation            | `internal` + compiler enforcement               |
| Documentation-friendly         | Clean `.h` files for public APIs                |
| Clang/GCC compatibility        | Same source files work in both modes            |
| Incremental adoption           | Easy to start with traditional builds           |
| Simple mental model            | Clear separation between public and private     |

## Usage Recommendations

1. Start with traditional multi-TU builds during development (faster feedback with Clang).
2. Use single-TU builds for release / performance-critical builds.
3. Keep shared types in `.h` files.
4. Use `internal` aggressively for implementation details.
5. Let your compiler enforce API consistency as a build step or during single-TU compilation.

---

This design gives you most of the benefits of Zig’s single compilation unit model while remaining fully compatible with standard C toolchains.
