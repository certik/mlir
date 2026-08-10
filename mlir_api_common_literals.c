// Shared integer/float/string literal implementation used by both
// mlir_api_impl.c (native backend) and mlir_api_impl_upstream.cpp
// (upstream backend). Payload structs are private to this file; all
// other translation units interact through the public get/set API in
// mlir_api.h.

#include <stdint.h>
#include <string.h>

#include <base/arena.h>

#include "mlir_api.h"

typedef struct {
    uint32_t width;
    int64_t value;
    bool is_larger_bits;
    uint32_t larger_word_count;
    uint64_t value_larger_bits[MLIR_LITERAL_LARGER_BITS_WORDS];
} IntegerLiteralStorage;

typedef struct {
    uint32_t width;
    MLIR_FloatEncoding encoding;
    double value;
    bool is_larger_bits;
    uint32_t larger_word_count;
    uint64_t value_larger_bits[MLIR_LITERAL_LARGER_BITS_WORDS];
} FloatLiteralStorage;

typedef struct {
    const uint8_t *bytes;
    size_t byte_count;
} StringLiteralStorage;

static IntegerLiteralStorage *resolve_integer_literal(MLIR_IntegerLiteralHandle h) {
    return h ? (IntegerLiteralStorage *)(uintptr_t)h : NULL;
}

static FloatLiteralStorage *resolve_float_literal(MLIR_FloatLiteralHandle h) {
    return h ? (FloatLiteralStorage *)(uintptr_t)h : NULL;
}

static StringLiteralStorage *resolve_string_literal(MLIR_StringLiteralHandle h) {
    return h ? (StringLiteralStorage *)(uintptr_t)h : NULL;
}

static uint32_t mlir_literal_words_for_width(uint32_t width) {
    if (width == 0) return 0;
    return (width + 63u) / 64u;
}

static void mlir_literal_clear_larger_words(uint64_t *dst) {
    for (int i = 0; i < MLIR_LITERAL_LARGER_BITS_WORDS; i++) dst[i] = 0;
}

static int64_t mlir_literal_sign_extend_from_word(uint32_t width,
                                                  uint64_t word0) {
    if (width >= 64) return (int64_t)word0;
    uint64_t mask = (1ULL << width) - 1ULL;
    uint64_t v = word0 & mask;
    if (width > 0 && (v >> (width - 1)) & 1ULL)
        v |= ~mask;
    return (int64_t)v;
}

MLIR_IntegerLiteralHandle MLIR_CreateIntegerLiteral(MLIR_Context *ctx) {
    if (!ctx || !ctx->arena) return MLIR_INVALID_LITERAL_HANDLE;
    IntegerLiteralStorage *lit = arena_new(ctx->arena, IntegerLiteralStorage);
    memset(lit, 0, sizeof(*lit));
    return (MLIR_IntegerLiteralHandle)(uintptr_t)lit;
}

MLIR_FloatLiteralHandle MLIR_CreateFloatLiteral(MLIR_Context *ctx) {
    if (!ctx || !ctx->arena) return MLIR_INVALID_LITERAL_HANDLE;
    FloatLiteralStorage *lit = arena_new(ctx->arena, FloatLiteralStorage);
    memset(lit, 0, sizeof(*lit));
    return (MLIR_FloatLiteralHandle)(uintptr_t)lit;
}

MLIR_StringLiteralHandle MLIR_CreateStringLiteral(MLIR_Context *ctx) {
    if (!ctx || !ctx->arena) return MLIR_INVALID_LITERAL_HANDLE;
    StringLiteralStorage *lit = arena_new(ctx->arena, StringLiteralStorage);
    memset(lit, 0, sizeof(*lit));
    return (MLIR_StringLiteralHandle)(uintptr_t)lit;
}

bool MLIR_IntegerLiteral_UsesLargerBits(MLIR_IntegerLiteralHandle lit) {
    IntegerLiteralStorage *body = resolve_integer_literal(lit);
    return body && body->is_larger_bits;
}

bool MLIR_IntegerLiteral_SetValue(MLIR_IntegerLiteralHandle lit, uint32_t width,
                                  int64_t value) {
    IntegerLiteralStorage *body = resolve_integer_literal(lit);
    if (!body || width == 0 || width > 64) return false;
    body->width = width;
    body->value = value;
    body->is_larger_bits = false;
    body->larger_word_count = 0;
    mlir_literal_clear_larger_words(body->value_larger_bits);
    return true;
}

bool MLIR_IntegerLiteral_SetLargerBits(MLIR_IntegerLiteralHandle lit,
                                       uint32_t width, const uint64_t *words,
                                       uint32_t word_count) {
    IntegerLiteralStorage *body = resolve_integer_literal(lit);
    if (!body || !words || width == 0) return false;
    uint32_t need = mlir_literal_words_for_width(width);
    if (need == 0 || need > MLIR_LITERAL_LARGER_BITS_WORDS) return false;
    if (word_count < need) return false;
    body->width = width;
    body->value = 0;
    body->is_larger_bits = true;
    body->larger_word_count = need;
    mlir_literal_clear_larger_words(body->value_larger_bits);
    for (uint32_t i = 0; i < need; i++) body->value_larger_bits[i] = words[i];
    return true;
}

bool MLIR_IntegerLiteral_GetWidth(MLIR_IntegerLiteralHandle lit, uint32_t *out_width) {
    IntegerLiteralStorage *body = resolve_integer_literal(lit);
    if (!body || body->width == 0 || !out_width) return false;
    *out_width = body->width;
    return true;
}

bool MLIR_IntegerLiteral_GetValue(MLIR_IntegerLiteralHandle lit, int64_t *out) {
    IntegerLiteralStorage *body = resolve_integer_literal(lit);
    if (!body || !out) return false;
    if (!body->is_larger_bits) {
        *out = body->value;
        return true;
    }
    if (body->width > 64 || body->larger_word_count == 0) return false;
    *out = mlir_literal_sign_extend_from_word(body->width,
                                              body->value_larger_bits[0]);
    return true;
}

bool MLIR_IntegerLiteral_GetLargerBits(MLIR_IntegerLiteralHandle lit,
                                       uint64_t *words, uint32_t words_cap,
                                       uint32_t *out_word_count) {
    IntegerLiteralStorage *body = resolve_integer_literal(lit);
    if (!body || !out_word_count) return false;
    if (!body->is_larger_bits) return false;
    if (words) {
        if (words_cap < body->larger_word_count) return false;
        for (uint32_t i = 0; i < body->larger_word_count; i++)
            words[i] = body->value_larger_bits[i];
    }
    *out_word_count = body->larger_word_count;
    return true;
}

bool MLIR_FloatLiteral_UsesLargerBits(MLIR_FloatLiteralHandle lit) {
    FloatLiteralStorage *body = resolve_float_literal(lit);
    return body && body->is_larger_bits;
}

bool MLIR_FloatLiteral_SetValue(MLIR_FloatLiteralHandle lit, uint32_t width,
                                MLIR_FloatEncoding encoding, double value) {
    FloatLiteralStorage *body = resolve_float_literal(lit);
    if (!body || width == 0 || width > 64) return false;
    body->width = width;
    body->encoding = encoding;
    body->value = value;
    body->is_larger_bits = false;
    body->larger_word_count = 0;
    mlir_literal_clear_larger_words(body->value_larger_bits);
    return true;
}

bool MLIR_FloatLiteral_SetLargerBits(MLIR_FloatLiteralHandle lit, uint32_t width,
                                     MLIR_FloatEncoding encoding,
                                     const uint64_t *words, uint32_t word_count) {
    FloatLiteralStorage *body = resolve_float_literal(lit);
    if (!body || !words || width == 0) return false;
    uint32_t need = mlir_literal_words_for_width(width);
    if (need == 0 || need > MLIR_LITERAL_LARGER_BITS_WORDS) return false;
    if (word_count < need) return false;
    body->width = width;
    body->encoding = encoding;
    body->value = 0.0;
    body->is_larger_bits = true;
    body->larger_word_count = need;
    mlir_literal_clear_larger_words(body->value_larger_bits);
    for (uint32_t i = 0; i < need; i++) body->value_larger_bits[i] = words[i];
    return true;
}

bool MLIR_FloatLiteral_GetWidth(MLIR_FloatLiteralHandle lit, uint32_t *out_width) {
    FloatLiteralStorage *body = resolve_float_literal(lit);
    if (!body || body->width == 0 || !out_width) return false;
    *out_width = body->width;
    return true;
}

bool MLIR_FloatLiteral_GetEncoding(MLIR_FloatLiteralHandle lit,
                                   MLIR_FloatEncoding *out_encoding) {
    FloatLiteralStorage *body = resolve_float_literal(lit);
    if (!body || body->width == 0 || !out_encoding) return false;
    *out_encoding = body->encoding;
    return true;
}

bool MLIR_FloatLiteral_GetValue(MLIR_FloatLiteralHandle lit, double *out) {
    FloatLiteralStorage *body = resolve_float_literal(lit);
    if (!body || !out) return false;
    if (body->is_larger_bits) return false;
    *out = body->value;
    return true;
}

bool MLIR_FloatLiteral_GetLargerBits(MLIR_FloatLiteralHandle lit,
                                     uint64_t *words, uint32_t words_cap,
                                     uint32_t *out_word_count) {
    FloatLiteralStorage *body = resolve_float_literal(lit);
    if (!body || !out_word_count) return false;
    if (!body->is_larger_bits) return false;
    if (words) {
        if (words_cap < body->larger_word_count) return false;
        for (uint32_t i = 0; i < body->larger_word_count; i++)
            words[i] = body->value_larger_bits[i];
    }
    *out_word_count = body->larger_word_count;
    return true;
}

bool MLIR_StringLiteral_SetBytes(MLIR_Context *ctx, MLIR_StringLiteralHandle lit,
                                 uint32_t element_width, const uint8_t *bytes,
                                 size_t byte_count) {
    StringLiteralStorage *body = resolve_string_literal(lit);
    (void)element_width;
    if (!ctx || !ctx->arena || !body || !bytes || byte_count == 0) return false;
    uint8_t *owned = arena_new_array(ctx->arena, uint8_t, byte_count);
    memcpy(owned, bytes, byte_count);
    body->bytes = owned;
    body->byte_count = byte_count;
    return true;
}

bool MLIR_StringLiteral_GetBytes(MLIR_StringLiteralHandle lit,
                                 const uint8_t **out_bytes, size_t *out_byte_count) {
    StringLiteralStorage *body = resolve_string_literal(lit);
    if (!body || !body->bytes || body->byte_count == 0 || !out_bytes || !out_byte_count)
        return false;
    *out_bytes = body->bytes;
    *out_byte_count = body->byte_count;
    return true;
}
