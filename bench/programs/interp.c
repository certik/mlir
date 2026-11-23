// Runtime kernel: a switch-dispatched bytecode interpreter (branchy, the shape
// of a compiler's inner loops).
#include <stdio.h>

#define OP_PUSH  0
#define OP_LOAD  1
#define OP_STORE 2
#define OP_ADD   3
#define OP_SUB   4
#define OP_MUL   5
#define OP_LT    6
#define OP_JZ    7
#define OP_JMP   8
#define OP_HALT  9
#define OP_MOD   10

static int code[64];
static long long stack[64];
static long long vars[8];

static int emit_at = 0;
static void op(int o) { code[emit_at++] = o; }
static void op1(int o, int arg) { code[emit_at++] = o; code[emit_at++] = arg; }

static long long run(void) {
    int pc = 0, sp = 0;
    for (;;) {
        int o = code[pc++];
        switch (o) {
        case OP_PUSH:  stack[sp++] = code[pc++]; break;
        case OP_LOAD:  stack[sp++] = vars[code[pc++]]; break;
        case OP_STORE: vars[code[pc++]] = stack[--sp]; break;
        case OP_ADD:   sp--; stack[sp - 1] = stack[sp - 1] + stack[sp]; break;
        case OP_SUB:   sp--; stack[sp - 1] = stack[sp - 1] - stack[sp]; break;
        case OP_MUL:   sp--; stack[sp - 1] = stack[sp - 1] * stack[sp]; break;
        case OP_MOD:   sp--; stack[sp - 1] = stack[sp - 1] % stack[sp]; break;
        case OP_LT:    sp--; stack[sp - 1] = stack[sp - 1] < stack[sp]; break;
        case OP_JZ:    { int t = code[pc++]; if (stack[--sp] == 0) pc = t; } break;
        case OP_JMP:   pc = code[pc]; break;
        case OP_HALT:  return vars[1];
        default:       return -1;
        }
    }
}

int main(void) {
    // i = 0; s = 0; while (i < 3000000) { s = (s + i * i) % 1000003; i = i + 1; }
    op1(OP_PUSH, 0); op1(OP_STORE, 0);
    op1(OP_PUSH, 0); op1(OP_STORE, 1);
    int loop = emit_at;
    op1(OP_LOAD, 0); op1(OP_PUSH, 3000000); op(OP_LT);
    op1(OP_JZ, 0); int patch = emit_at - 1;
    op1(OP_LOAD, 1); op1(OP_LOAD, 0); op1(OP_LOAD, 0); op(OP_MUL); op(OP_ADD);
    op1(OP_PUSH, 1000003); op(OP_MOD); op1(OP_STORE, 1);
    op1(OP_LOAD, 0); op1(OP_PUSH, 1); op(OP_ADD); op1(OP_STORE, 0);
    op1(OP_JMP, loop);
    code[patch] = emit_at;
    op(OP_HALT);
    printf("s = %lld\n", run());
    return 0;
}
