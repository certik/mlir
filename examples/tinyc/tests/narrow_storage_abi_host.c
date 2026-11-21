// Host side of narrow_storage_abi.tc, compiled by the C compiler at link time:
// the struct layout and the char / short / bool value ABI must match tinyC's.
#include <stdbool.h>
#include <stddef.h>

struct nsa_rec {
    char c;
    bool b;
    short s;
    unsigned char uc;
    unsigned short us;
    int i;
    char name[5];
    bool flags[3];
    short arr[3];
    signed char sc;
};

short nsa_host_g = -1234;

int nsa_host_layout(int *out) {
    int n = 0;
    out[n++] = (int)sizeof(struct nsa_rec);
    out[n++] = (int)_Alignof(struct nsa_rec);
    out[n++] = (int)offsetof(struct nsa_rec, c);
    out[n++] = (int)offsetof(struct nsa_rec, b);
    out[n++] = (int)offsetof(struct nsa_rec, s);
    out[n++] = (int)offsetof(struct nsa_rec, uc);
    out[n++] = (int)offsetof(struct nsa_rec, us);
    out[n++] = (int)offsetof(struct nsa_rec, i);
    out[n++] = (int)offsetof(struct nsa_rec, name);
    out[n++] = (int)offsetof(struct nsa_rec, flags);
    out[n++] = (int)offsetof(struct nsa_rec, arr);
    out[n++] = (int)offsetof(struct nsa_rec, sc);
    return n;
}

void nsa_host_fill(struct nsa_rec *r) {
    r->c = -5; r->b = true; r->s = -30000; r->uc = 200; r->us = 60000; r->i = 77;
    r->name[0] = 'a'; r->name[1] = 'b'; r->name[2] = 'c'; r->name[3] = 0; r->name[4] = 'z';
    r->flags[0] = false; r->flags[1] = true; r->flags[2] = true;
    r->arr[0] = 1; r->arr[1] = -2; r->arr[2] = 32767; r->sc = -128;
}

int nsa_host_check(const struct nsa_rec *r) {
    int bad = 0;
    if (r->c != 'x') bad |= 1;
    if (r->b != false) bad |= 2;
    if (r->s != -2) bad |= 4;
    if (r->uc != 255) bad |= 8;
    if (r->us != 65535) bad |= 16;
    if (r->i != -9) bad |= 32;
    if (r->name[0] != 'q' || r->name[4] != 'r') bad |= 64;
    if (r->flags[0] != true || r->flags[1] != false || r->flags[2] != true) bad |= 128;
    if (r->arr[0] != -1 || r->arr[1] != 2 || r->arr[2] != -32768) bad |= 256;
    if (r->sc != 127) bad |= 512;
    return bad;
}

short nsa_host_neg(short x) { return (short)-x; }
unsigned char nsa_host_inc(unsigned char x) { return (unsigned char)(x + 1); }
bool nsa_host_not(bool x) { return !x; }
signed char nsa_host_half(signed char x) { return (signed char)(x / 2); }
unsigned short nsa_host_sum(unsigned short a, char b, bool c, short d) {
    return (unsigned short)(a + b + c + d);
}

// Calls back into tinyC-compiled code with narrow arguments.
extern short nsa_tc_twice(short x);
extern bool nsa_tc_is_neg(signed char c);
extern unsigned char nsa_tc_low(int x);
int nsa_host_callback(void) {
    int bad = 0;
    if (nsa_tc_twice(-1000) != -2000) bad |= 1;
    if (nsa_tc_twice(20000) != (short)40000) bad |= 2;
    if (nsa_tc_is_neg(-3) != true) bad |= 4;
    if (nsa_tc_is_neg(3) != false) bad |= 8;
    if (nsa_tc_low(0x1234) != 0x34) bad |= 16;
    return bad;
}
