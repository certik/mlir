// Native (arm64/Darwin, LP64) runtime for the llvm Mach-O backend.
// The backend gets its runtime as ordinary tinyC-subset C, parsed into the
// user's module and lowered through the same llvm -> aarch64 path. The only
// primitives that cannot be expressed in C are the raw syscalls; those bind
// to libSystem dyld stubs by their underscore names (`_write`, `_exit`,
// `_mmap`, ...), which the Mach-O encoder already imports.
//
// Semantics mirror runtime_wasm.c exactly so output matches the wasm suite:
//   printI64(v)    -> decimal digits, NO trailing newline
//   printNewline() -> "\n"
//   printI32(v)    -> decimal digits + "\n"
//   printStr(s)    -> the string bytes + a trailing "\n"
//
// Each public runtime function is injected only if the user program does not
// already define it (forward declarations don't count), so user-provided
// implementations (e.g. selfhost's stdlib) win and we never hit a "two
// definitions" parse error.
//
// IMPORTANT: this header is parsed by tinyC itself during self-host (it is
// included by driver.c, which tinyC recompiles). It therefore must stay
// within the tinyC C subset — in particular it must NOT use global
// aggregate initializers (arrays of structs / arrays of string pointers /
// `char[]` string globals), none of which tinyC can parse or lower. The
// runtime function sources are added one-by-one via `tinyc_rt_add` using
// only local string literals and function-call arguments.
#ifndef TINYC_NATIVE_RUNTIME_H
#define TINYC_NATIVE_RUNTIME_H

#include <base/arena.h>
#include <base/string.h>
#include "tinyc.h"

// True if `prog` already DEFINES (not just forward-declares) a function
// named `name`.
static bool tinyc_user_defines(Program *prog, const char *name) {
    size_t nlen = strlen(name);
    for (size_t i = 0; i < prog->funcs.size; i++) {
        Func *f = prog->funcs.data[i];
        if (f->is_forward) continue;
        if (f->name.size == nlen && memcmp(f->name.str, name, nlen) == 0)
            return true;
    }
    return false;
}

// Append the source of one runtime function (named `name`) to `*combined`,
// unless the user program already defines it. `*any` tracks whether anything
// has been appended yet (so the first append seeds `combined`).
static void tinyc_rt_add(Arena *arena, Program *prog, string *combined,
                         bool *any, const char *name, const char *src) {
    if (tinyc_user_defines(prog, name)) return;
    string s = str_from_cstr_view((char *)src);
    *combined = *any ? str_concat(arena, *combined, s) : s;
    *any = true;
}

// Inject the native runtime into `prog` (parsed in-place). Only functions the
// user hasn't defined are added. No-op if nothing needs injecting.
//
// Definitions are ordered so that a function only calls ones added earlier.
static void tinyc_inject_native_runtime(Arena *arena, Program *prog,
                                        bool target_wasm32) {
    string combined = (string){0};
    bool any = false;

    tinyc_rt_add(arena, prog, &combined, &any, "printNewline",
        "void printNewline(void){ char c; c=10; _write(1,&c,1); }\n");
    tinyc_rt_add(arena, prog, &combined, &any, "printI64",
        "void printI64(long v){\n"
        "  char tmp[32]; char out[34]; unsigned long u; int neg; int n; int o; int i;\n"
        "  neg=0;\n"
        "  if(v<0){ neg=1; u=(unsigned long)(-(v+1))+1; } else { u=(unsigned long)v; }\n"
        "  n=0;\n"
        "  if(u==0){ tmp[n]=48; n=n+1; }\n"
        "  while(u){ tmp[n]=(char)(48+(u%10)); n=n+1; u=u/10; }\n"
        "  o=0;\n"
        "  if(neg){ out[o]=45; o=o+1; }\n"
        "  i=n-1;\n"
        "  while(i>=0){ out[o]=tmp[i]; o=o+1; i=i-1; }\n"
        "  _write(1,out,(long)o);\n"
        "}\n");
    tinyc_rt_add(arena, prog, &combined, &any, "printI32",
        "void printI32(int v){ printI64((long)v); printNewline(); }\n");
    tinyc_rt_add(arena, prog, &combined, &any, "printStr",
        "void printStr(char *s){\n"
        "  long n; char c;\n"
        "  n=0;\n"
        "  if(s){ while(s[n]) n=n+1; _write(1,s,n); }\n"
        "  c=10; _write(1,&c,1);\n"
        "}\n");
    // libc string/memory helpers. Signatures match the suite's extern decls.
    tinyc_rt_add(arena, prog, &combined, &any, "strlen",
        "long strlen(char *s){ long n; n=0; while(s[n]){ n=n+1; } return n; }\n");
    tinyc_rt_add(arena, prog, &combined, &any, "strcmp",
        "int strcmp(char *a, char *b){\n"
        "  long i; i=0;\n"
        "  while(a[i] && a[i]==b[i]){ i=i+1; }\n"
        "  return (int)(unsigned char)a[i] - (int)(unsigned char)b[i];\n"
        "}\n");
    tinyc_rt_add(arena, prog, &combined, &any, "memcmp",
        "int memcmp(void *a, void *b, long n){\n"
        "  char *pa; char *pb; long i;\n"
        "  pa=(char*)a; pb=(char*)b; i=0;\n"
        "  while(i<n){ if(pa[i] != pb[i]){ return (int)(unsigned char)pa[i] - (int)(unsigned char)pb[i]; } i=i+1; }\n"
        "  return 0;\n"
        "}\n");
    tinyc_rt_add(arena, prog, &combined, &any, "memchr",
        "void *memchr(void *s, int c, unsigned long n){\n"
        "  char *p; unsigned long i; p=(char*)s; i=0;\n"
        "  while(i<n){ if((unsigned char)p[i] == (unsigned char)c){ return p+i; } i=i+1; }\n"
        "  return (void*)0;\n"
        "}\n");
    // Heap allocator. The backend does not yet support module-level globals,
    // so malloc cannot keep a bump pointer; each call mmaps its own anonymous
    // region (rounded up to a 16 KiB page) and free is a no-op. Wasteful but
    // correct for the test suite.
    tinyc_rt_add(arena, prog, &combined, &any, "malloc",
        "void *malloc(long n){\n"
        "  long sz; char *p;\n"
        "  if(n<=0){ n=1; }\n"
        "  sz=(n+16383)/16384*16384;\n"
        "  p=_mmap(0, sz, 3, 4098, -1, 0);\n"
        "  return p;\n"
        "}\n");
    tinyc_rt_add(arena, prog, &combined, &any, "free",
        "void free(void *p){ }\n");
    // Variadic support. tinyC lowers `va_arg(ap, T)` to a call to one of these
    // helpers (the backend lowers va_start/va_end). On Darwin arm64 the va_list
    // buffer's first 8 bytes hold a `cur` pointer into the on-stack variadic
    // area; each helper reads the next 8-byte slot and advances cur.
    tinyc_rt_add(arena, prog, &combined, &any, "tinyc_va_arg_i32",
        "int tinyc_va_arg_i32(char **ap){\n"
        "  char *p; int *q; p=*ap; *ap=p+8; q=(int*)p; return *q;\n"
        "}\n");
    tinyc_rt_add(arena, prog, &combined, &any, "tinyc_va_arg_i64",
        "long tinyc_va_arg_i64(char **ap){\n"
        "  char *p; long *q; p=*ap; *ap=p+8; q=(long*)p; return *q;\n"
        "}\n");
    tinyc_rt_add(arena, prog, &combined, &any, "tinyc_va_arg_f64",
        "double tinyc_va_arg_f64(char **ap){\n"
        "  char *p; double *q; p=*ap; *ap=p+8; q=(double*)p; return *q;\n"
        "}\n");
    tinyc_rt_add(arena, prog, &combined, &any, "tinyc_va_arg_ptr",
        "char *tinyc_va_arg_ptr(char **ap){\n"
        "  char *p; char **q; p=*ap; *ap=p+8; q=(char**)p; return *q;\n"
        "}\n");
    tinyc_rt_add(arena, prog, &combined, &any, "tinyc_va_arg_struct",
        "void tinyc_va_arg_struct(char **ap, char *out, long size){\n"
        "  char *p; long words; long i; long *o; long *s;\n"
        "  p=*ap; words=(size+7)/8; o=(long*)out; s=(long*)p;\n"
        "  for(i=0;i<words;i=i+1){ o[i]=s[i]; }\n"
        "  *ap=p+words*8;\n"
        "}\n");
    // Integer formatting helpers used by printf.
    tinyc_rt_add(arena, prog, &combined, &any, "tinyc_rt_fmt_u64",
        "int tinyc_rt_fmt_u64(unsigned long v, char *buf, int base, int upper){\n"
        "  char tmp[32]; int n; int o; int i; char *D;\n"
        "  if(upper){ D=\"0123456789ABCDEF\"; } else { D=\"0123456789abcdef\"; }\n"
        "  n=0;\n"
        "  if(v==0){ tmp[n]='0'; n=n+1; }\n"
        "  while(v){ tmp[n]=D[v%(unsigned long)base]; n=n+1; v=v/(unsigned long)base; }\n"
        "  o=0; i=n-1;\n"
        "  while(i>=0){ buf[o]=tmp[i]; o=o+1; i=i-1; }\n"
        "  return o;\n"
        "}\n");
    tinyc_rt_add(arena, prog, &combined, &any, "tinyc_rt_fmt_i64",
        "int tinyc_rt_fmt_i64(long v, char *buf){\n"
        "  unsigned long u; int neg; int k;\n"
        "  neg=0;\n"
        "  if(v<0){ neg=1; u=(unsigned long)(-(v+1))+1; } else { u=(unsigned long)v; }\n"
        "  if(neg){ buf[0]='-'; }\n"
        "  k=tinyc_rt_fmt_u64(u, buf+neg, 10, 0);\n"
        "  return k+neg;\n"
        "}\n");
    // Floating-point formatting (mirrors runtime_wasm.c fmt_f64): up to 6
    // significant digits, trailing zeros trimmed, fixed notation for
    // exponents in [-4,6) and scientific otherwise.
    tinyc_rt_add(arena, prog, &combined, &any, "tinyc_rt_fmt_f64",
        "int tinyc_rt_fmt_f64(double v, char *buf){\n"
        "  int out; int exp10; double m; char digits[8]; char ebuf[8];\n"
        "  int di; int k; int ndig; int i; int en;\n"
        "  out=0;\n"
        "  if(v != v){ buf[0]='n'; buf[1]='a'; buf[2]='n'; return 3; }\n"
        "  if(v < 0.0){ buf[out]='-'; out=out+1; v=-v; }\n"
        "  if(v > 1e308){ buf[out]='i'; buf[out+1]='n'; buf[out+2]='f'; return out+3; }\n"
        "  if(v == 0.0){ buf[out]='0'; return out+1; }\n"
        "  exp10=0; m=v;\n"
        "  while(m >= 10.0){ m=m/10.0; exp10=exp10+1; }\n"
        "  while(m < 1.0){ m=m*10.0; exp10=exp10-1; }\n"
        "  for(di=0; di<8; di=di+1){ digits[di]=0; }\n"
        "  for(k=0;k<6;k=k+1){\n"
        "    int d; d=(int)m; if(d<0){ d=0; } if(d>9){ d=9; }\n"
        "    digits[k]=(char)(48+d); m=(m-(double)d)*10.0;\n"
        "  }\n"
        "  ndig=6; while(ndig>1 && digits[ndig-1]=='0'){ ndig=ndig-1; }\n"
        "  if(exp10 >= -4 && exp10 < 6){\n"
        "    if(exp10>=0){\n"
        "      int int_digits; int_digits=exp10+1;\n"
        "      for(i=0;i<int_digits;i=i+1){ if(i<ndig){ buf[out]=digits[i]; } else { buf[out]='0'; } out=out+1; }\n"
        "      if(int_digits<ndig){ buf[out]='.'; out=out+1;\n"
        "        for(i=int_digits;i<ndig;i=i+1){ buf[out]=digits[i]; out=out+1; } }\n"
        "    } else {\n"
        "      buf[out]='0'; out=out+1; buf[out]='.'; out=out+1;\n"
        "      for(i=0;i<(-exp10-1);i=i+1){ buf[out]='0'; out=out+1; }\n"
        "      for(i=0;i<ndig;i=i+1){ buf[out]=digits[i]; out=out+1; }\n"
        "    }\n"
        "  } else {\n"
        "    buf[out]=digits[0]; out=out+1;\n"
        "    if(ndig>1){ buf[out]='.'; out=out+1;\n"
        "      for(i=1;i<ndig;i=i+1){ buf[out]=digits[i]; out=out+1; } }\n"
        "    buf[out]='e'; out=out+1;\n"
        "    if(exp10<0){ buf[out]='-'; out=out+1; exp10=-exp10; } else { buf[out]='+'; out=out+1; }\n"
        "    en=0; if(exp10==0){ ebuf[en]='0'; en=en+1; }\n"
        "    while(exp10){ ebuf[en]=(char)(48+(exp10%10)); en=en+1; exp10=exp10/10; }\n"
        "    if(en<2){ buf[out]='0'; out=out+1; }\n"
        "    for(i=en-1;i>=0;i=i-1){ buf[out]=ebuf[i]; out=out+1; }\n"
        "  }\n"
        "  return out;\n"
        "}\n");
    tinyc_rt_add(arena, prog, &combined, &any, "printF64",
        "void printF64(double v){ char b[64]; int n; n=tinyc_rt_fmt_f64(v,b); _write(1,b,(long)n); }\n");
    tinyc_rt_add(arena, prog, &combined, &any, "tinyc_rt_fmt_f64_prec",
        "int tinyc_rt_fmt_f64_prec(double v, char *buf, int prec){\n"
        "  int out; double scale; int i; long ip; double rounded; double frac;\n"
        "  out=0;\n"
        "  if(v != v){ buf[0]='n'; buf[1]='a'; buf[2]='n'; return 3; }\n"
        "  if(v < 0.0){ buf[out]='-'; out=out+1; v=-v; }\n"
        "  scale=1.0;\n"
        "  for(i=0;i<prec;i=i+1){ scale=scale*10.0; }\n"
        "  rounded=(double)(long)(v*scale+0.5)/scale;\n"
        "  ip=(long)rounded;\n"
        "  out=out+tinyc_rt_fmt_i64(ip, buf+out);\n"
        "  if(prec>0){\n"
        "    buf[out]='.'; out=out+1;\n"
        "    frac=rounded-(double)ip;\n"
        "    if(frac<0.0){ frac=-frac; }\n"
        "    for(i=0;i<prec;i=i+1){\n"
        "      int d; frac=frac*10.0; d=(int)frac; if(d<0){ d=0; } if(d>9){ d=9; }\n"
        "      buf[out]=(char)(48+d); out=out+1; frac=frac-(double)d;\n"
        "    }\n"
        "  }\n"
        "  return out;\n"
        "}\n");
    tinyc_rt_add(arena, prog, &combined, &any, "printF32",
        "void printF32(float v){ char b[64]; int n; n=tinyc_rt_fmt_f64((double)v,b); _write(1,b,(long)n); }\n");
    // Minimal printf: supports %d %i %u %x %X %p %c %s %f %g %% (and length
    // modifiers l/ll). Semantics mirror runtime_wasm.c's vprintf_impl.
    tinyc_rt_add(arena, prog, &combined, &any, "printf",
        "int printf(char *fmt, ...){\n"
        "  char buf[64]; int total; int i; int prec; int lcount; char c;\n"
        "  __builtin_va_list ap; __builtin_va_start(ap, fmt);\n"
        "  total=0; i=0;\n"
        "  while(fmt[i]){\n"
        "    if(fmt[i] != '%'){ _write(1,fmt+i,1); total=total+1; i=i+1; continue; }\n"
        "    i=i+1;\n"
        "    prec=-1;\n"
        "    if(fmt[i]=='.'){ i=i+1; prec=0; while(fmt[i]>='0' && fmt[i]<='9'){ prec=prec*10+(fmt[i]-'0'); i=i+1; } }\n"
        "    lcount=0;\n"
        "    while(fmt[i]=='l'){ lcount=lcount+1; i=i+1; }\n"
        "    c=fmt[i];\n"
        "    if(c==0){ break; }\n"
        "    i=i+1;\n"
        "    if(c=='d' || c=='i'){\n"
        "      long v; int n;\n"
        "      if(lcount>=1){ v=__builtin_va_arg(ap,long); } else { v=(long)__builtin_va_arg(ap,int); }\n"
        "      n=tinyc_rt_fmt_i64(v,buf); _write(1,buf,(long)n); total=total+n;\n"
        "    } else if(c=='u'){\n"
        "      unsigned long v; int n;\n"
        "      v=(unsigned long)__builtin_va_arg(ap,unsigned long);\n"
        "      n=tinyc_rt_fmt_u64(v,buf,10,0); _write(1,buf,(long)n); total=total+n;\n"
        "    } else if(c=='x' || c=='X'){\n"
        "      unsigned long v; int n; int up; up=0; if(c=='X'){ up=1; }\n"
        "      v=(unsigned long)__builtin_va_arg(ap,unsigned long);\n"
        "      n=tinyc_rt_fmt_u64(v,buf,16,up); _write(1,buf,(long)n); total=total+n;\n"
        "    } else if(c=='p'){\n"
        "      char *p; int n; p=__builtin_va_arg(ap,char*);\n"
        "      _write(1,\"0x\",2); total=total+2;\n"
        "      n=tinyc_rt_fmt_u64((unsigned long)p,buf,16,0); _write(1,buf,(long)n); total=total+n;\n"
        "    } else if(c=='c'){\n"
        "      int v; char ch; v=__builtin_va_arg(ap,int); ch=(char)v; _write(1,&ch,1); total=total+1;\n"
        "    } else if(c=='s'){\n"
        "      char *s; long sn; s=__builtin_va_arg(ap,char*);\n"
        "      if(!s){ s=\"(null)\"; }\n"
        "      sn=0; while(s[sn]){ sn=sn+1; } _write(1,s,sn); total=total+(int)sn;\n"
        "    } else if(c=='f' || c=='F'){\n"
        "      double v; int n; int p; v=__builtin_va_arg(ap,double);\n"
        "      p=prec; if(p<0){ p=6; }\n"
        "      n=tinyc_rt_fmt_f64_prec(v,buf,p); _write(1,buf,(long)n); total=total+n;\n"
        "    } else if(c=='g' || c=='G'){\n"
        "      double v; int n; v=__builtin_va_arg(ap,double);\n"
        "      n=tinyc_rt_fmt_f64(v,buf); _write(1,buf,(long)n); total=total+n;\n"
        "    } else if(c=='%'){\n"
        "      _write(1,\"%\",1); total=total+1;\n"
        "    } else {\n"
        "      _write(1,\"%\",1); _write(1,&c,1); total=total+2;\n"
        "    }\n"
        "  }\n"
        "  __builtin_va_end(ap);\n"
        "  return total;\n"
        "}\n");

    if (!any) return;

    // Forward declarations for the raw syscall stubs the runtime binds to.
    const char *prelude =
        "long _write(long fd, char *buf, long n);\n"
        "char *_mmap(long addr, long len, long prot, long flags, long fd, long off);\n";
    string full = str_concat(arena,
                             str_from_cstr_view((char *)prelude),
                             combined);
    VecTcTok toks = tinyc_lex(arena, full);
    tinyc_parse_into(arena, prog, toks, target_wasm32);
}

#endif // TINYC_NATIVE_RUNTIME_H
