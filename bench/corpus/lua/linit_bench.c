/* Replacement for Lua's linit.c (the documented customization point): opens
 * the libraries that need no OS services beyond stdio. io, os and package are
 * left out (they need popen/tmpfile/system/remove/time/dlopen/...). */
#define linit_c
#define LUA_LIB
#include "lprefix.h"
#include <stddef.h>
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"

static const luaL_Reg bench_loadedlibs[] = {
  {LUA_GNAME, luaopen_base},
  {LUA_COLIBNAME, luaopen_coroutine},
  {LUA_TABLIBNAME, luaopen_table},
  {LUA_STRLIBNAME, luaopen_string},
  {LUA_MATHLIBNAME, luaopen_math},
  {LUA_UTF8LIBNAME, luaopen_utf8},
  {LUA_DBLIBNAME, luaopen_debug},
  {NULL, NULL}
};

LUALIB_API void luaL_openlibs (lua_State *L) {
  const luaL_Reg *lib;
  for (lib = bench_loadedlibs; lib->func; lib++) {
    luaL_requiref(L, lib->name, lib->func, 1);
    lua_pop(L, 1);
  }
}
