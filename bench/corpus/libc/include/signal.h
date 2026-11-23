/* signal.h shim: signal() is a no-op. */
#pragma once
typedef int sig_atomic_t;
typedef void (*sighandler_t)(int);
#define SIG_DFL ((sighandler_t)0)
#define SIG_IGN ((sighandler_t)1)
#define SIGINT 2
sighandler_t signal(int sig, sighandler_t handler);
