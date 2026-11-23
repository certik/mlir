// Stand-in for <sys/stat.h> used by bench/run.py: every compiler compiles the
// self-host sources with -nostdinc against corec-stdlib, so they all parse the
// same headers. examples/tinyc/driver.c only needs chmod() from here (and not
// at all under tinyC itself).
#pragma once
int chmod(const char *path, unsigned int mode);
