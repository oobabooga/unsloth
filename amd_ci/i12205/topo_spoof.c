// LD_PRELOAD shim: present a different max_waves_per_simd in the KFD topology
// that libhsakmt reads (topology.c fopen()s nodes/<n>/properties). The kernel
// keeps its own values, so this recreates, on a gfx1151, the userspace/KFD
// disagreement a GPU whose firmware reports 20 waves/SIMD has natively (#12205).
// Inert unless AMD_CI_SPOOF_WAVES is set. Each rewrite is appended to
// AMD_CI_SPOOF_LOG so a probe can prove the spoof engaged.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef FILE *(*fopen_fn)(const char *, const char *);

static int is_node_props(const char *p) {
    if (!p || !strstr(p, "/topology/nodes/")) return 0;
    size_t n = strlen(p), k = strlen("/properties");
    return n > k && strcmp(p + n - k, "/properties") == 0;
}

static void log_line(const char *msg) {
    const char *path = getenv("AMD_CI_SPOOF_LOG");
    if (!path) return;
    int fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);  // not fopen: no recursion
    if (fd < 0) return;
    ssize_t w = write(fd, msg, strlen(msg));
    (void)w;
    close(fd);
}

static FILE *maybe_spoof(fopen_fn real, const char *path, const char *mode) {
    const char *want = getenv("AMD_CI_SPOOF_WAVES");
    FILE *f = real(path, mode);
    if (!f || !want || !is_node_props(path) || mode[0] != 'r') return f;

    size_t cap = 1 << 16, len = 0;
    char *buf = malloc(cap + 64);
    if (!buf) return f;
    size_t r;
    while ((r = fread(buf + len, 1, cap - len, f)) > 0) len += r;
    fclose(f);
    buf[len] = '\0';

    unsigned long long simd_count = 0;
    char *s = strstr(buf, "\nsimd_count ");
    if (s) simd_count = strtoull(s + strlen("\nsimd_count "), NULL, 10);
    char *m = strstr(buf, "\nmax_waves_per_simd ");
    if (simd_count > 0 && m) {
        char *val = m + strlen("\nmax_waves_per_simd ");
        char *eol = strchr(val, '\n');
        if (!eol) eol = buf + len;
        char old[32] = {0};
        snprintf(old, sizeof(old), "%.*s", (int)(eol - val), val);
        size_t tail = (size_t)(buf + len - eol);
        size_t newlen = (size_t)(val - buf) + strlen(want) + tail;
        char *out = malloc(newlen + 1);  // leaked on purpose: fmemopen needs it alive
        if (!out) return fmemopen(buf, len, "r");
        memcpy(out, buf, (size_t)(val - buf));
        strcpy(out + (val - buf), want);
        memcpy(out + (val - buf) + strlen(want), eol, tail);
        out[newlen] = '\0';
        char msg[512];
        snprintf(msg, sizeof(msg), "pid=%d %s max_waves_per_simd %s -> %s\n",
                 (int)getpid(), path, old, want);
        log_line(msg);
        free(buf);
        return fmemopen(out, newlen, "r");
    }
    return fmemopen(buf, len, "r");
}

FILE *fopen(const char *path, const char *mode) {
    static fopen_fn real;
    if (!real) real = (fopen_fn)dlsym(RTLD_NEXT, "fopen");
    return maybe_spoof(real, path, mode);
}

FILE *fopen64(const char *path, const char *mode) {
    static fopen_fn real;
    if (!real) real = (fopen_fn)dlsym(RTLD_NEXT, "fopen64");
    return maybe_spoof(real, path, mode);
}
