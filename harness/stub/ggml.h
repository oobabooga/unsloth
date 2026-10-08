#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#define GGML_ASSERT(x) do { if (!(x)) { throw std::runtime_error(#x); } } while (0)
static inline int64_t ggml_time_ms() { return 0; }
