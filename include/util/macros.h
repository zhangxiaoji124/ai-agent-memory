#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define AMIO_UNUSED(x) (void)(x)

#define AMIO_CHECK(cond)                                                        \
  do {                                                                          \
    if (!(cond)) {                                                              \
      std::fprintf(stderr, "[AMIO_CHECK] %s:%d: %s\n", __FILE__, __LINE__,      \
                   #cond);                                                      \
      std::abort();                                                             \
    }                                                                           \
  } while (0)

#define AMIO_ASSERT(cond) AMIO_CHECK(cond)
