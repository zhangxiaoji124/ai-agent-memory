#include <cstdio>
#include "test.h"

int main() {
  int passed = 0;
  for (auto &tc : test_registry()) {
    std::fprintf(stderr, "[ RUN      ] %s\n", tc.name.c_str());
    tc.fn();
    std::fprintf(stderr, "[       OK ] %s\n", tc.name.c_str());
    passed++;
  }
  std::fprintf(stderr, "[  PASSED  ] %d tests\n", passed);
  return 0;
}
