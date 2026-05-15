#pragma once

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <utility>
#include <vector>

struct TestCase {
  std::string name;
  std::function<void()> fn;
};

inline std::vector<TestCase> &test_registry() {
  static std::vector<TestCase> r;
  return r;
}

struct RegisterTest {
  RegisterTest(const char *name, std::function<void()> fn) {
    test_registry().push_back(TestCase{std::string(name), std::move(fn)});
  }
};

#define TEST(name)                                                              \
  static void name();                                                           \
  static RegisterTest reg_##name(#name, &name);                                 \
  static void name()

#define EXPECT_TRUE(x)                                                          \
  do {                                                                          \
    if (!(x)) {                                                                 \
      std::fprintf(stderr, "[FAIL] %s:%d: EXPECT_TRUE(%s)\n", __FILE__,         \
                   __LINE__, #x);                                               \
      std::abort();                                                             \
    }                                                                           \
  } while (0)
