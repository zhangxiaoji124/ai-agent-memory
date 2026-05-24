#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "runtime/graph_reorder.h"

static const char *arg_value(int argc, char **argv, const char *key) {
  for (int i = 1; i + 1 < argc; i++) {
    if (std::strcmp(argv[i], key) == 0)
      return argv[i + 1];
  }
  return nullptr;
}

static bool arg_flag(int argc, char **argv, const char *key) {
  for (int i = 1; i < argc; i++) {
    if (std::strcmp(argv[i], key) == 0)
      return true;
  }
  return false;
}

int main(int argc, char **argv) {
  const char *input  = arg_value(argc, argv, "--input");
  const char *output = arg_value(argc, argv, "--output");
  const char *algo   = arg_value(argc, argv, "--algo");
  const char *wstr   = arg_value(argc, argv, "--window");

  if (!input || !output || arg_flag(argc, argv, "--help")) {
    std::fprintf(stderr,
                 "用法: reorder_index --input <src.index> --output <dst.index>\n"
                 "                   [--algo bfs|gorder] [--window N]\n"
                 "\n"
                 "  --algo bfs      BFS 广度优先重排（默认，快速）\n"
                 "  --algo gorder   Gorder 贪心重排（更优局部性，耗时约 3-5×）\n"
                 "  --window N      Gorder 滑动窗口大小（默认 64）\n"
                 "\n"
                 "  重排后索引与原格式兼容（v1/v2），可直接替换原索引使用。\n"
                 "  eval_disk 汇总日志会输出 locality 指标以量化改善幅度。\n");
    return 1;
  }

  const bool use_gorder = (algo && std::strcmp(algo, "gorder") == 0);
  const uint32_t window = wstr ? static_cast<uint32_t>(std::atoi(wstr)) : 64;

  std::string report;
  bool ok = false;

  if (use_gorder) {
    ok = amio::runtime::reorder_index_gorder(input, output, window, &report);
  } else {
    ok = amio::runtime::reorder_index_bfs(input, output, &report);
  }

  if (!ok) {
    std::fprintf(stderr, "reorder_index: 失败\n");
    return 2;
  }

  if (!report.empty()) {
    std::fprintf(stderr, "── 局部性指标 ──\n%s", report.c_str());
  }
  std::fprintf(stderr, "reorder_index: 成功 → %s\n", output);
  return 0;
}
