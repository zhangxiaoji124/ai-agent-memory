#!/usr/bin/env bash
# Linux 实机测试：多发行版依赖、工具链、构建、单测、双策略对比评测与逐查询指标日志。
# 用法: ./scripts/linux_run.sh help
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="${BUILD_DIR:-build}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
AMIO_ENABLE_URING="${AMIO_ENABLE_URING:-ON}"
SIFT_URL="${SIFT_URL:-http://corpus-texmex.irisa.fr/fvecs/sift.tar.gz}"

# 可用 CC/CXX 覆盖（例如 CC=clang CXX=clang++）
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"

die() { echo "错误: $*" >&2; exit 1; }

have_cmd() { command -v "$1" >/dev/null 2>&1; }

detect_distro() {
  if [[ -f /etc/os-release ]]; then
    # shellcheck source=/dev/null
    . /etc/os-release
    echo "${ID:-unknown}"
  else
    echo "unknown"
  fi
}

print_help() {
  cat <<'EOF'
Linux 运行脚本（agent-memory-io-cpp）

环境变量:
  BUILD_DIR            构建目录 (默认: build)
  JOBS                 并行编译线程数 (默认: nproc)
  AMIO_ENABLE_URING    ON/OFF，是否尝试链接 liburing (默认: ON)
  SIFT_URL             SIFT 数据集 tar.gz 地址
  CC / CXX             C/C++ 编译器 (默认: gcc / g++)
  CMP_INDEX            compare 子命令共用的索引路径 (默认: data/sift_compare.index)
  CMP_BASE_LIMIT       compare / eval 的 base 条数上限 (默认: 100000)
  CMP_QUERY_LIMIT      compare / eval 的 query 条数上限 (默认: 200)
  AMIO_RAM_BUDGET_GB   内存划分区分器预算 GB（默认: 物理内存 80% 或见 eval/build 的 --ram-budget-gb）
  AMIO_MEMORY_PROFILE  强制划分模式: HOST_RESIDENT|HYBRID_FLOAT|DISK_FIRST|BVEC_DISK_TIERED|BVEC_ULTRA_500G_100G

子命令:
  help              显示本说明
  deps              自动识别发行版并安装 cmake、C++ 编译器、liburing 开发包、wget（需 sudo）
  fetch-sift        下载并解压 SIFT 到 data/sift/
  policy            生成 data/agent_io_policy.json（优先 learn-fvecs；.bvecs 自动 learn-bvecs）
  build             cmake 配置并编译（使用当前 CC/CXX）
  test              运行 run_tests（含 memory_partition、960 维索引冒烟）
  bench             运行 bench_search 与 bench_write
  probe-dataset     探测向量文件类型/维度/体量并打印内存划分模式（dataset_loader）
  build-index       流式/内存建索引（支持 fvecs/bvecs，dim>128 自动 v2 外置向量区）
  eval              运行 eval_disk，参数原样透传（支持 --ram-budget-gb、--memory-profile）
  eval-quick        小规模 eval_disk（自动内存划分 + learned 策略）
  compare           同一索引上先后跑 builtin 与 learned，生成 logs/compare_* 汇总与逐查询 CSV
  smoke             build + test + bench
  all               deps + fetch-sift + policy + build + test + bench + eval-quick

向量与维度:
  - .fvecs / .bvecs / .ivecs（TexMex）；.bvecs 为 uint8（如 SIFT1B）
  - dim<=128: 索引 v1（向量内联 NodeBlock）；dim>128（如 GIST 960）: 索引 v2（外置向量区）
  - 大文件请用 build-index 或 eval --ram-budget-gb，勿整表载入内存

逐查询指标:
  eval_disk 默认将每条查询的延迟、recall、磁盘块读、预取提交量、缓存命中等写入
  logs/eval_disk_per_query.csv；关闭请传 --metrics-log none

示例:
  CC=clang CXX=clang++ ./scripts/linux_run.sh build
  ./scripts/linux_run.sh fetch-sift && ./scripts/linux_run.sh compare
  ./scripts/linux_run.sh probe-dataset data/sift/sift_base.fvecs
  ./scripts/linux_run.sh build-index -- --input data/sift/sift_base.fvecs --output data/sift.index
  AMIO_RAM_BUDGET_GB=100 ./scripts/linux_run.sh eval-quick
  ./scripts/linux_run.sh eval -- --ram-budget-gb 100 --memory-profile BVEC_ULTRA_500G_100G \
    --policy-mode learned --agent-policy data/agent_io_policy.json
EOF
}

run_apt_deps() {
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    have_cmd sudo || die "需要 root 或 sudo 以安装依赖"
    sudo apt-get update
    sudo apt-get install -y build-essential cmake wget ca-certificates liburing-dev python3
  else
    apt-get update
    apt-get install -y build-essential cmake wget ca-certificates liburing-dev python3
  fi
}

run_dnf_deps() {
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    have_cmd sudo || die "需要 sudo"
    sudo dnf install -y gcc gcc-c++ cmake wget ca-certificates liburing-devel make python3
  else
    dnf install -y gcc gcc-c++ cmake wget ca-certificates liburing-devel make python3
  fi
}

run_zypper_deps() {
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    have_cmd sudo || die "需要 sudo"
    sudo zypper --non-interactive install gcc gcc-c++ cmake wget ca-certificates liburing-devel make python3
  else
    zypper --non-interactive install gcc gcc-c++ cmake wget ca-certificates liburing-devel make python3
  fi
}

run_deps() {
  local id
  id="$(detect_distro)"
  echo "检测到 os-release ID=$id，安装编译依赖..."
  case "$id" in
    ubuntu|debian|linuxmint|pop)
      run_apt_deps
      ;;
    fedora|rhel|centos|rocky|almalinux|ol)
      run_dnf_deps
      ;;
    opensuse-leap|opensuse-tumbleweed|sles)
      run_zypper_deps
      ;;
    *)
      die "未识别的发行版 ID=$id。请手动安装: cmake、C++20 编译器(g++/gcc-c++)、liburing 开发包、wget，然后执行 build。"
      ;;
  esac
}

run_fetch_sift() {
  mkdir -p data
  local archive="data/sift.tar.gz"
  if [[ -f data/sift/sift_base.fvecs ]]; then
    echo "已存在 data/sift/sift_base.fvecs，跳过下载。"
    return 0
  fi
  if have_cmd wget; then
    wget -q "$SIFT_URL" -O "$archive"
  elif have_cmd curl; then
    curl -fsSL "$SIFT_URL" -o "$archive"
  else
    die "请安装 wget 或 curl 以下载数据集"
  fi
  tar -xzf "$archive" -C data/
  rm -f "$archive"
  echo "SIFT 已解压到 data/sift/"
}

run_policy() {
  local py="python3"
  have_cmd "$py" || die "未找到 python3，无法生成策略 JSON"
  mkdir -p data
  if [[ -f data/agent_io_policy.json ]]; then
    echo "已存在 data/agent_io_policy.json，跳过。"
    return 0
  fi
  if [[ -f data/sift/sift_learn.fvecs ]]; then
    echo "使用 learn 集: data/sift/sift_learn.fvecs 训练策略 JSON"
    "$py" learning/train_agent_policy.py --profile cursor --out data/agent_io_policy.json \
      --data-source learn-fvecs --learn-fvecs data/sift/sift_learn.fvecs
  elif [[ -f data/sift/sift_learn.bvecs ]]; then
    echo "使用 learn bvec: data/sift/sift_learn.bvecs"
    "$py" learning/train_agent_policy.py --profile cursor --out data/agent_io_policy.json \
      --data-source learn-bvecs --learn-fvecs data/sift/sift_learn.bvecs
  else
    echo "未找到 learn 集，使用合成轨迹生成策略（可先执行 fetch-sift）"
    "$py" learning/train_agent_policy.py --profile cursor --out data/agent_io_policy.json \
      --data-source synthetic
  fi
}

run_probe_dataset() {
  [[ -x "$BUILD_DIR/dataset_loader" ]] || die "请先构建: ./scripts/linux_run.sh build"
  if [[ $# -lt 1 ]]; then
    die "用法: ./scripts/linux_run.sh probe-dataset <path.fvecs|bvecs|ivecs>"
  fi
  "./$BUILD_DIR/dataset_loader" "$1"
}

run_build_index() {
  [[ -x "$BUILD_DIR/build_index" ]] || die "请先构建"
  mkdir -p data logs
  local ram_gb="${AMIO_RAM_BUDGET_GB:-}"
  local extra=()
  if [[ -n "$ram_gb" ]]; then
    extra+=(--ram-budget-gb "$ram_gb")
  fi
  if [[ $# -eq 0 ]]; then
    [[ -f data/sift/sift_base.fvecs ]] || die "请指定参数或先 fetch-sift"
    "./$BUILD_DIR/build_index" --input data/sift/sift_base.fvecs \
      --output data/sift_base.index "${extra[@]}"
    return 0
  fi
  "./$BUILD_DIR/build_index" "$@" "${extra[@]}"
}

run_build() {
  have_cmd cmake || die "未找到 cmake，请先执行: ./scripts/linux_run.sh deps"
  have_cmd "$CXX" || die "未找到 C++ 编译器: $CXX"
  cmake -S "$ROOT" -B "$BUILD_DIR" -DAMIO_ENABLE_URING="$AMIO_ENABLE_URING" \
    -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX"
  cmake --build "$BUILD_DIR" -j"$JOBS"
  echo "构建完成: $ROOT/$BUILD_DIR (CC=$CC CXX=$CXX)"
}

run_test() {
  [[ -x "$BUILD_DIR/run_tests" ]] || die "请先构建: ./scripts/linux_run.sh build"
  "./$BUILD_DIR/run_tests"
}

run_bench() {
  [[ -x "$BUILD_DIR/bench_search" ]] || die "请先构建"
  "./$BUILD_DIR/bench_search"
  "./$BUILD_DIR/bench_write"
}

run_eval() {
  [[ -x "$BUILD_DIR/eval_disk" ]] || die "请先构建"
  mkdir -p logs
  "./$BUILD_DIR/eval_disk" "$@"
}

run_eval_quick() {
  [[ -x "$BUILD_DIR/eval_disk" ]] || die "请先构建"
  mkdir -p logs
  local pm_args=(--policy-mode builtin)
  if [[ -f data/agent_io_policy.json ]]; then
    pm_args=(--policy-mode learned --agent-policy data/agent_io_policy.json)
  fi
  local ram_args=()
  if [[ -n "${AMIO_RAM_BUDGET_GB:-}" ]]; then
    ram_args+=(--ram-budget-gb "$AMIO_RAM_BUDGET_GB")
  fi
  local prof_args=()
  if [[ -n "${AMIO_MEMORY_PROFILE:-}" ]]; then
    prof_args+=(--memory-profile "$AMIO_MEMORY_PROFILE")
  fi
  "./$BUILD_DIR/eval_disk" "${pm_args[@]}" "${ram_args[@]}" "${prof_args[@]}" \
    --base-limit 5000 \
    --query-limit 20 \
    --k 10 \
    --ef-search 64 \
    --rebuild 1 \
    --recompute-gt 1 \
    --log logs/eval_quick_summary.log \
    --metrics-log logs/eval_quick_per_query.csv
}

run_compare() {
  [[ -x "$BUILD_DIR/eval_disk" ]] || die "请先构建"
  [[ -f data/sift/sift_base.fvecs ]] || die "缺少 SIFT，请先: ./scripts/linux_run.sh fetch-sift"
  [[ -f data/agent_io_policy.json ]] || die "缺少学习策略，请先: ./scripts/linux_run.sh policy"
  mkdir -p logs
  local idx="${CMP_INDEX:-data/sift_compare.index}"
  local bl="${CMP_BASE_LIMIT:-100000}"
  local ql="${CMP_QUERY_LIMIT:-200}"
  local ram_args=()
  if [[ -n "${AMIO_RAM_BUDGET_GB:-}" ]]; then
    ram_args+=(--ram-budget-gb "$AMIO_RAM_BUDGET_GB")
  fi
  echo "=== 1/2 内置策略 (builtin，不加载 JSON) ==="
  "./$BUILD_DIR/eval_disk" \
    --base data/sift/sift_base.fvecs \
    --query data/sift/sift_query.fvecs \
    --gt data/sift/sift_groundtruth.ivecs \
    --index "$idx" \
    --base-limit "$bl" \
    --query-limit "$ql" \
    --policy-mode builtin \
    "${ram_args[@]}" \
    --rebuild 1 \
    --log logs/compare_builtin_summary.log \
    --metrics-log logs/compare_builtin_per_query.csv
  echo "=== 2/2 学习策略 (learned + agent_io_policy.json) ==="
  "./$BUILD_DIR/eval_disk" \
    --base data/sift/sift_base.fvecs \
    --query data/sift/sift_query.fvecs \
    --gt data/sift/sift_groundtruth.ivecs \
    --index "$idx" \
    --base-limit "$bl" \
    --query-limit "$ql" \
    --policy-mode learned \
    --agent-policy data/agent_io_policy.json \
    "${ram_args[@]}" \
    --rebuild 0 \
    --log logs/compare_learned_summary.log \
    --metrics-log logs/compare_learned_per_query.csv
  echo "对比完成: logs/compare_*_summary.log 与 logs/compare_*_per_query.csv"
}

run_smoke() {
  run_build
  run_test
  run_bench
}

run_all() {
  run_deps
  run_fetch_sift
  run_policy
  run_build
  run_test
  run_bench
  run_eval_quick
}

cmd="${1:-help}"
case "$cmd" in
  help|-h|--help) print_help ;;
  deps) run_deps ;;
  fetch-sift) run_fetch_sift ;;
  policy) run_policy ;;
  build) run_build ;;
  test) run_test ;;
  bench) run_bench ;;
  probe-dataset)
    shift
    run_probe_dataset "$@"
    ;;
  build-index)
    shift
    run_build_index "$@"
    ;;
  eval)
    shift
    run_eval "$@"
    ;;
  eval-quick) run_eval_quick ;;
  compare) run_compare ;;
  smoke) run_smoke ;;
  all) run_all ;;
  *) die "未知子命令: $cmd（使用 help 查看用法）" ;;
esac
