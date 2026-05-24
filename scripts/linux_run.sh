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
GIST_URL="${GIST_URL:-http://corpus-texmex.irisa.fr/fvecs/gist.tar.gz}"

# 可用 CC/CXX 覆盖（例如 CC=clang CXX=clang++）
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"

# ── 数据集 profile（可用 AMIO_DATASET_PROFILE 或子命令参数覆盖）──────────────
# sift-sm   : 128 维，5k base  / 20 query   — 冒烟 / eval-quick 默认量级
# sift-md   : 128 维，100k base / 200 query — compare 默认
# sift-lg   : 128 维，1M base  / 500 query  — 大规模（需完整 SIFT）
# gist-sm   : 960 维，10k base / 50 query   — 高维 v2 冒烟
# gist-md   : 960 维，100k base / 200 query — 高维 v2 标准
# tiny      : 128 维，仓库内 tiny_learn_demo.fvecs（无 GT，recompute）
# custom    : 完全由 AMIO_BASE_PATH / AMIO_QUERY_PATH / AMIO_GT_PATH 指定
AMIO_DATASET_PROFILE="${AMIO_DATASET_PROFILE:-}"

# 路径/规模可单独覆盖（优先级高于 profile）
AMIO_BASE_PATH="${AMIO_BASE_PATH:-}"
AMIO_QUERY_PATH="${AMIO_QUERY_PATH:-}"
AMIO_GT_PATH="${AMIO_GT_PATH:-}"
AMIO_INDEX_PATH="${AMIO_INDEX_PATH:-}"
AMIO_BASE_LIMIT="${AMIO_BASE_LIMIT:-}"
AMIO_QUERY_LIMIT="${AMIO_QUERY_LIMIT:-}"
AMIO_EF_SEARCH="${AMIO_EF_SEARCH:-}"
AMIO_K="${AMIO_K:-}"

# compare 兼容旧变量名
CMP_INDEX="${CMP_INDEX:-}"
CMP_BASE_LIMIT="${CMP_BASE_LIMIT:-}"
CMP_QUERY_LIMIT="${CMP_QUERY_LIMIT:-}"

die() { echo "错误: $*" >&2; exit 1; }

have_cmd() { command -v "$1" >/dev/null 2>&1; }

# 解析后生效的数据集变量（由 resolve_dataset 填充）
DS_PROFILE=""
DS_DIM=""
DS_BASE=""
DS_QUERY=""
DS_GT=""
DS_INDEX=""
DS_BASE_LIMIT=""
DS_QUERY_LIMIT=""
DS_EF_SEARCH=""
DS_K=""
DS_HAS_GT="1"
DS_RAM_GB_HINT=""

resolve_dataset() {
  local profile="${1:-${AMIO_DATASET_PROFILE:-sift-md}}"

  DS_PROFILE="$profile"
  DS_DIM="128"
  DS_HAS_GT="1"
  DS_RAM_GB_HINT=""
  DS_K="10"
  DS_EF_SEARCH="128"

  case "$profile" in
    sift-sm)
      DS_DIM="128"
      DS_BASE="data/sift/sift_base.fvecs"
      DS_QUERY="data/sift/sift_query.fvecs"
      DS_GT="data/sift/sift_groundtruth.ivecs"
      DS_INDEX="data/sift_sm.index"
      DS_BASE_LIMIT="5000"
      DS_QUERY_LIMIT="20"
      DS_EF_SEARCH="64"
      ;;
    sift-md)
      DS_DIM="128"
      DS_BASE="data/sift/sift_base.fvecs"
      DS_QUERY="data/sift/sift_query.fvecs"
      DS_GT="data/sift/sift_groundtruth.ivecs"
      DS_INDEX="data/sift_compare.index"
      DS_BASE_LIMIT="100000"
      DS_QUERY_LIMIT="200"
      DS_EF_SEARCH="128"
      ;;
    sift-lg)
      DS_DIM="128"
      DS_BASE="data/sift/sift_base.fvecs"
      DS_QUERY="data/sift/sift_query.fvecs"
      DS_GT="data/sift/sift_groundtruth.ivecs"
      DS_INDEX="data/sift_lg.index"
      DS_BASE_LIMIT="1000000"
      DS_QUERY_LIMIT="500"
      DS_EF_SEARCH="128"
      DS_RAM_GB_HINT="32"
      ;;
    gist-sm)
      DS_DIM="960"
      DS_BASE="data/gist/gist_base.fvecs"
      DS_QUERY="data/gist/gist_query.fvecs"
      DS_GT="data/gist/gist_groundtruth.ivecs"
      DS_INDEX="data/gist_sm.index"
      DS_BASE_LIMIT="10000"
      DS_QUERY_LIMIT="50"
      DS_EF_SEARCH="128"
      DS_RAM_GB_HINT="16"
      ;;
    gist-md)
      DS_DIM="960"
      DS_BASE="data/gist/gist_base.fvecs"
      DS_QUERY="data/gist/gist_query.fvecs"
      DS_GT="data/gist/gist_groundtruth.ivecs"
      DS_INDEX="data/gist_md.index"
      DS_BASE_LIMIT="100000"
      DS_QUERY_LIMIT="200"
      DS_EF_SEARCH="192"
      DS_RAM_GB_HINT="64"
      ;;
    tiny)
      DS_DIM="128"
      DS_BASE="data/tiny_learn_demo.fvecs"
      DS_QUERY="data/tiny_learn_demo.fvecs"
      DS_GT=""
      DS_INDEX="data/tiny_demo.index"
      DS_BASE_LIMIT="500"
      DS_QUERY_LIMIT="20"
      DS_EF_SEARCH="64"
      DS_HAS_GT="0"
      ;;
    custom)
      DS_DIM="${AMIO_DIM:-auto}"
      DS_BASE="${AMIO_BASE_PATH:-}"
      DS_QUERY="${AMIO_QUERY_PATH:-}"
      DS_GT="${AMIO_GT_PATH:-}"
      DS_INDEX="${AMIO_INDEX_PATH:-data/custom.index}"
      DS_BASE_LIMIT="${AMIO_BASE_LIMIT:-100000}"
      DS_QUERY_LIMIT="${AMIO_QUERY_LIMIT:-200}"
      DS_EF_SEARCH="${AMIO_EF_SEARCH:-128}"
      [[ -z "$DS_BASE" ]] && die "custom profile 需设置 AMIO_BASE_PATH"
      [[ -z "$DS_QUERY" ]] && DS_QUERY="$DS_BASE"
      if [[ -z "$DS_GT" ]]; then
        DS_HAS_GT="0"
      fi
      ;;
    *)
      die "未知数据集 profile: $profile（可用: sift-sm sift-md sift-lg gist-sm gist-md tiny custom）"
      ;;
  esac

  # 环境变量覆盖（兼容 CMP_* / AMIO_*）
  [[ -n "$AMIO_BASE_PATH" ]] && DS_BASE="$AMIO_BASE_PATH"
  [[ -n "$AMIO_QUERY_PATH" ]] && DS_QUERY="$AMIO_QUERY_PATH"
  [[ -n "$AMIO_GT_PATH" ]] && DS_GT="$AMIO_GT_PATH" && DS_HAS_GT="1"
  [[ -n "$AMIO_INDEX_PATH" ]] && DS_INDEX="$AMIO_INDEX_PATH"
  [[ -n "$CMP_INDEX" ]] && DS_INDEX="$CMP_INDEX"
  [[ -n "$AMIO_BASE_LIMIT" ]] && DS_BASE_LIMIT="$AMIO_BASE_LIMIT"
  [[ -n "$CMP_BASE_LIMIT" ]] && DS_BASE_LIMIT="$CMP_BASE_LIMIT"
  [[ -n "$AMIO_QUERY_LIMIT" ]] && DS_QUERY_LIMIT="$AMIO_QUERY_LIMIT"
  [[ -n "$CMP_QUERY_LIMIT" ]] && DS_QUERY_LIMIT="$CMP_QUERY_LIMIT"
  [[ -n "$AMIO_EF_SEARCH" ]] && DS_EF_SEARCH="$AMIO_EF_SEARCH"
  [[ -n "$AMIO_K" ]] && DS_K="$AMIO_K"

  # trim ef (tiny profile typo guard)
  DS_EF_SEARCH="$(echo "$DS_EF_SEARCH" | tr -d '[:space:]')"
}

print_dataset_config() {
  echo "── 数据集 profile: ${DS_PROFILE} (dim≈${DS_DIM}) ──"
  echo "  base:        $DS_BASE  (limit=${DS_BASE_LIMIT})"
  echo "  query:       $DS_QUERY  (limit=${DS_QUERY_LIMIT})"
  echo "  gt:          ${DS_GT:-<recompute>}"
  echo "  index:       $DS_INDEX"
  echo "  k=${DS_K} ef_search=${DS_EF_SEARCH}"
  [[ -n "$DS_RAM_GB_HINT" ]] && echo "  建议 RAM 预算: ${DS_RAM_GB_HINT} GB (export AMIO_RAM_BUDGET_GB=${DS_RAM_GB_HINT})"
}

require_dataset_files() {
  [[ -f "$DS_BASE" ]] || die "缺少 base 向量文件: $DS_BASE"
  [[ -f "$DS_QUERY" ]] || die "缺少 query 向量文件: $DS_QUERY"
  if [[ "$DS_HAS_GT" == "1" ]]; then
    [[ -f "$DS_GT" ]] || die "缺少 groundtruth: $DS_GT"
  fi
}

common_ram_args() {
  local -n _out=$1
  _out=()
  local ram_gb="${AMIO_RAM_BUDGET_GB:-}"
  if [[ -z "$ram_gb" && -n "$DS_RAM_GB_HINT" ]]; then
    ram_gb="$DS_RAM_GB_HINT"
  fi
  if [[ -n "$ram_gb" ]]; then
    _out+=(--ram-budget-gb "$ram_gb")
  fi
}

common_profile_args() {
  local -n _out=$1
  _out=()
  if [[ -n "${AMIO_MEMORY_PROFILE:-}" ]]; then
    _out+=(--memory-profile "$AMIO_MEMORY_PROFILE")
  fi
}

gt_eval_args() {
  local -n _gt=$1
  _gt=()
  if [[ "$DS_HAS_GT" == "1" ]]; then
    _gt=(--gt "$DS_GT" --recompute-gt 0)
  else
    _gt=(--recompute-gt 1)
  fi
}

build_eval_cmd() {
  local -n _cmd=$1
  local policy_mode="${2:-builtin}"
  local rebuild="${3:-1}"
  local log_path="${4:-logs/eval_${DS_PROFILE}.log}"
  local metrics_path="${5:-logs/eval_${DS_PROFILE}_per_query.csv}"

  local ram_args=()
  local prof_args=()
  common_ram_args ram_args
  common_profile_args prof_args

  _cmd=(
    "./$BUILD_DIR/eval_disk"
    --base "$DS_BASE"
    --query "$DS_QUERY"
    --index "$DS_INDEX"
    --base-limit "$DS_BASE_LIMIT"
    --query-limit "$DS_QUERY_LIMIT"
    --k "$DS_K"
    --ef-search "$DS_EF_SEARCH"
    --policy-mode "$policy_mode"
    --rebuild "$rebuild"
    "${ram_args[@]}"
    "${prof_args[@]}"
    --log "$log_path"
    --metrics-log "$metrics_path"
  )

  if [[ "$DS_HAS_GT" == "1" ]]; then
    _cmd+=(--gt "$DS_GT" --recompute-gt 0)
  else
    _cmd+=(--recompute-gt 1)
  fi

  if [[ "$policy_mode" == "learned" ]]; then
    [[ -f data/agent_io_policy.json ]] || die "缺少 data/agent_io_policy.json，请先: ./scripts/linux_run.sh policy"
    _cmd+=(--agent-policy data/agent_io_policy.json)
  fi
}

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
  SIFT_URL / GIST_URL  TexMex 数据集下载地址
  CC / CXX             C/C++ 编译器 (默认: gcc / g++)

数据集 profile（AMIO_DATASET_PROFILE 或子命令第二参数）:
  sift-sm              128 维  5k base  /  20 query  / ef=64   冒烟
  sift-md              128 维 100k base / 200 query  / ef=128  标准对比（默认）
  sift-lg              128 维  1M base  / 500 query  / ef=128  大规模
  gist-sm              960 维  10k base /  50 query  / ef=128  高维 v2 冒烟
  gist-md              960 维 100k base / 200 query  / ef=192  高维 v2 标准
  tiny                 仓库 demo fvecs（无 GT，自动 recompute）
  custom               由 AMIO_BASE_PATH / AMIO_QUERY_PATH / AMIO_GT_PATH 指定

路径/规模覆盖（优先级高于 profile）:
  AMIO_BASE_PATH       base 向量文件
  AMIO_QUERY_PATH      query 向量文件
  AMIO_GT_PATH         groundtruth ivecs
  AMIO_INDEX_PATH      索引输出路径
  AMIO_BASE_LIMIT      base 条数上限
  AMIO_QUERY_LIMIT     query 条数上限
  AMIO_EF_SEARCH       检索 ef
  AMIO_K               Top-K
  AMIO_RAM_BUDGET_GB   内存划分预算 GB
  AMIO_MEMORY_PROFILE  强制 M0–M4 模式
  CMP_INDEX / CMP_BASE_LIMIT / CMP_QUERY_LIMIT  （compare 兼容旧名）

子命令:
  help              显示本说明
  datasets          列出所有 dataset profile 及默认参数
  deps              安装 cmake、g++、liburing、wget、python3
  fetch-sift        下载 SIFT → data/sift/
  fetch-gist        下载 GIST 960 → data/gist/
  fetch-all         fetch-sift + fetch-gist
  policy            生成 data/agent_io_policy.json
  build             cmake 配置并编译
  test              run_tests
  bench             bench_search + bench_write
  probe-dataset     探测向量文件类型/维度/体量（dataset_loader）
  build-index       建索引；默认 sift-md profile 的路径
  reorder           BFS/Gorder 图重排（见 reorder_index --help）
  eval              eval_disk，支持: eval [profile] [-- 额外参数]
  eval-quick        等价 eval sift-sm（可传 profile）
  eval-suite        依次跑 sift-sm / sift-md / gist-sm（若数据存在）
  compare           builtin vs learned；用法: compare [profile]
  compare-reorder   原始 vs 重排索引对比；用法: compare-reorder [profile]
  smoke             build + test + bench
  all               deps + fetch-sift + policy + build + test + bench + eval-quick

向量与维度:
  dim<=128 → 索引 v1（内联向量）；dim>128（GIST 960）→ v2（外置向量区）
  .fvecs float32 / .bvecs uint8 / .ivecs groundtruth

示例:
  ./scripts/linux_run.sh datasets
  ./scripts/linux_run.sh fetch-sift && ./scripts/linux_run.sh compare sift-md
  ./scripts/linux_run.sh eval gist-sm
  AMIO_RAM_BUDGET_GB=100 ./scripts/linux_run.sh eval-suite
  ./scripts/linux_run.sh build-index -- --input data/gist/gist_base.fvecs \
    --output data/gist_sm.index --ram-budget-gb 64 --max-vectors 10000
  AMIO_DATASET_PROFILE=custom AMIO_BASE_PATH=/data/big.bvecs \
    AMIO_QUERY_PATH=/data/q.bvecs AMIO_BASE_LIMIT=5000000 \
    ./scripts/linux_run.sh eval custom
EOF
}

run_list_datasets() {
  cat <<'EOF'
可用 dataset profile:

  profile    dim    base_limit  query_limit  ef_search  数据目录
  ─────────────────────────────────────────────────────────────────
  sift-sm    128         5000          20         64  data/sift/
  sift-md    128       100000         200        128  data/sift/     ← compare 默认
  sift-lg    128      1000000         500        128  data/sift/
  gist-sm    960        10000          50        128  data/gist/     ← 需 fetch-gist
  gist-md    960       100000         200        192  data/gist/
  tiny       128          500          20         64  data/tiny_learn_demo.fvecs
  custom     auto         自定义        自定义      自定义  AMIO_* 环境变量

用法:
  ./scripts/linux_run.sh eval sift-sm
  ./scripts/linux_run.sh compare gist-md
  export AMIO_DATASET_PROFILE=sift-lg && ./scripts/linux_run.sh compare
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
      die "未识别的发行版 ID=$id。请手动安装: cmake、C++20 编译器、liburing 开发包、wget。"
      ;;
  esac
}

fetch_archive() {
  local url="$1"
  local archive="$2"
  if have_cmd wget; then
    wget -q "$url" -O "$archive"
  elif have_cmd curl; then
    curl -fsSL "$url" -o "$archive"
  else
    die "请安装 wget 或 curl"
  fi
}

run_fetch_sift() {
  mkdir -p data
  if [[ -f data/sift/sift_base.fvecs ]]; then
    echo "已存在 data/sift/sift_base.fvecs，跳过下载。"
    return 0
  fi
  local archive="data/sift.tar.gz"
  echo "下载 SIFT: $SIFT_URL"
  fetch_archive "$SIFT_URL" "$archive"
  tar -xzf "$archive" -C data/
  rm -f "$archive"
  echo "SIFT 已解压到 data/sift/"
}

run_fetch_gist() {
  mkdir -p data
  if [[ -f data/gist/gist_base.fvecs ]]; then
    echo "已存在 data/gist/gist_base.fvecs，跳过下载。"
    return 0
  fi
  local archive="data/gist.tar.gz"
  echo "下载 GIST 960: $GIST_URL"
  fetch_archive "$GIST_URL" "$archive"
  tar -xzf "$archive" -C data/
  rm -f "$archive"
  echo "GIST 已解压到 data/gist/ (dim=960, 索引 v2)"
}

run_fetch_all() {
  run_fetch_sift
  run_fetch_gist
}

run_policy() {
  local py="python3"
  have_cmd "$py" || die "未找到 python3"
  mkdir -p data
  if [[ -f data/agent_io_policy.json ]]; then
    echo "已存在 data/agent_io_policy.json，跳过。"
    return 0
  fi
  if [[ -f data/sift/sift_learn.fvecs ]]; then
    echo "使用 learn 集: data/sift/sift_learn.fvecs"
    "$py" learning/train_agent_policy.py --profile cursor --out data/agent_io_policy.json \
      --data-source learn-fvecs --learn-fvecs data/sift/sift_learn.fvecs
  elif [[ -f data/sift/sift_learn.bvecs ]]; then
    echo "使用 learn bvec: data/sift/sift_learn.bvecs"
    "$py" learning/train_agent_policy.py --profile cursor --out data/agent_io_policy.json \
      --data-source learn-bvecs --learn-fvecs data/sift/sift_learn.bvecs
  else
    echo "未找到 learn 集，使用合成轨迹（可先 fetch-sift）"
    "$py" learning/train_agent_policy.py --profile cursor --out data/agent_io_policy.json \
      --data-source synthetic
  fi
}

run_probe_dataset() {
  [[ -x "$BUILD_DIR/dataset_loader" ]] || die "请先构建"
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
    resolve_dataset "${AMIO_DATASET_PROFILE:-sift-md}"
    print_dataset_config
    require_dataset_files
    "./$BUILD_DIR/build_index" --input "$DS_BASE" --output "$DS_INDEX" \
      --max-vectors "$DS_BASE_LIMIT" "${extra[@]}"
    return 0
  fi
  "./$BUILD_DIR/build_index" "$@" "${extra[@]}"
}

run_build() {
  have_cmd cmake || die "未找到 cmake，请先: ./scripts/linux_run.sh deps"
  have_cmd "$CXX" || die "未找到 C++ 编译器: $CXX"
  cmake -S "$ROOT" -B "$BUILD_DIR" -DAMIO_ENABLE_URING="$AMIO_ENABLE_URING" \
    -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX"
  cmake --build "$BUILD_DIR" -j"$JOBS"
  echo "构建完成: $ROOT/$BUILD_DIR (CC=$CC CXX=$CXX)"
}

run_test() {
  [[ -x "$BUILD_DIR/run_tests" ]] || die "请先构建"
  "./$BUILD_DIR/run_tests"
}

run_bench() {
  [[ -x "$BUILD_DIR/bench_search" ]] || die "请先构建"
  "./$BUILD_DIR/bench_search"
  "./$BUILD_DIR/bench_write"
}

run_eval_with_profile() {
  local profile="${1:-sift-sm}"
  shift || true
  [[ -x "$BUILD_DIR/eval_disk" ]] || die "请先构建"
  mkdir -p logs
  resolve_dataset "$profile"
  print_dataset_config
  require_dataset_files

  local policy="builtin"
  if [[ -f data/agent_io_policy.json ]]; then
    policy="learned"
  fi
  local cmd=()
  build_eval_cmd cmd "$policy" "1" "logs/eval_${DS_PROFILE}_summary.log" \
    "logs/eval_${DS_PROFILE}_per_query.csv"
  echo "运行 eval_disk (profile=$DS_PROFILE, policy=$policy) ..."
  "./$BUILD_DIR/eval_disk" "${cmd[@]}" "$@"
}

run_eval_quick() {
  local profile="${1:-sift-sm}"
  if [[ $# -gt 0 && "$1" != --* ]]; then
    shift
  fi
  run_eval_with_profile "$profile" "$@"
}

run_eval_suite() {
  [[ -x "$BUILD_DIR/eval_disk" ]] || die "请先构建"
  mkdir -p logs
  local profiles=(sift-sm sift-md gist-sm)
  local p
  for p in "${profiles[@]}"; do
    resolve_dataset "$p"
    if [[ ! -f "$DS_BASE" ]]; then
      echo "跳过 $p: 缺少 $DS_BASE"
      continue
    fi
    echo ""
    echo "======== eval-suite: $p ========"
    run_eval_with_profile "$p" || echo "警告: $p 评测失败"
  done
  echo ""
  echo "eval-suite 完成，日志: logs/eval_*_summary.log"
}

run_compare() {
  local profile="${1:-sift-md}"
  [[ -x "$BUILD_DIR/eval_disk" ]] || die "请先构建"
  resolve_dataset "$profile"
  print_dataset_config
  require_dataset_files
  [[ -f data/agent_io_policy.json ]] || die "缺少策略 JSON，请先: ./scripts/linux_run.sh policy"
  mkdir -p logs

  local ram_args=()
  local prof_args=()
  local gt_args=()
  common_ram_args ram_args
  common_profile_args prof_args
  gt_eval_args gt_args

  echo "=== 1/2 builtin ==="
  "./$BUILD_DIR/eval_disk" \
    --base "$DS_BASE" --query "$DS_QUERY" --index "$DS_INDEX" \
    --base-limit "$DS_BASE_LIMIT" --query-limit "$DS_QUERY_LIMIT" \
    --k "$DS_K" --ef-search "$DS_EF_SEARCH" \
    --policy-mode builtin \
    "${ram_args[@]}" "${prof_args[@]}" \
    --rebuild 1 \
    "${gt_args[@]}" \
    --log "logs/compare_${DS_PROFILE}_builtin_summary.log" \
    --metrics-log "logs/compare_${DS_PROFILE}_builtin_per_query.csv"

  echo "=== 2/2 learned ==="
  "./$BUILD_DIR/eval_disk" \
    --base "$DS_BASE" --query "$DS_QUERY" --index "$DS_INDEX" \
    --base-limit "$DS_BASE_LIMIT" --query-limit "$DS_QUERY_LIMIT" \
    --k "$DS_K" --ef-search "$DS_EF_SEARCH" \
    --policy-mode learned --agent-policy data/agent_io_policy.json \
    "${ram_args[@]}" "${prof_args[@]}" \
    --rebuild 0 \
    "${gt_args[@]}" \
    --log "logs/compare_${DS_PROFILE}_learned_summary.log" \
    --metrics-log "logs/compare_${DS_PROFILE}_learned_per_query.csv"

  echo "对比完成: logs/compare_${DS_PROFILE}_*_summary.log"
}

run_reorder() {
  [[ -x "$BUILD_DIR/reorder_index" ]] || die "请先构建"
  local src="${1:-data/sift_compare.index}"
  local dst="${2:-data/sift_base_reordered.index}"
  if [[ $# -ge 2 && "$1" == "--input" ]]; then
    src="$2"
    dst="${4:-${src%.index}_reordered.index}"
    [[ $# -ge 4 && "$3" == "--output" ]] && dst="$4"
  fi
  echo "图重排: $src → $dst"
  "./$BUILD_DIR/reorder_index" --input "$src" --output "$dst"
  echo "完成: $dst"
}

run_compare_reorder() {
  local profile="${1:-sift-md}"
  [[ -x "$BUILD_DIR/eval_disk" ]] || die "请先构建"
  [[ -x "$BUILD_DIR/reorder_index" ]] || die "请先构建"
  resolve_dataset "$profile"
  print_dataset_config
  require_dataset_files
  mkdir -p logs

  local idx_orig="$DS_INDEX"
  local idx_reordered="${DS_INDEX%.index}_reordered.index"
  local pm="builtin"
  local pm_extra=()
  if [[ -f data/agent_io_policy.json ]]; then
    pm="learned"
    pm_extra=(--agent-policy data/agent_io_policy.json)
  fi

  local ram_args=()
  local gt_args=()
  common_ram_args ram_args
  gt_eval_args gt_args

  echo "=== 1/3 构建/评测原始索引 ==="
  "./$BUILD_DIR/eval_disk" \
    --base "$DS_BASE" --query "$DS_QUERY" --index "$idx_orig" \
    --base-limit "$DS_BASE_LIMIT" --query-limit "$DS_QUERY_LIMIT" \
    --k "$DS_K" --ef-search "$DS_EF_SEARCH" \
    --policy-mode "$pm" "${pm_extra[@]}" "${ram_args[@]}" \
    --rebuild 1 \
    "${gt_args[@]}" \
    --log "logs/compare_${DS_PROFILE}_orig_summary.log" \
    --metrics-log "logs/compare_${DS_PROFILE}_orig_per_query.csv"

  echo "=== 2/3 BFS 重排 ==="
  "./$BUILD_DIR/reorder_index" --input "$idx_orig" --output "$idx_reordered"

  echo "=== 3/3 评测重排索引 ==="
  "./$BUILD_DIR/eval_disk" \
    --base "$DS_BASE" --query "$DS_QUERY" --index "$idx_reordered" \
    --base-limit "$DS_BASE_LIMIT" --query-limit "$DS_QUERY_LIMIT" \
    --k "$DS_K" --ef-search "$DS_EF_SEARCH" \
    --policy-mode "$pm" "${pm_extra[@]}" "${ram_args[@]}" \
    --rebuild 0 \
    "${gt_args[@]}" \
    --log "logs/compare_${DS_PROFILE}_reorder_summary.log" \
    --metrics-log "logs/compare_${DS_PROFILE}_reorder_per_query.csv"

  echo "对比完成: logs/compare_${DS_PROFILE}_orig_* vs logs/compare_${DS_PROFILE}_reorder_*"
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
  run_eval_quick sift-sm
}

cmd="${1:-help}"
shift || true

case "$cmd" in
  help|-h|--help) print_help ;;
  datasets|list-datasets) run_list_datasets ;;
  deps) run_deps ;;
  fetch-sift) run_fetch_sift ;;
  fetch-gist) run_fetch_gist ;;
  fetch-all) run_fetch_all ;;
  policy) run_policy ;;
  build) run_build ;;
  test) run_test ;;
  bench) run_bench ;;
  probe-dataset) run_probe_dataset "$@" ;;
  build-index) run_build_index "$@" ;;
  eval)
    profile="${1:-sift-md}"
    if [[ $# -gt 0 && "$1" != --* ]]; then
      shift
    fi
    if [[ $# -eq 0 ]]; then
      run_eval_with_profile "$profile"
    else
      [[ -x "$BUILD_DIR/eval_disk" ]] || die "请先构建"
      mkdir -p logs
      "./$BUILD_DIR/eval_disk" "$@"
    fi
    ;;
  eval-quick)
    profile="${1:-sift-sm}"
    [[ $# -gt 0 && "$1" != --* ]] && shift || true
    run_eval_quick "$profile" "$@"
    ;;
  eval-suite) run_eval_suite ;;
  compare)
    profile="${1:-sift-md}"
    run_compare "$profile"
    ;;
  reorder) run_reorder "$@" ;;
  compare-reorder)
    profile="${1:-sift-md}"
    run_compare_reorder "$profile"
    ;;
  smoke) run_smoke ;;
  all) run_all ;;
  *) die "未知子命令: $cmd（使用 help 或 datasets 查看用法）" ;;
esac
