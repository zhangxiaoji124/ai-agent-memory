#include "policy/agent_io_policy.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include "vector_store.h"

namespace amio::policy {

AgentIoPolicy AgentIoPolicy::from_config(const Config &cfg) {
  AgentIoPolicy p;
  p.prefetch_depth_upper = cfg.prefetch_depth;
  p.prefetch_depth_layer0 = cfg.prefetch_depth == 0 ? 0 : cfg.prefetch_depth;
  p.max_neighbor_fanout_layer0 = 32;
  p.use_layer_aware_prefetch = false;
  p.sort_prefetch_by_disk_offset = false;
  p.hot_insert_min_prior_misses = 1;
  // 从内存划分区分器传入的 θ（GoVector 阶段切换比例）
  p.partition_theta = cfg.partition.partition_theta;
  return p;
}

namespace {

void trim_inplace(std::string *s) {
  if (!s)
    return;
  while (!s->empty() && std::isspace(static_cast<unsigned char>(s->front())))
    s->erase(s->begin());
  while (!s->empty() && std::isspace(static_cast<unsigned char>(s->back())))
    s->pop_back();
}

bool extract_json_string(const std::string &json, const char *key,
                         std::string *out) {
  const std::string needle = std::string("\"") + key + "\"";
  size_t pos = json.find(needle);
  if (pos == std::string::npos)
    return false;
  pos = json.find(':', pos + needle.size());
  if (pos == std::string::npos)
    return false;
  ++pos;
  while (pos < json.size() &&
         std::isspace(static_cast<unsigned char>(json[pos])))
    ++pos;
  if (pos >= json.size() || json[pos] != '"')
    return false;
  ++pos;
  const size_t start = pos;
  while (pos < json.size() && json[pos] != '"') {
    if (json[pos] == '\\' && pos + 1 < json.size())
      pos += 2;
    else
      ++pos;
  }
  if (pos >= json.size())
    return false;
  *out = json.substr(start, pos - start);
  return true;
}

bool extract_json_uint(const std::string &json, const char *key, size_t *out) {
  const std::string needle = std::string("\"") + key + "\"";
  size_t pos = json.find(needle);
  if (pos == std::string::npos)
    return false;
  pos = json.find(':', pos + needle.size());
  if (pos == std::string::npos)
    return false;
  ++pos;
  while (pos < json.size() &&
         std::isspace(static_cast<unsigned char>(json[pos])))
    ++pos;
  char *end = nullptr;
  const unsigned long v = std::strtoul(json.c_str() + pos, &end, 10);
  if (end == json.c_str() + pos)
    return false;
  *out = static_cast<size_t>(v);
  return true;
}

bool extract_json_float(const std::string &json, const char *key, float *out) {
  const std::string needle = std::string("\"") + key + "\"";
  size_t pos = json.find(needle);
  if (pos == std::string::npos)
    return false;
  pos = json.find(':', pos + needle.size());
  if (pos == std::string::npos)
    return false;
  ++pos;
  while (pos < json.size() &&
         std::isspace(static_cast<unsigned char>(json[pos])))
    ++pos;
  char *end = nullptr;
  const float v = std::strtof(json.c_str() + pos, &end);
  if (end == json.c_str() + pos)
    return false;
  *out = v;
  return true;
}

bool extract_json_bool(const std::string &json, const char *key, bool *out) {
  const std::string needle = std::string("\"") + key + "\"";
  size_t pos = json.find(needle);
  if (pos == std::string::npos)
    return false;
  pos = json.find(':', pos + needle.size());
  if (pos == std::string::npos)
    return false;
  ++pos;
  while (pos < json.size() &&
         std::isspace(static_cast<unsigned char>(json[pos])))
    ++pos;
  if (json.compare(pos, 4, "true") == 0) {
    *out = true;
    return true;
  }
  if (json.compare(pos, 5, "false") == 0) {
    *out = false;
    return true;
  }
  return false;
}

} // namespace

bool AgentIoPolicy::merge_json_file(const std::string &path, AgentIoPolicy *out,
                                    std::string *error_message) {
  if (!out) {
    if (error_message)
      *error_message = "null output";
    return false;
  }
  std::ifstream ifs(path, std::ios::in | std::ios::binary);
  if (!ifs) {
    if (error_message)
      *error_message = "cannot open " + path;
    return false;
  }
  std::ostringstream oss;
  oss << ifs.rdbuf();
  std::string json = oss.str();

  std::string profile;
  if (extract_json_string(json, "agent_profile", &profile)) {
    trim_inplace(&profile);
    out->agent_profile = profile;
  }

  size_t v = 0;
  if (extract_json_uint(json, "prefetch_depth", &v)) {
    out->prefetch_depth_upper = v;
    out->prefetch_depth_layer0 = v;
  }
  if (extract_json_uint(json, "prefetch_depth_upper", &v))
    out->prefetch_depth_upper = v;
  if (extract_json_uint(json, "prefetch_depth_layer0", &v))
    out->prefetch_depth_layer0 = v;
  if (extract_json_uint(json, "max_neighbor_fanout_layer0", &v))
    out->max_neighbor_fanout_layer0 = v > 32 ? 32 : v;

  bool b = false;
  if (extract_json_bool(json, "use_layer_aware_prefetch", &b))
    out->use_layer_aware_prefetch = b;
  if (extract_json_bool(json, "sort_prefetch_by_disk_offset", &b))
    out->sort_prefetch_by_disk_offset = b;

  if (extract_json_uint(json, "hot_insert_min_prior_misses", &v)) {
    if (v < 1)
      v = 1;
    if (v > 64)
      v = 64;
    out->hot_insert_min_prior_misses = static_cast<uint32_t>(v);
  }

  float fv = 0.0f;
  if (extract_json_float(json, "partition_theta", &fv)) {
    if (fv < 0.0f)
      fv = 0.0f;
    if (fv > 1.0f)
      fv = 1.0f;
    out->partition_theta = fv;
  }

  return true;
}

} // namespace amio::policy
