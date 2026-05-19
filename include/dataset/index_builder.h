#pragma once

#include <cstdint>
#include <string>

namespace amio::dataset {

/// 从向量文件流式构建磁盘 HNSW 索引（bvec/fvecs，mmap 按批读，峰值内存可控）。
bool build_disk_index_streaming(const std::string &vector_path, const std::string &output,
                              uint32_t m, uint32_t ef_construction, uint64_t max_vectors,
                              uint64_t batch_vectors);

} // namespace amio::dataset
