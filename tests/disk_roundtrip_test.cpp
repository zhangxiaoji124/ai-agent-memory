#include <cstdio>
#include <string>
#include <vector>

#include "index/storage_layout.h"
#include "test.h"

TEST(disk_write_read_roundtrip) {
  const std::string path = "amio_test_roundtrip.index";
  (void)std::remove(path.c_str());

  amio::index::IndexFile f;
  EXPECT_TRUE(f.open_create_trunc(path));

  amio::index::IndexFileHeader h{};
  h.total_nodes = 2;
  h.dim = 32;
  h.entry_point = 0;
  EXPECT_TRUE(f.write_header(h));

  amio::index::NodeBlock n0{};
  n0.node_id = 0;
  n0.vec_dim = 32;
  n0.neighbor_counts[0] = 1;
  n0.neighbors[0][0] = 1;
  for (int i = 0; i < 32; i++)
    n0.vector[i] = static_cast<float>(i);
  EXPECT_TRUE(f.pwrite_node(0, n0));

  amio::index::NodeBlock n1{};
  n1.node_id = 1;
  n1.vec_dim = 32;
  n1.neighbor_counts[0] = 1;
  n1.neighbors[0][0] = 0;
  for (int i = 0; i < 32; i++)
    n1.vector[i] = static_cast<float>(i) * 2.0f;
  EXPECT_TRUE(f.pwrite_node(1, n1));
  f.close();

  amio::index::IndexFile r;
  EXPECT_TRUE(r.open_readonly(path));
  amio::index::IndexFileHeader rh{};
  EXPECT_TRUE(r.read_header(&rh));
  EXPECT_TRUE(rh.total_nodes == 2);
  amio::index::NodeBlock rr{};
  EXPECT_TRUE(r.pread_node(1, &rr));
  EXPECT_TRUE(rr.node_id == 1);

  r.close();
  (void)std::remove(path.c_str());
}
