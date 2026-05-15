FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
  cmake g++ wget \
  liburing-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .

RUN mkdir -p data && \
  wget -q http://corpus-texmex.irisa.fr/fvecs/sift.tar.gz -O data/sift.tar.gz && \
  tar -xzf data/sift.tar.gz -C data/ && \
  rm data/sift.tar.gz

RUN cmake -B build -DAMIO_ENABLE_URING=ON && cmake --build build -j$(nproc)

CMD ["./build/run_tests"]