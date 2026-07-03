# Native Linux x86_64 build of the C++ edge inference binary.
#   docker build -t yolo-edge-x86 -f cpp/Dockerfile cpp
#   docker run --rm -v $(pwd):/w yolo-edge-x86 /w/exports/best.onnx /w/sample.jpg --bench 200
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake curl ca-certificates libopencv-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*
ARG ORT_VERSION=1.23.2
RUN curl -fL -o /tmp/ort.tgz \
      https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz && \
    mkdir -p /opt/ort && tar -xzf /tmp/ort.tgz -C /opt/ort --strip-components=1 && rm /tmp/ort.tgz
WORKDIR /src
COPY . /src
RUN cmake -S . -B build -DONNXRUNTIME_ROOT_DIR=/opt/ort && cmake --build build -j"$(nproc)"
ENTRYPOINT ["/src/build/yolo_edge"]
