# Vendored inference core

Snapshot of `cpp/src/{common,ncnn_backend,stb_impl,ort_backend}.cpp`, `cpp/include/*.hpp` and the stb
headers from [skywalker-lt/yolo-master-edge](https://github.com/skywalker-lt/yolo-master-edge) `main`
(v1.2.0 line), so the Android runtime module builds on its own inside this example. The module's
CMakeLists picks this directory automatically; pass `-DYOLOMASTER_CORE_DIR=<path>` to build against a
checkout of the edge repository instead. Keep this snapshot in sync with that repository, do not edit it here.
