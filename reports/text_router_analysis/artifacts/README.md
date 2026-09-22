# 实验产物索引

[`artifact_index.csv`](artifact_index.csv) 提供实验文件名、条件、已记录的 SHA-256 和大小，供对应模型与预测产物。

当前索引覆盖 15 个 prediction 条目（3 个 seed × 5 个推理条件）和 9 个唯一 adapter checkpoint 条目（3 个 seed × 3 个训练 arm）。prediction 行的 `condition` 为推理条件：`native-random`、`native-fixed`、`true-new17`、`zero` 或 `wrong-new`。checkpoint 行的 `condition` 为训练 arm：`deterministic-random-routing`、`fixed-balanced-routing` 或 `true-text`；同一 checkpoint 可被该 seed 下的多个推理条件复用。

所有记录的 `provenance_scope` 都是 `historical_manifest`。prediction 的 `size_bytes` 来自历史清单；checkpoint 的大小在所查源元数据中未记录，因此留空。SHA-256 和大小是已记录值，本轮没有读取大文件 payload、执行当前文件重验或产生下载/传输承诺。
