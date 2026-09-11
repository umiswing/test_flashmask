#!/bin/bash
TS=$(date +%Y%m%d_%H%M%S)          # 只算一次，多张卡共用

# 三个数组按下标对应一个任务。576/512 是唯一有 kv-shared 反向的形状，把它的
# split / shared 拆到两张卡上并行；其余 D 只有 split 可测，用 sweep 也会被
# benchmark 自己跳过 shared。
GPUS=(1 2 3 4 5)                       # 每个任务用哪张卡
HEADDIMS=(128 192 256 576 576)         # 对应要跑的 D
KV_MODES=(split split split split shared)

FM_VERSION=4

# --vs_sparse_attn
for i in "${!HEADDIMS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" python3 benchmark_flashmask.py \
        --fm_version "$FM_VERSION"                                   \
        --head_dim "${HEADDIMS[$i]}"                                 \
        --kv_mode "${KV_MODES[$i]}"                                  \
        --dedup_static_masks                                         \
        --current_time "$TS" &
done
wait


# FM_VERSION=3

# for i in "${!HEADDIMS[@]}"; do
#     CUDA_VISIBLE_DEVICES="${GPUS[$i]}" python3 benchmark_flashmask.py \
#         --backend "cutedsl"                                          \
#         --fm_version "$FM_VERSION"                                   \
#         --head_dim "${HEADDIMS[$i]}"                                 \
#         --kv_mode "${KV_MODES[$i]}"                                  \
#         --current_time "$TS" &
# done
# wait
