#!/bin/bash
TS=$(date +%Y%m%d_%H%M%S)          # 只算一次，多张卡共用

GPUS=(1 2 3 4)                     # 每个 D 用哪张卡
HEADDIMS=(128 192 256 576)         # 对应要跑的 D


FM_VERSION=4

# --vs_sparse_attn
for i in "${!HEADDIMS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} python3 benchmark_flashmask.py \
        --fm_version "$FM_VERSION"                                  \
        --head_dim "${HEADDIMS[$i]}"                                \
        --kv_mode "sweep"                                           \
        --dedup_static_masks                                        \
        --current_time "$TS" &
done
wait


# FM_VERSION=3

# for i in "${!HEADDIMS[@]}"; do
#     CUDA_VISIBLE_DEVICES=${GPUS[$i]} python3 benchmark_flashmask.py \
#         --backend "cutedsl"                                         \
#         --fm_version "$FM_VERSION"                                  \
#         --head_dim "${HEADDIMS[$i]}"                                \
#         --current_time "$TS" &
# done
# wait
