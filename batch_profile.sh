#!/bin/bash

seq_len_base=1024
seq_lens=(2 4 8 16 32 64 128)

regex_filter="(.*flashmask.*|.*device_kernel.*)"
echo "[Warn] Note that regex is set for filtering kernel names, this could result in missing items. Be careful."
echo "[Warn] regex applied is '$regex_filter'"

echo "" &> profile_log.log
task_cnt=0
total_task=${#seq_lens[@]}
for seq_len in "${seq_lens[@]}"; do
    task_cnt=$((task_cnt + 1))
    seq_len_used=$((seq_len_base * seq_len))
    length_str="${seq_len}k"
    logging="`date` -- Task ($task_cnt/$total_task) Profiling with sequence length = $length_str ..."
    echo $logging
    echo $logging >> profile_log.log
    ncu --set "full" --nvtx --nvtx-include "flashmask/" \
        --kernel-name=regex:$regex_filter \
        -o fwd_${length_str}_global_swin \
        -f --import-source yes \
        ../flashmask_env/bin/python profile_flashmask.py --seqlen_q $seq_len_used --seqlen_k $seq_len_used >> profile_log.log
done