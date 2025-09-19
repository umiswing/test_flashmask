#!/bin/bash

seq_len_base=1024
seq_lens=(128 64 32 16 8 4 2)

regex_filter="(.*flashmask.*|.*device_kernel.*)"
echo "[Warn] Note that regex is set for filtering kernel names, this could result in missing items. Be careful."
echo "[Warn] regex applied is '$regex_filter'"

output_folder="./outputs/"
if [ ! -d $output_folder ]; then
    echo "Creating a folder for ncu batch prof at '$output_folder'"
    mkdir -p $output_folder
fi

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
    output_name="fwd_${length_str}_global_swin_ppt"
    ncu --set "full" --nvtx --nvtx-include "flashmask/" \
        --kernel-name=regex:$regex_filter \
        -o $output_name \
        -f --import-source yes \
        ../hqy_env/bin/python profile_flashmask.py --seqlen_q $seq_len_used --seqlen_k $seq_len_used --warmup_times 50 \
        --config ./configs/run_prof.conf \
        >> profile_log.log
    mv ${output_name}.ncu-rep $output_folder
done

echo "${total_task} items profiled. Batch profile output folder: $output_folder"
