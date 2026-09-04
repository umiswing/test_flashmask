export CUDA_VISIBLE_DEVICES=3
export FLAGS_cudnn_deterministic=1

# # run this if you want to update gt
# python test_md5sum.py

export FLAGS_alloc_fill_value=255
export FLAGS_use_system_allocator=1
export FLAGS_check_cuda_error=1

python3 -m pytest \
    test_md5sum.py \
    -v 2>&1 | tee test_md5.log

# if you update flash attention varlen
# python test_fwd_varlen_md5sum.py
# python test_bwd_varlen_md5sum.py

# python3 -m pytest \
#     test_fwd_varlen_md5sum.py \
#     test_bwd_varlen_md5sum.py \
#     -v 2>&1 | tee test_varlen_md5.log

