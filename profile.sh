# ncu --set "full" --nvtx --nvtx-include "flashmask/" -o bwd_128k_global_sliding_window -f --import-source yes ../flashmask_env/bin/python profile_flashmask.py

regex_filter="(.*flashmask.*|.*device_kernel.*)"
echo "[Warn] Note that regex is set for filtering kernel names, this could result in missing items. Be careful."
echo "[Warn] regex applied is '$regex_filter'"
ncu --set "full" --nvtx --nvtx-include "flashmask/" --kernel-name=regex:$regex_filter  -o fwd_32k_full -f --import-source yes ../flashmask_env/bin/python profile_flashmask.py
