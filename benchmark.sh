python benchmark_flashmask.py --fm_version 1
sleep 60
python benchmark_flashmask.py --fm_version 3
sleep 60
python benchmark_flexattention.py
sleep 60
python draw.py
python draw.py --baseline "flexattention"
