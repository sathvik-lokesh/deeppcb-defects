PY := .venv/bin/python -E -s   # -E: ignore PYTHONPATH (ROS on it breaks imports)
.PHONY: download prepare selftest selftest_eval start stop status
download: ; git clone --depth 1 https://github.com/tangsanli5201/DeepPCB.git data/raw/DeepPCB && git -C data/raw/DeepPCB checkout 08e98c4db5922613fb97176eb3d6497d48260cb1
prepare:  ; $(PY) prepare.py
selftest: ; $(PY) selftest.py
selftest_eval: ; $(PY) selftest_eval.py
# Whole queue in the background; survives closing the terminal. `make start` again resumes.
start:  ; mkdir -p runs && JOBS="$(JOBS)" setsid nohup ./queue.sh >> runs/queue.out 2>&1 & echo started
stop:   ; pkill -TERM -f "^bash ./queue.sh" ; pkill -TERM -f "^.venv/bin/python -E -s -u (train|evaluate).py" ; echo stopped
status: ; ls runs/*/DONE 2>/dev/null; tr '\r' '\n' < runs/queue.out | grep -a -o "[0-9]*/60 .\{0,40\}" | tail -1; tail -c 300 runs/queue.out | grep -a -o "runs/[A-Z0-9_a-z]*" | tail -1; ls runs/eval/*.json 2>/dev/null | wc -l; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
