#!/usr/bin/env bash
set -uo pipefail

# ── Configuration ──
UNITY_BIN=~/Unity/Hub/Editor/6000.4.3f1/Editor/Unity
SRC_PROJECT=/media/skr/storage/ten_bad/UnityProject_game
OUT_DIR=/media/skr/storage/ten_bad/TennisDataset_game_5000
TOTAL_EPISODES=5000
NUM_WORKERS=24
SPLIT=test
BASE_SEED=42

WORKER_DIR=/media/skr/storage/ten_bad/_worker_projects
LOG_DIR=/media/skr/storage/ten_bad/_worker_logs

EPISODES_PER_WORKER=$(( (TOTAL_EPISODES + NUM_WORKERS - 1) / NUM_WORKERS ))

echo "=== Parallel Unity Data Collection ==="
echo "Total episodes:  $TOTAL_EPISODES"
echo "Workers:         $NUM_WORKERS"
echo "Episodes/worker: $EPISODES_PER_WORKER"
echo "Output:          $OUT_DIR"
echo ""

# ── Phase A: Clone projects ──
echo "[Phase A] Cloning project for $NUM_WORKERS workers..."
mkdir -p "$WORKER_DIR" "$LOG_DIR" "$OUT_DIR"

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    dest="$WORKER_DIR/worker_$i"
    if [ -d "$dest" ]; then
        rm -rf "$dest/Temp" "$dest/Library/EditorInstance.json" 2>/dev/null || true
        echo "  worker_$i: reusing existing clone"
    else
        cp -a "$SRC_PROJECT" "$dest"
        rm -rf "$dest/Temp" "$dest/Library/EditorInstance.json" 2>/dev/null || true
        echo "  worker_$i: cloned"
    fi
done
echo "[Phase A] Done."
echo ""

# ── Phase B: Launch workers ──
echo "[Phase B] Launching $NUM_WORKERS Unity instances..."
PIDS=()

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    offset=$((i * EPISODES_PER_WORKER))
    remaining=$((TOTAL_EPISODES - offset))
    eps=$EPISODES_PER_WORKER
    if [ "$remaining" -lt "$eps" ]; then
        eps=$remaining
    fi
    if [ "$eps" -le 0 ]; then
        continue
    fi

    worker_seed=$((BASE_SEED + i * 10000))
    project="$WORKER_DIR/worker_$i"
    logfile="$LOG_DIR/worker_$i.log"

    "$UNITY_BIN" \
        -projectPath "$project" \
        -batchmode -nographics \
        -executeMethod TennisDataCollection.EditorTools.GameDatasetRunner.Run \
        -episodes "$eps" \
        -split "$SPLIT" \
        -outDir "$OUT_DIR" \
        -seed "$worker_seed" \
        -episodeIdOffset "$offset" \
        -logFile "$logfile" &

    PIDS+=($!)
    echo "  worker_$i: PID=$! episodes=$eps offset=$offset seed=$worker_seed"
done
echo "[Phase B] All workers launched."
echo ""

# ── Phase C: Monitor progress ──
echo "[Phase C] Monitoring progress..."
while true; do
    running=0
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            running=$((running + 1))
        fi
    done

    count=$(find "$OUT_DIR/$SPLIT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "  [$(date +%H:%M:%S)] episodes=$count/$TOTAL_EPISODES  workers_running=$running"

    if [ "$running" -eq 0 ]; then
        break
    fi
    sleep 30
done
echo ""

# ── Phase D: Validate ──
echo "[Phase D] Validating output..."
total_dirs=$(find "$OUT_DIR/$SPLIT" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "  Total episode directories: $total_dirs"

missing_files=0
for d in "$OUT_DIR/$SPLIT"/ep_*/; do
    for f in frames.csv meta.json camera.json; do
        if [ ! -f "$d/$f" ]; then
            echo "  MISSING: $d/$f"
            missing_files=$((missing_files + 1))
        fi
    done
done

if [ "$total_dirs" -eq "$TOTAL_EPISODES" ] && [ "$missing_files" -eq 0 ]; then
    echo "  PASS: All $TOTAL_EPISODES episodes with complete files."
else
    echo "  WARN: Expected $TOTAL_EPISODES dirs, got $total_dirs. Missing files: $missing_files"
fi

# Check worker exit codes
echo ""
echo "  Worker exit codes:"
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" 2>/dev/null
    code=$?
    echo "    worker_$i (PID=${PIDS[$i]}): exit $code"
done
echo ""

# ── Phase E: Cleanup ──
read -p "Remove worker clones ($WORKER_DIR)? [y/N] " ans
if [[ "$ans" =~ ^[Yy] ]]; then
    rm -rf "$WORKER_DIR"
    echo "  Cleaned up."
else
    echo "  Skipped cleanup. Remove manually: rm -rf $WORKER_DIR"
fi

echo ""
echo "=== Done. Output: $OUT_DIR ==="
