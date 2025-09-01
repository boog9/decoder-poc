#!/usr/bin/env bash
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# End-to-end перевірка пайплайна: court -> map -> detect -> track -> overlay -> mp4
# Образи: decoder-court:cuda, decoder-detect:latest, decoder-track:latest

set -euo pipefail

# ---------------------------- НАЛАШТУВАННЯ -------------------------------------
UIDGID="$(id -u):$(id -g)"
DOCKER_USER=${DOCKER_USER:---user "$(id -u):$(id -g)"}   # вимкніть, якщо ваші GPU-runtime не дружать з non-root

FRAMES_DIR="${FRAMES_DIR:-$(pwd)/frames}"
WEIGHTS_TCD="${WEIGHTS_TCD:-$(pwd)/weights/tcd.pth}"
FPS="${FPS:-30}"
FPS_MP4="${FPS_MP4:-30}"   # default MP4 export FPS
CRF="${CRF:-18}"           # default CRF; falls back to -1 if unsupported

OUT_DIR="$(pwd)"
COURT_JSON="${OUT_DIR}/court.json"
COURT_BY_NAME="${OUT_DIR}/court_by_name.json"
DETECTIONS_JSON="${OUT_DIR}/detections.json"
TRACKS_JSON="${OUT_DIR}/tracks.json"
PREVIEW_DIR="${OUT_DIR}/preview_tracks"
PREVIEW_MP4="${OUT_DIR}/preview_tracks.mp4"

# ROI-only previews
PREVIEW_COURT_DIR="${OUT_DIR}/preview_court"
PREVIEW_COURT_MP4="${OUT_DIR}/preview_court.mp4"

# Параметри моделей (узгоджено з останніми комітами/логами)
DETECT_FLAGS=${DETECT_FLAGS:-"--two-pass --nms-class-aware --multi-scale on --img-size 1536 --p-conf 0.30 --b-conf 0.05 --p-nms 0.60 --b-nms 0.70 --roi-json /app/court.json --roi-margin 8"}
TRACK_FLAGS=${TRACK_FLAGS:-"--fps ${FPS} --min-score 0.10"}
DRAW_FLAGS=${DRAW_FLAGS:-"--mode track --id --label --draw-court --draw-court-lines --roi-json /app/court.json"}

die() { echo "[ERROR] $*" >&2; exit 2; }

check_prereqs() {
  command -v docker >/dev/null 2>&1 || die "Потрібен Docker"
  [ -d "$FRAMES_DIR" ] || die "Не знайдено frames dir: $FRAMES_DIR"
  [ -f "$WEIGHTS_TCD" ] || die "Не знайдено ваги корту: $WEIGHTS_TCD"
}

check_space() {
  local avail_kb
  avail_kb=$(df -Pk "$OUT_DIR" | awk 'NR==2{print $4}')
  if [ "$avail_kb" -lt $((2*1024*1024)) ]; then
    echo "[WARN] Мало місця на диску: ~$((avail_kb/1024)) MB"
  fi
}

fix_permissions() {
  chown "$(id -u)":"$(id -g)" "$@" 2>/dev/null || true
  chmod 644 "$@" 2>/dev/null || true
}

# Визначаємо доступний entry для трекінгу:
# 1) шукаємо Python-модуль серед кандидатів;
# 2) якщо не знайшли — пробуємо консольний скрипт "track".
pick_track_entry() {
  local mod
  mod="$(docker run --rm $DOCKER_USER -v "$(pwd)":/app --entrypoint python decoder-track:latest - <<'PY' 2>/dev/null || true
import importlib.util as I
for c in ["src.track","src.track_objects","src.tracker","src.bytetrack","src.bytetrack_cli","src.run_tracker"]:
    if I.find_spec(c):
        print(c); break
PY
)"
  if [ -n "$mod" ]; then
    echo "MOD:${mod}"
    return 0
  fi
  # Перевіряємо наявність CLI "track"
  if docker run --rm $DOCKER_USER -v "$(pwd)":/app --entrypoint track decoder-track:latest --help >/dev/null 2>&1; then
    echo "CLI:track"; return 0
  fi
  echo "NONE"; return 1
}

py_host() { python - "$@"; }

summary_json() {
  local path="$1"
  py_host "$path" <<'PY'
import json,sys,os
p=sys.argv[1]
if not os.path.exists(p):
    print(f"[summary] {p}: not found"); sys.exit(0)
with open(p,"r",encoding="utf-8") as f:
    d=json.load(f)
n = len(d) if hasattr(d,'__len__') else 'n/a'
print(f"[summary] {os.path.basename(p)} type={type(d).__name__} len={n}")
if isinstance(d, list) and d:
    print("[summary] first keys:", list(d[0])[:8])
elif isinstance(d, dict) and d:
    print("[summary] first 5 keys:", list(d.keys())[:5])
PY
}

maybe_extract_frames() {
  if [ -z "$(find "$FRAMES_DIR" -maxdepth 1 -type f -name '*.*' | head -n1)" ]; then
    echo "[INFO] frames/ пустий — екстрагуємо з input.mp4 (опційно)"
    [ -f "$(pwd)/input.mp4" ] || die "Немає frames/ і немає input.mp4"
    docker run --rm $DOCKER_USER -v "$(pwd)":/app --entrypoint ffmpeg decoder-track:latest \
      -i /app/input.mp4 -vf "fps=${FPS}" /app/frames/frame_%06d.png
  fi
}

run_court() {
  echo "[STEP] Court detection -> ${COURT_JSON}"
  rm -f "$COURT_JSON"
  docker run --rm --gpus all $DOCKER_USER -v "$(pwd)":/app decoder-court:cuda \
    --frames-dir /app/frames \
    --output-json /app/court.json \
    --device cuda \
    --weights /app/weights/tcd.pth \
    --sample-rate 4 \
    --min-score 0.55 --score-metric max --mask-thr 0.30
  fix_permissions "$COURT_JSON"
  summary_json "$COURT_JSON"

  echo "[STEP] Court diagnostics (diag_court.py)"
  rm -f H_CHECK_TOP.txt 2>/dev/null || true
  docker run --rm $DOCKER_USER -v "$(pwd)":/app --entrypoint python \
    decoder-track:latest /app/tools/diag_court.py --court /app/court.json || true
  [ -f H_CHECK_TOP.txt ] && { chown "$(id -u)":"$(id -g)" H_CHECK_TOP.txt 2>/dev/null || true; chmod 644 H_CHECK_TOP.txt 2>/dev/null || true; }
}

run_court_map() {
  echo "[STEP] Court→Frame mapping -> ${COURT_BY_NAME}"
  ./scripts/run_court_map.sh
  fix_permissions "$COURT_BY_NAME"
  summary_json "$COURT_BY_NAME"
}

# Друкуємо підказку щодо доступних режимів оверлею (не блокує пайплайн)
overlay_modes_hint() {
  echo "[HINT] draw_overlay --mode choices:"
  docker run --rm $DOCKER_USER -v "$(pwd)":/app --entrypoint python \
    decoder-track:latest -m src.draw_overlay --help 2>&1 | grep -E -- '--mode.*\{.*\}' || true
}

run_detect() {
  echo "[STEP] Detection -> ${DETECTIONS_JSON}"
  rm -f "$DETECTIONS_JSON"
  docker run --gpus all --rm $DOCKER_USER -v "$(pwd)":/app \
    decoder-detect:latest detect \
      --frames-dir /app/frames \
      --output-json /app/detections.json \
      ${DETECT_FLAGS}
  fix_permissions "$DETECTIONS_JSON"
  summary_json "$DETECTIONS_JSON"

  # Підсумок по класах (підтримує list/dict та 'detections'/'objects', 'class'/'cls')
  py_host "$DETECTIONS_JSON" <<'PY'
import json,sys,collections
p=sys.argv[1]; d=json.load(open(p,'r',encoding='utf-8'))
ctr=collections.Counter()
def bump(rec):
    if not isinstance(rec,dict): return
    arr = rec.get('detections') or rec.get('objects') or []
    for o in arr:
        k = o.get('class', o.get('cls'))
        if k is not None:
            ctr[str(k)] += 1
if isinstance(d,list):
    for rec in d: bump(rec)
elif isinstance(d,dict):
    for _,rec in d.items(): bump(rec)
print("[summary] classes:", dict(ctr))
PY
}

run_track() {
  echo "[STEP] Tracking -> ${TRACKS_JSON}"
  rm -f "$TRACKS_JSON"
  local entry; entry="$(pick_track_entry)"
  case "$entry" in
    MOD:*)
      entry_mod="${entry#MOD:}"
      echo "[INFO] using tracker module: ${entry_mod}"
      docker run --gpus all --rm $DOCKER_USER -v "$(pwd)":/app \
        --entrypoint python decoder-track:latest \
          -m "${entry_mod}" \
          --detections-json /app/detections.json \
          --output-json /app/tracks.json \
          ${TRACK_FLAGS}
      ;;
    CLI:track)
      echo "[INFO] using tracker CLI: track"
      docker run --gpus all --rm $DOCKER_USER -v "$(pwd)":/app \
        --entrypoint track decoder-track:latest \
          --detections-json /app/detections.json \
          --output-json /app/tracks.json \
          ${TRACK_FLAGS}
      ;;
    *)
      echo "[ERROR] Не знайшов точку входу для трекінгу в decoder-track:latest" >&2
      echo "[DEBUG] src/ доступні модулі в контейнері:" >&2
      docker run --rm $DOCKER_USER -v "$(pwd)":/app --entrypoint python decoder-track:latest - <<'PY' 2>/dev/null || true
import pkgutil, json
mods=[m.name for m in pkgutil.iter_modules(['src'])]
print(json.dumps(mods, indent=2))
PY
      return 2
      ;;
  esac
  fix_permissions "$TRACKS_JSON"
  summary_json "$TRACKS_JSON"
}

run_overlay() {
  echo "[STEP] Tracks overlay (PNGs) -> ${PREVIEW_DIR}"
  rm -rf "$PREVIEW_DIR" "$PREVIEW_MP4"

  # 1) Рендер PNG-кадрів (track overlay)
  docker run --rm $DOCKER_USER -v "$(pwd)":/app --entrypoint python \
    decoder-track:latest \
      -m src.draw_overlay \
      --mode track \
      --frames-dir /app/frames \
      --tracks-json /app/tracks.json \
      --output-dir /app/preview_tracks \
      --draw-court --draw-court-lines --roi-json /app/court.json

  [ -d "$PREVIEW_DIR" ] && chmod -R a+r "$PREVIEW_DIR" 2>/dev/null || true

  # 2) Опційний експорт у MP4 (30 fps). Якщо CRF недоступний — фолбек на --crf -1
  echo "[STEP] Tracks overlay (MP4) -> ${PREVIEW_MP4}"
  set +e
  docker run --rm $DOCKER_USER -v "$(pwd)":/app --entrypoint python \
    decoder-track:latest \
      -m src.draw_overlay \
      --mode track \
      --frames-dir /app/frames \
      --tracks-json /app/tracks.json \
      --output-dir /app/preview_tracks \
      --export-mp4 /app/preview_tracks.mp4 \
      --fps "${FPS_MP4}" --crf "${CRF}" \
      --draw-court --draw-court-lines --roi-json /app/court.json
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[WARN] ffmpeg CRF може бути недоступний — повторюємо з --crf -1"
    docker run --rm $DOCKER_USER -v "$(pwd)":/app --entrypoint python \
      decoder-track:latest \
        -m src.draw_overlay \
        --mode track \
        --frames-dir /app/frames \
        --tracks-json /app/tracks.json \
        --output-dir /app/preview_tracks \
        --export-mp4 /app/preview_tracks.mp4 \
        --fps "${FPS_MP4}" --crf -1 \
        --draw-court --draw-court-lines --roi-json /app/court.json || true
  fi
  set -e

  [ -f "$PREVIEW_MP4" ] && fix_permissions "$PREVIEW_MP4"
  ls -lh "$PREVIEW_MP4" 2>/dev/null || echo "[INFO] MP4 не створено (перевірте логи або PNG-и у ${PREVIEW_DIR})"
  # Підказка щодо режимів
  overlay_modes_hint
}

# ---- КРОК 6: ROI-only overlay (кадри + MP4) ----------------------------------
run_roi_overlay() {
  echo "[STEP] ROI overlay -> ${PREVIEW_COURT_DIR} + ${PREVIEW_COURT_MP4}"
  rm -rf "$PREVIEW_COURT_DIR" "$PREVIEW_COURT_MP4"

  # лише контур майданчика, лінії та ROI; без треків/детекцій
  docker run --rm $DOCKER_USER -v "$(pwd)":/app --entrypoint python \
    decoder-track:latest \
      -m src.draw_overlay \
      --frames-dir /app/frames \
      --output-dir /app/preview_court \
      --only-court \
      --draw-court-lines \
      --roi-json /app/court.json \
      --export-mp4 /app/preview_court.mp4 \
      --fps ${FPS} --crf 18

  # права доступу
  if [ -f "$PREVIEW_COURT_MP4" ]; then fix_permissions "$PREVIEW_COURT_MP4"; fi
  if [ -d "$PREVIEW_COURT_DIR" ]; then chmod -R a+r "$PREVIEW_COURT_DIR" 2>/dev/null || true; fi

  # коротке зведення
  if [ -f "$PREVIEW_COURT_MP4" ]; then
    echo "[summary] ROI mp4:"; ls -lh "$PREVIEW_COURT_MP4"
  fi
}

# --------------------------------- MAIN ---------------------------------------
check_prereqs
check_space
maybe_extract_frames
run_court
run_court_map
run_detect
run_track
run_overlay
run_roi_overlay
echo "[OK] Пайплайн завершено"
