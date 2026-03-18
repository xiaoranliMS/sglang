#!/bin/bash
set -euo pipefail

BASE="${BASE:-http://localhost:30000}"
METRICS_BASE="${METRICS_BASE:-$BASE}"
MODEL="${MODEL:-/root/deepseek_tokenizer/DeepSeek-V3.2-NVFP4/}"
MAX_TOKENS="${MAX_TOKENS:-2000}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-1234}"
FLUSH="${FLUSH:-0}"
LOG_FILE="${LOG_FILE:-}"

PROMPTS=(
  "Write a detailed technical overview of FlashAttention and KV cache."
  "Compare Raft and Paxos with concrete failure-handling examples."
  "Design a PostgreSQL schema for e-commerce orders refunds and inventory."
  "Provide a step-by-step debugging plan for a Python memory leak in production."
  "Explain OAuth2 authorization code flow with PKCE and common security pitfalls."
  "Propose an architecture for real-time anomaly detection on streaming metrics."
  "Compare gRPC REST and WebSocket for low-latency online inference APIs."
  "Explain INT8 FP8 and NF4 quantization trade-offs for LLM serving."
  "Give a practical test strategy for a high-throughput message queue consumer."
  "Write a concise production incident postmortem template with examples."
)

gen_ids_from_prompt() {
local prompt="$1"
MODEL="$MODEL" PROMPT="$prompt" python - <<'PY'
import json
import os
import pathlib
import sys

from transformers import AutoTokenizer

repo_python = pathlib.Path.cwd() / "python"
if repo_python.exists():
    sys.path.insert(0, str(repo_python))

model = os.environ["MODEL"]
prompt = os.environ["PROMPT"]

tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
messages = [{"role": "user", "content": prompt}]

if getattr(tok, "chat_template", None):
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
else:
    # DeepSeek-V3.x fallback: align with SGLang DS encoding path
    from sglang.srt.entrypoints.openai.encoding_dsv32 import encode_messages

    rendered = encode_messages(
        [{"role": "system", "content": ""}] + messages, thinking_mode="chat"
    )
    ids = tok.encode(rendered)

print(json.dumps(ids))
PY
}

maybe_flush() {
  if [[ "$FLUSH" == "1" ]]; then
    curl -sS -X POST "$BASE/flush_cache" -H "Content-Type: application/json" -d "{}" >/dev/null || true
  fi
}

calc_avg() {
  printf '%s\n' "$@" | awk '{s+=$1} END {if (NR>0) printf "%.6f", s/NR; else print "0.000000"}'
}

calc_avg_numeric() {
  printf '%s\n' "$@" | awk '
    /^-?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$/ {s+=$1; n++}
    END {if (n>0) printf "%.6f", s/n; else print "n/a"}'
}

get_log_size() {
  if [[ -n "$LOG_FILE" && -f "$LOG_FILE" ]]; then
    wc -c < "$LOG_FILE"
  else
    echo 0
  fi
}

extract_accept_rate_stats_from_log_since() {
  local start_size="${1:-0}"
  if [[ -z "$LOG_FILE" || ! -f "$LOG_FILE" ]]; then
    echo "n/a 0"
    return
  fi
  local start_byte=$((start_size + 1))
  if (( start_byte < 1 )); then
    start_byte=1
  fi
  tail -c +"$start_byte" "$LOG_FILE" 2>/dev/null | awk '
    /accept rate:/ {
      line = $0
      sub(/.*accept rate:[[:space:]]*/, "", line)
      sub(/,.*/, "", line)
      if (line ~ /^-?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$/) {
        s += line
        n += 1
      }
    }
    END {
      if (n > 0) {
        printf "%.6f %d\n", s / n, n
      } else {
        print "n/a 0"
      }
    }'
}

get_spec_accept_rate_stats() {
  local start_size="${1:-0}"
  local v c

  # Prefer per-request log-segment averaging when log file is available.
  if [[ -n "$LOG_FILE" && -f "$LOG_FILE" ]]; then
    read -r v c <<< "$(extract_accept_rate_stats_from_log_since "$start_size")"
    if [[ "$v" != "n/a" ]]; then
      echo "$v $c log"
      return
    fi
  fi

  # Fallback to Prometheus gauge snapshot.
  local m
  m=$(curl -fsS "$METRICS_BASE/metrics" 2>/dev/null | awk '
    /^sglang:spec_accept_rate([[:space:]]|{)/ {v=$NF}
    END {if (v!="") print v}')
  if [[ -n "${m:-}" ]]; then
    echo "$m 1 metrics"
    return
  fi

  echo "n/a 0 none"
}

echo "BASE=$BASE"
echo "METRICS_BASE=$METRICS_BASE"
echo "MODEL=$MODEL"
echo "MAX_TOKENS=$MAX_TOKENS temperature=$TEMPERATURE top_p=$TOP_P seed=$SEED"
echo "FLUSH=$FLUSH"
echo "LOG_FILE=${LOG_FILE:-<none>}"
echo "cases=${#PROMPTS[@]}"
echo

chat_messages_times=()
chat_input_ids_times=()
completion_text_times=()
chat_messages_accept_rates=()
chat_input_ids_accept_rates=()
completion_text_accept_rates=()

for i in "${!PROMPTS[@]}"; do
  idx=$((i + 1))
  prompt="${PROMPTS[$i]}"

  echo "========== case $idx/${#PROMPTS[@]} =========="
  echo "prompt=$prompt"

  messages_json=$(jq -nc --arg p "$prompt" '[{"role":"user","content":$p}]')
  ids=$(gen_ids_from_prompt "$prompt")
  echo "ids_len=$(jq length <<< "$ids")"

  maybe_flush
  s1=$(get_log_size)
  t1=$(curl -sS "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":$messages_json,\"max_tokens\":$MAX_TOKENS,\"temperature\":$TEMPERATURE,\"top_p\":$TOP_P,\"seed\":$SEED,\"stream\":false,\"separate_reasoning\":false,\"stream_reasoning\":false}" \
    -o "/tmp/chat_messages_${idx}.json" \
    -w "%{time_total}")
  read -r r1 r1_cnt r1_src <<< "$(get_spec_accept_rate_stats "$s1")"
  echo "chat(messages) time_total=${t1}s"
  echo "chat(messages) spec_accept_rate_avg=${r1} (samples=${r1_cnt}, source=${r1_src})"
  chat_messages_times+=("$t1")
  chat_messages_accept_rates+=("$r1")

  maybe_flush
  s2=$(get_log_size)
  t2=$(curl -sS "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[],\"input_ids\":$ids,\"max_tokens\":$MAX_TOKENS,\"temperature\":$TEMPERATURE,\"top_p\":$TOP_P,\"seed\":$SEED,\"stream\":false,\"separate_reasoning\":false,\"stream_reasoning\":false}" \
    -o "/tmp/chat_input_ids_${idx}.json" \
    -w "%{time_total}")
  read -r r2 r2_cnt r2_src <<< "$(get_spec_accept_rate_stats "$s2")"
  echo "chat(input_ids) time_total=${t2}s"
  echo "chat(input_ids) spec_accept_rate_avg=${r2} (samples=${r2_cnt}, source=${r2_src})"
  chat_input_ids_times+=("$t2")
  chat_input_ids_accept_rates+=("$r2")

  maybe_flush
  s3=$(get_log_size)
  t3=$(curl -sS "$BASE/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"$prompt\",\"max_tokens\":$MAX_TOKENS,\"temperature\":$TEMPERATURE,\"top_p\":$TOP_P,\"seed\":$SEED}" \
    -o "/tmp/completion_text_${idx}.json" \
    -w "%{time_total}")
  read -r r3 r3_cnt r3_src <<< "$(get_spec_accept_rate_stats "$s3")"
  echo "completion(prompt_text) time_total=${t3}s"
  echo "completion(prompt_text) spec_accept_rate_avg=${r3} (samples=${r3_cnt}, source=${r3_src})"
  completion_text_times+=("$t3")
  completion_text_accept_rates+=("$r3")

  echo "usage(chat/messages):   $(jq -c '{prompt: .usage.prompt_tokens, completion: .usage.completion_tokens, finish: .choices[0].finish_reason}' "/tmp/chat_messages_${idx}.json")"
  echo "usage(chat/input_ids):  $(jq -c '{prompt: .usage.prompt_tokens, completion: .usage.completion_tokens, finish: .choices[0].finish_reason}' "/tmp/chat_input_ids_${idx}.json")"
  echo "usage(completion/text): $(jq -c '{prompt: .usage.prompt_tokens, completion: .usage.completion_tokens, finish: .choices[0].finish_reason}' "/tmp/completion_text_${idx}.json")"
  echo
done

avg_chat_messages=$(calc_avg "${chat_messages_times[@]}")
avg_chat_input_ids=$(calc_avg "${chat_input_ids_times[@]}")
avg_completion_text=$(calc_avg "${completion_text_times[@]}")
avg_chat_messages_accept=$(calc_avg_numeric "${chat_messages_accept_rates[@]}")
avg_chat_input_ids_accept=$(calc_avg_numeric "${chat_input_ids_accept_rates[@]}")
avg_completion_text_accept=$(calc_avg_numeric "${completion_text_accept_rates[@]}")

delta_chat_msg_vs_ids=$(awk -v a="$avg_chat_messages" -v b="$avg_chat_input_ids" 'BEGIN {if (b>0) printf "%.2f", (a-b)*100/b; else print "0.00"}')
delta_chat_msg_vs_comp=$(awk -v a="$avg_chat_messages" -v b="$avg_completion_text" 'BEGIN {if (b>0) printf "%.2f", (a-b)*100/b; else print "0.00"}')
delta_chat_ids_vs_comp=$(awk -v a="$avg_chat_input_ids" -v b="$avg_completion_text" 'BEGIN {if (b>0) printf "%.2f", (a-b)*100/b; else print "0.00"}')

echo "========== summary =========="
echo "avg chat(messages):      ${avg_chat_messages}s"
echo "avg chat(input_ids):     ${avg_chat_input_ids}s"
echo "avg completion(text):    ${avg_completion_text}s"
echo "avg spec_accept_rate chat(messages):   ${avg_chat_messages_accept}"
echo "avg spec_accept_rate chat(input_ids):  ${avg_chat_input_ids_accept}"
echo "avg spec_accept_rate completion(text): ${avg_completion_text_accept}"
echo "chat(messages) vs chat(input_ids): ${delta_chat_msg_vs_ids}%"
echo "chat(messages) vs completion:      ${delta_chat_msg_vs_comp}%"
echo "chat(input_ids) vs completion:     ${delta_chat_ids_vs_comp}%"
