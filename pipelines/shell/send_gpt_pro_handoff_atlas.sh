#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  send_gpt_pro_handoff_atlas.sh --prompt TEXT [options]
  send_gpt_pro_handoff_atlas.sh --prompt-file PATH [options]
  some_command | send_gpt_pro_handoff_atlas.sh --stdin [options]

Options:
  --prompt TEXT         Prompt text to send.
  --prompt-file PATH    Read prompt text from a file.
  --stdin               Read prompt text from stdin.
  --url URL             ChatGPT URL to open in Atlas. Default: https://chatgpt.com/
  --timeout SECONDS     Max wait time after opening Atlas. Default: 120.
  --wait-seconds SECS   Delay after opening the tab before pasting. Default: 6.
  --no-open-tab         Skip opening a new Atlas tab.
  --dry-run             Do not send; only write the temporary handoff file.
  --keep-file           Keep the temporary handoff file after execution.
  --print-final-file    Print the final handoff markdown to stdout.
  --help                Show this help.
EOF
}

prompt=""
prompt_file=""
read_stdin=0
chat_url="https://chatgpt.com/"
timeout_sec=120
wait_seconds=6
open_tab=1
dry_run=0
keep_file=0
print_final_file=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt)
      [[ $# -ge 2 ]] || { echo "missing value for --prompt" >&2; exit 1; }
      prompt="$2"
      shift 2
      ;;
    --prompt-file)
      [[ $# -ge 2 ]] || { echo "missing value for --prompt-file" >&2; exit 1; }
      prompt_file="$2"
      shift 2
      ;;
    --stdin)
      read_stdin=1
      shift
      ;;
    --url)
      [[ $# -ge 2 ]] || { echo "missing value for --url" >&2; exit 1; }
      chat_url="$2"
      shift 2
      ;;
    --timeout)
      [[ $# -ge 2 ]] || { echo "missing value for --timeout" >&2; exit 1; }
      timeout_sec="$2"
      shift 2
      ;;
    --wait-seconds)
      [[ $# -ge 2 ]] || { echo "missing value for --wait-seconds" >&2; exit 1; }
      wait_seconds="$2"
      shift 2
      ;;
    --no-open-tab)
      open_tab=0
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --keep-file)
      keep_file=1
      shift
      ;;
    --print-final-file)
      print_final_file=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

input_modes=0
[[ -n "$prompt" ]] && ((input_modes += 1))
[[ -n "$prompt_file" ]] && ((input_modes += 1))
[[ "$read_stdin" -eq 1 ]] && ((input_modes += 1))
[[ "$input_modes" -eq 1 ]] || {
  echo "choose exactly one of --prompt, --prompt-file, or --stdin" >&2
  exit 1
}

if [[ -n "$prompt_file" ]]; then
  [[ -f "$prompt_file" ]] || { echo "prompt file not found: $prompt_file" >&2; exit 1; }
  prompt="$(cat "$prompt_file")"
elif [[ "$read_stdin" -eq 1 ]]; then
  prompt="$(cat)"
fi

[[ -n "${prompt//[$' \t\r\n']}" ]] || {
  echo "prompt is empty" >&2
  exit 1
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
atlas_cli="$HOME/.codex/skills/atlas/scripts/atlas_cli.py"
if [[ ! -f "$atlas_cli" ]]; then
  atlas_cli="$HOME/.codex/skills/atlas/scripts/atlas_cli.py"
fi
[[ -f "$atlas_cli" ]] || {
  echo "Atlas CLI not found: $atlas_cli" >&2
  exit 1
}

tmp_base="$(mktemp /tmp/gpt-pro-handoff-atlas.XXXXXX)"
tmp_md="${tmp_base}.md"
mv "$tmp_base" "$tmp_md"
cleanup() {
  if [[ "$keep_file" -ne 1 && "$dry_run" -ne 1 ]]; then
    rm -f "$tmp_md"
  fi
}
trap cleanup EXIT

printf '%s
' "$prompt" > "$tmp_md"

if [[ "$print_final_file" -eq 1 ]]; then
  cat "$tmp_md"
fi

echo "Handoff file: $tmp_md" >&2

if [[ "$dry_run" -eq 1 ]]; then
  echo "Dry run: not sending" >&2
  exit 0
fi

if [[ "$open_tab" -eq 1 ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv run --python 3.12 python "$atlas_cli" open-tab "$chat_url"
  else
    python3 "$atlas_cli" open-tab "$chat_url"
  fi
fi

sleep "$wait_seconds"
pbcopy < "$tmp_md"

osascript <<'APPLESCRIPT'
tell application "System Events"
  keystroke "v" using command down
  delay 0.3
  key code 36
end tell
APPLESCRIPT

echo "Sent via Atlas: $chat_url"
