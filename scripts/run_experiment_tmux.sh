#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_experiment_tmux.sh \
    --session NAME --log LOG_PATH --meta-dir META_DIR \
    --config CONFIG_PATH --output-dir OUTPUT_DIR -- COMMAND [ARG ...]
EOF
}

session=""
log_path=""
meta_dir=""
config_path=""
output_dir=""
while (($#)); do
  case "$1" in
    --session) session="${2:-}"; shift 2 ;;
    --log) log_path="${2:-}"; shift 2 ;;
    --meta-dir) meta_dir="${2:-}"; shift 2 ;;
    --config) config_path="${2:-}"; shift 2 ;;
    --output-dir) output_dir="${2:-}"; shift 2 ;;
    --) shift; break ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$session" || -z "$log_path" || -z "$meta_dir" ||
      -z "$config_path" || -z "$output_dir" || $# -eq 0 ]]; then
  usage >&2
  exit 2
fi
if [[ ! "$session" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  printf 'Unsafe tmux session name: %s\n' "$session" >&2
  exit 2
fi

project_root="$(git rev-parse --show-toplevel)"
cd "$project_root"
config_path="$(realpath -m "$config_path")"
log_path="$(realpath -m "$log_path")"
meta_dir="$(realpath -m "$meta_dir")"
output_dir="$(realpath -m "$output_dir")"
if [[ ! -f "$config_path" ]]; then
  printf 'Config does not exist: %s\n' "$config_path" >&2
  exit 1
fi
if tmux has-session -t "=$session" 2>/dev/null; then
  printf 'Refusing duplicate launch: tmux session exists: %s\n' "$session" >&2
  exit 1
fi

mkdir -p "$meta_dir" "$(dirname "$log_path")" "$output_dir"
pid_file="$meta_dir/pid"
if [[ -s "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; then
  printf 'Refusing duplicate launch: recorded PID is alive.\n' >&2
  exit 1
fi
if [[ -e "$meta_dir/command.txt" || -e "$meta_dir/started_at.txt" ]]; then
  printf 'Existing metadata must be archived before launch: %s\n' "$meta_dir" >&2
  exit 1
fi

printf '%q ' "$@" >"$meta_dir/command.txt"
printf '\n' >>"$meta_dir/command.txt"
git rev-parse HEAD >"$meta_dir/git_commit.txt"
git status --short >"$meta_dir/git_status.txt"
printf '%s\n' "$config_path" >"$meta_dir/config.txt"
printf '%s\n' "$output_dir" >"$meta_dir/output_dir.txt"
hostname >"$meta_dir/host.txt"
date --iso-8601=seconds >"$meta_dir/started_at.txt"

runner="$meta_dir/run.sh"
{
  printf '#!/usr/bin/env bash\nset -o pipefail\n'
  printf 'cd %q\nsource %q\n' "$project_root" \
    "$project_root/scripts/activate_streampetr.sh"
  printf 'printf "%%s\\n" "$$" > %q\n' "$pid_file"
  printf 'exec > >(tee -a %q) 2>&1\n' "$log_path"
  printf '%q ' "$@"
  printf '\nstatus=$?\n'
  printf 'date --iso-8601=seconds > %q\n' "$meta_dir/finished_at.txt"
  printf 'printf "%%s\\n" "$status" > %q\n' "$meta_dir/exit_status.txt"
  printf 'exit "$status"\n'
} >"$runner"
chmod 700 "$runner"

tmux new-session -d -s "$session" "bash $(printf '%q' "$runner")"
printf 'Started tmux session: %s\nLog: %s\nMetadata: %s\n' \
  "$session" "$log_path" "$meta_dir"

