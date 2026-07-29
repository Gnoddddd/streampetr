#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_experiment_tmux.sh \
    --session NAME \
    --log LOG_PATH \
    --meta-dir META_DIR \
    --config CONFIG_PATH \
    --output-dir OUTPUT_DIR \
    -- COMMAND [ARG ...]

Starts one detached tmux session and records enough metadata to prevent
accidental duplicate launches. It does not interpret the experiment command.
EOF
}

session=""
log_path=""
meta_dir=""
config_path=""
output_dir=""

while (($#)); do
  case "$1" in
    --session)
      session="${2:-}"
      shift 2
      ;;
    --log)
      log_path="${2:-}"
      shift 2
      ;;
    --meta-dir)
      meta_dir="${2:-}"
      shift 2
      ;;
    --config)
      config_path="${2:-}"
      shift 2
      ;;
    --output-dir)
      output_dir="${2:-}"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
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
if ! command -v tmux >/dev/null 2>&1; then
  printf 'tmux is not installed or not on PATH.\n' >&2
  exit 1
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
  printf 'Refusing duplicate launch: tmux session exists: %s\n' \
    "$session" >&2
  exit 1
fi

mkdir -p "$meta_dir" "$(dirname "$log_path")" "$output_dir"
pid_file="$meta_dir/pid"
if [[ -s "$pid_file" ]]; then
  old_pid="$(<"$pid_file")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    printf 'Refusing duplicate launch: recorded PID %s is alive.\n' \
      "$old_pid" >&2
    exit 1
  fi
  printf 'Existing metadata has a stopped PID; archive it first: %s\n' \
    "$meta_dir" >&2
  exit 1
fi
if [[ -e "$meta_dir/command.txt" || -e "$meta_dir/started_at.txt" ]]; then
  printf 'Existing metadata must be archived before launch: %s\n' \
    "$meta_dir" >&2
  exit 1
fi

command_file="$meta_dir/command.txt"
runner="$meta_dir/run.sh"
printf '%q ' "$@" >"$command_file"
printf '\n' >>"$command_file"
git rev-parse HEAD >"$meta_dir/git_commit.txt"
git status --short >"$meta_dir/git_status.txt"
printf '%s\n' "$config_path" >"$meta_dir/config.txt"
printf '%s\n' "$output_dir" >"$meta_dir/output_dir.txt"
hostname >"$meta_dir/host.txt"
date --iso-8601=seconds >"$meta_dir/started_at.txt"

{
  printf '#!/usr/bin/env bash\n'
  printf 'set -o pipefail\n'
  printf 'cd %q\n' "$project_root"
  printf 'source %q\n' "$project_root/scripts/activate_streampetr.sh"
  printf 'printf "%%s\\n" "$$" > %q\n' "$pid_file"
  printf 'exec > >(tee -a %q) 2>&1\n' "$log_path"
  printf 'printf "started_at=%%s\\n" "$(date --iso-8601=seconds)"\n'
  printf 'printf "git_commit=%%s\\n" "$(git rev-parse HEAD)"\n'
  printf '%q ' "$@"
  printf '\n'
  printf 'status=$?\n'
  printf 'date --iso-8601=seconds > %q\n' \
    "$meta_dir/finished_at.txt"
  printf 'printf "%%s\\n" "$status" > %q\n' \
    "$meta_dir/exit_status.txt"
  printf 'printf "finished_at=%%s exit_status=%%s\\n" "$(date --iso-8601=seconds)" "$status"\n'
  printf 'exit "$status"\n'
} >"$runner"
chmod 700 "$runner"

tmux new-session -d -s "$session" "bash $(printf '%q' "$runner")"
printf 'Started tmux session: %s\n' "$session"
printf 'Attach with: tmux attach-session -t %q\n' "$session"
printf 'Log: %s\n' "$log_path"
printf 'Metadata: %s\n' "$meta_dir"
