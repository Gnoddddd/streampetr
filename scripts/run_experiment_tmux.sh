#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 SESSION_NAME COMMAND [ARG ...]" >&2
  exit 2
fi

session_name="$1"
shift
project_root="$(git rev-parse --show-toplevel)"
run_root="${EVIDENCE3D_RUN_ROOT:-$project_root/outputs/tmux_runs}"
run_dir="$run_root/$session_name"

if tmux has-session -t "$session_name" 2>/dev/null; then
  echo "Refusing duplicate tmux session: $session_name" >&2
  exit 3
fi
if [[ -f "$run_dir/pid" ]]; then
  prior_pid="$(tr -dc '0-9' < "$run_dir/pid")"
  if [[ -n "$prior_pid" ]] && kill -0 "$prior_pid" 2>/dev/null; then
    echo "Refusing duplicate live PID $prior_pid for $session_name" >&2
    exit 4
  fi
fi

mkdir -p "$run_dir"
printf '%q ' "$@" > "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"
git rev-parse HEAD > "$run_dir/git_commit.txt"
date --iso-8601=seconds > "$run_dir/started_at.txt"
printf '%s\n' "$project_root" > "$run_dir/project_root.txt"

quoted_command="$(printf '%q ' "$@")"
runner="$run_dir/runner.sh"
{
  printf '#!/usr/bin/env bash\n'
  printf 'set -o pipefail\n'
  printf 'cd %q\n' "$project_root"
  printf 'printf \"%%s\\\\n\" \"\\$\\$\" > %q\n' "$run_dir/pid"
  printf 'source scripts/activate_streampetr.sh\n'
  printf '%s 2>&1 | tee %q\n' "$quoted_command" "$run_dir/stdout_stderr.log"
  printf 'status=${PIPESTATUS[0]}\n'
  printf 'printf \"%%s\\\\n\" \"$status\" > %q\n' "$run_dir/exit_code"
  printf 'date --iso-8601=seconds > %q\n' "$run_dir/finished_at.txt"
  printf 'exit \"$status\"\n'
} > "$runner"
chmod +x "$runner"

tmux new-session -d -s "$session_name" "$runner"
echo "Started tmux session $session_name"
echo "Metadata/logs: $run_dir"
