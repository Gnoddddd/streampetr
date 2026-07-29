# 长实验 tmux 运行约定

训练、评估和长测试必须在独立 tmux 会话中启动。每个实验使用唯一
session、日志路径和 metadata 目录；重连后先检查已有会话与 PID，禁止
重复启动同一实验。

## 启动模板

以下仅为未来命令模板，本次任务不执行：

```bash
cd /home/research/research/evidence3d
source scripts/activate_streampetr.sh

scripts/run_experiment_tmux.sh \
  --session s24-main-smoke \
  --log outputs/stage2/s2_4/main_smoke/console.log \
  --meta-dir outputs/stage2/s2_4/main_smoke/run_meta \
  --config configs/evidence_conserving/mini_stage2_s2_4_main_smoke.py \
  --output-dir outputs/stage2/s2_4/main_smoke \
  -- python tools/train.py \
     --config configs/evidence_conserving/mini_stage2_s2_4_main_smoke.py
```

脚本会拒绝复用已有 tmux session，也会在 metadata 中已有活 PID 时拒绝
启动。它保存：

- `pid`、`command.txt`、`git_commit.txt`、`git_status.txt`
- `config.txt`、`output_dir.txt`、`host.txt`
- `started_at.txt`、`finished_at.txt`、`exit_status.txt`
- 实际 runner 和 stdout/stderr 日志

## 断线重连

```bash
tmux list-sessions
ps -eo pid,ppid,etimes,cmd |
  egrep 'tools/(train|evaluate)\.py|pytest' |
  grep -v grep
tmux attach-session -t s24-main-smoke
```

只查看日志：

```bash
tail -n 100 -f outputs/stage2/s2_4/main_smoke/console.log
```

若 tmux session 消失，先检查 metadata 中 PID：

```bash
META=outputs/stage2/s2_4/main_smoke/run_meta
cat "$META/pid"
ps -fp "$(cat "$META/pid")"
cat "$META/exit_status.txt" 2>/dev/null || true
```

仅在确认 session 不存在、PID 不存活且旧运行已明确结束后，才能使用新
session 名或归档旧 metadata 后重新启动。SSH/VS Code 断线不是重新启动
实验的理由。
