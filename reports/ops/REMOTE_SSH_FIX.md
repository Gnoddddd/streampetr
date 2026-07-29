# Evidence3D Remote-SSH 修复准备

检查日期：2026-07-29
远端：WSL2 Ubuntu 22.04，`research@192.168.100.10:2222`

## WSL 端只读检查结论

- `0.0.0.0:2222` 与 `[::]:2222` 均在监听。
- `ssh.service` 为 `active (running)`，自 2026-07-29 10:20:50 CST
  持续运行；监听进程为 `/usr/sbin/sshd -D`。
- 检查时存在来自 `192.168.100.41` 的两个 2222 ESTABLISHED 连接，
  远端 VS Code Server 及 extension host 正常运行。
- `/etc/ssh/sshd_config.d/99-wsl-remote.conf` 明确设置：
  `Port 2222`、IPv4/IPv6 监听、`PubkeyAuthentication yes`、
  `PasswordAuthentication yes`、`PermitRootLogin no`。
- 主配置和 include 中没有 `Match`、`DisableForwarding yes`、
  `AllowTcpForwarding no` 或限制性的 `PermitOpen`。
- 未显式覆盖的 OpenSSH 默认值为：
  `AllowTcpForwarding yes`、`PermitOpen any`、`MaxSessions 10`、
  `ClientAliveInterval 0`、`ClientAliveCountMax 3`、`TCPKeepAlive yes`。
- `sudo -n` 返回“a password is required”，因此无法用
  `sshd -T` 读取 host key 后的最终展开配置，也无法读取其他用户的
  system journal。普通用户可见的 `journalctl -u ssh` 没有日志记录；
  不能据此声称系统日志中不存在断线或 forwarding 拒绝。
- `/home/research/.ssh` 权限为 `0700`，但当前没有
  `~/.ssh/authorized_keys`。因此密码登录反复出现是预期现象，公钥尚未
  安装到该账号。

现有证据不支持“WSL sshd 禁止 TCP forwarding”这一判断。监听、已有
SSH 会话和 VS Code Server 均正常，更可能的问题位于 Mac 本地：
没有专用公钥、陈旧 ControlMaster socket、本地动态端口冲突，或
Mac 到 WSL 地址的链路中断。以下命令均应在 **Mac 终端**执行；本文只
生成操作步骤，没有执行这些 Mac 命令。

## 1. 创建或复用专用 ed25519 密钥

```bash
KEY="$HOME/.ssh/evidence3d_wsl_ed25519"
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
if [ ! -f "$KEY" ]; then
  ssh-keygen -t ed25519 -a 64 -f "$KEY" -C "evidence3d-wsl-research"
fi
chmod 600 "$KEY"
chmod 644 "${KEY}.pub"
ssh-add --apple-use-keychain "$KEY"
```

不要覆盖已有同名私钥。若 `ssh-add --apple-use-keychain` 在旧版 macOS
不可用，可使用 `ssh-add -K "$KEY"`。

## 2. 将公钥安装到 research

优先使用 `ssh-copy-id`；这一步预计只需最后输入一次 research 密码：

```bash
KEY="$HOME/.ssh/evidence3d_wsl_ed25519"
ssh-copy-id -i "${KEY}.pub" -p 2222 research@192.168.100.10
```

若 Mac 没有 `ssh-copy-id`，使用下面的幂等替代命令：

```bash
KEY="$HOME/.ssh/evidence3d_wsl_ed25519"
PUBKEY="$(cat "${KEY}.pub")"
printf '%s\n' "$PUBKEY" |
  ssh -p 2222 research@192.168.100.10 \
  'umask 077; mkdir -p "$HOME/.ssh"; touch "$HOME/.ssh/authorized_keys"; IFS= read -r key; grep -qxF "$key" "$HOME/.ssh/authorized_keys" || printf "%s\n" "$key" >> "$HOME/.ssh/authorized_keys"; chmod 700 "$HOME/.ssh"; chmod 600 "$HOME/.ssh/authorized_keys"'
unset PUBKEY
```

## 3. 配置专用 Host 别名

编辑 Mac 的 `~/.ssh/config`，加入且只保留一个同名 Host 块：

```sshconfig
Host evidence3d-wsl-key
    HostName 192.168.100.10
    User research
    Port 2222
    IdentityFile ~/.ssh/evidence3d_wsl_ed25519
    IdentitiesOnly yes
    AddKeysToAgent yes
    UseKeychain yes
    ServerAliveInterval 20
    ServerAliveCountMax 6
    TCPKeepAlive yes
    ControlMaster auto
    ControlPath ~/.ssh/cm-%C
    ControlPersist 10m
```

然后执行：

```bash
chmod 600 "$HOME/.ssh/config"
ssh -G evidence3d-wsl-key |
  egrep '^(hostname|user|port|identityfile|identitiesonly|controlmaster|controlpath|controlpersist|serveraliveinterval|serveralivecountmax|tcpkeepalive) '
```

## 4. 验证免密与 forwarding

免密验证必须成功且不得出现密码提示：

```bash
ssh -o BatchMode=yes -o PasswordAuthentication=no \
  evidence3d-wsl-key 'printf "key-auth-ok\n"'
```

验证本地 TCP forwarding 能创建：

```bash
ssh -vvv -o BatchMode=yes -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:51888:127.0.0.1:2222 \
  evidence3d-wsl-key 'printf "forwarding-ok\n"'
```

若第二条命令成功，WSL 端 forwarding 路径可用；VS Code 的
`Failed to set up dynamic port forwarding` 应优先从 Mac 本地端口、
ControlMaster 和 VS Code Remote-SSH 日志排查。

## 5. VS Code 用户设置

在 Mac 的 VS Code `settings.json` 中设置：

```json
{
  "remote.SSH.useLocalServer": false,
  "remote.SSH.connectTimeout": 60,
  "remote.SSH.remotePlatform": {
    "evidence3d-wsl-key": "linux"
  }
}
```

连接时选择 `Remote-SSH: Connect to Host...` →
`evidence3d-wsl-key`。不要删除远端 `~/.vscode-server`，也不要通过关闭
TCP forwarding 来掩盖隧道错误。

## 6. 安全清理陈旧 ControlMaster socket

先请求现有 master 正常退出，再只删除该别名解析出的确切 socket：

```bash
ssh -O check evidence3d-wsl-key || true
ssh -O exit evidence3d-wsl-key || true
CONTROL_PATH="$(ssh -G evidence3d-wsl-key |
  awk '$1 == "controlpath" {print $2; exit}')"
if [ -n "$CONTROL_PATH" ] && [ -S "$CONTROL_PATH" ]; then
  rm -f -- "$CONTROL_PATH"
fi
unset CONTROL_PATH
```

不要使用 `rm -rf ~/.ssh`，也不要删除私钥、`known_hosts` 或整个 VS Code
Server 目录。

## 7. 失败时的安全恢复顺序

1. 关闭该 WSL Host 的 VS Code 窗口，但不要 kill 远端 server。
2. 执行上面的精确 ControlMaster 清理。
3. 检查地址与端口：

   ```bash
   route -n get 192.168.100.10
   nc -vz -G 5 192.168.100.10 2222
   ```

4. 重新执行 BatchMode 免密验证；若失败，用
   `ssh -vvv evidence3d-wsl-key` 判断是 key、网络还是 host-key 问题。
5. BatchMode 成功后，再执行 forwarding 验证。
6. 两项均成功后才重新连接 VS Code。
7. 若 `192.168.100.10` 因 WSL/宿主网络重建而变化，先在 WSL 中只读
   确认实际地址，再更新 Mac Host 别名；不要盲目重启 sshd。
8. 只有取得管理员授权并保存诊断证据后，才考虑服务端配置或服务重启。
