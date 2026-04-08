#!/bin/bash
# ============================================================
# awesome-algorithm-auto-tools — 自托管 Runner 安装
#
# 如果你已经在 ClaudeNotice 项目配置过 self-hosted runner，
# 有两种方式复用：
#
#   方式 A（推荐）：组织级 Runner
#     将已有 runner 改为组织级，所有仓库共享
#     GitHub → Settings → Actions → Runners → 移到 Organization
#
#   方式 B：为本仓库单独注册一个 Runner
#     运行本脚本：./scripts/setup_runner.sh
#
# 使用前：
#   1. 打开 https://github.com/BinHPdev/awesome-algorithm-auto-tools
#   2. Settings → Actions → Runners → New self-hosted runner
#   3. 复制 token，粘贴到下面
# ============================================================

set -e

GITHUB_REPO="https://github.com/BinHPdev/awesome-algorithm-auto-tools"
RUNNER_TOKEN="YOUR_RUNNER_TOKEN_FROM_GITHUB"
RUNNER_NAME="algo-tools-runner"
RUNNER_DIR="$HOME/actions-runner-algo"

echo "======================================"
echo "  Awesome Algorithm Auto Tools Runner"
echo "======================================"

command -v claude >/dev/null || { echo "❌ 需要安装 Claude Code"; exit 1; }
command -v git    >/dev/null || { echo "❌ 需要安装 git"; exit 1; }

mkdir -p "$RUNNER_DIR"
cd "$RUNNER_DIR"

RUNNER_VERSION=$(curl -s https://api.github.com/repos/actions/runner/releases/latest | grep '"tag_name"' | sed 's/.*"v\([^"]*\)".*/\1/')
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
  PKG="actions-runner-osx-x64-${RUNNER_VERSION}.tar.gz"
else
  PKG="actions-runner-osx-arm64-${RUNNER_VERSION}.tar.gz"
fi

if [ ! -f "run.sh" ]; then
  echo "📦 下载 Runner v${RUNNER_VERSION}..."
  curl -sOL "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${PKG}"
  tar xzf "$PKG"
  rm "$PKG"
fi

echo "⚙️  配置..."
./config.sh \
  --url "$GITHUB_REPO" \
  --token "$RUNNER_TOKEN" \
  --name "$RUNNER_NAME" \
  --labels "self-hosted,macos" \
  --work "_work" \
  --unattended \
  --replace

echo "🚀 安装为系统服务..."
./svc.sh install
./svc.sh start

echo ""
echo "======================================"
echo "  ✅ Runner 安装完成！"
echo "  状态: cd $RUNNER_DIR && ./svc.sh status"
echo "  每周五中午会自动运行更新"
echo "======================================"
