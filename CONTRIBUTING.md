# 参与 SuuClaw

欢迎报告问题、改进文档和提交修复。较大的功能或接口变更，请先通过 Issue
说明使用场景与方案，避免提交后才发现方向不一致。

## 开发环境

需要 Python 3.10 或更新版本，以及 uv。运行浏览器界面的 JavaScript 回归测试
还需要 Node.js；用户启动工作台不需要 Node.js 或前端构建。

```bash
git clone https://github.com/SorenProjections/SuuClaw.git
cd SuuClaw
uv sync --locked --all-extras
uv run pytest
uv run ruff check src tests scripts
uv build
uv run python scripts/check_distribution.py
```

已有 `.env` 请保留。测试使用离线模型和模拟消息通道，不需要真实模型或飞书密钥。
不要把真实密钥、聊天记录、运行日志或工作区文件放进提交。

## 提交修改

1. 从 `main` 创建分支，保持每个 PR 的目标明确。
2. 说明修复的问题、最终行为，以及运行过的检查。
3. 行为变更应有相关回归验证；文档或视觉修改附必要截图即可。
4. 修改依赖时同步更新 `uv.lock`，不要只修改本地环境。
5. 提交前检查 `git diff --check` 和 `git status --short`。

不要在公开 Issue 或 PR 中粘贴 `.env`。日志需要先去除密钥、私人路径和会话内容。
漏洞报告请使用 [安全报告说明](SECURITY.md)。

## AI 辅助贡献

可以使用 AI 辅助设计、编程和测试，不要求采用特定工具。
提交者仍应理解所提交的代码，检查行为和授权来源，并运行相关验证。
不要提交没有复现步骤的自动生成缺陷报告，或未经检查的大批量代码修改。

## 许可

提交贡献时，你同意按本项目的 [MIT 许可证](LICENSE) 提供该贡献。
请确认你有权提交这些内容；引入第三方代码或资源时注明来源并保留其许可要求。
本项目暂不要求另行签署 CLA。
