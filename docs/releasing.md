# 发布 SuuClaw

当前准备流程只提交源码和文档，不自动改变仓库可见性、创建 Release 或上传 PyPI。

## 验证发布内容

```bash
uv sync --locked --all-extras
uv run pytest
uv run ruff check src tests scripts
uv build
uv run python scripts/check_distribution.py
git diff --check
git status --short
```

检查文档和界面截图不包含私人信息。发布包应包含标准 MIT 文件与前端静态资源，
不得包含 `.env`、会话数据、个人工作区、虚拟环境或开发缓存。

依赖通过包管理器单独安装，其许可证仍适用于对应依赖；SuuClaw 的 MIT 声明不改变
这些依赖的许可。以后若将依赖源码或第三方素材直接打包进项目，需要同时核对并
保留相应版权与许可文件。

## 首次公开

1. 确认 `LICENSE` 中的版权持有人信息准确，当前为 Soren (SorenProjections)。
2. 确认 GitHub CI 检查通过，再将仓库设为公开。
3. 公开后启用 Security 页面的 Private vulnerability reporting。
4. 将版本号与 `src/suuclaw/__init__.py` 保持一致，并整理 CHANGELOG。
5. 为确认发布的提交创建版本标签和 GitHub Release，首次可使用 `v0.1.0`。

公开后，其他人可以按 MIT 条款使用、修改和再发布代码；已有克隆和下载的副本
不会因以后重新设为私有而消失。

## 可选事项

PyPI 发布、Docker 镜像、官网、徽章、赞助和商标注册都不是 MIT 发布的必需步骤。
先让用户能够通过源码安装与文档启动，再按需求增加分发渠道。

AI 辅助开发不需要另选一种开源许可证。项目许可、所用 AI 工具的服务条款，以及
输出内容是否受版权保护属于不同事项。本项目不把“AI 生成”当作原创性或权利归属
保证；维护者需要理解最终代码、检查来源，并负责发布质量。
