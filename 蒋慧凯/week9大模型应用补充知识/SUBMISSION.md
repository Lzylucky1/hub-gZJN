# 作业提交清单

> 本文档说明 week9 作业需要提交的内容和提交方式。请在 WSL2 中跑通项目、生成自己的输出后，再按步骤提交。

---

## 一、提交前必须完成

1. **WSL2 + Ubuntu 22.04 安装完成**（见 `WSL2_INSTALL_GUIDE.md`）。
2. **Python 依赖安装完成**：`pip install -r requirements.txt`。
3. **模型权重已就位**：`pretrain_models/Qwen2-0.5B-Instruct`。
4. **vLLM server 启动成功**，API 调用返回正常结果。
5. **运行 `python run_all.py`** 完成 5 个约束解码 demo。
6. **（可选）运行 `python src/bench_throughput.py`** 生成吞吐对比图。

---

## 二、需要保留的输出证据

请将以下文件/截图放入 `outputs/` 目录：

| 文件 | 说明 | 是否必须 |
|------|------|----------|
| `outputs/screenshots/server_started.png` | vLLM server 启动成功界面 | ✅ |
| `outputs/screenshots/api_test.png` | `curl` 调用返回正常 JSON | ✅ |
| `outputs/screenshots/function_call_result.png` | `demo_function_call.py` 运行结果 | ✅ |
| `outputs/function_call_results.json` | 自己运行生成的 JSON | ✅ |
| `outputs/screenshots/throughput_result.png` | 吞吐对比结果 | 可选 |
| `outputs/throughput_results.json` | 自己运行生成的 JSON | 可选 |
| `outputs/throughput_comparison.png` | 自己运行生成的图 | 可选 |

---

## 三、不提交的大文件

以下文件/目录**保留在本地**，不要提交到 git：

- `pretrain_models/` — 模型权重
- `data/` — 数据集（本项目无）
- `*.pt` / `*.pth` / `*.bin` / `*.safetensors` — 模型权重
- `__pycache__/` — Python 缓存

---

## 四、提交步骤

### 4.1 开发目录确认

确认 `week9大模型应用补充知识/homework/` 中文件齐全、运行无误。

### 4.2 复制到个人提交目录

```bash
# 源目录（开发目录）
SRC="Z:\LearningDocs\八斗AI\week9大模型应用补充知识\homework"

# 目标目录（个人提交目录）
DST="D:\codes\python code\bdai-nlp-study\蒋慧凯\week9"

# 复制
mkdir -p "$DST"
cp -r "$SRC"/* "$DST"/
```

### 4.3 Git 提交

```bash
cd "D:\codes\python code\bdai-nlp-study"

git status

# 只 add 自己的目录
git add "蒋慧凯/week9/"

# 确认没有他人文件
git status --short | grep -v "蒋慧凯"

# 提交
git commit -m "feat(week9): vLLM部署与约束解码作业"

# 推送
git push origin main
```

---

## 五、提交 checklist

- [ ] WSL2 中成功启动 vLLM server
- [ ] `run_all.py` 5 个 demo 全部成功
- [ ] `outputs/screenshots/` 中有部署和运行截图
- [ ] `outputs/function_call_results.json` 是自己运行的结果
- [ ] 已删除/排除模型权重等大文件
- [ ] 只 `git add` 了 `蒋慧凯/week9/` 目录
- [ ] `git status` 没有他人文件
- [ ] commit message 规范：`feat(week9): ...`

---

## 六、注意

- 不要夸大项目规模，这是教学 demo。
- 量化数据请与自己的运行结果一致。
- 约束解码是 vLLM/xgrammar 提供的能力，不要写“自研”。
