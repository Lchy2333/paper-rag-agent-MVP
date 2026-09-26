# paper-rag-agent-MVP

学术论文检索 + RAG agent 的最小可行原型（MVP）。

## 环境准备

- Python 3.14（系统已装，位于 `C:\Python314`）
- [uv](https://docs.astral.sh/uv/) 用于虚拟环境管理

### 创建并激活虚拟环境

```bash
uv venv .venv

# git-bash / MSYS
source .venv/Scripts/activate

# 或 Windows PowerShell / cmd
.venv\Scripts\activate
```

### 安装依赖

```bash
uv pip install -r requirements.txt
```

## 项目结构

```
main.py             # 入口：组装 client / registry / dispatcher
core/               # 核心抽象层（与业务无关）
  tool.py             # ToolBase / ToolResult
  registry.py         # ToolRegistry
  dispatcher.py       # Dispatcher（含重试）
  retry.py            # retry_call / is_transient
tools/              # 具体工具
  RAGtools.json       # RAG 接口声明（一个接口 = 一段配置）
  http_tool.py        # 通用 HTTP 工具（配置驱动，无需子类）
agent/              # agent 循环
  loop.py             # ask_LLM
```

## 使用

RAG 服务地址通过环境变量注入（默认 `http://localhost:8000`）：

```bash
$env:RAG_BASE_URL = "http://your-rag-server:8000"
python main.py
```

RAG 接口接入：在 `tools/RAGtools.json` 中新增一段配置即可，无需改代码。配置项：

- `description`：给 LLM 看的函数描述
- `path`：接口路径，`/api/v1/documents/:id` 形式的 `:param` 会替换为参数值
- `method`：HTTP 方法（GET / POST / ...）
- `schema`：OpenAI 工具调用参数 schema（`type` / `properties` / `required`）
