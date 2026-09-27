import json
import os
import sys
from pathlib import Path

import requests
from openai import OpenAI

from core import Dispatcher, ToolRegistry
from core.graph import StateGraph, START, CompileGraph
from agent.graph_agent import call_llm, call_tool, route_after_llm
from tools.http_tool import HttpTool

sys.stdout.reconfigure(encoding="utf-8")

RAG_BASE_URL = os.getenv("RAG_BASE_URL", "http://localhost:8080")
TOOLS_CONFIG = Path(__file__).parent / "tools" / "RAGtools.json"
ENV_FILE = Path(__file__).parent / ".env"


def load_dotenv(path: Path) -> None:
    """极简 .env 加载器：读取 key=value，已有环境变量优先，不覆盖。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_dotenv(ENV_FILE)

ARK_BASE_URL = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/plan/v3")
ARK_API_KEY = os.getenv("ARK_API_KEY", "")
ARK_MODEL = os.getenv("ARK_MODEL", "deepseek-v4-flash")
SYSTEM_PROMPT = """
    你是一个严谨的论文检索助手，负责帮助用户在其私有论文库中查找信息。

    【工作流程】
    1. 先用 RAG 工具检索，再根据检索结果回答。
    2. 一次提问可能涉及多个子问题，可调用多次工具、合并后综合回答。

    【回答原则】
    - 回答必须严格基于检索到的参考文献内容，并引用对应来源。
    - 若工具返回结果为空或与问题无关，如实告知用户"未找到相关内容"，并给出建议（如更换关键词、调整检索范围）。
    - 严禁在没有任何参考文献支撑的前提下编造或自行作答。

    【输出风格】
    - 中文回答，条理清晰，先给结论再展开。
    - 内容较长时使用分点或小节组织。
    - 不确定的内容明确标注，不用模糊表述蒙混。
"""

def build_agent() -> tuple:
    registry = ToolRegistry()
    session = requests.Session()
    register_rag_tools(registry, session)
    dispatcher = Dispatcher(registry)
    return dispatcher, build_client()


def build_client() -> OpenAI:
    return OpenAI(
        base_url=ARK_BASE_URL,
        api_key=ARK_API_KEY
    )

def build_graph(client: OpenAI, dispatcher: Dispatcher) -> CompileGraph:
    graph = StateGraph()
    graph.add_node("call_llm", call_llm(client, dispatcher))
    graph.add_node("call_tool", call_tool(dispatcher))

    graph.add_edge(START, "call_llm")
    graph.add_edge("call_tool", "call_llm")

    graph.add_conditional_edge("call_llm", route_after_llm)

    return graph.compile()

def register_rag_tools(registry: ToolRegistry, session: requests.Session) -> None:
    configs = json.loads(TOOLS_CONFIG.read_text(encoding="utf-8"))
    for name, cfg in configs.items():
        registry.register(HttpTool(session, RAG_BASE_URL, cfg, name))


if __name__ == "__main__":
    dispatcher, client = build_agent()
    app = build_graph(client, dispatcher)

    print("环境注册完成，请输入问题（输入 q | quit | exit 退出）：")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() in ("q", "quit", "exit"):
            break

        initial_state = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ]
        }
        final_state = app.invoke(initial_state) 
        print(final_state["messages"][-1]["content"])
