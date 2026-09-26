import json
import os
import sys
from pathlib import Path

import requests
from openai import OpenAI

from agent.loop import ask_LLM
from core import Dispatcher, ToolRegistry
from tools.http_tool import HttpTool

sys.stdout.reconfigure(encoding="utf-8")

RAG_BASE_URL = os.getenv("RAG_BASE_URL", "http://localhost:8080")
TOOLS_CONFIG = Path(__file__).parent / "tools" / "RAGtools.json"


def build_client() -> OpenAI:
    return OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="true",
    )


def register_rag_tools(registry: ToolRegistry, session: requests.Session) -> None:
    configs = json.loads(TOOLS_CONFIG.read_text(encoding="utf-8"))
    for name, cfg in configs.items():
        registry.register(HttpTool(session, RAG_BASE_URL, cfg, name))


def build_agent() -> tuple:
    registry = ToolRegistry()
    session = requests.Session()
    register_rag_tools(registry, session)
    dispatcher = Dispatcher(registry)
    return dispatcher, build_client()


if __name__ == "__main__":
    dispatcher, client = build_agent()
    print(ask_LLM("当前RAG数据库中有哪些论文？", dispatcher, client))
