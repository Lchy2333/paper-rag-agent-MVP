import json
import os
import sys
from pathlib import Path

import requests
from openai import OpenAI

from agent.loop import Loop
from core import Dispatcher, ToolRegistry
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


def build_client() -> OpenAI:
    return OpenAI(
        base_url=ARK_BASE_URL,
        api_key=ARK_API_KEY
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
    loop = Loop(dispatcher, client, model=ARK_MODEL)
    print("环境注册完成，请输入问题（输入 q | quit | exit 退出）：")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() in ("q", "quit", "exit"):
            break
        print(loop.chat(user_input))
