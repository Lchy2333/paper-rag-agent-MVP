from typing import List, Dict
from openai import OpenAI

from core.dispatcher import Dispatcher
from core.retry import is_transient, retry_call

class Loop:
    def __init__(self, dispatcher: Dispatcher, client: OpenAI,
                 system_prompt: str = """
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
                 - 不确定的内容明确标注，不用模糊表述蒙混。""",
                 model: str = "deepseek-v4-flash",
                 max_step: int = 5):
        self.dispatcher = dispatcher
        self.client = client
        self.system_prompt = system_prompt
        self.model = model
        self.messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        self.max_step = max_step

    def call_llm(self):
        return self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tool_choice="auto",
            tools=self.dispatcher.registry.list_schemas(),
        )

    def chat(self, user_query: str) -> str:
        self.messages.append(
            {"role": "user", "content": user_query}
        )

        for _ in range(self.max_step):
            try:
                response = retry_call(
                    self.call_llm,
                    retryable=is_transient,
                    max_attempts=4,
                    jitter_ratio=0.1,
                )
            except Exception as e:
                return f"LLM 调用失败: {e}"

            llm_response = response.choices[0].message
            print(f"llm_response: {llm_response}")

            if llm_response.tool_calls:
                print(f"tool_calls: {llm_response.tool_calls}")

                # 归一化成普通 dict，保证 content 是字符串
                self.messages.append({
                    "role": "assistant",
                    "content": llm_response.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in (llm_response.tool_calls or [])
                    ]
                })

                for tool_call in llm_response.tool_calls:
                    tool_name = tool_call.function.name
                    args_json = tool_call.function.arguments
                    print(f"Calling tool: {tool_name}({args_json})")

                    # 统一走 Dispatcher
                    result = self.dispatcher.dispatch(tool_name, args_json)

                    # 不管成功失败，都把结果回填给模型，让模型决定要不要继续或纠错
                    content = result.data if result.success else f"工具执行失败: {result.error}"
                    self.messages.append({
                        "role": "tool",
                        "content": str(content),
                        "tool_call_id": tool_call.id,
                    })
            else:
                self.messages.append({
                    "role": "assistant",
                    "content": llm_response.content
                })
                return llm_response.content

        return "LLM calling failed!"
