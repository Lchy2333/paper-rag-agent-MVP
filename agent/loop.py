from typing import List, Dict

from core.dispatcher import Dispatcher
from core.retry import is_transient, retry_call


def ask_LLM(user_query: str, dispatcher: Dispatcher, client,
            system_prompt: str = "你是一个论文检索助手，能根据RAG提供的工具执行用户的指令。") -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    # LLM 请求属于 agent 层，也用通用重试（传输层暂时失败才重试）
    def call_llm():
        return client.chat.completions.create(
            model="qwen3:8b",
            messages=messages,
            tool_choice="auto",
            tools=dispatcher.registry.list_schemas(),
        )

    max_try = 3
    for _ in range(max_try):
        try:
            response = retry_call(
                call_llm,
                retryable=is_transient,
                max_attempts=4,
                jitter_ratio=0.1,
            )
        except Exception as e:
            return f"LLM 调用失败: {e}"

        llm_response = response.choices[0].message
        print(f"llm_response: {llm_response}")

        # 归一化成普通 dict，保证 content 是字符串
        messages.append({
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
            ] if llm_response.tool_calls else None,
        })

        if llm_response.tool_calls:
            print(f"tool_calls: {llm_response.tool_calls}")
            for tool_call in llm_response.tool_calls:
                tool_name = tool_call.function.name
                args_json = tool_call.function.arguments
                print(f"Calling tool: {tool_name}({args_json})")

                # 统一走 Dispatcher
                result = dispatcher.dispatch(tool_name, args_json)

                # 不管成功失败，都把结果回填给模型，让模型决定要不要继续或纠错
                content = result.data if result.success else f"工具执行失败: {result.error}"
                messages.append({
                    "role": "tool",
                    "content": str(content),
                    "tool_call_id": tool_call.id,
                })
        else:
            return llm_response.content

    return "LLM calling failed!"
