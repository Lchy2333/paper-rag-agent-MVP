from openai import OpenAI
from typing import Dict

from core.graph import END
from core.dispatcher import Dispatcher

def call_llm(client: OpenAI, dispatcher: Dispatcher) -> Dict:
    def node(state: Dict):
        response =  client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=state["messages"],
            tool_choice="auto",
            tools=dispatcher.registry.list_schemas(),
        )
        msg = response.choices[0].message
        print(f"llm_response: {msg}")

        if msg.tool_calls:
            print(f"tool_calls: {msg.tool_calls}")
            assistant_msg = {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in (msg.tool_calls or [])
                ]
            }
        else:
            assistant_msg = {
                "role": "assistant",
                "content": msg.content or "",
            }
        return {"messages": state["messages"] + [assistant_msg]}
    return node

def call_tool(dispatcher: Dispatcher):
    def node(state: Dict):
        tool_call_msg = state["messages"][-1]
        tool_msgs = []
        for tool_call in tool_call_msg["tool_calls"]:
            name = tool_call["function"]["name"]
            args = tool_call["function"]["arguments"]
            print(f"Calling tool: {name}({args})")
            res = dispatcher.dispatch(name, args)
            content = res.data if res.success else f"工具执行失败: {res.error}"
            tool_msgs.append({
                "role": "tool",
                "content": str(content),
                "tool_call_id": tool_call["id"],
            })
        return {"messages": state["messages"] + tool_msgs}
    return node

def route_after_llm(state):
    if state["messages"][-1].get("tool_calls"):
        return "call_tool"
    return END
