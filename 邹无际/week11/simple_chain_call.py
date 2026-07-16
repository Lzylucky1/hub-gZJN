import json


def chain_call(client, model, messages, tools, tool_dispatch):
   
    while True:
        # resp = model.chat(msgs, tools)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        msg = resp.choices[0].message
        
        # 如果没有 tool_call，退出循环
        if not msg.tool_calls:
            break
        
        # 有 tool_call，执行并回填
        messages.append(msg)
        
        for tc in msg.tool_calls:
            # result = run(resp.tool_call)
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            
            # 执行工具
            fn = tool_dispatch.get(name)
            if fn:
                result = fn(**args)
            else:
                result = f"未知工具: {name}"
            
            # msgs.append(result)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result)
            })
    
    return resp
