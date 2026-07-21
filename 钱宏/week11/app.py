"""
app.py — Flask Web 服务

提供 HTTP 接口和前端页面，支持通过浏览器查询天气。

使用方式：
  pip install flask
  python app.py
  然后访问 http://localhost:5000

功能：
  GET  /           — 返回前端 HTML 页面
  POST /api/chat   — 接收问题，调用 run() 函数，返回 JSON 结果
"""

import json
import sys
from pathlib import Path

from flask import Flask, render_template, request, jsonify

sys.path.insert(0, str(Path(__file__).parent))

from run_function_call import build_client, run

app = Flask(__name__)

# 初始化 LLM 客户端（复用 run_function_call.py 的配置）
client, model = build_client("dashscope")


@app.route('/')
def index():
    """返回前端 HTML 页面"""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    接收用户问题，调用 run() 函数执行链式工具调用，返回结果。
    
    请求体：{"question": "银川今天天气怎么样"}
    响应体：{"answer": "...", "tool_calls": [...], "elapsed": 8.5}
    """
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "缺少 question 参数"}), 400
        
        question = data['question'].strip()
        if not question:
            return jsonify({"error": "问题不能为空"}), 400
        
        # 调用核心逻辑，verbose=False 避免打印到 stdout
        result = run(client, model, question, verbose=False)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": f"服务器内部错误：{str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
