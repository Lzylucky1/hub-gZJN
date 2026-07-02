import time
import asyncio
import aiohttp
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==================== 配置参数 ====================
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"          # 模型名称
VLLM_API_URL = "http://localhost:8000/v1/chat/completions"
CONCURRENT_REQUESTS = 32                       # vLLM 并发请求数
TOTAL_REQUESTS = 100                           # vLLM 总请求数（实际会循环发送）
PROMPT = "请详细介绍人工智能的发展历程，包括重要里程碑和关键技术。"
MAX_TOKENS = 128                               # 最大生成 token 数
# =================================================

# ----- 1. vLLM 并发测试 -----
async def send_vllm_request(session, prompt):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0
    }
    try:
        async with session.post(VLLM_API_URL, json=payload) as resp:
            if resp.status == 200:
                data = await resp.json()
                return len(data["choices"][0]["message"]["content"].split())
            else:
                return 0
    except Exception:
        return 0

async def benchmark_vllm():
    print(f"[vLLM] 并发数: {CONCURRENT_REQUESTS}, 总请求: {TOTAL_REQUESTS}")
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(TOTAL_REQUESTS):
            tasks.append(send_vllm_request(session, PROMPT))
        
        start = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start
        
        success = [r for r in results if r > 0]
        total_tokens = sum(success)
        throughput = total_tokens / elapsed if elapsed > 0 else 0
        req_per_sec = len(success) / elapsed if elapsed > 0 else 0
        
        print(f"[vLLM] 成功请求: {len(success)}/{TOTAL_REQUESTS}")
        print(f"[vLLM] 总耗时: {elapsed:.2f}s")
        print(f"[vLLM] 吞吐量: {req_per_sec:.2f} req/s, {throughput:.2f} tokens/s")
        return elapsed, total_tokens, len(success)

# ----- 2. HuggingFace 单线程测试 -----
def benchmark_hf():
    print(f"\n[HF] 加载模型: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
    
    # 预热
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)
    
    # 正式测试单条生成
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)
    elapsed = time.time() - start
    
    gen_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    latency = elapsed
    tokens_per_sec = gen_tokens / elapsed if elapsed > 0 else 0
    
    print(f"[HF] 单次生成耗时: {latency:.2f}s, 生成 token 数: {gen_tokens}, 速度: {tokens_per_sec:.2f} tokens/s")
    
    # 模拟串行处理多个请求（用于对比吞吐）
    serial_requests = CONCURRENT_REQUESTS
    print(f"[HF] 模拟串行处理 {serial_requests} 个请求...")
    start_serial = time.time()
    with torch.no_grad():
        for _ in range(serial_requests):
            model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)
    elapsed_serial = time.time() - start_serial
    total_tokens_serial = serial_requests * gen_tokens
    throughput_serial = total_tokens_serial / elapsed_serial
    print(f"[HF] 串行 {serial_requests} 请求总耗时: {elapsed_serial:.2f}s, 吞吐量: {throughput_serial:.2f} tokens/s")
    
    return latency, tokens_per_sec, elapsed_serial, throughput_serial

# ----- 主流程 -----
async def main():
    print("=" * 60)
    print("vLLM vs HuggingFace 推理速度对比")
    print("=" * 60)
    
    # 测试 vLLM
    vllm_elapsed, vllm_tokens, vllm_success = await benchmark_vllm()
    
    # 测试 HF
    hf_latency, hf_single_tps, hf_serial_time, hf_serial_tps = benchmark_hf()
    
    # 输出对比表格
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    print(f"{'指标':<25} {'vLLM':<20} {'HF Transformers':<20}")
    print("-" * 60)
    print(f"{'单请求延迟':<25} {'(并发测量不准)':<20} {hf_latency:.2f}s")
    print(f"{'并发/串行处理请求数':<25} {CONCURRENT_REQUESTS:<20} {CONCURRENT_REQUESTS:<20}")
    print(f"{'处理耗时':<25} {vllm_elapsed:.2f}s (并发) {'':<10} {hf_serial_time:.2f}s (串行)")
    vllm_tps = vllm_tokens / vllm_elapsed if vllm_elapsed > 0 else 0
    print(f"{'吞吐量 (tokens/s)':<25} {vllm_tps:.2f} {hf_serial_tps:.2f}")
    if hf_serial_tps > 0:
        speedup = vllm_tps / hf_serial_tps
        print(f"{'加速比':<25} {speedup:.1f}x")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
