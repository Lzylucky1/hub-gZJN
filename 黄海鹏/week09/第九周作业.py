vLLM 与 Transformers 吞吐量对比测试报告

一、 测试背景与描述
为了评估大语言模型在不同推理框架下的实际吞吐表现，本次测试选用 Qwen2-0.5B-Instruct 模型，
并构造了 50 条长短混合的游戏领域问答 Prompt（涵盖 MOBA、RPG、数值策划、伤害计算等场景），
设定最大生成长度为 100 tokens。 测试对比了原生 Transformers 的串行推理、Batch=8 的
批处理推理，以及 vLLM 框架的批处理推理。

二、 测试结果对比
模式	                          总耗时    	QPS (请求/秒)	tokens/s (生成速度)	相对 vLLM 速度
--------------------------------------------------------------------------------
[A] transformers 串行          78.41s    0.64             52                 0.02×
[B] transformers batch=8      14.56s    3.43             281                0.10×
[C] vLLM 批处理                1.44s     34.74            2511               1.00×


三、 核心结论与分析
原生 Transformers 的局限性：串行模式下 QPS 仅为 0.64，吞吐极低；手动 Batch=8 虽然利用了
GPU 并行度提升了约 5 倍速度，但由于不同请求生成长度不一，较短请求必须等待同 Batch 内最长请求
完成才能释放资源（Padding 碎片与等待开销），导致吞吐量依然受限。
vLLM 的显著加速效果：vLLM 相对串行模式实现了 54.5 倍 的惊人加速，相对手动 Batch 模式也达到
了 10.1 倍 的提升，生成速度高达 2511 tokens/s。
关键机制：vLLM 之所以能实现数量级的性能飞跃，主要得益于 PagedAttention 和 Continuous Batching
机制。PagedAttention 按 Block 动态管理 KV Cache，消除了内存碎片；Continuous Batching 实现了
请求级别的调度，已完成的请求立即让出资源给新请求，无需等待整个 Batch 结束，从而最大化了 GPU 利用率。
