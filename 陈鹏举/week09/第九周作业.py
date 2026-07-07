{
  "n_prompts": 50,
  "max_new_tokens": 100,
  "batch_size": 8,
  "results": {
    "serial": {
      "time": 92.45,
      "gen_tokens": 241400,
      "qps": 0.5408328826392645,
      "tps": 2611.141157382369
    },
    "batch": {
      "time": 34.18,
      "gen_tokens": 238500,
      "qps": 1.462843768285547,
      "tps": 6977.76477472206
    },
    "vllm": {
      "time": 11.27,
      "gen_tokens": 236400,
      "qps": 4.43655723158828,
      "tps": 20976.042590949426
    }
  }
}
