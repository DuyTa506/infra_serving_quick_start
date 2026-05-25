FROM vllm/vllm-openai:latest

ENV HF_HOME=/root/.cache/huggingface

EXPOSE 8001

# Shell form allows $API_KEY expansion from environment
CMD vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
    --host 0.0.0.0 \
    --port 8001 \
    --api-key "$API_KEY" \
    --revision 95a723d08a9490559dae23d0cff1d9466213d989 \
    --tensor-parallel-size 2 \
    --max-model-len 131072 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.75 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --safetensors-load-strategy prefetch \
    --disable-custom-all-reduce \
    --language-model-only \
    --trust-remote-code
