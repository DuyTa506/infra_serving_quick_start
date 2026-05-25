FROM vllm/vllm-openai:latest

ENV HF_HOME=/root/.cache/huggingface

EXPOSE 8001

# Shell form allows $API_KEY expansion from environment
CMD vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
    --host 0.0.0.0 \
    --port 8001 \
    --api-key "$API_KEY" \
    --tensor-parallel-size 2 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.75 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
    --language-model-only \
    --trust-remote-code
