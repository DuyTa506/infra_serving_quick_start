FROM vllm/vllm-openai:latest

ENV HF_HOME=/root/.cache/huggingface

EXPOSE 8002

# Shell form allows $API_KEY expansion from environment
CMD vllm serve BAAI/bge-reranker-v2-m3 \
    --host 0.0.0.0 \
    --port 8002 \
    --api-key "$API_KEY" \
    --dtype float16 \
    --max-model-len 512 \
    --gpu-memory-utilization 0.15 \
    --trust-remote-code
