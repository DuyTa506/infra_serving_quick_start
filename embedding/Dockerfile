FROM vllm/vllm-openai:latest

ENV HF_HOME=/root/.cache/huggingface

EXPOSE 8000

# Shell form allows $API_KEY expansion from environment
CMD vllm serve BAAI/bge-large-en-v1.5 \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key "$API_KEY" \
    --dtype float16 \
    --max-model-len 512 \
    --gpu-memory-utilization 0.15 \
    --trust-remote-code
