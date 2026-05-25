FROM vllm/vllm-openai:latest

ENV HF_HOME=/root/.cache/huggingface

EXPOSE 8002

# Shell form allows $API_KEY expansion from environment
CMD vllm serve BAAI/bge-reranker-v2-m3 \
    --host 0.0.0.0 \
    --port 8002 \
    --api-key "$API_KEY" \
    --revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e \
    --dtype float16 \
    --max-model-len 512 \
    --gpu-memory-utilization 0.15 \
    --trust-remote-code
