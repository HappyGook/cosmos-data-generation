# Synthetic cabin clip generation with Cosmos3-Nano

## Files

- **`start_server.sh`** launches the Cosmos3-Nano model as a Docker container (vLLM-Omni, tensor-parallel across 2 GPUs), 
waits until it's actually ready, logs to `./server_logs/`.
- **`main.py`** is an HTTP client. Reads `resources/prompts.csv`, 
prepares `resources/cabin.jpeg`, sends concurrent requests to the running server, saves clips to `outputs/`, logs to `outputs/generation.log`.

## Requirements

- Docker + `nvidia-container-toolkit` (so `docker run --gpus all` works)
- 2 GPUs, 24GB+ each
- HF auth cached locally: `uvx hf@latest auth login` (once)
- Python: `pip install httpx pandas pillow`

## Run order

1. `./start_server.sh` — wait for `server is READY at http://localhost:8000` (first run downloads the checkpoint, can take a while)
2. `python main.py` — processes every row in `prompts.csv`, skips clips already in `outputs/`
3. When done: `docker stop cosmos3-nano-server` to free the GPUs
