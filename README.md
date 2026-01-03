# voice-assistant

## Whisper Server for Voice Command Transcription

To run the Whisper server on the Jetson:
```
~ $ source ~/stt_server/.venv/bin/activate && nohup uvicorn server:app --host 0.0.0.0 --port 8008 &
```
This should be made more resistant to failure with systemd or even just a cron job.

## Wakeword Listener

On the Pi, run (assuming the code was cloned to ~/voiceassistant):

```
~ $ cd ~/voiceassistant
~ $ source .wakeword-venv/bin/activate && nohup python hey_george_listener.py &
```

The listener uses environment defaults from `~/.ha_env` plus the following optional variables:

```
export STT_URL="http://<jetson-ip>:8008/stt"
export LLM_URL="http://<jetson-ip>:8000/v1/chat/completions"
export LLM_MODEL="local-model"
export LLM_API_KEY="local-anything"
```

## LLM Server

To run the LLM server on the Jetson:
```
~ $ git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh
~ $ sudo mkdir -p /opt/models
~ $ sudo chown -R $USER:$USER /opt/models
~ $ cd /opt/models
~ $ get https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf -O model.gguf
```
Now for a sanity check:
```
~ $ sudo docker run --rm \
  --runtime nvidia \
  --network host \
  --shm-size=8g \
  -v /opt/models:/models:ro \
  dustynv/llama_cpp:b5283-r36.4-cu128-24.04 \
  python3 -c 'from llama_cpp import Llama; llm=Llama(model_path="/models/model.gguf", n_ctx=1024); out=llm("Say hello in Portuguese, Croatian, and English.", max_tokens=64); print(out["choices"][0]["text"])'
```
If the output had downloads, times, and the response from the LLM, the container is being able to use the downloaded model. Now start the daemon:
```
~ $ sudo docker run -d --runtime nvidia --restart unless-stopped \
  --name jetson-llm \
  --network host \
  --shm-size=8g \
  -v /opt/models:/models:ro \
  dustynv/llama_cpp:b5283-r36.4-cu128-24.04 \
  bash -lc 'llama-server -m /models/model.gguf --host 0.0.0.0 --port 8000 -c 1024 -ngl 999 -b 256 --flash-attn'
```
On a different device connected to the same network, export the Jetson's IP to the environment JETSON_IP and call the client with:
```
~ $ curl -X POST "http://$JETSON_IP:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-anything" \
  -d '{
    "model": "local-model",
    "temperature": 0.0,
    "max_tokens": 64,
    "messages": [
      {
        "role": "user",
        "content": "Say hello in Portuguese, Croatian, and English."
      }
    ]
  }'
```
You should get back a response with the answer.
