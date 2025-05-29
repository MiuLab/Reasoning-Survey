```bash

cd sampling # optional if you're not in the current directory
conda create -n tts_sampling python=3.10.13
conda activate tts_sampling
pip install requirements.txt

python3 vllm_inference.py --dataset_name math500 --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --n 250 --max_tokens 512 # Optional: add --bf16 if bf16 available

```