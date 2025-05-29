# Revisiting Test-Time Scaling: A Survey and a Diversity Aware Method for Efficient Reasoning

## Architecture

### Introduction
Large Language Models (LLMs) have become central to modern NLP applications such as generation, translation, and question answering. Their success largely stems from transformer-based architectures and large-scale pretraining, which endow models with strong fluency and generalization. However, standard autoregressive decoding imposes a fixed inference routine that limits their performance on complex reasoning tasks. As model sizes grow, the training cost escalates, yet the marginal gains diminish. To mitigate this, Test-Time Scaling (TTS) has emerged as a promising direction: it enhances model performance by allocating more compute during inference, allowing adaptation to input complexity without retraining.

While TTS has shown effectiveness, its performance is often tied to the model's intrinsic capacity for generation diversity—a factor not yet well understood or explicitly optimized. In particular, models optimized for reasoning, such as distilled variants, tend to exhibit reduced output variance, which may dampen the gains from TTS. This raises an open question: Can diversity-aware fine-tuning improve TTS effectiveness for reasoning models?

To address this, we first conduct a strategy-oriented survey of recent TTS methods, categorizing them into three major families: Sampling, Search, and Trajectory Optimization and identify diversity as a critical enabler of TTS success. Next, we propose a simple yet effective fine-tuning method, ADAPT (A Diversity Aware Prefix fine-Tuning), which enhances early-stage output diversity via prefix-tuned sampling.

We evaluate ADAPT on a compact reasoning model under Best-of-N sampling. As shown in , ADAPT achieves 80% accuracy with eight times fewer samples, outperforming all baseline models in efficiency while retaining strong peak performance.
### Related Work:  Other related Survey
### Improvement Method
- Performance
  - prompting: COT, TOT…
  - post-training: SFT, RL, MCTS
  - tool using: external web browser,…
- Efficiency
    - Training-wise
      - Data
      - Epoch
    - Inference-wise
### Evaluation & Dataset & Benchmark
- Evaluation method
  - cost
  - thinking-time
  - readability
  - human feedback
### Conclusion & Future Direction
- Problem & challenge
  -While ADAPT demonstrates strong performance under Best-of-N sampling, several limitations remain. First, all experiments are conducted on a single reasoning domain—mathematical problem solving. It remains unclear whether similar diversity-induced gains would generalize to broader tasks such as commonsense or multi-hop QA. Second, our evaluations focus on a relatively small model; scaling effects and interactions with larger architectures are left for future work.

  -Third, although ADAPT improves sample efficiency, it does not directly optimize diversity metrics (e.g., self-BLEU, pairwise entropy), and its diversity-enhancing effect is inferred only through indirect accuracy gains. Explicit diversity measurements could provide more rigorous support for the core hypothesis. Finally, we fix the prefix length and data mixture ratio throughout; exploring how these hyperparameters impact diversity and performance may yield further improvements.
  - safety
  - overthinking
  - theory
