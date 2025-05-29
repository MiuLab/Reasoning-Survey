# Revisiting Test-Time Scaling: A Survey and a Diversity Aware Method for Efficient Reasoning

## Architecture

### Introduction
Large Language Models (LLMs) have become central to modern NLP applications such as generation, translation, and question answering. Their success largely stems from transformer-based architectures and large-scale pretraining, which endow models with strong fluency and generalization. However, standard autoregressive decoding imposes a fixed inference routine that limits their performance on complex reasoning tasks. As model sizes grow, the training cost escalates, yet the marginal gains diminish. To mitigate this, Test-Time Scaling (TTS) has emerged as a promising direction: it enhances model performance by allocating more compute during inference, allowing adaptation to input complexity without retraining.

While TTS has shown effectiveness, its performance is often tied to the model's intrinsic capacity for generation diversity—a factor not yet well understood or explicitly optimized. In particular, models optimized for reasoning, such as distilled variants, tend to exhibit reduced output variance, which may dampen the gains from TTS. This raises an open question: Can diversity-aware fine-tuning improve TTS effectiveness for reasoning models?

To address this, we first conduct a strategy-oriented survey of recent TTS methods, categorizing them into three major families: Sampling, Search, and Trajectory Optimization and identify diversity as a critical enabler of TTS success. Next, we propose a simple yet effective fine-tuning method, ADAPT (A Diversity Aware Prefix fine-Tuning), which enhances early-stage output diversity via prefix-tuned sampling.

We evaluate ADAPT on a compact reasoning model under Best-of-N sampling. As shown in , ADAPT achieves 80% accuracy with eight times fewer samples, outperforming all baseline models in efficiency while retaining strong peak performance. (The figure shows Accuracy vs. Inference Cost in log scale, Each point represents a language model.)
<div align=center>
<img src="https://github.com/MiuLab/Reasoning-Survey/blob/main/images/output-14.png" width="60%" height="60%">
</div>

### Related Work:  Other related Survey
### Test Time Scaling
- Sampling
- Search
  - CoT
  - Hidden-layer Search
  - Self-improvement
- Trajectory Optimization

  We reviewed recent advances in optimizing test-time reasoning (TTS) for large language models by controlling the sequence of reasoning steps to balance accuracy and computational cost. We categorized current methods into reinforcement learning (RL) and distillation-based approaches. 
  - RL

      RL techniques use reward-based feedback, such as step-wise rewards and meta-reinforcement tuning (MRT), to adapt compute usage and encourage concise, effective reasoning. While RL can improve sampling efficiency and optimize for fewer reasoning steps, recent work has shown it may limit reasoning diversity, primarily reweighting the base model’s outputs and risking reward hacking. Newer approaches like min-form credit assignment address these challenges by achieving high accuracy with significantly fewer steps.
  - Distillation

      Distillation methods transfer structured and diverse reasoning strategies from large teacher models to smaller student models, enabling concise inference without reliance on long chains of steps. These methods leverage teacher exploration (including multiple reasoning paths or tree-structured chains of thought), improved supervision formats, and curated datasets to ensure informativeness and generalization. Distilled models often match or surpass RL-trained models within similar compute budgets and generalize better to new tasks.

### ADAPT


### Experiment

### Conclusion & Future Direction
- Problem & challenge
  - While ADAPT demonstrates strong performance under Best-of-N sampling, several limitations remain. First, all experiments are conducted on a single reasoning domain—mathematical problem solving. It remains unclear whether similar diversity-induced gains would generalize to broader tasks such as commonsense or multi-hop QA.
  - Second, our evaluations focus on a relatively small model; scaling effects and interactions with larger architectures are left for future work.
  - Third, although ADAPT improves sample efficiency, it does not directly optimize diversity metrics (e.g., self-BLEU, pairwise entropy), and its diversity-enhancing effect is inferred only through indirect accuracy gains. Explicit diversity measurements could provide more rigorous support for the core hypothesis. Finally, we fix the prefix length and data mixture ratio throughout; exploring how these hyperparameters impact diversity and performance may yield further improvements.

### Applications in Real-World Domains

### Dataset


## Legacy (For reference, free to remove)
- 
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
We evaluate model performance using four metrics. acc_maj denotes the final accuracy obtained via majority voting over N sampled outputs. Improvement measures the absolute increase in acc_maj relative to the baseline performance at N=2. Gain per generation quantifies the average accuracy gain when doubling the sample size (e.g., from N=2 to N=4). Finally, Min N to hit threshold refers to the smallest sample count N required to reach a target acc_maj, such as 80%.
  - cost
  - thinking-time
  - readability
  - human feedback
- Datasets
  - The training dataset consists primarily of diverse responses, supplemented with a smaller subset of outputs generated by the target model, which may exhibit lower generative diversity. This latter subset is included to mitigate potential catastrophic forgetting and to preserve the model's original capabilities.

    In our experiments, the dataset includes 90\% responses generated by Qwen2.5-Math-1.5B and 10\% inference outputs from DeepSeek-R1-Distill-Qwen-1.5B (our target model). For the Qwen-derived examples, we employ a custom prompt format designed to encourage varied initial reasoning steps, whereas the DeepSeek-generated samples retain their original chat template.Since all training targets are produced by existing models, this fine-tuning process can be viewed as a form of targeted knowledge transfer or self-supervised learning.

### Conclusion & Future Direction
- Problem & challenge
  - While ADAPT demonstrates strong performance under Best-of-N sampling, several limitations remain. First, all experiments are conducted on a single reasoning domain—mathematical problem solving. It remains unclear whether similar diversity-induced gains would generalize to broader tasks such as commonsense or multi-hop QA. Second, our evaluations focus on a relatively small model; scaling effects and interactions with larger architectures are left for future work.

    Third, although ADAPT improves sample efficiency, it does not directly optimize diversity metrics (e.g., self-BLEU, pairwise entropy), and its diversity-enhancing effect is inferred only through indirect accuracy gains. Explicit diversity measurements could provide more rigorous support for the core hypothesis. Finally, we fix the prefix length and data mixture ratio throughout; exploring how these hyperparameters impact diversity and performance may yield further improvements.
  - safety
  - overthinking
  - theory
