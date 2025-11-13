# Quick Start Guide

## Prerequisites

1. **Tinker API Access**
   - Sign up at https://thinkingmachines.ai/tinker
   - Create API key from https://tinker-console.thinkingmachines.ai
   - Export: `export TINKER_API_KEY=your_api_key`

2. **Docker**
   - Install Docker CE: https://docs.docker.com/get-docker/
   - Add user to docker group:
     ```bash
     sudo usermod -aG docker $USER
     # Log out and back in
     ```

3. **Python Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Step 1: Test Setup

Run the test script to validate your environment:

```bash
python test_setup.py
```

Expected output:
```
✅ Environment creation test passed!
✅ Dataset loading test passed!
✅ Code extraction test passed!
✅ ALL TESTS PASSED!
```

## Step 2: Prepare Dataset

### Option A: Use CVDP Example Dataset

```bash
cd /Users/yecao/Downloads/project/cvdp_benchmark
DATASET_PATH="example_dataset/cvdp_v1.0.1_example_agentic_code_generation_no_commercial_with_solutions.jsonl"
```

### Option B: Create Small Subset for Testing

```python
from utils import create_small_cvdp_subset

create_small_cvdp_subset(
    input_jsonl="/path/to/full_cvdp_dataset.jsonl",
    output_jsonl="./workspaces/small_dataset.jsonl",
    num_problems=10,
    categories=["easy"]  # Optional: filter by category
)
```

## Step 3: Run Training

### Standard RL Training

```bash
python train_rl.py \
  --cvdp_jsonl_path $DATASET_PATH \
  --model_name meta-llama/Llama-3.2-1B \
  --batch_size 4 \
  --group_size 2 \
  --learning_rate 1e-4 \
  --lora_rank 32 \
  --max_tokens 4096 \
  --log_path /tmp/rtl-rl-test
```

### On-Policy Distillation

```bash
python train_distillation.py \
  --cvdp_jsonl_path $DATASET_PATH \
  --student_model gpt-oss-20b \
  --teacher_model Qwen/Qwen3-Coder-30B \
  --batch_size 4 \
  --group_size 2 \
  --learning_rate 1e-4 \
  --kl_penalty_coef 1.0 \
  --lora_rank 32 \
  --max_tokens 8192 \
  --log_path /tmp/rtl-distill-test
```

## Step 4: Monitor Training

Training outputs:
- `{log_path}/metrics.jsonl` - Training metrics
- `{log_path}/logtree.html` - Detailed execution traces
- `{log_path}/checkpoints.jsonl` - Checkpoint metadata

Key metrics to watch:
- `format_valid`: Code extraction success rate (target: >90%)
- `syntax_valid`: Code compilation rate (target: >60%)
- `tests_passed`: Test passing rate (target: >30%)
- `reward`: Overall reward (target: increasing trend)

## Step 5: Analyze Results

```python
from utils import analyze_cvdp_results

results = analyze_cvdp_results("/tmp/rtl-rl-test/metrics.jsonl")
print(results)
```

## Common Issues

### Issue: Docker timeout

**Symptom**: "Docker execution timeout"

**Solution**: Increase timeout
```bash
python train_rl.py --timeout_seconds 600 ...
```

### Issue: Out of memory

**Symptom**: "CUDA out of memory"

**Solution**: Reduce batch size or max tokens
```bash
python train_rl.py --batch_size 2 --max_tokens 2048 ...
```

### Issue: Slow training

**Symptom**: Training takes hours

**Solution**:
1. Use smaller dataset (10-50 problems)
2. Reduce group_size to 2
3. Check Docker parallelism

### Issue: Low pass rate

**Symptom**: Tests passing rate < 5%

**Solution**:
1. Check reward shaping coefficients
2. Try supervised pre-training first
3. Use easier problems (category="easy")
4. Increase training steps

## Next Steps

1. **Baseline Evaluation**
   - Run model without training on test set
   - Establish baseline metrics

2. **Hyperparameter Tuning**
   - Try different learning rates (1e-5 to 1e-3)
   - Adjust reward coefficients
   - Experiment with KL penalty (0.5 to 2.0)

3. **Scale Up**
   - Increase dataset size (50 → 100 → 500 problems)
   - Run longer training (10-20 epochs)
   - Use multiple GPUs if available

4. **Evaluation**
   - Test on held-out problems
   - Compare student vs. teacher performance
   - Analyze error patterns

## Resources

- [CVDP Benchmark](https://github.com/nvidia/cvdp_benchmark)
- [Tinker Docs](https://tinker-docs.thinkingmachines.ai/)
- [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation)
- [Main README](./README.md)
