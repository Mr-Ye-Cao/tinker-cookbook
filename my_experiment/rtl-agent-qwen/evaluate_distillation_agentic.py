"""
Evaluation Script for RTL Agent (Qwen Version)

Runs an evaluation of the RTL agent on a set of problems using a remote API endpoint.
Reuses the CVDPAgenticEnvQwen environment for full agentic capabilities (Docker, tools, etc.).
Uses standard OpenAI Async Client for compatibility with vLLM/Text-Generation-Inference endpoints.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import List, Dict, Any

import chz
from openai import AsyncOpenAI
import tinker
# We still import tinker renderers even if we don't use the client
from tinker_cookbook import renderers, model_info
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

from cvdp_agentic_env_qwen import CVDPAgenticEnvQwen

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@chz.chz
class EvalConfig:
    """Configuration for agentic evaluation"""
    
    # Model & API
    student_model: str = "Qwen/Qwen3-8B"  # Model name for API requests
    tokenizer_name: str = "Qwen/Qwen3-8B" # Model name for local tokenization
    api_base: str = "http://localhost:8000/v1"  # Remote API endpoint
    api_key: str = "EMPTY"
    
    # Dataset
    cvdp_jsonl_path: str = "cvdp_16_easy_problems.jsonl"
    limit: int | None = None  # Limit number of problems (optional)
    temperature: float = 1.0 # Temperature for sampling
    
    # Environment
    workspace_dir: str = "/tmp/cvdp_eval_workspace"
    docker_image: str = "gpt-oss-20b-agent-base:latest"
    timeout_seconds: int = 30
    max_turns: int = 50
    max_tokens: int = 4096  # Max generation tokens per turn
    
    # Execution
    concurrency: int = 2  # Number of parallel environments
    
    # Logging
    log_dir: str = "./logs/eval"


class AgenticEvaluator:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.tokenizer = get_tokenizer(config.tokenizer_name)
        
        # Auto-detect renderer if possible, or default to qwen3_keep_thinking
        try:
            renderer_name = model_info.get_recommended_renderer_name(config.tokenizer_name)
        except:
            renderer_name = "qwen3_keep_thinking"
            
        logger.info(f"Using renderer: {renderer_name}")
        self.renderer = renderers.get_renderer(renderer_name, self.tokenizer)
        
        # Initialize OpenAI Async Client
        self.client = AsyncOpenAI(
            base_url=config.api_base,
            api_key=config.api_key
        )
        logger.info(f"Initialized OpenAI Configured Client: {config.api_base}")

        # Load global agentic system message
        self.system_message = None
        agentic_msg_path = os.path.join(os.path.dirname(__file__), "AGENTIC_SYSTEM_MESSAGE.txt")
        if os.path.exists(agentic_msg_path):
            with open(agentic_msg_path, 'r') as f:
                self.system_message = f.read().strip()
            logger.info(f"Loaded agentic system message from {agentic_msg_path}")
        else:
            logger.warning(f"Agentic message file not found: {agentic_msg_path}. Agent might behave as non-agentic.")


    def load_problems(self) -> List[Dict]:
        """Load problems from JSONL"""
        problems = []
        with open(self.config.cvdp_jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    problems.append(json.loads(line))
        
        if self.config.limit:
            problems = problems[:self.config.limit]
            
        logger.info(f"Loaded {len(problems)} problems from {self.config.cvdp_jsonl_path}")
        return problems

    async def run_single_problem(self, problem: Dict, worker_id: int, log_dir_base: str) -> Dict[str, Any]:
        """Run a single evaluation episode"""
        problem_id = problem["id"]
        task_start_time = time.time()
        turn_times = []  # List of (turn_number, duration_seconds)
        logger.info(f"[Worker {worker_id}] Starting problem {problem_id}")
        
        # Create unique log path for this run
        run_log_path = os.path.join(log_dir_base, f"{problem_id}_{worker_id}")
        os.makedirs(run_log_path, exist_ok=True)
        
        # Extract difficulty from categories (e.g., ['cid005', 'medium'] -> 'medium')
        categories = problem.get("categories", [])
        difficulty = categories[1] if len(categories) > 1 else "unknown"

        # Create environment
        env = CVDPAgenticEnvQwen(
            problem_id=problem_id,
            prompt=problem["prompt"],
            context_files=problem.get("context", {}),
            harness_config=problem.get("harness", {}),
            workspace_dir=self.config.workspace_dir,
            renderer=self.renderer,
            system_message=self.system_message, # Force global agentic message
            docker_image=self.config.docker_image,
            timeout_seconds=self.config.timeout_seconds,
            max_turns=self.config.max_turns,
            log_path=run_log_path,
            difficulty=difficulty,
        )
        
        try:
            # step 1: Initial observation
            observation, stop_condition = await env.initial_observation()
            
            episode_done = False
            total_reward = 0.0
            turns = 0
            
            while not episode_done and turns < self.config.max_turns:
                turns += 1
                turn_start_time = time.time()

                # step 2: Sample from model via API
                logger.info(f"[Worker {worker_id}] {problem_id} Turn {turns}: Sampling from model...")
                
                # Convert tinker observation (tokens) back to text prompt for OpenAI API
                # This is slightly inefficient but ensures compat with standard OpenAI endpoints
                # that expect text/messages, not raw token IDs.
                # Ideally we pass messages list, but environment gives us pre-rendered prompts.
                # Let's verify what `initial_observation` returns. 
                # It returns `model_input` which is `tinker.ModelInput`.
                # We can get the text from `env.conversation_history` instead which is cleaner.
                
                # Use environment's message history directly to construct OpenAI messages
                messages = []
                for msg in env.conversation_history:
                    role = msg["role"]
                    content = msg["content"]
                    # Map tinker roles to standard OpenAI roles if needed
                    # usually they match (system, user, assistant)
                    messages.append({"role": role, "content": content})
                
                try:
                    generation_start = time.time()
                    response = await self.client.chat.completions.create(
                        model=self.config.student_model,
                        messages=messages,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        # We don't pass stop sequences here as they might be token-based in Tinker
                        # but string-based in OpenAI.
                        # env.stop_condition is list of lists of token ids.
                        # Simple OpenAI usage usually doesn't need detailed stop tokens for this task
                        # since model is trained to output properly formatted tool calls.
                    )
                    generation_time = time.time() - generation_start

                    generated_text = response.choices[0].message.content
                    if not generated_text:
                        logger.error(f"[Worker {worker_id}] {problem_id}: Empty response")
                        break

                    # Tokenize the response to feed back into env.step (which expects action=tokens or text)
                    # CVDPAgenticEnvQwen.step handles text input too!
                    # lines 614-630: if isinstance(action, list): decode... else: text
                    # So we can pass text directly!
                    action = generated_text

                except Exception as e:
                    logger.error(f"[Worker {worker_id}] {problem_id}: Sampling failed: {e}")
                    break

                # step 3: Environment step
                env_start = time.time()
                step_result = await env.step(action)
                env_time = time.time() - env_start

                # Record turn timing (generation vs environment separately)
                turn_duration = time.time() - turn_start_time
                turn_times.append({
                    "turn": turns,
                    "generation_seconds": round(generation_time, 2),
                    "env_execution_seconds": round(env_time, 2),
                    "total_seconds": round(turn_duration, 2)
                })
                logger.info(f"[Worker {worker_id}] {problem_id} Turn {turns}: gen={generation_time:.2f}s env={env_time:.2f}s total={turn_duration:.2f}s")

                # observation = step_result.next_observation # Ignore next obs tokens, we use history next loop
                stop_condition = step_result.next_stop_condition
                episode_done = step_result.episode_done
                total_reward += step_result.reward
                
                if episode_done:
                    metrics = step_result.metrics
                    task_duration = time.time() - task_start_time
                    logger.info(f"[Worker {worker_id}] {problem_id} Finished. Reward: {total_reward}, Tests Passed: {metrics.get('tests_passed')}, Duration: {task_duration:.2f}s")
                    return {
                        "problem_id": problem_id,
                        "reward": total_reward,
                        "turns": turns,
                        "metrics": metrics,
                        "success": bool(metrics.get("tests_passed", 0.0) > 0),
                        "timing": {
                            "task_duration_seconds": round(task_duration, 2),
                            "turn_times": turn_times,
                            "avg_generation_seconds": round(sum(t["generation_seconds"] for t in turn_times) / len(turn_times), 2) if turn_times else 0,
                            "avg_env_execution_seconds": round(sum(t["env_execution_seconds"] for t in turn_times) / len(turn_times), 2) if turn_times else 0,
                            "avg_turn_total_seconds": round(sum(t["total_seconds"] for t in turn_times) / len(turn_times), 2) if turn_times else 0,
                            "total_generation_seconds": round(sum(t["generation_seconds"] for t in turn_times), 2) if turn_times else 0,
                            "total_env_execution_seconds": round(sum(t["env_execution_seconds"] for t in turn_times), 2) if turn_times else 0
                        }
                    }
            
            # If we exit loop without done (e.g. max turns check inside env handled it, but let's be safe)
            task_duration = time.time() - task_start_time
            logger.info(f"[Worker {worker_id}] {problem_id} Timeout after {turns} turns, Duration: {task_duration:.2f}s")
            return {
                "problem_id": problem_id,
                "reward": total_reward,
                "turns": turns,
                "metrics": {"status": "timeout"},
                "success": False,
                "timing": {
                    "task_duration_seconds": round(task_duration, 2),
                    "turn_times": turn_times,
                    "avg_generation_seconds": round(sum(t["generation_seconds"] for t in turn_times) / len(turn_times), 2) if turn_times else 0,
                    "avg_env_execution_seconds": round(sum(t["env_execution_seconds"] for t in turn_times) / len(turn_times), 2) if turn_times else 0,
                    "avg_turn_total_seconds": round(sum(t["total_seconds"] for t in turn_times) / len(turn_times), 2) if turn_times else 0,
                    "total_generation_seconds": round(sum(t["generation_seconds"] for t in turn_times), 2) if turn_times else 0,
                    "total_env_execution_seconds": round(sum(t["env_execution_seconds"] for t in turn_times), 2) if turn_times else 0
                }
            }
            
        except Exception as e:
            task_duration = time.time() - task_start_time
            logger.exception(f"[Worker {worker_id}] {problem_id} Failed with error after {task_duration:.2f}s")
            return {
                "problem_id": problem_id,
                "error": str(e),
                "success": False,
                "timing": {
                    "task_duration_seconds": round(task_duration, 2),
                    "turn_times": turn_times,
                    "avg_generation_seconds": round(sum(t.get("generation_seconds", 0) for t in turn_times) / len(turn_times), 2) if turn_times else 0,
                    "avg_env_execution_seconds": round(sum(t.get("env_execution_seconds", 0) for t in turn_times) / len(turn_times), 2) if turn_times else 0,
                    "avg_turn_total_seconds": round(sum(t.get("total_seconds", 0) for t in turn_times) / len(turn_times), 2) if turn_times else 0,
                    "total_generation_seconds": round(sum(t.get("generation_seconds", 0) for t in turn_times), 2) if turn_times else 0,
                    "total_env_execution_seconds": round(sum(t.get("env_execution_seconds", 0) for t in turn_times), 2) if turn_times else 0
                }
            }
        finally:
            # Ensure container is cleaned up (env handles it in __aexit__ or manual stop)
            await env._stop_docker_container()

    async def run(self):
        """Main execution loop"""
        eval_start_time = time.time()
        problems = self.load_problems()
        
        # Setup timestamped log dir
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_log_dir = f"{self.config.log_dir}_{timestamp}"
        os.makedirs(run_log_dir, exist_ok=True)

        # Configure logging to file for high-level overview
        ml_log.configure_logging_module(os.path.join(run_log_dir, "logs.log"))
        logger.info("=" * 80)
        logger.info("RTL Agent - Agentic Evaluation")
        logger.info("=" * 80)
        logger.info(f"Model: {self.config.student_model}")
        logger.info(f"API: {self.config.api_base}")
        logger.info(f"Dataset: {self.config.cvdp_jsonl_path}")
        logger.info(f"Max turns: {self.config.max_turns}")
        logger.info(f"Log directory: {run_log_dir}")
        logger.info("=" * 80)

        semaphore = asyncio.Semaphore(self.config.concurrency)
        results = []
        
        async def worker(problem, idx):
            async with semaphore:
                # Pass run_log_dir explicitly to run_single_problem
                return await self.run_single_problem(problem, idx, run_log_dir)
        
        tasks = [worker(p, i) for i, p in enumerate(problems)]
        results = await asyncio.gather(*tasks)
        
        # Aggregate results
        eval_total_time = time.time() - eval_start_time
        success_count = sum(1 for r in results if r.get("success", False))
        total = len(results)
        pass_rate = success_count / total if total > 0 else 0

        # Calculate timing statistics
        task_durations = [r.get("timing", {}).get("task_duration_seconds", 0) for r in results if r.get("timing")]
        all_generation_times = []
        all_env_times = []
        all_turn_totals = []
        for r in results:
            if r.get("timing") and r["timing"].get("turn_times"):
                for t in r["timing"]["turn_times"]:
                    all_generation_times.append(t.get("generation_seconds", 0))
                    all_env_times.append(t.get("env_execution_seconds", 0))
                    all_turn_totals.append(t.get("total_seconds", 0))

        timing_summary = {
            "total_eval_seconds": round(eval_total_time, 2),
            "total_eval_minutes": round(eval_total_time / 60, 2),
            "total_eval_hours": round(eval_total_time / 3600, 3),
            "avg_task_seconds": round(sum(task_durations) / len(task_durations), 2) if task_durations else 0,
            "min_task_seconds": round(min(task_durations), 2) if task_durations else 0,
            "max_task_seconds": round(max(task_durations), 2) if task_durations else 0,
            "total_turns": len(all_turn_totals),
            # Separate generation vs environment timing
            "avg_generation_seconds": round(sum(all_generation_times) / len(all_generation_times), 2) if all_generation_times else 0,
            "avg_env_execution_seconds": round(sum(all_env_times) / len(all_env_times), 2) if all_env_times else 0,
            "avg_turn_total_seconds": round(sum(all_turn_totals) / len(all_turn_totals), 2) if all_turn_totals else 0,
            "total_generation_seconds": round(sum(all_generation_times), 2),
            "total_env_execution_seconds": round(sum(all_env_times), 2)
        }

        summary = {
            "config": chz.asdict(self.config),
            "timestamp": timestamp,
            "total_problems": total,
            "success_count": success_count,
            "pass_rate": pass_rate,
            "timing_summary": timing_summary,
            "results": results
        }

        summary_path = os.path.join(run_log_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info("="*80)
        logger.info(f"EVALUATION COMPLETE")
        logger.info(f"Pass Rate: {success_count}/{total} ({pass_rate:.2%})")
        logger.info(f"Total Time: {timing_summary['total_eval_minutes']:.2f} minutes ({timing_summary['total_eval_hours']:.3f} hours)")
        logger.info(f"Avg Task Time: {timing_summary['avg_task_seconds']:.2f}s")
        logger.info(f"Total Turns: {timing_summary['total_turns']}")
        logger.info(f"Avg Turn Breakdown: gen={timing_summary['avg_generation_seconds']:.2f}s | env={timing_summary['avg_env_execution_seconds']:.2f}s | total={timing_summary['avg_turn_total_seconds']:.2f}s")
        logger.info(f"Total Time Breakdown: gen={timing_summary['total_generation_seconds']:.2f}s | env={timing_summary['total_env_execution_seconds']:.2f}s")
        logger.info(f"Results saved to: {summary_path}")
        logger.info("="*80)


async def main(config: EvalConfig):
    evaluator = AgenticEvaluator(config)
    await evaluator.run()


if __name__ == "__main__":
    config = chz.entrypoint(EvalConfig)
    asyncio.run(main(config))
