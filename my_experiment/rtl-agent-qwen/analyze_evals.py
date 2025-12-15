#!/usr/bin/env python3
"""
Generate markdown report from evaluation logs with correct difficulty mapping.
"""

import os
import re
import json
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
LOGS_DIR = SCRIPT_DIR / "logs"
OUTPUT_DIR = SCRIPT_DIR / "analysis_results"
DATASET_FILE = SCRIPT_DIR / "cvdp_full_agentic_noncommercial.jsonl"

def load_difficulty_mapping():
    """Load problem ID to difficulty mapping from dataset."""
    mapping = {}
    with open(DATASET_FILE, "r") as f:
        for line in f:
            data = json.loads(line)
            problem_id = data["id"]
            categories = data.get("categories", [])
            # Find difficulty in categories
            for cat in categories:
                if cat in ["easy", "medium", "hard"]:
                    mapping[problem_id] = cat
                    break
            else:
                mapping[problem_id] = "unknown"
    return mapping

def parse_logs(difficulty_map):
    """Parse all log files and extract model/problem/pass_rate data."""
    results = defaultdict(lambda: defaultdict(list))
    all_problems = set()
    all_models = set()

    for eval_folder in os.listdir(LOGS_DIR):
        log_path = LOGS_DIR / eval_folder / "logs.log"
        if not log_path.is_file():
            continue

        with open(log_path, "r") as f:
            content = f.read()

        model_match = re.search(r"Model: (\S+)", content)
        if not model_match:
            continue
        model_name = model_match.group(1)
        all_models.add(model_name)

        episode_pattern = r"EPISODE END \[([^\]]+)\]:.*?pass_rate=(\d+\.?\d*)%"
        for match in re.finditer(episode_pattern, content):
            problem_name = match.group(1)
            pass_rate = float(match.group(2))
            
            # Normalize problem name (remove difficulty suffix if present)
            base_problem = problem_name
            for suffix in [", easy", ", medium", ", hard"]:
                if problem_name.endswith(suffix):
                    base_problem = problem_name[:-len(suffix)]
                    break
            
            all_problems.add(base_problem)
            results[model_name][base_problem].append(pass_rate)

    return results, sorted(all_problems), sorted(all_models)

def format_cell(rates):
    if not rates:
        return "-"
    rate_strs = []
    for r in rates:
        if r == 100.0:
            rate_strs.append("✓")
        elif r == 0.0:
            rate_strs.append("✗")
        else:
            rate_strs.append(f"{r:.0f}%")
    return f"`[{len(rates)}]` {' '.join(rate_strs)}"

def is_solved(rates):
    return any(r == 100.0 for r in rates) if rates else False

def main():
    print("Loading difficulty mapping...")
    difficulty_map = load_difficulty_mapping()
    print(f"Loaded {len(difficulty_map)} problem difficulty mappings")
    
    print("Parsing logs...")
    results, problems, models = parse_logs(difficulty_map)
    print(f"Found {len(models)} models and {len(problems)} unique problems")
    
    model_short = {
        "qwen3-8b": "base",
        "qwen3-finetuned-easy": "ft-easy",
        "qwen3-finetuned-hard": "ft-hard",
    }

    # Group problems by difficulty
    problems_by_difficulty = defaultdict(list)
    for p in problems:
        diff = difficulty_map.get(p, "unknown")
        problems_by_difficulty[diff].append(p)
    
    print(f"Problems by difficulty: {', '.join(f'{k}={len(v)}' for k,v in problems_by_difficulty.items())}")

    md_lines = []
    def add(line=""):
        md_lines.append(line)

    add("# Evaluation Results Analysis")
    add()

    # =======================================================================
    # HIGH-LEVEL SUMMARY
    # =======================================================================
    add("## High-Level Summary by Difficulty")
    add()
    add("| Difficulty | Total | base | ft-easy | ft-hard |")
    add("|------------|-------|------|---------|---------|")

    for difficulty in ["easy", "medium", "hard"]:
        probs = problems_by_difficulty.get(difficulty, [])
        if not probs:
            continue
        
        row = f"| **{difficulty}** | {len(probs)} |"
        for model in models:
            evaluated = [p for p in probs if p in results[model]]
            solved = [p for p in evaluated if is_solved(results[model][p])]
            total_evals = sum(len(results[model].get(p, [])) for p in probs)
            
            if total_evals > 0:
                cell = f"{len(solved)}/{len(evaluated)} ({100*len(solved)/len(evaluated):.0f}%)"
            else:
                cell = "-"
            row += f" {cell} |"
        add(row)

    # Add totals
    all_probs = [p for d in ["easy", "medium", "hard"] for p in problems_by_difficulty.get(d, [])]
    row = f"| **TOTAL** | {len(all_probs)} |"
    for model in models:
        evaluated = [p for p in all_probs if p in results[model]]
        solved = [p for p in evaluated if is_solved(results[model][p])]
        total_evals = sum(len(results[model].get(p, [])) for p in all_probs)
        
        if total_evals > 0:
            cell = f"{len(solved)}/{len(evaluated)} ({100*len(solved)/len(evaluated):.0f}%)"
        else:
            cell = "-"
        row += f" {cell} |"
    add(row)

    add()
    add("---")
    add()

    # =======================================================================
    # DETAILED MATRICES BY DIFFICULTY
    # =======================================================================
    for difficulty in ["easy", "medium", "hard"]:
        probs = problems_by_difficulty.get(difficulty, [])
        if not probs:
            continue
        
        add(f"## {difficulty.upper()} Problems ({len(probs)} total)")
        add()
        add("| Problem | base | ft-easy | ft-hard | Solved By |")
        add("|---------|------|---------|---------|-----------|")
        
        for problem in sorted(probs):
            short_name = problem.replace("cvdp_agentic_", "")
            row = f"| `{short_name}` |"
            solvers = []
            for model in models:
                rates = results[model].get(problem, [])
                cell = format_cell(rates)
                row += f" {cell} |"
                if is_solved(rates):
                    solvers.append(model_short.get(model, model))
            
            solver_str = ", ".join(solvers) if solvers else "-"
            row += f" {solver_str} |"
            add(row)
        
        # Summary
        add()
        add(f"**{difficulty.upper()} Summary:**")
        add()
        add("| Model | Evaluated | Solved | Solve Rate | Total Evals | Passes | Pass Rate |")
        add("|-------|-----------|--------|------------|-------------|--------|-----------|")
        
        for model in models:
            evaluated = [p for p in probs if p in results[model]]
            solved = [p for p in evaluated if is_solved(results[model][p])]
            total_evals = sum(len(results[model].get(p, [])) for p in probs)
            total_passes = sum(1 for p in evaluated for r in results[model][p] if r == 100.0)
            
            if evaluated:
                add(f"| {model_short.get(model, model)} | {len(evaluated)} | {len(solved)} | {100*len(solved)/len(evaluated):.1f}% | {total_evals} | {total_passes} | {100*total_passes/total_evals:.1f}% |")
            else:
                add(f"| {model_short.get(model, model)} | 0 | 0 | - | 0 | 0 | - |")
        
        add()
        add("---")
        add()

    # =======================================================================
    # UNIQUE SOLVES BY DIFFICULTY
    # =======================================================================
    add("## Unique Solves (Problems only one model solved)")
    add()
    
    for difficulty in ["easy", "medium", "hard"]:
        probs = problems_by_difficulty.get(difficulty, [])
        if not probs:
            continue
            
        add(f"### {difficulty.upper()}")
        add()
        
        for model in models:
            unique = []
            for p in probs:
                if is_solved(results[model].get(p, [])):
                    other_solved = any(is_solved(results[m].get(p, [])) for m in models if m != model)
                    if not other_solved:
                        unique.append(p)
            
            if unique:
                add(f"**{model_short.get(model, model)}** ({len(unique)} unique):")
                for p in sorted(unique):
                    short_name = p.replace("cvdp_agentic_", "")
                    add(f"- `{short_name}`")
                add()
        
        # Problems all models solved
        all_solved = [p for p in probs if all(is_solved(results[m].get(p, [])) for m in models if p in results[m])]
        # Actually check if at least 2 models solved it
        multi_solved = []
        for p in probs:
            solvers = [m for m in models if is_solved(results[m].get(p, []))]
            if len(solvers) >= 2:
                multi_solved.append((p, solvers))
        
        if multi_solved:
            add(f"**Solved by multiple models:**")
            for p, solvers in sorted(multi_solved):
                short_name = p.replace("cvdp_agentic_", "")
                solver_names = ", ".join(model_short.get(m, m) for m in solvers)
                add(f"- `{short_name}` ({solver_names})")
            add()

    # =======================================================================
    # OVERALL SUMMARY
    # =======================================================================
    add("---")
    add()
    add("## Overall Model Comparison")
    add()
    add("| Model | Easy Solved | Medium Solved | Hard Solved | Total Solved |")
    add("|-------|-------------|---------------|-------------|--------------|")
    
    for model in models:
        row = f"| {model_short.get(model, model)} |"
        total_solved = 0
        for difficulty in ["easy", "medium", "hard"]:
            probs = problems_by_difficulty.get(difficulty, [])
            evaluated = [p for p in probs if p in results[model]]
            solved = [p for p in evaluated if is_solved(results[model][p])]
            total_solved += len(solved)
            if evaluated:
                row += f" {len(solved)}/{len(evaluated)} ({100*len(solved)/len(evaluated):.0f}%) |"
            else:
                row += f" - |"
        row += f" {total_solved} |"
        add(row)

    # Write markdown
    OUTPUT_DIR.mkdir(exist_ok=True)
    md_content = "\n".join(md_lines)
    output_path = OUTPUT_DIR / "analysis_report.md"
    with open(output_path, "w") as f:
        f.write(md_content)
    
    print(f"\nSaved to: {output_path}")

if __name__ == "__main__":
    main()
