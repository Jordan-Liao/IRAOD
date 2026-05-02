#!/usr/bin/env python3
"""AgentForge benchmark script for IRAOD.
Finds the latest checkpoint in work_dirs/, evaluates it, writes results/benchmark.json.
"""
import glob
import json
import os
import re
import subprocess
import sys


def find_latest_checkpoint():
    """Find the most recently modified latest.pth in work_dirs/."""
    candidates = glob.glob('work_dirs/**/latest.pth', recursive=True)
    if not candidates:
        # Also try epoch_*.pth
        candidates = glob.glob('work_dirs/**/epoch_*.pth', recursive=True)
    if not candidates:
        print('[benchmark] No checkpoint found in work_dirs/', file=sys.stderr)
        return None, None
    # Sort by modification time, newest first
    candidates.sort(key=os.path.getmtime, reverse=True)
    ckpt = candidates[0]
    work_dir = os.path.dirname(ckpt)
    return ckpt, work_dir


def find_config(work_dir):
    """Find the config file for a work_dir.
    Strategy 1: Look for .py config saved in work_dir by train.py
    Strategy 2: Match work_dir name to configs/*.py
    """
    # Strategy 1: config .py in work_dir (mmdet saves it)
    py_files = glob.glob(os.path.join(work_dir, '*.py'))
    for f in py_files:
        name = os.path.basename(f)
        if not name.startswith('_') and not name.startswith('test'):
            return f

    # Strategy 2: match dir name to configs/
    dir_name = os.path.basename(work_dir)
    config_path = f'configs/{dir_name}.py'
    if os.path.exists(config_path):
        return config_path

    # Strategy 3: try base config name (strip timestamps/suffixes)
    # e.g. exp_rsar_ut_nocga_thr06_full_exec-001_... -> exp_rsar_ut_nocga_thr06_full
    base = dir_name
    # Remove agentforge experiment suffixes
    base = re.sub(r'_exec-\d+.*$', '', base)
    base = re.sub(r'_idea-\d+.*$', '', base)
    base = re.sub(r'_\d{8}T\d+Z.*$', '', base)
    config_path = f'configs/{base}.py'
    if os.path.exists(config_path):
        return config_path

    # Fallback: default config
    default = 'configs/exp_rsar_ut_nocga_thr06_full.py'
    if os.path.exists(default):
        return default

    print(f'[benchmark] No config found for {work_dir}', file=sys.stderr)
    return None


def parse_mAP(output):
    """Parse mAP from test.py output.
    Handles:
    - OrderedDict([('mAP', 0.667)])
    - {'mAP': 0.667}
    - mAP: 0.667
    """
    # Try OrderedDict format
    m = re.search(r"mAP['\"]?,\s*([0-9.]+)", output)
    if m:
        return float(m.group(1))
    # Try table format: | mAP | 0.667 |
    m = re.search(r'mAP\s*\|\s*([0-9.]+)', output)
    if m:
        return float(m.group(1))
    # Try plain format
    m = re.search(r'mAP:\s*([0-9.]+)', output)
    if m:
        return float(m.group(1))
    return None


def parse_eval_json(work_dir):
    """Try to read mAP from eval_*.json written by test.py."""
    eval_files = sorted(glob.glob(os.path.join(work_dir, 'eval_*.json')),
                        key=os.path.getmtime, reverse=True)
    for ef in eval_files:
        try:
            with open(ef) as f:
                data = json.load(f)
            metric = data.get('metric', {})
            if isinstance(metric, dict) and 'mAP' in metric:
                return float(metric['mAP'])
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return None


def main():
    ckpt, work_dir = find_latest_checkpoint()
    if not ckpt:
        # Write zero score
        os.makedirs('results', exist_ok=True)
        with open('results/benchmark.json', 'w') as f:
            json.dump({'mAP': 0.0}, f)
        sys.exit(1)

    config = find_config(work_dir)
    if not config:
        os.makedirs('results', exist_ok=True)
        with open('results/benchmark.json', 'w') as f:
            json.dump({'mAP': 0.0}, f)
        sys.exit(1)

    print(f'[benchmark] checkpoint: {ckpt}')
    print(f'[benchmark] config: {config}')
    print(f'[benchmark] work_dir: {work_dir}')

    # Run evaluation
    cmd = [
        'conda', 'run', '-n', 'iraod', '--no-capture-output',
        'python', 'test.py', config, ckpt,
        '--eval', 'mAP',
        '--work-dir', work_dir,
    ]
    env = {**os.environ, 'PYTHONPATH': '.'}
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800, env=env
        )
        output = result.stdout + result.stderr
        print(output[-2000:] if len(output) > 2000 else output)
    except subprocess.TimeoutExpired:
        print('[benchmark] Evaluation timed out', file=sys.stderr)
        os.makedirs('results', exist_ok=True)
        with open('results/benchmark.json', 'w') as f:
            json.dump({'mAP': 0.0}, f)
        sys.exit(1)

    # Parse mAP - try eval JSON first, then stdout
    mAP = parse_eval_json(work_dir)
    if mAP is None:
        mAP = parse_mAP(output)
    if mAP is None:
        print('[benchmark] Failed to parse mAP from output', file=sys.stderr)
        mAP = 0.0

    # Write results
    os.makedirs('results', exist_ok=True)
    with open('results/benchmark.json', 'w') as f:
        json.dump({'mAP': round(mAP, 6)}, f, indent=2)
    print(f'[benchmark] mAP = {mAP:.6f}')
    print(f'[benchmark] Results written to results/benchmark.json')


if __name__ == '__main__':
    main()
