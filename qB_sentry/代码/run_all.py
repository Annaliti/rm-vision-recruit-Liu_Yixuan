"""一键复现B题全部代码与证据。使用当前Python环境；任何步骤失败即停止。

python run_all.py              # 使用现有点云预处理缓存（若有）
python run_all.py --rebuild    # 从原始PCD重新处理
python run_all.py --no-gif     # 跳过第二步搜索GIF；决策脚本仍生成自己的短GIF
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()
    env = dict(os.environ, OMP_NUM_THREADS="4", MPLCONFIGDIR=str(ROOT/".cache/matplotlib"))
    scripts = [
        ["inspect_map.py"], ["verify_step1.py"], ["run_planning.py"], ["verify_step2.py"],
        ["plot_planning.py"]+(["--no-gif"] if args.no_gif else []),
        ["build_cloud_map.py"]+([] if args.rebuild else ["--reuse-prepared"]),
        ["compare_cloud_planning.py"], ["verify_cloud.py"], ["plot_cloud_results.py"],
        ["heuristic_experiments.py"], ["decision_sim.py"], ["verify_decision.py"],
    ]
    runs = []
    logs = ROOT/"results_full/logs"
    logs.mkdir(parents=True, exist_ok=True)
    for script in scripts:
        tick = time.perf_counter()
        print(f"正在运行：{' '.join(script)}", flush=True)
        result = subprocess.run([sys.executable, "-B", *script], cwd=ROOT, env=env,
                                capture_output=True, text=True)
        (logs/f"{Path(script[0]).stem}.txt").write_text(result.stdout+result.stderr)
        runs.append(dict(script=script, returncode=result.returncode, seconds=time.perf_counter()-tick))
        if result.returncode:
            print(result.stdout+result.stderr)
            raise SystemExit(f"步骤失败，请查看日志：{script[0]}")
        print(f"完成：{script[0]} ({runs[-1]['seconds']:.1f}s)", flush=True)
    for number in range(1, 6):
        shutil.copyfile(ROOT/f"paths/path{number}.csv", ROOT/f"path{number}.csv")
    shutil.copyfile(ROOT/"results_step2/unreachable.json", ROOT/"unreachable.json")
    (ROOT/"results_full/pipeline.json").write_text(json.dumps(dict(passed=True, python=sys.executable, runs=runs), ensure_ascii=False, indent=2)+"\n")
    print("全部步骤已完成；path1.csv至path5.csv已复制到项目根目录；第6、7组见unreachable.json。")


if __name__ == "__main__":
    main()
