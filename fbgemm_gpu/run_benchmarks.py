####################################################################################################
#!/usr/bin/env python3
"""
run_benchmarks.py

Run predefined benchmark commands for a given kernel name.
Optimized for LONG multi-line shell command strings.

Examples:
  python run_benchmarks.py my_kernel wrt4_bwd1
  python run_benchmarks.py my_kernel wrt4_bwd1 wrt3_bwd2
  python run_benchmarks.py my_kernel all
  python run_benchmarks.py --list
  LOG_DIR=./out python run_benchmarks.py my_kernel wrt3_bwd1 --dry-run

Notes:
- By default, shell execution is ENABLED (suitable for pipelines/&&/env vars).
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple
import subprocess
import textwrap
import sqlite3

# ── Configuration ──────────────────────────────────────────────────────────────

ALLOWED: Tuple[str] = (
    "wrt4_bwd1",
    "wrt4_bwd5",
    "wrt4_bwd6",
    "wrt1_bwd13",
    "wrt1_bwd10",
    "wrt1_bwd4",
)


@dataclass(frozen=True)
class CommandSpec:
    """
    A task's command builder for LONG shell commands.
    Return a list of shell command STRINGS (multi-line allowed),
    which will be rendered with kernel_name, iters, and warmup_runs.
    """

    build: Callable[
        [str, int, int], List[str]
    ]  # kernel_name, iters, warmup_runs -> [cmd_str, cmd_str, ...]


def sh(cmd: str) -> str:
    """Normalize indentation and strip trailing whitespace from a multi-line shell command."""
    return textwrap.dedent(cmd).strip() + "\n"


def _spec_wrt1_bwd13(kernel_name: str, iters: int, warmup_runs: int) -> List[str]:
    return [
        sh(
            f"""python bench/tbe/split_table_batched_embeddings_benchmark.py device-with-spec --num-embeddings-list 305789,3212710,100000,13200000,2972580,305789,305789,2446310,305789,3057890,790661,2879130,2,204082,3852380,3212700,13200000 --bag-size-list 1,33,1,1,31,8,1,1,15,1,14,9,1,97,95,51,1 --bag-size-sigma-list 0,6,0,0,6,1,0,0,2,0,2,1,0,19,18,10,0 --embedding-dim-list 320,160,16,64,320,12,320,160,24,12,60,320,60,36,160,160,64 --weights-precision fp16 --output-dtype fp16  --warmup-runs {warmup_runs} --iters {iters} --batch-size 229376 --flush-gpu-cache-size-mb 512 --alpha 1.15
        """
        ),
    ]


def _spec_wrt1_bwd10(kernel_name: str, iters: int, warmup_runs: int) -> List[str]:
    return [
        sh(
            f"""python bench/tbe/split_table_batched_embeddings_benchmark.py device-with-spec --num-embeddings-list 13200000,363,590355,321271,2,207087,3209510,305789,13200000,3210880,1669610,191,13200000,611,3212700,1212200,305789,688670,3057890,741,2 --bag-size-list 1,1,4,188,1,17,32,14,1,12,1,1,1,1,8,1,1,24,116,1,1 --bag-size-sigma-list 0,0,0,37,0,3,6,2,0,2,0,0,0,0,1,0,0,4,23,0,0 --embedding-dim-list 64,12,320,160,4,160,320,320,64,160,320,12,64,12,160,320,320,320,160,12,4 --weights-precision fp16 --output-dtype fp16  --warmup-runs {warmup_runs} --iters {iters} --batch-size 229376 --flush-gpu-cache-size-mb 512 --alpha 1.15
        """
        ),
    ]


def _spec_wrt1_bwd4(kernel_name: str, iters: int, warmup_runs: int) -> List[str]:
    return [
        sh(
            f"""python bench/tbe/split_table_batched_embeddings_benchmark.py device-with-spec --num-embeddings-list 305789,3212710,1624700,305789,3007170,3057890,13200000,25000000,305789,13200000,1607740,2,305789,321271,13200000,305789,305789 --bag-size-list 1,33,93,1,10,1,1,1,74,1,27,1,9,141,1,9,1 --bag-size-sigma-list 0,6,18,0,1,0,0,0,14,0,5,0,1,28,0,1,0 --embedding-dim-list 320,160,160,160,160,12,64,16,160,64,320,60,48,320,64,320,320 --weights-precision fp16 --output-dtype fp16  --warmup-runs {warmup_runs} --iters {iters} --batch-size 229376 --flush-gpu-cache-size-mb 512 --alpha 1.15
        """
        ),
    ]


def _spec_wrt4_bwd1(kernel_name: str, iters: int, warmup_runs: int) -> List[str]:
    return [
        sh(
            f"""python bench/tbe/split_table_batched_embeddings_benchmark.py device-with-spec --num-embeddings-list 500000,1000000,1000000,1000000,100001,1000000,1000000,1000000,50001,1000000,50001,300000,1000000,1000000,10001,50001,1000000,500000,1000000,1000000,1000000,100001,1000000,1000000,1000000,50001,1000000,50001,300000,1000000,1000000,10001,50001,1000000,500000,1000000,1000000,1000000,100001,1000000,1000000,1000000,50001,1000000,50001,300000,1000000,1000000,10001,50001,1000000,500000,1000000,1000000,1000000,100001,1000000,1000000,1000000,50001,1000000,50001,300000,1000000,1000000,10001,50001,1000000,500000,1000000,1000000,1000000,100001,1000000,1000000,1000000,50001,1000000,50001,300000,1000000,1000000,10001,50001,1000000,500000,1000000,1000000,1000000,100001,1000000,1000000,1000000,50001,1000000,50001,300000,1000000,1000000,10001,50001,1000000,500000,1000000,1000000,1000000,100001,1000000,1000000,1000000,50001,1000000,50001,300000,1000000,1000000,10001,50001,1000000,500000,1000000,1000000,1000000,100001,1000000,1000000,1000000,50001,1000000,50001,300000,1000000,1000000,10001,50001,1000000,500000,1000000,1000000,1000000,100001,1000000,1000000,1000000,50001,1000000,50001,300000,1000000,1000000,10001,50001,1000000 --bag-size-list 194,1,14,1,10,15,1,1,1,15,21,1,164,79,37,50,1,194,1,14,1,10,15,1,1,1,15,21,1,164,79,37,50,1,194,1,14,1,10,15,1,1,1,15,21,1,164,79,37,50,1,194,1,14,1,10,15,1,1,1,15,21,1,164,79,37,50,1,194,1,14,1,10,15,1,1,1,15,21,1,164,79,37,50,1,194,1,14,1,10,15,1,1,1,15,21,1,164,79,37,50,1,194,1,14,1,10,15,1,1,1,15,21,1,164,79,37,50,1,194,1,14,1,10,15,1,1,1,15,21,1,164,79,37,50,1,194,1,14,1,10,15,1,1,1,15,21,1,164,79,37,50,1 --bag-size-sigma-list 38,0,2,0,1,2,0,0,0,2,4,0,32,15,7,9,0,38,0,2,0,1,2,0,0,0,2,4,0,32,15,7,9,0,38,0,2,0,1,2,0,0,0,2,4,0,32,15,7,9,0,38,0,2,0,1,2,0,0,0,2,4,0,32,15,7,9,0,38,0,2,0,1,2,0,0,0,2,4,0,32,15,7,9,0,38,0,2,0,1,2,0,0,0,2,4,0,32,15,7,9,0,38,0,2,0,1,2,0,0,0,2,4,0,32,15,7,9,0,38,0,2,0,1,2,0,0,0,2,4,0,32,15,7,9,0,38,0,2,0,1,2,0,0,0,2,4,0,32,15,7,9,0 --embedding-dim-list 256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256 --weights-precision fp16 --output-dtype fp16  --warmup-runs {warmup_runs} --iters {iters} --batch-size 65536 --flush-gpu-cache-size-mb 512 --alpha 1.15  --weighted
        """
        ),
    ]


def _spec_wrt4_bwd5(kernel_name: str, iters: int, warmup_runs: int) -> List[str]:
    return [
        sh(
            f"""python bench/tbe/split_table_batched_embeddings_benchmark.py device-with-spec --num-embeddings-list 1000000,2000000,2000000,2000000,1000000,2000000,2000000,2000000,2000000,2000000,2000000,2000000,1000000,2000000,1000000 --bag-size-list 1,32,978,77,1,1,1,2,82,85,1042,89,1,983,1 --bag-size-sigma-list 0,6,195,15,0,0,0,0,16,16,208,17,0,196,0 --embedding-dim-list 256,256,256,256,256,256,256,256,256,256,256,256,256,256,256 --weights-precision fp16 --output-dtype fp16  --warmup-runs {warmup_runs} --iters {iters} --batch-size 65536 --flush-gpu-cache-size-mb 512 --alpha 1.15  --weighted
        """
        ),
    ]


def _spec_wrt4_bwd6(kernel_name: str, iters: int, warmup_runs: int) -> List[str]:
    return [
        sh(
            f"""python bench/tbe/split_table_batched_embeddings_benchmark.py device-with-spec --num-embeddings-list 2000000,2000000,2000000,2000000,2000000,2000000,2000000,2000000,1000000,2000000,2000000,1000000,2000000,1000000,1000000 --bag-size-list 2,97,1,32,983,978,1042,48,1,80,1,1,85,1,1 --bag-size-sigma-list 0,19,0,6,196,195,208,9,0,15,0,0,16,0,0 --embedding-dim-list 256,256,256,256,256,256,256,256,256,256,256,256,256,256,256 --weights-precision fp16 --output-dtype fp16  --warmup-runs {warmup_runs} --iters {iters} --batch-size 65536 --flush-gpu-cache-size-mb 512 --alpha 1.15  --weighted
        """
        ),
    ]


COMMAND_SPECS: Dict[str, CommandSpec] = {
    "wrt1_bwd4": CommandSpec(build=_spec_wrt1_bwd4),
    "wrt1_bwd10": CommandSpec(build=_spec_wrt1_bwd10),
    "wrt1_bwd13": CommandSpec(build=_spec_wrt1_bwd13),
    "wrt4_bwd1": CommandSpec(build=_spec_wrt4_bwd1),
    "wrt4_bwd5": CommandSpec(build=_spec_wrt4_bwd5),
    "wrt4_bwd6": CommandSpec(build=_spec_wrt4_bwd6),
}

# ── Utilities ─────────────────────────────────────────────────────────────────


def now_stamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    return dt.datetime.now().strftime(fmt)


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def validate_keywords(keys: Iterable[str]) -> Tuple[List[str], List[str]]:
    keys_list = list(keys)
    invalid = [k for k in keys_list if k not in ALLOWED]
    valid = [k for k in keys_list if k in ALLOWED]
    return valid, invalid


def ensure_log_dir(env_var: str = "LOG_DIR", default: str = "logs") -> Path:
    log_dir = Path(os.environ.get(env_var, default))
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def ensure_traces_dir(log_dir: Path) -> Path:
    """Create traces directory for RPD files."""
    traces_dir = log_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    return traces_dir


def create_tracer_command(original_command: str, rpd_filename: str) -> str:
    """Wrap the original command with runTracer.sh to generate RPD trace file."""
    return f"runTracer.sh -o {rpd_filename} {original_command}"


def process_rpd_file(rpd_path: Path, kernel_name: str, logger: logging.Logger) -> None:
    """Process RPD file and extract kernel information matching the kernel name."""
    try:
        import pandas as pd
    except ImportError:
        logger.error(
            "Error: pandas is required for RPD trace file processing. Please install pandas with: pip install pandas"
        )
        return

    try:
        if not rpd_path.exists():
            logger.warning(f"RPD file not found: {rpd_path}")
            return

        logger.info(f"Processing RPD file: {rpd_path}")

        conn = sqlite3.connect(str(rpd_path))

        # Read the top table
        df_top = pd.read_sql_query("SELECT * from top", conn)

        # Read the busy table
        df_busy = pd.read_sql_query("SELECT * from busy", conn)

        # Read kernel information
        df_kernel_info = pd.read_sql_query(
            """
            SELECT s1.string AS api_name, k.stream, k.gridX, k.gridY, k.gridZ,
                   k.workgroupX, k.workgroupY, k.workgroupZ, s2.string AS kernel_name
            FROM rocpd_kernelapi k
            LEFT JOIN rocpd_string s1 ON k.api_ptr_id = s1.id
            LEFT JOIN rocpd_string s2 ON k.kernelName_id = s2.id;
            """,
            conn,
        ).drop_duplicates()

        conn.close()

        # Get top 100 entries
        df_top_100 = df_top.head(100)

        # Find kernel name containing the search term
        matching_kernels = [x for x in df_top_100.Name if kernel_name in x]

        if not matching_kernels:
            logger.info(f"No kernels found containing '{kernel_name}' in {rpd_path}")
            return

        # Process each matching kernel
        for kernel_match in matching_kernels:
            logger.info(f"Processing kernel: {kernel_match}")

            # Print the row from top table
            kernel_row = df_top_100.loc[df_top_100["Name"] == kernel_match]
            logger.info(f"Top table data for '{kernel_match}':")
            logger.info(f"{kernel_row.to_string()}")

            # Also print kernel info if available
            kernel_info_row = df_kernel_info.loc[
                df_kernel_info["kernel_name"] == kernel_match
            ]
            if not kernel_info_row.empty:
                logger.info(f"Kernel info for '{kernel_match}':")
                logger.info(f"{kernel_info_row.to_string()}")
            else:
                logger.info(f"No detailed kernel info found for '{kernel_match}'")

    except Exception as e:
        logger.error(f"Error processing RPD file {rpd_path}: {e}")


def process_trace_files(
    traces_dir: Path, kernel_name: str, logger: logging.Logger
) -> None:
    """Process all RPD files in the traces directory."""
    if not traces_dir.exists():
        logger.warning(f"Traces directory not found: {traces_dir}")
        return

    rpd_files = list(traces_dir.glob("*.rpd"))
    if not rpd_files:
        logger.info(f"No RPD files found in {traces_dir}")
        return

    logger.info(f"Found {len(rpd_files)} RPD files to process")

    for rpd_file in sorted(rpd_files):
        logger.info(f"{'='*60}")
        process_rpd_file(rpd_file, kernel_name, logger)
        logger.info(f"{'='*60}")


# ── Logging ───────────────────────────────────────────────────────────────────


def init_combined_logger(combined_log_path: Path) -> logging.Logger:
    logger = logging.getLogger("run")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)

    fh = logging.FileHandler(combined_log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)

    return logger


def open_task_logger(task_log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"task:{task_log_path.stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(task_log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    return logger


# ── Execution ─────────────────────────────────────────────────────────────────


def run_one_command(
    command: str,
    task_logger: logging.Logger,
    combined_logger: logging.Logger,
    dry_run: bool,
) -> int:
    if dry_run:
        for line in command.rstrip("\n").splitlines():
            msg = f"DRY-RUN: {line}"
            combined_logger.info(msg)
            task_logger.info(msg)
        return 0

    proc = subprocess.Popen(
        command,
        shell=True,  # ENABLE shell for long commands by default
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        executable="/bin/bash",  # ensure bash semantics for 'set -euo pipefail'
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        combined_logger.info(line)
        task_logger.info(line)

    proc.wait()
    return int(proc.returncode or 0)


def run_task(
    keyword: str,
    kernel_name: str,
    log_dir: Path,
    combined_logger: logging.Logger,
    dry_run: bool,
    iters: int,
    warmup_runs: int,
    enable_trace: bool,
) -> int:
    if keyword not in COMMAND_SPECS:
        combined_logger.info(f"INTERNAL ERROR: unmapped keyword '{keyword}'")
        return 2

    ts = now_stamp()
    task_log_path = log_dir / f"{keyword}_{ts}.log"
    task_logger = open_task_logger(task_log_path)

    header = f"==> [{keyword}] Logging to {task_log_path}"
    combined_logger.info(header)
    task_logger.info(header)

    commands = COMMAND_SPECS[keyword].build(kernel_name, iters, warmup_runs)

    # Set up tracing if enabled
    traces_dir = None
    if enable_trace:
        traces_dir = ensure_traces_dir(log_dir)
        combined_logger.info(
            f"[{keyword}] Tracing enabled - RPD files will be saved to {traces_dir}"
        )
        task_logger.info(
            f"[{keyword}] Tracing enabled - RPD files will be saved to {traces_dir}"
        )

    rc = 0
    for i, c in enumerate(commands, start=1):
        combined_logger.info(f"[{keyword}] Command {i}/{len(commands)}:")
        task_logger.info(f"[{keyword}] Command {i}/{len(commands)}:")

        # Apply tracing if enabled
        command_to_execute = c
        if enable_trace and traces_dir:
            rpd_filename = traces_dir / f"{keyword}_{ts}_cmd{i}.rpd"
            command_to_execute = create_tracer_command(c.strip(), str(rpd_filename))
            combined_logger.info(
                f"[{keyword}] Command {i} will generate trace: {rpd_filename}"
            )
            task_logger.info(
                f"[{keyword}] Command {i} will generate trace: {rpd_filename}"
            )

        for ln in command_to_execute.rstrip("\n").splitlines():
            combined_logger.info(f"  {ln}")
            task_logger.info(f"  {ln}")

        rc = run_one_command(command_to_execute, task_logger, combined_logger, dry_run)
        if rc != 0:
            combined_logger.info(f"[{keyword}] Command {i} failed with rc={rc}")
            task_logger.info(f"[{keyword}] Command {i} failed with rc={rc}")
            break

    return rc


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run long benchmark commands for allowed keywords; logs per-task and combined."
    )
    p.add_argument(
        "kernel_name",
        nargs="?",
        help="Kernel name to inject into commands (required unless using --list).",
    )
    p.add_argument(
        "keys",
        nargs="*",
        help="Keywords to run (one or more) or 'all'.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List allowed keywords and exit.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    p.add_argument(
        "--log-dir",
        default=None,
        help="Directory for logs (overrides LOG_DIR env var).",
    )
    p.add_argument(
        "--iters",
        type=int,
        default=1,
        help="Number of iterations to run (default: 1).",
    )
    p.add_argument(
        "--warmup-runs",
        type=int,
        default=5,
        help="Number of warmup runs (default: 5).",
    )
    p.add_argument(
        "--trace",
        action="store_true",
        help="Enable tracing with runTracer.sh (generates RPD files).",
    )
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        print("\n".join(ALLOWED))
        return 0

    if not args.kernel_name:
        print(
            "Error: kernel_name is required (first positional argument).",
            file=sys.stderr,
        )
        return 2

    if not args.keys:
        print(
            "Error: no keywords provided. Use --list to see allowed keywords.",
            file=sys.stderr,
        )
        return 2

    keys = list(ALLOWED) if "all" in args.keys else args.keys
    keys = dedupe_preserve_order(keys)

    valid, invalid = validate_keywords(keys)
    if invalid:
        print(f"Error: invalid keyword(s): {' '.join(invalid)}", file=sys.stderr)
        print(f"Allowed: {' '.join(ALLOWED)}", file=sys.stderr)
        return 2

    log_dir = Path(args.log_dir) if args.log_dir else ensure_log_dir()
    run_id = now_stamp()
    combined_log_path = log_dir / f"run_{run_id}.log"
    combined_logger = init_combined_logger(combined_log_path)

    # Header
    for line in [
        f"Run ID: {run_id}",
        f"Kernel: {args.kernel_name}",
        f"Keywords: {' '.join(valid)}",
        f"Iterations: {args.iters}",
        f"Warmup runs: {args.warmup_runs}",
        f"Tracing: {'enabled' if args.trace else 'disabled'}",
        f"Logs dir: {log_dir}",
        f"Dry-run: {int(args.dry_run)}",
        f"Start: {dt.datetime.now().isoformat()}",
        "-" * 64,
    ]:
        combined_logger.info(line)

    failures: List[Tuple[str, int]] = []
    for k in valid:
        rc = run_task(
            keyword=k,
            kernel_name=args.kernel_name,
            log_dir=log_dir,
            combined_logger=combined_logger,
            dry_run=args.dry_run,
            iters=args.iters,
            warmup_runs=args.warmup_runs,
            enable_trace=args.trace,
        )
        if rc != 0:
            failures.append((k, rc))

    combined_logger.info("-" * 64)
    combined_logger.info(f"End: {dt.datetime.now().isoformat()}")

    # Process RPD files if tracing was enabled
    if args.trace:
        combined_logger.info("-" * 64)
        combined_logger.info("Processing RPD trace files...")
        traces_dir = log_dir / "traces"
        process_trace_files(traces_dir, args.kernel_name, combined_logger)
        combined_logger.info("RPD trace file processing completed.")

    if failures:
        msg = f"Completed with {len(failures)} failure(s): " + ", ".join(
            f"{k}[rc={rc}]" for k, rc in failures
        )
        print(msg, file=sys.stderr)
        combined_logger.info(msg)
        return 1

    ok = "All tasks completed successfully."
    print(ok)
    combined_logger.info(ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
