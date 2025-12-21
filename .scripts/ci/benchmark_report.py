#!/usr/bin/env python
"""Generate benchmark reports from pytest-benchmark JSON output."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def load_benchmarks(path: str | Path) -> dict:
    """Load benchmarks from JSON file."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {b["name"]: b["stats"] for b in data.get("benchmarks", [])}


def format_time(seconds: float) -> str:
    """Format time in human-readable units."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.2f}µs"
    if seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    return f"{seconds:.4f}s"


def generate_report(
    pr_path: str | Path,
    base_path: str | Path | None = None,
    pr_ref: str = "PR",
    base_ref: str = "base",
) -> str:
    """
    Generate markdown benchmark report.

    Parameters
    ----------
    pr_path
        Path to PR benchmark JSON file.
    base_path
        Path to base benchmark JSON file (optional, for comparison mode).
    pr_ref
        Name of PR branch/ref for display.
    base_ref
        Name of base branch/ref for display.

    Returns
    -------
    Markdown formatted report string.
    """
    pr_benchmarks = load_benchmarks(pr_path)

    if not pr_benchmarks:
        return "❌ No benchmark results found!"

    # Comparison mode
    if base_path:
        base_benchmarks = load_benchmarks(base_path)
        return _generate_comparison_report(pr_benchmarks, base_benchmarks, pr_ref, base_ref)

    # Simple report mode (no comparison)
    return _generate_simple_report(pr_benchmarks, pr_ref)


def _generate_simple_report(benchmarks: dict, ref: str) -> str:
    """Generate a simple benchmark report without comparison."""
    lines = [
        "## 📊 Benchmark Results\n",
        f"Results for `{ref}`\n",
        "| Benchmark | Mean | Std Dev | Min | Max | Rounds |",
        "|-----------|------|---------|-----|-----|--------|",
    ]

    for name, stats in sorted(benchmarks.items()):
        mean = format_time(stats["mean"])
        stddev = format_time(stats["stddev"])
        min_time = format_time(stats["min"])
        max_time = format_time(stats["max"])
        rounds = stats["rounds"]

        lines.append(f"| `{name}` | {mean} | ±{stddev} | {min_time} | {max_time} | {rounds} |")

    lines.extend(
        [
            "",
            "<details>",
            "<summary>📈 Raw Statistics</summary>",
            "",
            "```",
        ]
    )

    for name, stats in sorted(benchmarks.items()):
        lines.append(f"\n{name}:")
        lines.append(f"  mean:   {stats['mean']:.6f}s ± {stats['stddev']:.6f}s")
        lines.append(f"  min:    {stats['min']:.6f}s")
        lines.append(f"  max:    {stats['max']:.6f}s")
        lines.append(f"  rounds: {stats['rounds']}")
        if "iterations" in stats:
            lines.append(f"  iterations: {stats['iterations']}")

    lines.extend(
        [
            "```",
            "</details>",
        ]
    )

    return "\n".join(lines)


def _generate_comparison_report(
    pr_benchmarks: dict,
    base_benchmarks: dict,
    pr_ref: str,
    base_ref: str,
) -> str:
    """Generate a comparison benchmark report."""
    lines = [
        "## 📊 Benchmark Results\n",
        f"Comparing `{pr_ref}` against `{base_ref}`\n",
        "| Benchmark | PR (mean) | Base (mean) | Change |",
        "|-----------|-----------|-------------|--------|",
    ]

    for name, pr_stats in sorted(pr_benchmarks.items()):
        pr_mean = pr_stats["mean"]
        pr_str = format_time(pr_mean)

        if name in base_benchmarks:
            base_mean = base_benchmarks[name]["mean"]
            base_str = format_time(base_mean)
            change = ((pr_mean - base_mean) / base_mean) * 100

            if change > 10:
                change_str = f"🔴 +{change:.1f}%"
            elif change < -10:
                change_str = f"🟢 {change:.1f}%"
            else:
                change_str = f"⚪ {change:+.1f}%"
        else:
            base_str = "N/A"
            change_str = "🆕 New"

        lines.append(f"| `{name}` | {pr_str} | {base_str} | {change_str} |")

    # Check for removed benchmarks
    removed = set(base_benchmarks.keys()) - set(pr_benchmarks.keys())
    if removed:
        lines.append("")
        lines.append("**Removed benchmarks:** " + ", ".join(f"`{n}`" for n in sorted(removed)))

    lines.extend(
        [
            "",
            "<details>",
            "<summary>📈 Detailed Statistics</summary>",
            "",
            "```",
        ]
    )

    for name, stats in sorted(pr_benchmarks.items()):
        lines.append(f"\n{name}:")
        lines.append(f"  mean:   {stats['mean']:.6f}s ± {stats['stddev']:.6f}s")
        lines.append(f"  min:    {stats['min']:.6f}s")
        lines.append(f"  max:    {stats['max']:.6f}s")
        lines.append(f"  rounds: {stats['rounds']}")

    lines.extend(
        [
            "```",
            "</details>",
            "",
            "**Legend:** 🔴 >10% slower | 🟢 >10% faster | ⚪ within 10% | 🆕 new benchmark",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Generate benchmark reports")
    parser.add_argument(
        "pr_results",
        help="Path to PR benchmark JSON file",
    )
    parser.add_argument(
        "--base-results",
        help="Path to base benchmark JSON file (enables comparison mode)",
    )
    parser.add_argument(
        "--pr-ref",
        default=os.environ.get("GITHUB_HEAD_REF", "PR"),
        help="PR branch/ref name for display",
    )
    parser.add_argument(
        "--base-ref",
        default=os.environ.get("GITHUB_BASE_REF", "main"),
        help="Base branch/ref name for display",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--github-summary",
        action="store_true",
        help="Also write to GITHUB_STEP_SUMMARY if available",
    )

    args = parser.parse_args()

    report = generate_report(
        pr_path=args.pr_results,
        base_path=args.base_results,
        pr_ref=args.pr_ref,
        base_ref=args.base_ref,
    )

    # Output to file or stdout
    if args.output:
        Path(args.output).write_text(report)
        print(f"Report written to {args.output}")
    else:
        print(report)

    # Optionally write to GitHub step summary
    if args.github_summary and "GITHUB_STEP_SUMMARY" in os.environ:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(report)
            f.write("\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
