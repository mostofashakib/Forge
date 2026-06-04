from __future__ import annotations
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from forge.benchmark.env_quality import EnvQualityMetrics

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    output_dir: Path


class BenchmarkReport:
    def __init__(self, config: ReportConfig) -> None:
        self._cfg = config
        self._cfg.output_dir.mkdir(parents=True, exist_ok=True)
        (self._cfg.output_dir / "figures").mkdir(exist_ok=True)

    def write_env_quality(self, metrics: list[EnvQualityMetrics]) -> None:
        out = self._cfg.output_dir

        # summary.json
        summary = {
            "env_quality": [
                {
                    "env_name": m.env_name,
                    "state_coverage_score": m.state_coverage_score,
                    "reward_density": m.reward_density,
                    "dead_end_rate": m.dead_end_rate,
                    "action_diversity": m.action_diversity,
                    "num_episodes": m.num_episodes,
                    "num_steps": m.num_steps,
                }
                for m in metrics
            ]
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))

        # env_quality.csv
        csv_path = out / "env_quality.csv"
        fieldnames = [
            "env_name", "state_coverage_score", "reward_density",
            "dead_end_rate", "action_diversity", "num_episodes", "num_steps",
        ]
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in metrics:
                writer.writerow({
                    "env_name": m.env_name,
                    "state_coverage_score": round(m.state_coverage_score, 4),
                    "reward_density": round(m.reward_density, 4),
                    "dead_end_rate": round(m.dead_end_rate, 4),
                    "action_diversity": round(m.action_diversity, 4),
                    "num_episodes": m.num_episodes,
                    "num_steps": m.num_steps,
                })

        self._write_quality_figure(metrics)

    def _write_quality_figure(self, metrics: list[EnvQualityMetrics]) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            env_names = [m.env_name for m in metrics]
            x = np.arange(len(env_names))
            width = 0.2

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(x - 1.5 * width, [m.state_coverage_score for m in metrics], width, label="Coverage")
            ax.bar(x - 0.5 * width, [m.reward_density for m in metrics], width, label="Reward density")
            ax.bar(x + 0.5 * width, [1 - m.dead_end_rate for m in metrics], width, label="1 - Dead-end rate")
            ax.bar(x + 1.5 * width, [m.action_diversity for m in metrics], width, label="Action diversity")

            ax.set_xticks(x)
            ax.set_xticklabels(env_names, rotation=30, ha="right")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Score")
            ax.set_title("Environment Quality Metrics")
            ax.legend()
            fig.tight_layout()
            fig.savefig(str(self._cfg.output_dir / "figures" / "env_quality.pdf"))
            plt.close(fig)
        except ImportError:
            logger.warning("[report] matplotlib not installed — skipping figure generation")
        except Exception as exc:
            logger.warning("[report] figure generation failed: %s", exc)
