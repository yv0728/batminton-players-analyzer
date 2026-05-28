from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    data_dir: Path
    artifacts_dir: Path
    reports_dir: Path

    @staticmethod
    def from_root(project_root: Path) -> "ProjectPaths":
        return ProjectPaths(
            project_root=project_root,
            data_dir=project_root / "data",
            artifacts_dir=project_root / "artifacts",
            reports_dir=project_root / "reports",
        )


SKILL_LABELS = ["beginner", "intermediate", "pro"]
