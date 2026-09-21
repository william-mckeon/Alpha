"""Catalog access for pinned MCPMark tasks; external-account services stay opt-in."""
from pathlib import Path


def load_task_catalog(tasks_dir: Path, *, service: str = "filesystem") -> list[str]:
    if service != "filesystem":
        raise ValueError("only isolated filesystem tasks are currently authorized")
    directory = tasks_dir / service
    tasks = []
    for meta in sorted(directory.glob("*/*/meta.json")):
        if not meta.with_name("verify.py").is_file() or not meta.with_name("description.md").is_file():
            raise ValueError(f"incomplete MCPMark task: {meta.parent}")
        tasks.append(f"{service}/{meta.parent.relative_to(directory).as_posix()}")
    if not tasks:
        raise ValueError("MCPMark filesystem catalog is empty")
    return tasks


def build_command(python: Path, *, task_id: str, model_alias: str, run_id: str, output_dir: Path, generation: dict) -> list[str]:
    """Official filesystem pipeline; routing/retry/isolation audits remain prerequisites."""
    if not task_id.startswith("filesystem/") or len(task_id.split("/")) != 3 or ".." in task_id.split("/"):
        raise ValueError("MCPMark execution is restricted to one filesystem/category/task")
    return [str(python), "-m", "pipeline", "--mcp", "filesystem", "--tasks", task_id.removeprefix("filesystem/"), "--models", model_alias, "--exp-name", run_id, "--k", "1", "--timeout", str(generation["wall_time_seconds"]), "--output-dir", str(output_dir)]
