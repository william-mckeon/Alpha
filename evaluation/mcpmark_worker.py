"""One frozen filesystem task in a disposable container; official MCPMark grading."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess

from evaluation.fixtures import extract_snapshot
from evaluation.worker_adapters import install_mcpmark_loop, install_mcpmark_verifier_timeout, isolated_worker_environment, register_mcpmark_route, validate_gateway


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    args = parser.parse_args()
    # This entry point cannot launch tools against a user's native filesystem.
    work = Path("/work")
    if os.name != "posix" or not Path("/.dockerenv").exists() or not work.is_dir():
        raise ValueError("MCP worker requires a disposable container with /work")
    suites = json.loads((ROOT / "evaluation/suites.json").read_text())["suites"]
    frozen = set(suites["smoke"]["mcp"]["task_ids"]) | set(suites["qualification"]["mcp"]["task_ids"])
    if args.task not in frozen or len(args.task.split("/")) != 3 or not args.task.startswith("filesystem/"):
        raise ValueError("MCP worker requires one frozen filesystem task")
    _, category, task_id = args.task.split("/")
    vendor = ROOT / "evaluation/vendor/mcpmark"
    expected = json.loads((ROOT / "evaluation/harnesses/mcpmark.json").read_text())["revision"]
    git = ["git", "-c", "safe.directory=" + str(vendor), "-C", str(vendor)]
    if subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() != expected or subprocess.run([*git, "diff", "--quiet", "HEAD"]).returncode:
        raise ValueError("MCPMark source differs from its clean pin")
    validate_gateway(os.environ["ARCUS_WORKER_BASE_URL"], os.environ["ARCUS_WORKER_TOKEN"])
    output = work / "results"
    if output.exists():
        raise ValueError("MCP trial artifacts cannot be reused or resumed")
    evidence = extract_snapshot(args.archive, work / "fixtures", expected_sha256=args.archive_sha256)
    fixture = work / "fixtures" / category
    if not fixture.is_dir():
        raise ValueError("snapshot does not contain the exact task category directory")
    os.environ["FILESYSTEM_TEST_ROOT"] = str(work / "fixtures")
    os.environ["TMPDIR"] = str(work)
    import tempfile
    tempfile.tempdir = str(work)
    from src.factory import ServiceRegistry
    from src.model_config import ModelConfig
    from src.agents.mcpmark_agent import MCPMarkAgent, MCPStdioServer
    from src.evaluator import MCPEvaluator
    components = ServiceRegistry.get_components("filesystem")

    class FrozenState(components.state_manager_class):
        def _get_project_root(self):
            # Upstream places backups under project root; keep all mutations in /work.
            return work

        def __init__(self, **kwargs):
            super().__init__(test_root=fixture, cleanup_on_exit=False)

        def _set_dynamic_test_root(self, task):
            if task.category_id != category or str(task.task_id) != task_id:
                raise ValueError("fixture task identity mismatch")
            self.test_root = fixture

        def _download_and_extract_test_environment(self):
            raise ValueError("mutable fixture downloads are forbidden")

    ServiceRegistry._components_cache["filesystem"] = replace(components, state_manager_class=FrozenState)
    protocol = json.loads((ROOT / "evaluation/protocol.json").read_text())
    generation = protocol["generation"]
    from evaluation.execution import source_fingerprint
    evidence["protocol_snapshot"] = protocol
    evidence["execution_fingerprint"] = source_fingerprint(ROOT)
    evidence["loop_overlay"] = install_mcpmark_loop(MCPMarkAgent, generation)
    evidence["verifier_overlay"] = install_mcpmark_verifier_timeout(components.task_manager_class, generation["verifier_timeout_seconds"])
    alias = register_mcpmark_route(ModelConfig, candidate_id=args.candidate)
    server_entry = ROOT / "evaluation/node-filesystem/node_modules/@modelcontextprotocol/server-filesystem/dist/index.js"
    if not server_entry.is_file():
        raise ValueError("pinned filesystem MCP server is not installed")

    def pinned_stdio(agent):
        directory = Path(agent.service_config["test_directory"]).resolve()
        if not directory.is_relative_to(work) or agent.mcp_service != "filesystem":
            raise ValueError("filesystem MCP authority escaped the isolated workspace")
        server = MCPStdioServer(command="node", args=[str(server_entry), str(directory)], timeout=45)
        # Upstream merges os.environ; replace params.env rather than overlaying it.
        server.params.env = isolated_worker_environment(os.environ)
        return server

    MCPMarkAgent._create_stdio_server = pinned_stdio
    evaluator = MCPEvaluator("filesystem", alias, timeout=generation["wall_time_seconds"], exp_name="attempt", output_dir=output)
    original_execute = evaluator.agent.execute_sync

    def completed_execute(*positional, **kwargs):
        result = original_execute(*positional, **kwargs)
        (work / "model-completed.json").write_text('{"model_completed": true}\n')
        return result

    evaluator.agent.execute_sync = completed_execute
    selected = evaluator.task_manager.filter_tasks(category + "/" + task_id)
    if len(selected) != 1 or selected[0].category_id != category or str(selected[0].task_id) != task_id:
        raise ValueError("official task filter did not resolve the exact frozen task")
    (work / "worker-provenance.json").write_text(json.dumps(evidence, indent=2) + "\n")
    evaluator.run_evaluation(category + "/" + task_id)


if __name__ == "__main__":
    main()
