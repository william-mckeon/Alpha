"""Process-local upstream adaptations; never change pinned vendor source files."""
from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
import time
import statistics
from urllib.parse import urlsplit


def isolated_worker_environment(parent) -> dict[str, str]:
    environment = dict(parent)
    for secret in ("OPENROUTER_API_KEY", "TAVILY_API_KEY", "ARCUS_PROXY_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ARCUS_WORKER_TOKEN", "LLM_API_KEY", "LLM_BASE_URL"):
        environment.pop(secret, None)
    environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHON_DOTENV_DISABLED="1")
    return environment


def singleton_stdev(values):
    """Reporting convention for one-attempt diagnostics; not a verifier change."""
    return 0.0 if len(values) == 1 else statistics.stdev(values)


def validate_gateway(url: str, token: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "host.docker.internal"} or parsed.path.rstrip("/") != "/v1" or parsed.username or parsed.password or parsed.query or parsed.fragment or not token:
        raise ValueError("worker adapter requires an authenticated local Arcus gateway")


def native_bfcl_handler(base_handler: type, *, gateway_url: str, proxy_token: str, generation: dict) -> type:
    """Preserve official compilation/parsing, replace only routing and retry boundary."""
    validate_gateway(gateway_url, proxy_token)
    class ArcusNativeHandler(base_handler):
        def _compile_tools(self, inference_data, test_entry):
            names = [f["name"].replace(".", "_") for f in test_entry["function"]]
            if len(names) != len(set(names)):
                raise ValueError("BFCL tool names collide after OpenAI normalization")
            return super()._compile_tools(inference_data, test_entry)

        def _build_client_kwargs(self):
            return {"base_url": gateway_url, "api_key": proxy_token, "max_retries": 0, "timeout": generation.get("trial_timeout_seconds", 9000)}

        def generate_with_backoff(self, **kwargs):
            # No inherited rate-limit decorator and no SDK retries. Gateway owns policy.
            kwargs.update(temperature=generation["temperature"], top_p=generation["top_p"], max_tokens=generation["max_output_tokens"])
            started = time.monotonic()
            result = self.client.chat.completions.create(**kwargs)
            return result, time.monotonic() - started
    return ArcusNativeHandler


def register_bfcl_route(mapping: dict, config_type: type, base_handler: type, *, candidate_id: str, gateway_url: str, proxy_token: str, generation: dict) -> str:
    if not candidate_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in candidate_id):
        raise ValueError("invalid candidate ID")
    alias = "arcus-" + candidate_id + "-FC"
    if alias in mapping:
        raise ValueError("BFCL Arcus route already registered")
    handler = native_bfcl_handler(base_handler, gateway_url=gateway_url, proxy_token=proxy_token, generation=generation)
    mapping[alias] = config_type(model_name=candidate_id, display_name=alias, url="https://openrouter.ai", org="Arcus evaluation adaptation", license="see candidate-specific license audit", model_handler=handler, is_fc_model=True, underscore_to_dot=True)
    return alias


def adapted_mcp_loop(source: str, generation: dict) -> tuple[ast.Module, dict]:
    """Auditable AST overlay for the two specific pinned MCPMark retry assignments."""
    tree = ast.parse(textwrap.dedent(source))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.AsyncFunctionDef) or tree.body[0].name != "_execute_litellm_tool_loop":
        raise ValueError("unexpected MCPMark loop signature")
    changed = {"failure_limit": 0, "completion_policy": 0}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        target = node.targets[0].id
        if target == "max_consecutive_failures":
            if not isinstance(node.value, ast.Constant) or node.value.value != 3:
                raise ValueError("upstream retry assignment changed; adaptation refused")
            node.value = ast.Constant(1)
            changed["failure_limit"] += 1
        elif target == "completion_kwargs":
            if not isinstance(node.value, ast.Dict):
                raise ValueError("upstream completion assignment changed")
            policy = {"num_retries": 0, "temperature": generation["temperature"], "top_p": generation["top_p"], "max_tokens": generation["max_output_tokens"]}
            if any(isinstance(key, ast.Constant) and key.value in policy for key in node.value.keys):
                raise ValueError("upstream already defines completion policy; adaptation refused")
            node.value.keys.extend(ast.Constant(key) for key in policy)
            node.value.values.extend(ast.Constant(value) for value in policy.values())
            changed["completion_policy"] += 1
    if changed != {"failure_limit": 1, "completion_policy": 1}:
        raise ValueError("expected exactly two MCPMark policy adaptations")
    ast.fix_missing_locations(tree)
    return tree, {"source_sha256": hashlib.sha256(source.encode()).hexdigest(), "overlay_sha256": hashlib.sha256(ast.unparse(tree).encode()).hexdigest(), "changes": changed}


def install_mcpmark_loop(agent_type: type, generation: dict) -> dict:
    original = agent_type._execute_litellm_tool_loop
    source = inspect.getsource(original)
    expected = "f2264f1af21cf6d616de38a87561b1b4c0f61539f369202305ee09b6dcab5736"
    if hashlib.sha256(source.encode()).hexdigest() != expected:
        raise ValueError("MCPMark loop does not match the audited pinned source")
    tree, evidence = adapted_mcp_loop(source, generation)
    namespace = dict(original.__globals__)
    exec(compile(tree, "<arcus-mcpmark-policy-overlay>", "exec"), namespace)
    agent_type._execute_litellm_tool_loop = namespace[original.__name__]
    agent_type.MAX_TURNS = generation["max_agent_iterations"]
    return evidence


def register_mcpmark_route(config_type: type, *, candidate_id: str) -> str:
    if not candidate_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in candidate_id):
        raise ValueError("invalid candidate ID")
    alias = "arcus-" + candidate_id
    if alias in config_type.MODEL_CONFIGS:
        raise ValueError("MCPMark Arcus route already registered")
    config_type.MODEL_CONFIGS[alias] = {"provider": "openai", "api_key_var": "ARCUS_WORKER_TOKEN", "base_url_var": "ARCUS_WORKER_BASE_URL", "litellm_input_model_name": "openai/" + candidate_id}
    return alias


def install_mcpmark_verifier_timeout(manager_type: type, timeout: int) -> dict:
    """Change only the pinned verifier deadline, preserving command/env/grading."""
    original = manager_type.run_verification
    source = inspect.getsource(original)
    if hashlib.sha256(source.encode()).hexdigest() != "62b0b0908f5bedeecdafd6ebbfb9549d239bf906c79a2833067b09be777c6254":
        raise ValueError("MCPMark verifier method differs from audited source")
    tree = ast.parse(textwrap.dedent(source))
    changed = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == "timeout":
                    if not isinstance(keyword.value, ast.Constant) or keyword.value.value != 300:
                        raise ValueError("upstream verifier timeout changed")
                    keyword.value = ast.Constant(timeout)
                    changed += 1
    if changed != 1 or type(timeout) is not int or timeout <= 0:
        raise ValueError("expected exactly one positive verifier deadline adaptation")
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    exec(compile(tree, "<arcus-mcpmark-verifier-deadline>", "exec"), namespace)
    manager_type.run_verification = namespace[original.__name__]
    return {"source_sha256": hashlib.sha256(source.encode()).hexdigest(), "timeout_seconds": timeout, "overlay_sha256": hashlib.sha256(ast.unparse(tree).encode()).hexdigest()}
