"""Shared launch helpers for deploy_policy Python controller nodes.

The policy controllers are installed as executable Python scripts. Their
``#!/usr/bin/env python3`` shebang normally resolves through ``PATH``, which is
unsafe when an Isaac/Conda shell is active: ROS 2 Humble's apt ``rclpy`` C
extension is built for CPython 3.10, while Isaac Lab environments are commonly
Python 3.11. These helpers validate and explicitly prefix the controller
interpreter before the node process starts, producing an actionable launch error
instead of a low-level ``_rclpy_pybind11`` import failure.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import textwrap
from typing import Iterable, Mapping, Sequence

from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PYTHON_EXECUTABLE_ARGUMENT = "python_executable"
_PROBE_TIMEOUT_SEC = 30


def declare_python_executable_argument() -> DeclareLaunchArgument:
    """Declare the common Python interpreter override for controller launches."""

    return DeclareLaunchArgument(
        PYTHON_EXECUTABLE_ARGUMENT,
        default_value="",
        description=(
            "Python interpreter used to run deploy_policy Python nodes. "
            "Leave empty to use PATH's python3. For apt ROS 2 Humble policy "
            "controllers, use a Python 3.10 interpreter that can import both "
            "rclpy and torch."
        ),
    )


def python_node_with_preflight(
    *,
    package: str,
    executable: str,
    name: str,
    parameters: Sequence[object],
    required_modules: Sequence[str],
    output: str = "screen",
    arguments: Sequence[object] | None = None,
) -> OpaqueFunction:
    """Return an action that validates the runtime and launches a Python node.

    ``launch_ros.actions.Node`` executes installed scripts directly, so a script
    shebang can silently select a Conda interpreter. This wrapper resolves the
    requested interpreter, verifies that required imports work in that exact
    process environment, then uses the interpreter as a command prefix.
    """

    def _launch_setup(context, *_, **__):
        python_executable = _resolve_python_executable(context)
        probe = probe_python_runtime(python_executable, required_modules)
        failures = _failed_modules(probe, required_modules)
        if failures:
            raise RuntimeError(format_preflight_error(probe, failures))

        return [
            Node(
                package=package,
                executable=executable,
                name=name,
                output=output,
                parameters=list(parameters),
                arguments=list(arguments or []),
                prefix=f"{shlex.quote(probe.get('executable') or python_executable)} ",
            )
        ]

    return OpaqueFunction(function=_launch_setup)


def _resolve_python_executable(context) -> str:
    configured = LaunchConfiguration(PYTHON_EXECUTABLE_ARGUMENT).perform(context).strip()
    if configured:
        return configured
    return shutil.which("python3") or "python3"


def probe_python_runtime(python_executable: str, required_modules: Iterable[str]) -> dict:
    """Probe imports in ``python_executable`` and return a JSON-serializable dict."""

    modules = list(dict.fromkeys(required_modules))
    probe_code = r'''
import importlib
import importlib.util
import json
import sys
import traceback

modules = {}
for module_name in sys.argv[1:]:
    record = {"ok": False, "spec": None, "error": None, "traceback_tail": None}
    try:
        spec = importlib.util.find_spec(module_name)
        record["spec"] = getattr(spec, "origin", None) if spec else None
        importlib.import_module(module_name)
        record["ok"] = True
    except BaseException as exc:  # import-time native-library failures can be broad
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback_tail"] = traceback.format_exc(limit=6)
    modules[module_name] = record

print(json.dumps({
    "ok": all(record["ok"] for record in modules.values()),
    "executable": sys.executable,
    "version": sys.version.split()[0],
    "version_info": list(sys.version_info[:3]),
    "prefix": sys.prefix,
    "modules": modules,
}, ensure_ascii=False))
'''

    try:
        result = subprocess.run(
            [python_executable, "-c", probe_code, *modules],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_PROBE_TIMEOUT_SEC,
        )
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "executable": python_executable,
            "version": None,
            "modules": {module: {"ok": False, "error": "not probed"} for module in modules},
            "probe_error": f"Python executable not found: {exc.filename}",
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "executable": python_executable,
            "version": None,
            "modules": {module: {"ok": False, "error": "probe timed out"} for module in modules},
            "probe_error": f"Runtime probe exceeded {_PROBE_TIMEOUT_SEC}s",
        }

    stdout = result.stdout.strip()
    if result.returncode != 0 or not stdout:
        return {
            "ok": False,
            "executable": python_executable,
            "version": None,
            "modules": {module: {"ok": False, "error": "not probed"} for module in modules},
            "probe_error": (result.stderr or stdout or f"probe exited {result.returncode}").strip(),
        }

    try:
        return json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "executable": python_executable,
            "version": None,
            "modules": {module: {"ok": False, "error": "not probed"} for module in modules},
            "probe_error": f"Could not parse runtime probe JSON: {exc}: {stdout[-500:]}",
        }


def _failed_modules(probe: Mapping[str, object], required_modules: Iterable[str]) -> dict[str, Mapping[str, object]]:
    module_results = probe.get("modules") or {}
    failures = {}
    for module_name in required_modules:
        result = module_results.get(module_name) if isinstance(module_results, Mapping) else None
        if not isinstance(result, Mapping) or not result.get("ok"):
            failures[module_name] = result or {"ok": False, "error": "missing probe result"}
    return failures


def format_preflight_error(probe: Mapping[str, object], failures: Mapping[str, Mapping[str, object]]) -> str:
    """Build a concise, actionable launch error for runtime import failures."""

    executable = probe.get("executable") or "<unknown>"
    version = probe.get("version") or "<unknown>"
    prefix = probe.get("prefix") or "<unknown>"
    lines = [
        "deploy_policy Python runtime preflight failed.",
        f"Interpreter: {executable} (Python {version}, prefix {prefix})",
    ]
    if probe.get("probe_error"):
        lines.append(f"Probe error: {probe['probe_error']}")

    lines.append("Failed required imports:")
    for module_name, result in failures.items():
        error = result.get("error") or "unknown import failure"
        spec = result.get("spec")
        spec_text = f"; spec={spec}" if spec else ""
        lines.append(f"- {module_name}: {error}{spec_text}")

    all_error_text = "\n".join(str(result.get("error") or "") for result in failures.values())
    if "_rclpy_pybind11" in all_error_text:
        lines.append("")
        lines.append(
            "Detected ROS 2 rclpy C-extension ABI mismatch. apt ROS 2 Humble on Ubuntu 22.04 "
            "uses CPython 3.10; Conda/Isaac Python 3.11 cannot load that extension."
        )

    lines.append("")
    lines.append("Use one Python runtime that can import all required modules.")
    lines.append("Recommended for this repository:")
    lines.extend(
        textwrap.dedent(
            """
            1. Do not launch deploy_policy from env_isaaclab.
            2. Create/use a Python 3.10 ROS policy runtime with torch installed, for example:
               cd /path/to/RC2026_SIM/ros2_ws
               python3.10 -m venv --system-site-packages .venv-ros2-policy
               source /opt/ros/humble/setup.zsh
               source .venv-ros2-policy/bin/activate
               python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
               source install/setup.zsh
            3. Relaunch with:
               ros2 launch deploy_policy go2w_controller.launch.py use_sim_time:=true \\
                 python_executable:=$PWD/.venv-ros2-policy/bin/python3
            """
        ).strip().splitlines()
    )
    return "\n".join(lines)
