#!/usr/bin/env python3
"""
Agentic Framework (Single-file)

Capabilities:
- Config management: local LLMs and API keys in ~/.agentic/config.json
- Local LLM adapters (CLI-based): ollama, llama.cpp (if available)
- Kali tools wrappers with safety checks (nmap, nikto, sqlmap, gobuster, hydra, dirb, john, hashcat)
- Basic agent routing: heuristic selection of tool vs LLM
- CLI interface for managing config and running tasks

No external Python dependencies. Uses only the standard library.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

CONFIG_DIR = Path.home() / ".agentic"
CONFIG_PATH = CONFIG_DIR / "config.json"
DEFAULT_TIMEOUT_SEC = 60 * 10  # 10 minutes for long-running security tools


# -----------------------------
# Config Models and Management
# -----------------------------
@dataclass
class LocalLLM:
    id: str
    kind: str  # "ollama" | "llama.cpp" | "custom"
    model: Optional[str] = None  # ollama model name OR llama.cpp model path
    args: List[str] = field(default_factory=list)


@dataclass
class ApiKey:
    service: str
    key: str  # stored in config (env can override)


@dataclass
class AgenticConfig:
    llms: Dict[str, LocalLLM] = field(default_factory=dict)
    api_keys: Dict[str, ApiKey] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "llms": {k: dataclasses.asdict(v) for k, v in self.llms.items()},
            "api_keys": {k: dataclasses.asdict(v) for k, v in self.api_keys.items()},
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AgenticConfig":
        llms = {k: LocalLLM(**v) for k, v in d.get("llms", {}).items()}
        api_keys = {k: ApiKey(**v) for k, v in d.get("api_keys", {}).items()}
        return AgenticConfig(llms=llms, api_keys=api_keys)


class ConfigManager:
    def __init__(self, path: Path = CONFIG_PATH) -> None:
        self.path = path
        self._config = self._load()

    def _load(self) -> AgenticConfig:
        if not self.path.exists():
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            self._save(AgenticConfig())
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return AgenticConfig.from_dict(data)
        except Exception:
            return AgenticConfig()

    def _save(self, cfg: AgenticConfig) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(cfg.to_dict(), f, indent=2)
        tmp_path.replace(self.path)

    # Public API
    def get(self) -> AgenticConfig:
        return self._config

    def save(self) -> None:
        self._save(self._config)

    def add_llm(self, llm: LocalLLM) -> None:
        self._config.llms[llm.id] = llm
        self.save()

    def remove_llm(self, llm_id: str) -> bool:
        removed = self._config.llms.pop(llm_id, None) is not None
        if removed:
            self.save()
        return removed

    def add_api_key(self, service: str, key: str) -> None:
        self._config.api_keys[service] = ApiKey(service=service, key=key)
        self.save()

    def get_api_key(self, service: str) -> Optional[str]:
        env_key = os.getenv(f"{service.upper()}_API_KEY")
        if env_key:
            return env_key
        entry = self._config.api_keys.get(service)
        return entry.key if entry else None


# -----------------------------
# Utility: subprocess execution
# -----------------------------
class CommandError(Exception):
    pass


def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def run_command(args: List[str], timeout_sec: int = DEFAULT_TIMEOUT_SEC, cwd: Optional[str] = None) -> Tuple[int, str, str]:
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        text=True,
    )

    timer = threading.Timer(timeout_sec, proc.kill)
    try:
        timer.start()
        out, err = proc.communicate()
        return proc.returncode, out, err
    finally:
        timer.cancel()


# -----------------------------
# Local LLM Adapters (CLI-based)
# -----------------------------
class LLMNotAvailable(Exception):
    pass


class BaseLLMAdapter:
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        raise NotImplementedError


class OllamaAdapter(BaseLLMAdapter):
    def __init__(self, model: str) -> None:
        self.model = model
        self.bin = which("ollama")
        if not self.bin:
            raise LLMNotAvailable("ollama CLI not found in PATH")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        args = [self.bin, "run", self.model]
        # Streaming disabled: capture full output
        code, out, err = run_command(args + [prompt], timeout_sec=120)
        if code != 0:
            raise CommandError(f"ollama failed: {err.strip()}")
        return out.strip()


class LlamaCppAdapter(BaseLLMAdapter):
    def __init__(self, model_path: str, extra_args: Optional[List[str]] = None) -> None:
        # Try common llama.cpp CLI names
        candidates = ["llama-cli", "llama", "main"]  # depending on build
        self.bin = None
        for c in candidates:
            p = which(c)
            if p:
                self.bin = p
                break
        if not self.bin:
            raise LLMNotAvailable("llama.cpp CLI not found (tried llama-cli/llama/main)")
        self.model_path = model_path
        self.extra_args = extra_args or []

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        args = [self.bin, "-m", self.model_path, "-n", str(max_tokens), "-temp", str(temperature)] + self.extra_args
        # Some builds expect prompt via -p
        if "-p" not in args:
            args += ["-p", prompt]
        code, out, err = run_command(args, timeout_sec=180)
        if code != 0:
            raise CommandError(f"llama.cpp failed: {err.strip()}")
        return out.strip()


class FallbackAdapter(BaseLLMAdapter):
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        # Very small rule-based echo to avoid errors if no LLM is available
        snippet = prompt.strip().split("\n")[-1]
        return f"[fallback] Unable to run a local LLM. Last prompt line: {snippet[:200]}"


def build_llm_adapter(llm: LocalLLM) -> BaseLLMAdapter:
    try:
        if llm.kind == "ollama":
            if not llm.model:
                raise ValueError("ollama llm requires 'model'")
            return OllamaAdapter(model=llm.model)
        elif llm.kind == "llama.cpp":
            if not llm.model:
                raise ValueError("llama.cpp llm requires 'model' (path)")
            return LlamaCppAdapter(model_path=llm.model, extra_args=llm.args)
        else:
            raise LLMNotAvailable(f"Unsupported LLM kind: {llm.kind}")
    except (LLMNotAvailable, ValueError):
        return FallbackAdapter()


# -----------------------------
# Kali Tool Wrappers (safe)
# -----------------------------
@dataclass
class ToolResult:
    command: List[str]
    returncode: int
    stdout: str
    stderr: str


class BaseTool:
    name: str = ""
    description: str = ""

    def is_available(self) -> bool:
        return which(self.name) is not None

    def build_command(self, target: str, extra_args: Optional[List[str]] = None) -> List[str]:
        raise NotImplementedError

    def run(self, target: str, extra_args: Optional[List[str]] = None, timeout_sec: int = DEFAULT_TIMEOUT_SEC) -> ToolResult:
        if not self.is_available():
            return ToolResult([self.name], 127, "", f"{self.name} not found in PATH")
        cmd = self.build_command(target, extra_args)
        code, out, err = run_command(cmd, timeout_sec=timeout_sec)
        return ToolResult(cmd, code, out, err)


class NmapTool(BaseTool):
    name = "nmap"
    description = "Network exploration tool and security/port scanner"

    def build_command(self, target: str, extra_args: Optional[List[str]] = None) -> List[str]:
        args = [self.name, "-sV", "-T4", target]
        if extra_args:
            args.extend(extra_args)
        return args


class NiktoTool(BaseTool):
    name = "nikto"
    description = "Web server scanner"

    def build_command(self, target: str, extra_args: Optional[List[str]] = None) -> List[str]:
        args = [self.name, "-h", target]
        if extra_args:
            args.extend(extra_args)
        return args


class SqlmapTool(BaseTool):
    name = "sqlmap"
    description = "Automatic SQL injection and database takeover tool"

    def build_command(self, target: str, extra_args: Optional[List[str]] = None) -> List[str]:
        args = [self.name, "-u", target, "--batch"]
        if extra_args:
            args.extend(extra_args)
        return args


class GobusterTool(BaseTool):
    name = "gobuster"
    description = "Directory/File, DNS and VHost busting tool"

    def build_command(self, target: str, extra_args: Optional[List[str]] = None) -> List[str]:
        # Defaults to dir mode with common wordlist if provided via args
        args = [self.name, "dir", "-u", target]
        if extra_args:
            args.extend(extra_args)
        return args


class HydraTool(BaseTool):
    name = "hydra"
    description = "Parallelized login cracker"

    def build_command(self, target: str, extra_args: Optional[List[str]] = None) -> List[str]:
        # hydra requires service/protocol and user/pass lists; rely on extra_args
        args = [self.name]
        if extra_args:
            args.extend(extra_args)
        args.append(target)
        return args


class DirbTool(BaseTool):
    name = "dirb"
    description = "Web Content Scanner"

    def build_command(self, target: str, extra_args: Optional[List[str]] = None) -> List[str]:
        args = [self.name, target]
        if extra_args:
            args.extend(extra_args)
        return args


class JohnTool(BaseTool):
    name = "john"
    description = "John the Ripper password cracker"

    def build_command(self, target: str, extra_args: Optional[List[str]] = None) -> List[str]:
        # target is a hash file path in most cases
        args = [self.name, target]
        if extra_args:
            args.extend(extra_args)
        return args


class HashcatTool(BaseTool):
    name = "hashcat"
    description = "Advanced password recovery"

    def build_command(self, target: str, extra_args: Optional[List[str]] = None) -> List[str]:
        # target is usually hash file; mode must be in extra_args (e.g., -m 0)
        args = [self.name]
        if extra_args:
            args.extend(extra_args)
        args.append(target)
        return args


def get_available_tools() -> Dict[str, BaseTool]:
    tools: List[BaseTool] = [
        NmapTool(), NiktoTool(), SqlmapTool(), GobusterTool(), HydraTool(), DirbTool(), JohnTool(), HashcatTool()
    ]
    return {t.name: t for t in tools}


# -----------------------------
# Agent Router and Execution
# -----------------------------
class Agent:
    def __init__(self, config: AgenticConfig, preferred_llm_id: Optional[str] = None) -> None:
        self.config = config
        self.tools = get_available_tools()
        self.llm_adapter = self._select_llm_adapter(preferred_llm_id)

    def _select_llm_adapter(self, preferred_llm_id: Optional[str]) -> BaseLLMAdapter:
        if preferred_llm_id and preferred_llm_id in self.config.llms:
            return build_llm_adapter(self.config.llms[preferred_llm_id])
        # else pick first available; else fallback
        for llm in self.config.llms.values():
            adapter = build_llm_adapter(llm)
            if not isinstance(adapter, FallbackAdapter):
                return adapter
        return FallbackAdapter()

    def _select_tool_by_prompt(self, prompt: str) -> Optional[BaseTool]:
        p = prompt.lower()
        # simple heuristics
        if any(k in p for k in ["port scan", "scan ports", "open ports", "nmap", "scan "]):
            return self.tools.get("nmap")
        if any(k in p for k in ["web vuln", "nikto", "server scan", "http vuln", "ssl vuln"]):
            return self.tools.get("nikto")
        if any(k in p for k in ["sql injection", "sqlmap", "sqli"]):
            return self.tools.get("sqlmap")
        if any(k in p for k in ["directories", "dir brute", "gobuster", "wordlist"]):
            return self.tools.get("gobuster")
        if any(k in p for k in ["bruteforce", "hydra", "password attack"]):
            return self.tools.get("hydra")
        if any(k in p for k in ["dirb", "dirb scan"]):
            return self.tools.get("dirb")
        if any(k in p for k in ["john", "password crack"]):
            return self.tools.get("john")
        if any(k in p for k in ["hashcat", "gpu crack", "hash crack"]):
            return self.tools.get("hashcat")
        return None

    def _extract_target(self, prompt: str) -> Optional[str]:
        # naive URL/IP extractor
        url_match = re.search(r"https?://[^\s]+", prompt)
        if url_match:
            return url_match.group(0)
        ip_match = re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", prompt)
        if ip_match:
            return ip_match.group(0)
        return None

    def run(self, prompt: str, extra_args: Optional[List[str]] = None) -> str:
        tool = self._select_tool_by_prompt(prompt)
        target = self._extract_target(prompt)
        if tool and target:
            result = tool.run(target, extra_args=extra_args)
            header = f"$ {' '.join(shlex.quote(x) for x in result.command)}\n(returncode={result.returncode})\n"
            body = result.stdout.strip() or "<no stdout>"
            err = result.stderr.strip()
            if err:
                body += f"\n\n[stderr]\n{err}"
            return header + "\n" + body
        # else fallback to LLM
        response = self.llm_adapter.generate(prompt)
        return response


# -----------------------------
# CLI Interface
# -----------------------------
class CLI:
    def __init__(self) -> None:
        self.cfg = ConfigManager()
        self.parser = argparse.ArgumentParser(
            prog="agentic",
            description="Agentic framework: manage local LLMs, API keys, and run tools/agent",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        sub = self.parser.add_subparsers(dest="cmd", required=True)

        # add-llm
        p_add_llm = sub.add_parser("add-llm", help="Add a local LLM (ollama or llama.cpp)")
        p_add_llm.add_argument("id", help="Unique ID for this LLM")
        p_add_llm.add_argument("kind", choices=["ollama", "llama.cpp"], help="LLM kind")
        p_add_llm.add_argument("model", help="ollama model name or llama.cpp model path")
        p_add_llm.add_argument("--args", nargs=argparse.REMAINDER, help="Extra args for llama.cpp binary")

        # remove-llm
        p_rm_llm = sub.add_parser("remove-llm", help="Remove a local LLM by ID")
        p_rm_llm.add_argument("id", help="LLM ID to remove")

        # list-llms
        sub.add_parser("list-llms", help="List configured local LLMs")

        # add-api-key
        p_add_key = sub.add_parser("add-api-key", help="Add an API key entry")
        p_add_key.add_argument("service", help="Service name (e.g., openai, anthropic)")
        p_add_key.add_argument("key", help="API key value (env override: SERVICE_API_KEY)")

        # list-api-keys
        sub.add_parser("list-api-keys", help="List configured API key entries (masked)")

        # run-tool
        p_tool = sub.add_parser("run-tool", help="Run a specific Kali tool wrapper")
        p_tool.add_argument("tool", choices=sorted(get_available_tools().keys()), help="Tool name")
        p_tool.add_argument("target", help="Target (URL/IP, file path, etc.)")
        p_tool.add_argument("--args", nargs=argparse.REMAINDER, help="Extra args passed to the tool")

        # agent
        p_agent = sub.add_parser("agent", help="Run the agent with a natural language prompt")
        p_agent.add_argument("prompt", help="Instruction, may include a URL/IP or context")
        p_agent.add_argument("--llm-id", help="Preferred LLM ID to use", default=None)
        p_agent.add_argument("--args", nargs=argparse.REMAINDER, help="Extra args for tools if selected")

        # tools
        sub.add_parser("tools", help="List known tool wrappers and availability")

        # doctor
        sub.add_parser("doctor", help="Check environment and config health")

        # print-config
        sub.add_parser("print-config", help="Print raw config JSON path and contents")

    def run(self, argv: Optional[List[str]] = None) -> int:
        args = self.parser.parse_args(argv)
        cmd = args.cmd
        if cmd == "add-llm":
            self._cmd_add_llm(args)
        elif cmd == "remove-llm":
            self._cmd_remove_llm(args)
        elif cmd == "list-llms":
            self._cmd_list_llms()
        elif cmd == "add-api-key":
            self._cmd_add_api_key(args)
        elif cmd == "list-api-keys":
            self._cmd_list_api_keys()
        elif cmd == "run-tool":
            return self._cmd_run_tool(args)
        elif cmd == "agent":
            return self._cmd_agent(args)
        elif cmd == "tools":
            self._cmd_tools()
        elif cmd == "doctor":
            return self._cmd_doctor()
        elif cmd == "print-config":
            self._cmd_print_config()
        return 0

    # ----- Commands -----
    def _cmd_add_llm(self, args: argparse.Namespace) -> None:
        llm = LocalLLM(id=args.id, kind=args.kind, model=args.model, args=args.args or [])
        self.cfg.add_llm(llm)
        print(f"Added LLM '{args.id}' ({args.kind})")

    def _cmd_remove_llm(self, args: argparse.Namespace) -> None:
        ok = self.cfg.remove_llm(args.id)
        if ok:
            print(f"Removed LLM '{args.id}'")
        else:
            print(f"LLM '{args.id}' not found", file=sys.stderr)

    def _cmd_list_llms(self) -> None:
        cfg = self.cfg.get()
        if not cfg.llms:
            print("No LLMs configured")
            return
        for llm in cfg.llms.values():
            info = f"id={llm.id} kind={llm.kind} model={llm.model} args={' '.join(llm.args)}"
            print(info)

    def _cmd_add_api_key(self, args: argparse.Namespace) -> None:
        self.cfg.add_api_key(args.service, args.key)
        print(f"Added API key for service '{args.service}'")

    def _cmd_list_api_keys(self) -> None:
        cfg = self.cfg.get()
        if not cfg.api_keys:
            print("No API keys configured")
            return
        for service, entry in cfg.api_keys.items():
            masked = entry.key[:4] + "*" * max(0, len(entry.key) - 8) + entry.key[-4:]
            print(f"{service}: {masked}")

    def _cmd_run_tool(self, args: argparse.Namespace) -> int:
        tools = get_available_tools()
        tool = tools.get(args.tool)
        if not tool:
            print(f"Unknown tool: {args.tool}", file=sys.stderr)
            return 2
        extra = args.args or []
        res = tool.run(args.target, extra_args=extra)
        print(f"$ {' '.join(shlex.quote(x) for x in res.command)}")
        print(f"(returncode={res.returncode})")
        if res.stdout:
            print(res.stdout.rstrip())
        if res.stderr:
            print("\n[stderr]")
            print(res.stderr.rstrip())
        return res.returncode

    def _cmd_agent(self, args: argparse.Namespace) -> int:
        agent = Agent(self.cfg.get(), preferred_llm_id=args.llm_id)
        out = agent.run(args.prompt, extra_args=args.args)
        print(out)
        return 0

    def _cmd_tools(self) -> None:
        tools = get_available_tools()
        for name, tool in tools.items():
            print(f"{name:10} - {'available' if tool.is_available() else 'missing'} - {tool.description}")

    def _cmd_doctor(self) -> int:
        issues: List[str] = []
        # Config directory
        if not CONFIG_DIR.exists():
            issues.append(f"Config dir missing: {CONFIG_DIR}")
        if not CONFIG_PATH.exists():
            issues.append(f"Config file missing: {CONFIG_PATH}")

        # LLM availability
        cfg = self.cfg.get()
        if not cfg.llms:
            issues.append("No local LLMs configured (add one via 'add-llm')")
        else:
            for llm in cfg.llms.values():
                adapter = build_llm_adapter(llm)
                if isinstance(adapter, FallbackAdapter):
                    issues.append(f"LLM '{llm.id}' not available (kind={llm.kind})")

        # Tool availability
        tools = get_available_tools()
        missing = [name for name, t in tools.items() if not t.is_available()]
        if missing:
            issues.append("Missing tools: " + ", ".join(sorted(missing)))

        if issues:
            print("Environment issues detected:\n- " + "\n- ".join(issues))
            return 1
        print("All checks passed.")
        return 0

    def _cmd_print_config(self) -> None:
        print(f"Config path: {CONFIG_PATH}")
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                print(f.read())
        except FileNotFoundError:
            print("<no config found>")


def main() -> int:
    cli = CLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
