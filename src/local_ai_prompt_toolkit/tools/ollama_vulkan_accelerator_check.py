"""Local hardware accelerator auditor diagnosing Vulkan/CUDA drivers and Ollama GPU offloading status."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


def _detect_gpus() -> List[Dict[str, str]]:
    """Query system hardware for installed GPU controllers."""
    gpus: List[Dict[str, str]] = []

    # Windows PowerShell Get-CimInstance check
    if sys.platform == "win32":
        try:
            cmd = ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, AdapterRAM | ConvertTo-Json"]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if proc.returncode == 0 and proc.stdout.strip():
                data = json.loads(proc.stdout)
                if isinstance(data, dict):
                    data = [data]
                for item in data:
                    ram = item.get("AdapterRAM")
                    vram_mb = round(int(ram) / (1024 * 1024)) if ram else 0
                    gpus.append({
                        "name": str(item.get("Name", "Generic Display Adapter")),
                        "driver_version": str(item.get("DriverVersion", "Unknown")),
                        "vram_mb": str(vram_mb),
                    })
        except Exception:
            pass

    # Fallback to nvidia-smi if available
    if not gpus and shutil.which("nvidia-smi"):
        try:
            proc = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
            if proc.returncode == 0:
                for line in proc.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        gpus.append({
                            "name": parts[0],
                            "driver_version": parts[1],
                            "vram_mb": parts[2],
                        })
        except Exception:
            pass

    return gpus


def _check_vulkan_libraries() -> Dict[str, bool]:
    """Inspect system library paths for Vulkan runtime components."""
    vulkan_found = False
    search_paths = [
        os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "System32", "vulkan-1.dll"),
        os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "SysWOW64", "vulkan-1.dll"),
    ]
    for p in search_paths:
        if os.path.isfile(p):
            vulkan_found = True
            break

    return {
        "vulkan_runtime_present": vulkan_found,
        "ollama_igpu_enable_env": os.environ.get("OLLAMA_IGPU_ENABLE") == "1",
    }


def execute_ollama_vulkan_accelerator_check(
    host: str = "http://127.0.0.1:11434",
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Audit GPU accelerators, Vulkan runtime presence, and Ollama server status.

    Args:
        host: Base URL of local Ollama daemon.
        output_path: Optional destination to write JSON audit report.

    Returns:
        Structured dictionary reporting driver support and offloading readiness.
    """
    gpus = _detect_gpus()
    vulkan_state = _check_vulkan_libraries()

    daemon_online = False
    active_models: List[str] = []
    daemon_version = "unknown"

    try:
        req = urllib.request.Request(f"{host.rstrip('/')}/api/version")
        with urllib.request.urlopen(req, timeout=3) as resp:
            ver_data = json.loads(resp.read().decode("utf-8"))
            daemon_version = ver_data.get("version", "unknown")
            daemon_online = True

        req_ps = urllib.request.Request(f"{host.rstrip('/')}/api/ps")
        with urllib.request.urlopen(req_ps, timeout=3) as resp_ps:
            ps_data = json.loads(resp_ps.read().decode("utf-8"))
            active_models = [m.get("name", "") for m in ps_data.get("models", [])]
    except Exception:
        daemon_online = False

    has_discrete_gpu = any("nvidia" in g["name"].lower() or "rtx" in g["name"].lower() for g in gpus)
    has_amd_igpu = any("amd" in g["name"].lower() or "radeon" in g["name"].lower() for g in gpus)

    recommendations: List[str] = []
    if has_amd_igpu and not vulkan_state["ollama_igpu_enable_env"]:
        recommendations.append("Set environment variable OLLAMA_IGPU_ENABLE=1 to offload models to AMD Radeon unified memory.")
    if not vulkan_state["vulkan_runtime_present"]:
        recommendations.append("Install latest GPU display drivers containing official Vulkan 1.3 runtime libraries.")
    if not daemon_online:
        recommendations.append("Start local Ollama server via 'ollama serve' to enable LLM API inference.")

    result = {
        "status": "success",
        "daemon_online": daemon_online,
        "daemon_version": daemon_version,
        "gpus_detected": gpus,
        "gpu_count": len(gpus),
        "vulkan_runtime_detected": vulkan_state["vulkan_runtime_present"],
        "ollama_igpu_env_set": vulkan_state["ollama_igpu_enable_env"],
        "active_models_in_memory": active_models,
        "recommendations": recommendations,
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        with open(output_path, mode="w", encoding="utf-8") as out_f:
            json.dump(result, out_f, indent=2)

    return result


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Local hardware accelerator auditor diagnosing Vulkan/CUDA and Ollama offloading."
    )
    parser.add_argument("--host", default="http://127.0.0.1:11434", help="Ollama daemon URL")
    parser.add_argument("--output", "-o", help="Path to write JSON diagnostic report")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        res = execute_ollama_vulkan_accelerator_check(
            host=args.host,
            output_path=args.output,
        )
        print(f"Accelerators Detected: {res['gpu_count']} GPUs | Vulkan Runtime: {res['vulkan_runtime_detected']} | Ollama Online: {res['daemon_online']}")
        for g in res["gpus_detected"]:
            print(f"  - {g['name']} (VRAM: {g['vram_mb']} MB, Driver: {g['driver_version']})")
        for rec in res["recommendations"]:
            print(f"  [Recommendation] {rec}")
        return 0
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
