"""Runtime and accelerator diagnostics. PyTorch is optional for Phase 1 checks."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any


def inspect_runtime() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "working_directory": str(Path.cwd()),
        "pytorch_installed": False,
        "cuda_available": False,
        "gpu_count": 0,
        "gpus": [],
    }

    try:
        import torch  # type: ignore[import-not-found]

        report["pytorch_installed"] = True
        report["pytorch_version"] = torch.__version__
        report["cuda_build"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if report["cuda_available"]:
            report["gpu_count"] = torch.cuda.device_count()
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                report["gpus"].append(
                    {
                        "index": index,
                        "name": props.name,
                        "total_memory_gib": round(props.total_memory / (1024**3), 2),
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                )
    except ImportError:
        report["pytorch_note"] = "PyTorch is not installed; install a build matching the cloud GPU."
    except (RuntimeError, OSError) as exc:
        report["pytorch_error"] = f"{type(exc).__name__}: {exc}"

    return report


def print_runtime_report(report: dict[str, Any]) -> None:
    print("=" * 68)
    print("VYREX AI — RUNTIME DIAGNOSTICS")
    print("=" * 68)
    for key, value in report.items():
        if key == "gpus":
            print("GPUS:")
            if not value:
                print("  (none detected)")
            for gpu in value:
                print(
                    f"  [{gpu['index']}] {gpu['name']} | "
                    f"{gpu['total_memory_gib']} GiB | CC {gpu['compute_capability']}"
                )
        else:
            print(f"{key.replace('_', ' ').upper()}: {value}")
    print("=" * 68)
