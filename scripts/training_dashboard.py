import csv
import re
import time
import numpy as np
import torch
from collections import deque
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "oanc_production_v1"
GPU_FILE = LOG_DIR / "gpu_metrics.csv"
TRAIN_FILE = LOG_DIR / "training_metrics.csv"
CHECKPOINT_FILE = ROOT / "checkpoints" / "oanc_production_v1" / "latest.pt"
_checkpoint_cache = {"key": None, "value": None}
CHECKPOINT_HISTORY = {"step": None, "time": None, "rate": None}

MAX_GPU_POINTS = 1800
REFRESH_MS = 2000
MAX_STEPS = 10000

BG = "#080D19"
PANEL = "#101827"
CYAN = "#22D3EE"
PINK = "#F472B6"
VIOLET = "#A78BFA"
GREEN = "#34D399"
AMBER = "#FBBF24"
WHITE = "#E5EDF7"
MUTED = "#91A1B7"
GRID = "#263449"


def number(value):
    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    return float(match.group()) if match else None



def read_checkpoint():
    """Read checkpoint metadata safely and only when the file changes."""
    try:
        stat = CHECKPOINT_FILE.stat()
        key = (stat.st_mtime_ns, stat.st_size)

        if _checkpoint_cache["key"] == key:
            return _checkpoint_cache["value"]

        allowed = [
            np._core.multiarray._reconstruct,
            np.ndarray,
            np.dtype,
            np.dtypes.UInt32DType,
        ]

        with torch.serialization.safe_globals(allowed):
            state = torch.load(
                CHECKPOINT_FILE,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )

        result = {
            "step": int(state.get("step", 0)),
            "best_validation_loss": state.get("best_validation_loss"),
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%H:%M:%S"),
        }

        _checkpoint_cache["key"] = key
        _checkpoint_cache["value"] = result
        return result

    except FileNotFoundError:
        return None
    except Exception as exc:
        print(f"Checkpoint monitor warning: {exc}")
        return _checkpoint_cache["value"]

def read_training():
    records = {}

    if not TRAIN_FILE.exists():
        return []

    try:
        with TRAIN_FILE.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    step = int(row["step"])
                    record = records.setdefault(step, {})
                    for field in ("train_loss", "lr", "validation_loss"):
                        value = row.get(field)
                        if value not in (None, ""):
                            record[field] = float(value)
                except (ValueError, KeyError, TypeError):
                    continue
    except OSError:
        pass

    return [(step, records[step]) for step in sorted(records)]


def read_gpu():
    data = deque(maxlen=MAX_GPU_POINTS)

    if not GPU_FILE.exists():
        return data

    try:
        with GPU_FILE.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.reader(f):
                if len(row) < 6:
                    continue

                try:
                    stamp = row[0].strip()
                    timestamp = None

                    for fmt in ("%Y/%m/%d %H:%M:%S.%f",
                                "%Y/%m/%d %H:%M:%S"):
                        try:
                            timestamp = datetime.strptime(stamp, fmt)
                            break
                        except ValueError:
                            continue

                    values = [number(v) for v in row[1:6]]

                    if timestamp and all(v is not None for v in values):
                        data.append((timestamp, *values))
                except (ValueError, IndexError):
                    continue
    except OSError:
        pass

    return data


def rolling_mean(values, window=15):
    if not values:
        return []

    result = []
    running = 0.0
    queue = deque()

    for value in values:
        queue.append(value)
        running += value

        if len(queue) > window:
            running -= queue.popleft()

        result.append(running / len(queue))

    return result


def neon(ax, x, y, color, label, smooth=False, raw_label="Raw telemetry"):
    if len(x) < 1:
        return

    if smooth and len(y) > 2:
        ax.plot(x, y, color=color, alpha=0.20,
                linewidth=0.8, label=raw_label)
        y = rolling_mean(y)

    line, = ax.plot(
        x, y, color=color, linewidth=2.0,
        label=label, zorder=3
    )

    line.set_path_effects([
        pe.Stroke(linewidth=7, foreground=color, alpha=0.12),
        pe.Stroke(linewidth=4, foreground=color, alpha=0.22),
        pe.Normal()
    ])

    if len(x) > 1:
        ax.fill_between(x, y, min(y), color=color, alpha=0.045)


def style_axis(ax, title, xlabel, ylabel):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=WHITE, fontsize=10, pad=10)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, alpha=0.65, linewidth=0.65)

    for spine in ax.spines.values():
        spine.set_color(GRID)

    ax.set_axisbelow(True)


def add_legend(ax):
    handles, labels = ax.get_legend_handles_labels()

    if handles:
        legend = ax.legend(
            loc="best", fontsize=7, framealpha=0.75,
            facecolor="#172337", edgecolor=GRID
        )
        for item in legend.get_texts():
            item.set_color(WHITE)


plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": GRID,
    "text.color": WHITE,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.family": "DejaVu Sans",
})

fig, axes = plt.subplots(3, 2, figsize=(16, 10))
fig.canvas.manager.set_window_title("VYREX AI | Live Production Monitor")

ax_loss, ax_lr, ax_util, ax_temp, ax_memory, ax_power = axes.flat

progress_ax = fig.add_axes([0.08, 0.905, 0.84, 0.014])
progress_ax.set_xlim(0, MAX_STEPS)
progress_ax.set_ylim(-0.6, 0.6)
progress_ax.axis("off")

fig.suptitle(
    "VYREX AI  |  PRODUCTION TRAINING SYSTEM",
    fontsize=15, fontweight="bold", color=CYAN, y=0.975
)

status = fig.text(
    0.5, 0.943, "Initializing monitor...",
    ha="center", fontsize=9, color=MUTED
)

footer = fig.text(
    0.5, 0.018, "OANC PRODUCTION V1  |  LIVE TELEMETRY",
    ha="center", fontsize=8, color=VIOLET
)


def update(_):
    records = read_training()
    checkpoint = read_checkpoint()
    gpu = list(read_gpu())

    for ax in axes.flat:
        ax.clear()

    steps = [step for step, row in records if "train_loss" in row]
    losses = [row["train_loss"] for step, row in records if "train_loss" in row]

    rates = [(step, row["lr"]) for step, row in records if "lr" in row]
    validations = [
        (step, row["validation_loss"])
        for step, row in records if "validation_loss" in row
    ]

    csv_latest = max([step for step, row in records] or [0])
    checkpoint_step = checkpoint["step"] if checkpoint else 0
    latest = max(csv_latest, checkpoint_step)

    # Estimate speed only from observed checkpoint advancement.
    now = time.monotonic()
    previous_step = CHECKPOINT_HISTORY["step"]
    previous_time = CHECKPOINT_HISTORY["time"]

    if checkpoint_step > 0 and previous_step is not None:
        delta_steps = checkpoint_step - previous_step
        delta_time = now - previous_time
        if delta_steps > 0 and delta_time > 0:
            CHECKPOINT_HISTORY["rate"] = delta_steps / delta_time

    if checkpoint_step > 0 and (
        previous_step is None or checkpoint_step > previous_step
    ):
        CHECKPOINT_HISTORY["step"] = checkpoint_step
        CHECKPOINT_HISTORY["time"] = now

    rate = CHECKPOINT_HISTORY["rate"]
    eta_text = "CALIBRATING"
    if rate and rate > 0:
        remaining_seconds = max(0, MAX_STEPS - checkpoint_step) / rate
        eta_minutes = int(remaining_seconds // 60)
        eta_hours = eta_minutes // 60
        eta_minutes %= 60
        eta_text = (
            f"{eta_hours}h {eta_minutes}m"
            if eta_hours else f"{eta_minutes}m"
        )

    checkpoint_age = None
    if CHECKPOINT_FILE.exists():
        checkpoint_age = (
            datetime.now().timestamp() - CHECKPOINT_FILE.stat().st_mtime
        )

    gpu_age = None
    if gpu:
        gpu_age = max(0, (datetime.now() - gpu[-1][0]).total_seconds())

    gpu_temp = gpu[-1][4] if gpu else None
    health = "OK"
    if checkpoint_age is None:
        health = "NO CHECKPOINT"
    elif checkpoint_age > 300:
        health = "CHECKPOINT STALE"
    elif gpu_age is None or gpu_age > 15:
        health = "GPU TELEMETRY STALE"
    elif gpu_temp is not None and gpu_temp >= 85:
        health = "HIGH GPU TEMP"

    status_color = GREEN if health == "OK" else AMBER

    best_candidates = [v for _, v in validations]
    if checkpoint and checkpoint["best_validation_loss"] is not None:
        best_candidates.append(float(checkpoint["best_validation_loss"]))
    best_val = min(best_candidates, default=None)
    latest_val = validations[-1] if validations else None

    # 1. Training and validation loss
    if steps:
        neon(ax_loss, steps, losses, CYAN, "Training loss", smooth=True, raw_label="Raw training loss")

    if validations:
        val_steps = [x[0] for x in validations]
        val_losses = [x[1] for x in validations]
        neon(ax_loss, val_steps, val_losses, PINK, "Validation loss")

    style_axis(ax_loss, "TRAINING / VALIDATION LOSS",
               "Training step", "Cross-entropy loss")
    add_legend(ax_loss)

    # 2. Learning rate
    if rates:
        neon(
            ax_lr,
            [x[0] for x in rates],
            [x[1] for x in rates],
            VIOLET,
            "Learning rate"
        )

    style_axis(ax_lr, "LEARNING RATE SCHEDULE",
               "Training step", "Learning rate")
    add_legend(ax_lr)

    # 3-6. GPU telemetry
    if gpu:
        times = [x[0] for x in gpu]
        utilization = [x[1] for x in gpu]
        memory = [x[2] for x in gpu]
        temperature = [x[4] for x in gpu]
        power = [x[5] for x in gpu]

        neon(ax_util, times, utilization, CYAN,
             "GPU utilization", smooth=True)
        ax_util.set_ylim(0, 105)
        style_axis(ax_util, "GPU UTILIZATION",
                   "Time", "Utilization (%)")
        add_legend(ax_util)

        neon(ax_temp, times, temperature, PINK,
             "GPU temperature", smooth=True)
        style_axis(ax_temp, "GPU TEMPERATURE",
                   "Time", "Temperature (°C)")
        add_legend(ax_temp)

        memory_percent = [
            (used / total * 100) if total > 0 else 0
            for used, total in zip(memory, [x[3] for x in gpu])
        ]
        neon(ax_memory, times, memory_percent, GREEN,
             "VRAM utilization", smooth=True)
        ax_memory.set_ylim(0, 105)
        style_axis(ax_memory, "GPU MEMORY",
                   "Time", "VRAM utilization (%)")
        add_legend(ax_memory)

        neon(ax_power, times, power, AMBER,
             "GPU power", smooth=True)
        style_axis(ax_power, "GPU POWER",
                   "Time", "Power (W)")
        add_legend(ax_power)

        for ax in (ax_util, ax_temp, ax_memory, ax_power):
            ax.xaxis.set_major_locator(
                mdates.AutoDateLocator(minticks=3, maxticks=5)
            )
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter("%H:%M:%S")
            )
            ax.tick_params(axis="x", rotation=15)

    else:
        for ax, title, ylabel in (
            (ax_util, "GPU UTILIZATION", "Utilization (%)"),
            (ax_temp, "GPU TEMPERATURE", "Temperature (°C)"),
            (ax_memory, "GPU MEMORY", "VRAM used (MiB)"),
            (ax_power, "GPU POWER", "Power (W)"),
        ):
            style_axis(ax, title, "Time", ylabel)
            ax.text(
                0.5, 0.5, "Waiting for GPU telemetry...",
                transform=ax.transAxes, ha="center",
                color=MUTED, fontsize=9
            )

    val_text = f"{latest_val[1]:.4f} @ {latest_val[0]:,}" if latest_val else "Awaiting evaluation"
    best_text = f"{best_val:.4f}" if best_val is not None else "N/A"

    progress = min(100.0, latest / MAX_STEPS * 100)

    # Render progress independently of the six data charts.
    progress_ax.clear()
    progress_ax.barh(
        0, MAX_STEPS, height=0.72, color=GRID, zorder=1
    )
    progress_ax.barh(
        0, min(latest, MAX_STEPS), height=0.72,
        color=CYAN, zorder=2
    )
    progress_ax.set_xlim(0, MAX_STEPS)
    progress_ax.set_ylim(-0.6, 0.6)
    progress_ax.axis("off")

    checkpoint_text = (
        f"{checkpoint_step:,} @ {checkpoint['modified']}"
        if checkpoint else "waiting"
    )

    status.set_color(status_color)
    status.set_text(
        f"STEP {latest:,}/{MAX_STEPS:,}  |  "
        f"{progress:.1f}%  |  "
        f"ETA: {eta_text}  |  "
        f"HEALTH: {health}  |  "
        f"GPU AGE: {gpu_age:.1f}s  |  "
        f"REFRESH: {REFRESH_MS // 1000}s"
    )

    footer.set_text(
        f"OANC PRODUCTION V1  |  "
        f"RECORDED TRAINING POINTS: {len(steps)}  |  "
        f"VALIDATION POINTS: {len(validations)}  |  "
        f"BEST VAL: {best_text}  |  "
        f"CHECKPOINT: {checkpoint_text}  |  "
        f"GPU SAMPLES: {len(gpu)}"
    )

    fig.subplots_adjust(
        left=0.075, right=0.97,
        top=0.875, bottom=0.13,
        hspace=0.40, wspace=0.22
    )


ani = FuncAnimation(
    fig, update, interval=REFRESH_MS,
    cache_frame_data=False
)

update(0)
plt.show()
