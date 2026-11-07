"""Headless, grayscale diagnostics for optional training-trial alignment."""

import numpy as np


def plot_trial_alignment(path, state):
    """Plot at most twelve retained examples; return whether a file was written."""
    examples = state.alignment_examples[:12]
    if not examples:
        return False
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(12, 3 * len(examples)), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(examples), 2, squeeze=False)
    for (activation, cues), example in zip(axes, examples):
        record = example["record"]
        trace = example.get("trace")
        if trace is not None:
            hi = example["search"][1]
            left, right = example["baseline"][0], min(len(trace), hi + 13)
            activation.plot(np.arange(left, right), trace[left:right],
                            color="black", linewidth=1, label="Activation")
            activation.axhline(example["threshold"], color="0.4", linestyle="--", label="Threshold")
            activation.axvspan(*example["baseline"], color="0.9", label="Baseline")
            activation.set_xlim(left, right - 1)
        else:
            activation.text(0.05, 0.5, "No usable detection trace", transform=activation.transAxes)
        for key, label, style in (("cue_onset", "Cue onset", ":"), ("emg_onset", "EMG onset", "-.")):
            value = example.get(key)
            if value is not None:
                activation.axvline(value, color="0.3", linestyle=style, label=label)
        before, after = example["before"], example["after"]
        active = np.flatnonzero(np.any(before != 0, axis=0))
        for i, dof in enumerate(active):
            gray = str(0.6 * i / max(1, len(active) - 1))
            cues.plot(before[:, dof], color=gray, linestyle="--", label=f"DOF {dof} before")
            cues.plot(np.arange(len(after)) + example["after_start"], after[:, dof],
                      color=gray, label=f"DOF {dof} after")
        title = f"{record['trial_id']}: {record['status']}, shift={record['residual_shift']}"
        if record["reason"]:
            title += f" ({record['reason']})"
        activation.set_title(title, loc="left", fontsize=9)
        cues.set_title("Kinematic cues", loc="left", fontsize=9)
        for ax in (activation, cues):
            ax.set_xlabel("Sample after global alignment, before residual cropping")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(frameon=False, fontsize=7)
    fig.savefig(path, dpi=130)
    fig.clear()
    return True
