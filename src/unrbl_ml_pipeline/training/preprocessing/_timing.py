"""Explicit sample-order fallback for recordings without a usable clock."""

import os

import numpy as np


def usable_clock(file):
    return (
        file.nip_time is not None
        and np.isfinite(file.nip_time).all()
        and file.nominal_interval is not None
        and np.isfinite(file.nominal_interval)
        and file.nominal_interval > 0
    )


def apply_timing_policy(files, state, config):
    """Leave real clocks intact; optionally reconstruct unusable clocks by row order.

    Intervals remain in the recording timestamp's units. Inference uses the median
    of per-recording positive-delta medians, giving recordings equal influence.
    It uses only eligible loaded clocks, never features or target values.
    """
    data = config.get("data", {})
    fallback = data.get("allow_missing_timestamps", False)
    if not isinstance(fallback, bool):
        raise TypeError("data.allow_missing_timestamps must be boolean")
    explicit = data.get("sample_interval")
    if explicit is not None:
        if (
            isinstance(explicit, bool)
            or not np.isfinite(float(explicit))
            or float(explicit) <= 0
        ):
            raise ValueError(
                "data.sample_interval must be a finite positive timestamp interval"
            )
        explicit = float(explicit)
    records = {r["path"]: r for r in state.selection_report}
    donors = [
        {"path": os.path.abspath(f.path), "median_interval": float(f.nominal_interval)}
        for f in files
        if usable_clock(f)
    ]
    missing = [f for f in files if not usable_clock(f)]
    interval = (
        explicit
        if explicit is not None
        else (
            float(np.median([r["median_interval"] for r in donors])) if donors else None
        )
    )
    if fallback and missing and interval is None:
        raise ValueError(
            "No usable timestamps to estimate the sample interval; set data.sample_interval "
            "or --sample-interval in timestamp units with --allow-missing-timestamps"
        )
    if fallback and missing and state.progress is not None:
        source = (
            "explicit override"
            if explicit is not None
            else f"estimated from {len(donors)} recorded clocks"
        )
        state.progress.message(
            f"  Timing fallback: {interval:g} timestamp units/sample ({source})."
        )
    kept = []
    for f in files:
        record = records[os.path.abspath(f.path)]
        if usable_clock(f):
            f.timestamp_info = {
                "source": "recorded",
                "nominal_interval": float(f.nominal_interval),
            }
        elif fallback:
            reason = (
                "missing timestamps"
                if f.nip_time is None
                else (
                    "nonfinite timestamps"
                    if not np.isfinite(f.nip_time).all()
                    else "no positive timestamp progression"
                )
            )
            f.original_nip_time = None if f.nip_time is None else f.nip_time.copy()
            f.nip_time = np.arange(len(f.features), dtype=np.float64) * interval
            f.nominal_interval = interval
            f.timestamp_info = {
                "source": "sample_order",
                "nominal_interval": interval,
                "interval_source": "explicit"
                if explicit is not None
                else "median_of_recording_medians",
                "reason": reason,
                "origin": 0.0,
                "assumption": "Consecutive regularly sampled rows; unrecorded gaps cannot be detected.",
            }
            if state.progress is not None:
                state.progress.file(
                    f.path,
                    config,
                    decision="ROW ORDER",
                    reason=f"{reason}; assuming {interval:g} timestamp units/sample",
                )
        elif data.get("require_usable_timestamps"):
            reason = "Unusable acquisition timestamps; enable --allow-missing-timestamps to assume regular sample order"
            state.rejected.append((f.path, reason))
            record.update(
                status="excluded", reason=reason, timing={"source": "unusable"}
            )
            if state.progress is not None:
                state.progress.file(f.path, config, decision="SKIP", reason=reason)
            continue
        else:
            f.timestamp_info = {"source": "unusable"}
        record["timing"] = f.timestamp_info.copy()
        kept.append(f)
    state.fitted_preprocessing["timing"] = {
        "allow_missing_timestamps": fallback,
        "proxy_interval": interval if fallback and missing else None,
        "interval_source": (
            "explicit" if explicit is not None else "median_of_recording_medians"
        )
        if fallback
        else None,
        "donors": donors if fallback and explicit is None else [],
        "sample_order_files": [
            os.path.abspath(f.path)
            for f in kept
            if f.timestamp_info["source"] == "sample_order"
        ],
        "assumption": "Proxy timestamps assume consecutive regularly sampled rows; file/trial boundaries still reset state."
        if fallback
        else None,
    }
    return kept
