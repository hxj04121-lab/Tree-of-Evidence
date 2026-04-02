# Optimized anomaly scoring rules
def _anomaly_score_optimized(self, text: str) -> int:
    low = (text or "").lower()
    score = 0

    # Very strong signals (+3)
    critical = [
        "fatal", "critical", "segfault", "panic", "oom",
        "out of memory", "no space left", "disk full",
        "permission denied", "access denied", "corrupt", "killed", "aborting",
    ]
    for k in critical:
        if k in low:
            score += 3

    # Strong signals (+2)
    strong = [
        "exception", "i/o error", "ioerror", "checksum", "bad crc",
        "timed out", "timeout", "connection refused", "connection reset",
        "unreachable", "failed", "failure", "java.io.ioexception",
        "java.net.socketexception", "cannot", "unable to", "could not",
    ]
    for k in strong:
        if k in low:
            score += 2

    # Mild signals (+1)
    mild = ["warn", "warning", "error", "retry", "retries", "disconnect"]
    for k in mild:
        if k in low:
            score += 1

    # === BGL whitelist (Normal patterns, expanded to reduce FP) ===
    # "error" + "corrected" is typically INFO-level normal recovery
    if "corrected" in low and "error" in low and not any(x in low for x in ["uncorrected", "failed", "fatal", "exception"]):
        score -= 4
    # alignment exceptions are common BGL INFO messages
    if "alignment exceptions" in low or "alignment exception" in low:
        score -= 4
    # "generating core" is normal BGL process dump, not anomaly
    if "generating core" in low and not any(x in low for x in ["fatal", "critical", "panic"]):
        score -= 4
    # "re-synch" / "resync" are normal sync operations
    if ("re-synch" in low or "resync" in low or "synchronization" in low) and not any(x in low for x in ["failed", "timeout", "error"]):
        score -= 3
    # CIOD error reading is typically benign
    if "ciod" in low and "error reading" in low:
        score -= 3
    # instruction/data cache parity error (INFO) is hardware self-correction
    if ("instruction cache" in low or "data cache" in low) and "parity error" in low and " info " in low:
        score -= 3
    # "tree receiver" related re-synch is normal state sync
    if "tree receiver" in low and "re-synch" in low:
        score -= 3
    # "generating" related without fatal or strong signals
    if "generating" in low and not any(x in low for x in ["failed", "fatal", "critical", "exception"]):
        score -= 2
    # "detected" + INFO is typically status notification, not anomaly
    if "detected" in low and " info " in low and not any(x in low for x in ["failed", "error", "fatal"]):
        score -= 2

    # === HDFS whitelist ===
    # PacketResponder terminating is typically normal completion
    if "packetresponder" in low and "terminating" in low and not any(
        x in low for x in ["error", "failed", "failure", "exception", "timeout", "aborting"]
    ):
        score -= 4
    # "got exception while serving" without severe errors is typically client disconnect
    if "got exception while serving" in low and not any(
        x in low for x in [
            "ioexception", "java.io.", "i/o error", "ioerror", "timed out", "timeout",
            "connection reset", "connection refused", "unreachable", "aborting", "killed",
            "corrupt", "checksum", "bad crc", "permission denied", "access denied",
            "no space", "out of memory", "oom",
        ]
    ):
        score -= 4
    # "receiving/received block" INFO is normal operation
    if ("receiving block" in low or "received block" in low) and "info" in low:
        score -= 1
    # "allocateblock" INFO is normal allocation
    if "allocateblock" in low and " info " in low:
        score -= 1

    return max(score, 0)


# Optimized judgment logic
def _looks_anomalous_optimized_hdfs(self, text: str) -> bool:
    """HDFS: lower threshold to improve recall"""
    return self._anomaly_score(text) >= 1  # lowered from 2 to 1


def _looks_anomalous_optimized_bgl(self, text: str) -> bool:
    """BGL: raise threshold to reduce false positives"""
    return self._anomaly_score(text) >= 3  # raised from 2 to 3




