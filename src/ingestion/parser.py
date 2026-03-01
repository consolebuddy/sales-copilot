"""Parse sales call transcripts into structured data."""

import re
import os
from dataclasses import dataclass, field


@dataclass
class DialogueTurn:
    timestamp: str
    speaker: str        # Full label, e.g. "AE (Jordan)" or "Prospect (Priya – RevOps Director)"
    speaker_name: str   # e.g. "Jordan", "Priya"
    role: str           # e.g. "AE", "Prospect", "SE", "CISO"
    text: str


@dataclass
class CallTranscript:
    call_id: str
    call_type: str
    file_name: str
    participants: list[str]
    turns: list[DialogueTurn]
    duration: str = ""
    stage_directions: list[str] = field(default_factory=list)


# Matches lines like: [00:05] Prospect (Priya – RevOps Director):  Hey Jordan...
# Also handles: [01:05] SE (Luis):  Thanks Jordan.
# And shorthand: [00:21] Prospect:  Perfect.
_TURN_PATTERN = re.compile(
    r"^\[(\d{2}:\d{2})\]\s+(.+?):\s{2}(.+)$"
)

# Matches stage directions like: *Call ends.* or *screen share: ROI.xlsx*
_STAGE_DIRECTION = re.compile(r"^\[(\d{2}:\d{2})\]\s+(\*.+\*)\s*$")
_BARE_STAGE_DIRECTION = re.compile(r"^\*(.+)\*\s*$")

# Extract name and role from speaker label
# "AE (Jordan)" -> role=AE, name=Jordan
# "Prospect (Priya – RevOps Director)" -> role=Prospect, name=Priya
# "CISO (Elena)" -> role=CISO, name=Elena
# "AE" -> role=AE, name=AE
_SPEAKER_DETAIL = re.compile(
    r"^(\w+(?:\s+\w+)?)\s*(?:\(([^)]+)\))?\s*$"
)

# Map to infer call type from filename
_CALL_TYPE_MAP = {
    "demo": "Demo",
    "pricing": "Pricing",
    "objection": "Objection Handling",
    "negotiation": "Negotiation",
}


def _parse_speaker(speaker_label: str) -> tuple[str, str]:
    """Extract (speaker_name, role) from speaker label."""
    match = _SPEAKER_DETAIL.match(speaker_label.strip())
    if not match:
        return speaker_label.strip(), "Unknown"

    role = match.group(1).strip()
    detail = match.group(2)

    if detail:
        # "Priya – RevOps Director" -> name = "Priya"
        name = detail.split("–")[0].split("-")[0].strip()
    else:
        name = role  # e.g. just "AE" with no parenthetical

    return name, role


def _infer_call_type(file_name: str) -> str:
    """Infer call type from filename."""
    lower = file_name.lower()
    for keyword, label in _CALL_TYPE_MAP.items():
        if keyword in lower:
            return label
    return "General"


def _extract_call_id(file_name: str) -> str:
    """Extract call ID from filename like 'call_1.txt' or '1_demo_call.txt'."""
    match = re.search(r"(\d+)", os.path.basename(file_name))
    return match.group(1) if match else os.path.splitext(os.path.basename(file_name))[0]


def parse_transcript(file_path: str) -> CallTranscript:
    """Parse a transcript file into a CallTranscript dataclass."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    file_name = os.path.basename(file_path)
    call_id = _extract_call_id(file_name)
    call_type = _infer_call_type(file_name)

    turns: list[DialogueTurn] = []
    stage_directions: list[str] = []
    participants_set: set[str] = set()

    for line in lines:
        line = line.rstrip()
        if not line:
            continue

        # Check for stage directions with timestamps: [05:12] *screen share: ROI.xlsx*
        sd_match = _STAGE_DIRECTION.match(line)
        if sd_match:
            stage_directions.append(f"[{sd_match.group(1)}] {sd_match.group(2)}")
            continue

        # Check for bare stage directions: *Call ends.*
        bare_sd = _BARE_STAGE_DIRECTION.match(line)
        if bare_sd:
            stage_directions.append(f"*{bare_sd.group(1)}*")
            continue

        # Try to match a dialogue turn
        turn_match = _TURN_PATTERN.match(line)
        if turn_match:
            timestamp = turn_match.group(1)
            speaker_label = turn_match.group(2).strip()
            text = turn_match.group(3).strip()

            speaker_name, role = _parse_speaker(speaker_label)
            participants_set.add(f"{speaker_name} ({role})")

            turns.append(DialogueTurn(
                timestamp=timestamp,
                speaker=speaker_label,
                speaker_name=speaker_name,
                role=role,
                text=text,
            ))

    duration = turns[-1].timestamp if turns else "00:00"

    return CallTranscript(
        call_id=call_id,
        call_type=call_type,
        file_name=file_name,
        participants=sorted(participants_set),
        turns=turns,
        duration=duration,
        stage_directions=stage_directions,
    )
