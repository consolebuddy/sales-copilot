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


# Known speaker roles that appear at the start of a turn
# Roles and names that can appear as the first word of a speaker label.
# Includes common B2B sales roles plus names used in shorthand turns.
_KNOWN_ROLES = {
    # Roles
    "AE", "SE", "Prospect", "Prospective", "CISO", "VP",
    # Names used as speaker labels in shorthand turns
    "Maya", "Asha", "Arjun", "Elena", "Luis", "Priya", "Sara",
    "Pricing",  # "Pricing (Maya)" shorthand in call 4
}

# Matches lines like: [00:05] Prospect (Priya – RevOps Director):  Hey Jordan...
# Also handles: [01:05] SE (Luis):  Thanks Jordan.
# And shorthand: [00:21] Prospect:  Perfect.
# The speaker label must start with a known role word to avoid matching
# parenthetical stage directions like "SE (reads on-screen):  ..."
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
    r"^(\w+(?:\s+\w+){0,2})\s*(?:\(([^)]+)\))?\s*$"
)

# Map to infer call type from filename
_CALL_TYPE_MAP = {
    "demo": "Demo",
    "pricing": "Pricing",
    "objection": "Objection Handling",
    "negotiation": "Negotiation",
}


def _is_valid_speaker(speaker_label: str) -> bool:
    """Check if the speaker label starts with a known role word."""
    first_word = speaker_label.strip().split()[0].split("(")[0].strip()
    return first_word in _KNOWN_ROLES


def _parse_speaker(speaker_label: str) -> tuple[str, str]:
    """Extract (speaker_name, role) from speaker label.

    Returns (speaker_name, role). The parenthetical may be a real name
    (e.g. "Jordan") or a stage note (e.g. "reads on-screen", "smiling").
    We detect stage notes and ignore them.
    """
    match = _SPEAKER_DETAIL.match(speaker_label.strip())
    if not match:
        return speaker_label.strip(), "Unknown"

    role = match.group(1).strip()
    detail = match.group(2)

    if detail:
        # Detect parenthetical stage notes (lowercase first word, contains spaces
        # with common action words)
        detail_stripped = detail.strip()
        _STAGE_WORDS = {"reads", "smiling", "laughing", "typing", "nodding",
                        "pausing", "sighing", "whispering", "leaves"}
        first_word = detail_stripped.split()[0].lower() if detail_stripped else ""
        if first_word in _STAGE_WORDS:
            # This is a stage note, not a name — use the last known name for this role
            return role, role

        # "Priya – RevOps Director" -> name = "Priya"
        name = detail_stripped.split("–")[0].split("-")[0].strip()
        # "Pricing Strategist" -> just use first word as name
        name = name.split(",")[0].strip()
    else:
        name = role  # e.g. just "AE" with no parenthetical

    return name, role


def _infer_call_type(file_name: str, content: str = "") -> str:
    """Infer call type from filename or content keywords."""
    lower = file_name.lower()
    for keyword, label in _CALL_TYPE_MAP.items():
        if keyword in lower:
            return label
    # Fallback: scan first ~1000 chars of content for clues
    if content:
        snippet = content[:1000].lower()
        # Check negotiation first — these calls mention pricing too
        if any(w in snippet for w in ("final stretch", "commercial terms",
                                       "redline", "sign-off path")):
            return "Negotiation"
        # Check pricing before objection — pricing calls may mention security
        if any(w in snippet for w in ("pricing", "discount", "sku", "price card")):
            return "Pricing"
        if any(w in snippet for w in ("security", "privacy", "legal concern",
                                       "data residency", "encryption")):
            return "Objection Handling"
        if any(w in snippet for w in ("demo", "product demo", "live product")):
            return "Demo"
    return "General"


def _extract_call_id(file_name: str) -> str:
    """Extract call ID from filename like 'call_1.txt' or '1_demo_call.txt'."""
    match = re.search(r"(\d+)", os.path.basename(file_name))
    return match.group(1) if match else os.path.splitext(os.path.basename(file_name))[0]


def _format_participants(participants_map: dict[str, str]) -> list[str]:
    """Format participants map into clean display strings like 'Jordan (AE)'."""
    result = []
    seen_names = set()
    for name, label in participants_map.items():
        # Skip generic role-only entries if we already have a named version
        if name in ("AE", "SE", "Prospect") and any(
            v.startswith(name) for k, v in participants_map.items() if k != name
        ):
            continue
        if name in seen_names:
            continue
        seen_names.add(name)
        # Extract role from the label for cleaner display
        _, role = _parse_speaker(label)
        if name == role:
            result.append(name)
        else:
            result.append(f"{name} ({role})")
    return result


def parse_transcript(file_path: str) -> CallTranscript:
    """Parse a transcript file into a CallTranscript dataclass."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines()
    file_name = os.path.basename(file_path)
    call_id = _extract_call_id(file_name)
    call_type = _infer_call_type(file_name, content)

    turns: list[DialogueTurn] = []
    stage_directions: list[str] = []
    # Track participants by name -> most descriptive role seen
    participants_map: dict[str, str] = {}

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

            # Skip false matches where the "speaker" is not a real role
            if not _is_valid_speaker(speaker_label):
                continue

            speaker_name, role = _parse_speaker(speaker_label)
            # Keep the most descriptive label for each name
            # e.g., "Priya – RevOps Director" > "Prospect" > "Priya"
            existing = participants_map.get(speaker_name, "")
            if len(speaker_label) > len(existing):
                participants_map[speaker_name] = speaker_label

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
        participants=sorted(_format_participants(participants_map)),
        turns=turns,
        duration=duration,
        stage_directions=stage_directions,
    )
