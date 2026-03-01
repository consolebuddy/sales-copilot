"""Tests for the transcript parser."""

import os
import tempfile
import pytest

from src.ingestion.parser import parse_transcript, _parse_speaker, _extract_call_id


SAMPLE_TRANSCRIPT = """\
[00:00] AE (Jordan):  Good morning, Priya!  Appreciate you carving out a full hour.

[00:05] Prospect (Priya – RevOps Director):  Hey Jordan.  Busy as always.

[00:11] AE:  Before we jump in, quick agenda check.

[00:21] Prospect:  Perfect.

[00:23] SE (Luis):  Thanks Jordan.  Priya, can you see my browser?

[05:12] *screen share: ROI.xlsx*

[06:37] AE:  Pleasure, Priya.  Talk soon!

*Call ends.*
"""


@pytest.fixture
def transcript_file(tmp_path):
    fp = tmp_path / "call_1.txt"
    fp.write_text(SAMPLE_TRANSCRIPT, encoding="utf-8")
    return str(fp)


def test_parse_turn_count(transcript_file):
    call = parse_transcript(transcript_file)
    assert len(call.turns) == 6  # 6 dialogue turns (stage directions excluded)


def test_call_id_extraction(transcript_file):
    call = parse_transcript(transcript_file)
    assert call.call_id == "1"


def test_speaker_extraction(transcript_file):
    call = parse_transcript(transcript_file)
    names = [t.speaker_name for t in call.turns]
    assert "Jordan" in names
    assert "Priya" in names
    assert "Luis" in names


def test_role_extraction(transcript_file):
    call = parse_transcript(transcript_file)
    roles = {t.role for t in call.turns}
    assert "AE" in roles
    assert "Prospect" in roles
    assert "SE" in roles


def test_timestamp_parsing(transcript_file):
    call = parse_transcript(transcript_file)
    assert call.turns[0].timestamp == "00:00"
    assert call.turns[1].timestamp == "00:05"


def test_duration(transcript_file):
    call = parse_transcript(transcript_file)
    assert call.duration == "06:37"


def test_stage_directions(transcript_file):
    call = parse_transcript(transcript_file)
    assert len(call.stage_directions) == 2  # screen share + call ends


def test_participants(transcript_file):
    call = parse_transcript(transcript_file)
    assert len(call.participants) >= 3  # Jordan, Priya, Luis


def test_parse_speaker_with_title():
    name, role = _parse_speaker("Prospect (Priya – RevOps Director)")
    assert name == "Priya"
    assert role == "Prospect"


def test_parse_speaker_simple():
    name, role = _parse_speaker("AE (Jordan)")
    assert name == "Jordan"
    assert role == "AE"


def test_parse_speaker_no_parens():
    name, role = _parse_speaker("AE")
    assert name == "AE"
    assert role == "AE"


def test_extract_call_id_numbered():
    assert _extract_call_id("call_1.txt") == "1"
    assert _extract_call_id("3_objection_call.txt") == "3"
