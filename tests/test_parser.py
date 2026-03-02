"""Tests for the transcript parser."""

import pytest

from src.ingestion.parser import (
    parse_transcript,
    _parse_speaker,
    _extract_call_id,
    _infer_call_type,
    _is_valid_speaker,
)


# ── Fixtures ─────────────────────────────────────────────────────────

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

MULTI_SPEAKER_TRANSCRIPT = """\
[00:00] AE (Jordan):  Hi everyone.
[00:10] Prospect (Priya – RevOps Director):  Hey.
[00:20] Prospect (Dan – Finance VP):  Hello.
[00:30] SE (Luis):  Let's begin.
[00:40] CISO (Elena):  Ready.
[00:50] Maya (Pricing Strategist):  Starting price discussion.
"""

STAGE_DIRECTION_TRANSCRIPT = """\
[00:00] AE (Jordan):  Let's start.
[00:10] SE (reads on-screen):  *"Summary: The deal is moving forward."*
[00:20] Prospect (smiling):  Great news.
[00:30] AE:  Thanks everyone.
*Call ends.*
[01:00] *screen share: demo.pptx*
"""

EMPTY_TRANSCRIPT = ""

SINGLE_TURN_TRANSCRIPT = """\
[00:00] AE (Jordan):  Hello world.
"""


@pytest.fixture
def transcript_file(tmp_path):
    fp = tmp_path / "call_1.txt"
    fp.write_text(SAMPLE_TRANSCRIPT, encoding="utf-8")
    return str(fp)


@pytest.fixture
def multi_speaker_file(tmp_path):
    fp = tmp_path / "call_10.txt"
    fp.write_text(MULTI_SPEAKER_TRANSCRIPT, encoding="utf-8")
    return str(fp)


@pytest.fixture
def stage_direction_file(tmp_path):
    fp = tmp_path / "call_7.txt"
    fp.write_text(STAGE_DIRECTION_TRANSCRIPT, encoding="utf-8")
    return str(fp)


@pytest.fixture
def empty_file(tmp_path):
    fp = tmp_path / "call_0.txt"
    fp.write_text(EMPTY_TRANSCRIPT, encoding="utf-8")
    return str(fp)


@pytest.fixture
def single_turn_file(tmp_path):
    fp = tmp_path / "call_42.txt"
    fp.write_text(SINGLE_TURN_TRANSCRIPT, encoding="utf-8")
    return str(fp)


# ── Basic parsing ────────────────────────────────────────────────────

class TestBasicParsing:
    def test_turn_count(self, transcript_file):
        call = parse_transcript(transcript_file)
        assert len(call.turns) == 6

    def test_call_id_extraction(self, transcript_file):
        call = parse_transcript(transcript_file)
        assert call.call_id == "1"

    def test_duration(self, transcript_file):
        call = parse_transcript(transcript_file)
        assert call.duration == "06:37"

    def test_timestamp_order(self, transcript_file):
        call = parse_transcript(transcript_file)
        timestamps = [t.timestamp for t in call.turns]
        assert timestamps == sorted(timestamps)

    def test_text_content_preserved(self, transcript_file):
        call = parse_transcript(transcript_file)
        assert "Good morning" in call.turns[0].text
        assert "Busy as always" in call.turns[1].text

    def test_file_name_stored(self, transcript_file):
        call = parse_transcript(transcript_file)
        assert call.file_name == "call_1.txt"


# ── Speaker extraction ───────────────────────────────────────────────

class TestSpeakerExtraction:
    def test_named_speaker(self, transcript_file):
        call = parse_transcript(transcript_file)
        names = [t.speaker_name for t in call.turns]
        assert "Jordan" in names
        assert "Priya" in names
        assert "Luis" in names

    def test_role_extraction(self, transcript_file):
        call = parse_transcript(transcript_file)
        roles = {t.role for t in call.turns}
        assert "AE" in roles
        assert "Prospect" in roles
        assert "SE" in roles

    def test_shorthand_speaker(self, transcript_file):
        """When speaker uses shorthand like 'AE:' without parenthetical."""
        call = parse_transcript(transcript_file)
        shorthand_turns = [t for t in call.turns if t.speaker == "AE"]
        assert len(shorthand_turns) >= 1

    def test_multi_speaker_call(self, multi_speaker_file):
        call = parse_transcript(multi_speaker_file)
        names = {t.speaker_name for t in call.turns}
        assert names >= {"Jordan", "Priya", "Dan", "Luis", "Elena"}

    def test_participants_list(self, multi_speaker_file):
        call = parse_transcript(multi_speaker_file)
        assert len(call.participants) >= 5


# ── _parse_speaker unit tests ────────────────────────────────────────

class TestParseSpeaker:
    def test_role_with_name(self):
        name, role = _parse_speaker("AE (Jordan)")
        assert name == "Jordan"
        assert role == "AE"

    def test_role_with_name_and_title(self):
        name, role = _parse_speaker("Prospect (Priya – RevOps Director)")
        assert name == "Priya"
        assert role == "Prospect"

    def test_role_with_name_and_hyphen_title(self):
        name, role = _parse_speaker("Prospect (Dan - Finance VP)")
        assert name == "Dan"
        assert role == "Prospect"

    def test_bare_role(self):
        name, role = _parse_speaker("AE")
        assert name == "AE"
        assert role == "AE"

    def test_two_word_role(self):
        name, role = _parse_speaker("VP CS (Asha)")
        assert name == "Asha"
        assert role == "VP CS"

    def test_ciso_role(self):
        name, role = _parse_speaker("CISO (Elena)")
        assert name == "Elena"
        assert role == "CISO"

    def test_stage_direction_smiling(self):
        """'Prospect (smiling)' should return role as speaker name."""
        name, role = _parse_speaker("Prospect (smiling)")
        assert role == "Prospect"
        assert name == "Prospect"  # stage note, not a real name

    def test_stage_direction_reads(self):
        name, role = _parse_speaker("SE (reads on-screen)")
        assert role == "SE"
        assert name == "SE"


# ── _extract_call_id ─────────────────────────────────────────────────

class TestExtractCallId:
    def test_call_underscore(self):
        assert _extract_call_id("call_1.txt") == "1"

    def test_numbered_prefix(self):
        assert _extract_call_id("3_objection_call.txt") == "3"

    def test_double_digit(self):
        assert _extract_call_id("call_12.txt") == "12"

    def test_no_number(self):
        result = _extract_call_id("notes.txt")
        assert result == "notes"  # fallback to stem


# ── _infer_call_type ─────────────────────────────────────────────────

class TestInferCallType:
    def test_demo_from_filename(self):
        assert _infer_call_type("1_demo_call.txt") == "Demo"

    def test_pricing_from_filename(self):
        assert _infer_call_type("2_pricing_call.txt") == "Pricing"

    def test_objection_from_filename(self):
        assert _infer_call_type("3_objection_call.txt") == "Objection Handling"

    def test_negotiation_from_filename(self):
        assert _infer_call_type("4_negotiation_call.txt") == "Negotiation"

    def test_generic_filename_with_pricing_content(self):
        content = "Goal today is to dig into SKUs, discount structure..."
        assert _infer_call_type("call_2.txt", content) == "Pricing"

    def test_generic_filename_with_security_content(self):
        content = "Purpose of today's call is to address security, privacy concerns"
        assert _infer_call_type("call_3.txt", content) == "Objection Handling"

    def test_generic_filename_with_negotiation_content(self):
        content = "Great to see a full house for what we hope is the final stretch"
        assert _infer_call_type("call_4.txt", content) == "Negotiation"

    def test_generic_filename_with_demo_content(self):
        content = "Luis will run a live product demo, then we'll map next steps"
        assert _infer_call_type("call_1.txt", content) == "Demo"

    def test_unknown_falls_back_to_general(self):
        assert _infer_call_type("call_99.txt", "just a regular chat") == "General"


# ── _is_valid_speaker ────────────────────────────────────────────────

class TestIsValidSpeaker:
    def test_known_roles(self):
        assert _is_valid_speaker("AE")
        assert _is_valid_speaker("SE (Luis)")
        assert _is_valid_speaker("Prospect (Priya – RevOps Director)")
        assert _is_valid_speaker("CISO (Elena)")
        assert _is_valid_speaker("Maya")

    def test_invalid_speakers(self):
        assert not _is_valid_speaker("Audio plays")
        assert not _is_valid_speaker("screen share")


# ── Stage directions ─────────────────────────────────────────────────

class TestStageDirections:
    def test_timestamped_stage_directions_excluded(self, stage_direction_file):
        """Bare *...* stage directions should not appear as turns."""
        call = parse_transcript(stage_direction_file)
        speakers = [t.speaker for t in call.turns]
        # No turn should have a stage direction marker as its speaker
        assert not any(s.startswith("*") for s in speakers)

    def test_stage_directions_collected(self, stage_direction_file):
        call = parse_transcript(stage_direction_file)
        assert len(call.stage_directions) >= 2  # *Call ends.* + *screen share*

    def test_smiling_filtered(self, stage_direction_file):
        """'Prospect (smiling)' should still parse as a turn with role=Prospect."""
        call = parse_transcript(stage_direction_file)
        turn_speakers = [t.speaker for t in call.turns]
        assert "Prospect (smiling)" in turn_speakers

    def test_reads_on_screen_filtered(self, stage_direction_file):
        """'SE (reads on-screen)' should still parse as a turn with role=SE."""
        call = parse_transcript(stage_direction_file)
        turn_speakers = [t.speaker for t in call.turns]
        assert "SE (reads on-screen)" in turn_speakers


# ── Edge cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_file(self, empty_file):
        call = parse_transcript(empty_file)
        assert call.turns == []
        assert call.duration == "00:00"
        assert call.participants == []

    def test_single_turn(self, single_turn_file):
        call = parse_transcript(single_turn_file)
        assert len(call.turns) == 1
        assert call.turns[0].speaker_name == "Jordan"
        assert call.duration == "00:00"

    def test_call_id_from_single_turn(self, single_turn_file):
        call = parse_transcript(single_turn_file)
        assert call.call_id == "42"


# ── Real transcript integration ──────────────────────────────────────

class TestRealTranscripts:
    """Parse the actual sample transcripts to ensure no regressions."""

    @pytest.fixture(params=["call_1.txt", "call_2.txt", "call_3.txt", "call_4.txt"])
    def real_file(self, request):
        import os
        path = os.path.join(
            os.path.dirname(__file__), "..", "transcripts", request.param
        )
        if not os.path.isfile(path):
            pytest.skip(f"{request.param} not found")
        return path

    def test_real_transcript_parses(self, real_file):
        call = parse_transcript(real_file)
        assert len(call.turns) > 10
        assert call.call_id.isdigit()
        assert call.duration != "00:00"

    def test_real_transcript_has_participants(self, real_file):
        call = parse_transcript(real_file)
        assert len(call.participants) >= 2

    def test_real_transcript_jordan_present(self, real_file):
        """Jordan (AE) appears in all 4 sample calls."""
        call = parse_transcript(real_file)
        names = {t.speaker_name for t in call.turns}
        assert "Jordan" in names
