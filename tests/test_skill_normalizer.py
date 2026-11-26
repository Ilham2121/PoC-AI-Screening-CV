"""Unit tests untuk normalisasi software dan keahlian teknis."""

from src.skill_normalizer import match_required_skills, normalize_skills


def test_normalizes_typos_and_variants():
    raw = ["AUTOCAD", "Skechup", "Sketch Up", "Revit Architecture", "VRAY", "enscape 3d", "Lumion"]
    normalized = normalize_skills(raw)
    assert "AutoCAD" in normalized
    assert "SketchUp" in normalized
    assert "Revit" in normalized
    assert "V-Ray" in normalized
    assert "Enscape" in normalized
    assert "Lumion" in normalized
    # Pastikan tidak ada duplikat SketchUp
    assert normalized.count("SketchUp") == 1


def test_matches_required_software_against_normalized_skills():
    matched, missing = match_required_skills(
        candidate_skills=["Autocad", "Skechup", "Enscape"],
        required_skills=["AutoCAD", "SketchUp", "Revit"],
    )
    assert matched == ["AutoCAD", "SketchUp"]
    assert missing == ["Revit"]
