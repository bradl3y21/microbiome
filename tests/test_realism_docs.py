"""docs/REALISM.md must describe the code that exists: no undocumented parameter, trait or table, no dead knob."""
import dataclasses
import re
from pathlib import Path

from microbiome.genome.physiology import RULES, SCALAR_TRAITS
from microbiome.sim import realism
from microbiome.sim.realism import RealismConfig

ROOT = Path(__file__).resolve().parent.parent
DOC = (ROOT / "docs" / "REALISM.md").read_text(encoding="utf-8")
SOURCE = (ROOT / "src" / "microbiome" / "sim" / "realism.py").read_text(encoding="utf-8")


def test_every_config_parameter_is_documented_with_its_actual_default():
    for field in dataclasses.fields(RealismConfig):
        assert f"| `{field.name}` |" in DOC, f"{field.name} is not in the parameter table"
        row = next(line for line in DOC.splitlines() if line.startswith(f"| `{field.name}` |"))
        value = getattr(RealismConfig(), field.name)
        shown = f"{value:.4g}" if isinstance(value, float) else str(value)
        assert f"| {shown} |" in row, f"{field.name}: documented default differs from {shown}"


def test_the_documentation_lists_no_parameter_that_does_not_exist():
    documented = set(re.findall(r"^\| `([a-z0-9_]+)` \| [^|]+ \| [^|]", DOC, flags=re.M))
    known = {f.name for f in dataclasses.fields(RealismConfig)} | set(SCALAR_TRAITS)
    assert documented <= known, sorted(documented - known)


def test_no_config_parameter_is_dead_every_one_is_read_by_the_code():
    outside_the_definition = SOURCE.split("class RealismConfig")[0] + SOURCE.split("class Physiology")[1]
    for field in dataclasses.fields(RealismConfig):
        assert re.search(rf"(cfg|config)\.{field.name}\b", outside_the_definition), f"{field.name} is never read by the code"


def test_every_physiology_trait_and_the_rule_count_are_documented():
    for trait in SCALAR_TRAITS:
        assert f"| `{trait}` |" in DOC, trait
    assert f"{len(RULES)} gene-to-physiology rules" in DOC


def test_the_test_files_the_documentation_names_all_exist():
    for name in re.findall(r"`(tests/[a-z_]+\.py)`", DOC):
        assert (ROOT / name).is_file(), name
    for short in ("kinetics", "movement", "chemistry", "life", "world", "controller", "ui"):
        assert (ROOT / "tests" / f"test_realism_{short}.py").is_file(), short
    assert (ROOT / "tests" / "test_physiology.py").is_file() and (ROOT / "tests" / "test_transformation.py").is_file()


def test_the_default_channel_set_matches_the_documentation():
    from microbiome.genome.physiology import CHANNEL_NAMES

    from microbiome.genome.physiology import REALISM_CHANNELS

    assert tuple(CHANNEL_NAMES) == ("glucose", "lactose", "oxygen", "acetate", "bacilysin", "ai2", "pyocyanin")
    assert "**glucose, lactose, oxygen, acetate, bacilysin, ai2, pyocyanin**" in DOC
    assert REALISM_CHANNELS.size == len(CHANNEL_NAMES)
