"""Validation tests, including bundled agent skill installation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from squidpy._skills import install as skill_install
from squidpy._validators import (
    assert_in_range,
    assert_isinstance,
    assert_key_in_adata,
    assert_key_in_sdata,
    assert_non_empty_sequence,
    assert_non_negative,
    assert_positive,
    check_tuple_needles,
    get_valid_values,
    normalize_choice,
)


# assert_positive
class TestAssertPositive:
    def test_positive_value(self):
        assert_positive(1.0, name="x")
        assert_positive(0.001, name="x")

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            assert_positive(0, name="x")

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="positive"):
            assert_positive(-1, name="x")


# assert_non_negative
class TestAssertNonNegative:
    def test_non_negative_value(self):
        assert_non_negative(0, name="x")
        assert_non_negative(1, name="x")

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            assert_non_negative(-0.1, name="x")


# assert_in_range
class TestAssertInRange:
    def test_in_range(self):
        assert_in_range(0.5, 0, 1, name="x")
        assert_in_range(0, 0, 1, name="x")
        assert_in_range(1, 0, 1, name="x")

    def test_out_of_range(self):
        with pytest.raises(ValueError, match="interval"):
            assert_in_range(1.1, 0, 1, name="x")
        with pytest.raises(ValueError, match="interval"):
            assert_in_range(-0.1, 0, 1, name="x")


# assert_non_empty_sequence
class TestAssertNonEmptySequence:
    def test_list(self):
        assert assert_non_empty_sequence(["a", "b"], name="items") == ["a", "b"]

    def test_scalar_conversion(self):
        assert assert_non_empty_sequence("a", name="items") == ["a"]

    def test_no_scalar_conversion(self):
        with pytest.raises(TypeError, match="sequence"):
            assert_non_empty_sequence(42, name="items", convert_scalar=False)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No items"):
            assert_non_empty_sequence([], name="items")


# get_valid_values
class TestGetValidValues:
    def test_valid(self):
        assert get_valid_values(["a", "b"], ["a", "b", "c"]) == ["a", "b"]

    def test_partial(self):
        assert get_valid_values(["a", "z"], ["a", "b"]) == ["a"]

    def test_none_valid(self):
        with pytest.raises(ValueError, match="No valid values"):
            get_valid_values(["z"], ["a", "b"])


# check_tuple_needles
class TestCheckTupleNeedles:
    def test_valid_needles(self):
        result = check_tuple_needles([("a", "b")], ["a", "b", "c"], "Value `{}` not found.")
        assert result == [("a", "b")]

    def test_invalid_needle_reraise(self):
        with pytest.raises(ValueError, match="z"):
            check_tuple_needles([("z", "b")], ["a", "b"], "Value `{}` not found.")

    def test_invalid_needle_no_reraise(self):
        result = check_tuple_needles([("z", "b")], ["a", "b"], "Value `{}` not found.", reraise=False)
        assert result == []

    def test_wrong_length(self):
        with pytest.raises(ValueError, match="length"):
            check_tuple_needles([("a",)], ["a"], "msg {}")

    def test_not_sequence(self):
        with pytest.raises(TypeError, match="Sequence"):
            check_tuple_needles([42], ["a"], "msg {}")


# assert_isinstance
class TestAssertIsinstance:
    def test_correct_type(self):
        assert_isinstance("hello", str, name="x")
        assert_isinstance(42, int, name="x")

    def test_tuple_of_types(self):
        assert_isinstance("hello", (str, int), name="x")
        assert_isinstance(42, (str, int), name="x")

    def test_wrong_type(self):
        with pytest.raises(TypeError, match="str"):
            assert_isinstance(42, str, name="x")

    def test_wrong_type_tuple(self):
        with pytest.raises(TypeError, match="str or int"):
            assert_isinstance(3.14, (str, int), name="x")


# normalize_choice
class TestNormalizeChoice:
    @pytest.mark.parametrize(("value", "expected"), [("a", "a"), ("A", "a"), ("x", "X"), (None, None), (4, 4)])
    def test_valid(self, value, expected):
        assert normalize_choice(value, ["a", "X", None, 4], name="x") == expected

    @pytest.mark.parametrize("value", ["z", 5, ["a"]])
    def test_invalid(self, value):
        with pytest.raises(ValueError, match=r"Expected `x` to be one of `\['a', 'b'\]`"):
            normalize_choice(value, ["a", "b"], name="x")


# assert_key_in_adata
class TestAssertKeyInAdata:
    def test_key_present(self):
        adata = MagicMock()
        adata.obs = {"cell_type": [1, 2, 3]}
        assert_key_in_adata(adata, "cell_type", attr="obs")

    def test_key_missing(self):
        adata = MagicMock()
        adata.obs = {"cell_type": [1, 2, 3]}
        with pytest.raises(KeyError, match="missing_key"):
            assert_key_in_adata(adata, "missing_key", attr="obs")

    def test_extra_msg(self):
        adata = MagicMock()
        adata.obs = {}
        with pytest.raises(KeyError, match="Run this first"):
            assert_key_in_adata(adata, "key", attr="obs", extra_msg="Run this first.")

    def test_lists_available_keys(self):
        adata = MagicMock()
        adata.obs = {"a": 1, "b": 2}
        with pytest.raises(KeyError, match="Available keys"):
            assert_key_in_adata(adata, "missing", attr="obs")

    def test_container_without_keys_method(self):
        """Fallback to list(container) when .keys() is not available."""
        adata = MagicMock()
        adata.obsm = ["X_pca", "X_umap"]  # list has no .keys()
        with pytest.raises(KeyError, match="X_spatial"):
            assert_key_in_adata(adata, "X_spatial", attr="obsm")


# assert_key_in_sdata
class TestAssertKeyInSdata:
    def test_key_present(self):
        sdata = MagicMock()
        sdata.images = {"image1": "data"}
        assert_key_in_sdata(sdata, "image1", attr="images")

    def test_key_missing(self):
        sdata = MagicMock()
        sdata.images = {"image1": "data"}
        with pytest.raises(KeyError, match="missing"):
            assert_key_in_sdata(sdata, "missing", attr="images")

    def test_extra_msg(self):
        sdata = MagicMock()
        sdata.labels = {}
        with pytest.raises(KeyError, match="Please provide"):
            assert_key_in_sdata(sdata, "mask", attr="labels", extra_msg="Please provide a mask.")

    def test_lists_available_keys(self):
        sdata = MagicMock()
        sdata.images = {"img1": "data", "img2": "data"}
        with pytest.raises(KeyError, match="Available keys"):
            assert_key_in_sdata(sdata, "missing", attr="images")


# assert_isinstance edge cases
class TestAssertIsinstanceEdgeCases:
    def test_bool_is_subclass_of_int(self):
        """bool is a subclass of int - assert_isinstance(True, int) passes."""
        assert_isinstance(True, int, name="x")

    def test_none_type(self):
        with pytest.raises(TypeError, match="str"):
            assert_isinstance(None, str, name="x")


class TestSkillInstallation:
    @pytest.fixture
    def skill_home(self, tmp_path, monkeypatch):
        directory = tmp_path / "home"
        directory.mkdir()
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: directory))
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
        return directory

    def test_bundle_and_links(self):
        source = skill_install._source_dir()
        text = (source / "SKILL.md").read_text()
        metadata = yaml.safe_load(text.split("---", 2)[1])
        assert metadata["name"] == "squidpy" and metadata["description"]
        references = list((source / "references").glob("*.md"))
        assert references
        for path in [source / "SKILL.md", *references]:
            for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text()):
                if "://" not in target and not target.startswith("#"):
                    resolved = (path.parent / target.split("#", 1)[0]).resolve()
                    assert resolved.is_relative_to(source.resolve()) and resolved.is_file()

    def test_install_check_and_refresh(self, tmp_path):
        destination = tmp_path / "squidpy"
        command = ["--dest", str(destination)]
        assert skill_install.main([*command, "--check"]) == 1
        assert not destination.exists()
        assert skill_install.main(command) == 0
        source = skill_install._source_dir()
        expected = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
        assert {
            path.relative_to(destination): path.read_bytes() for path in destination.rglob("*") if path.is_file()
        } == (expected)
        assert skill_install.main([*command, "--check"]) == 0
        marker = destination / "SKILL.md"
        modified = b"locally edited instructions\n"
        marker.write_bytes(modified)
        stale = destination / "references" / "stale.md"
        stale.write_text("old reference")
        assert skill_install.main([*command, "--check"]) == 1
        assert marker.read_bytes() == modified
        assert skill_install.main(command) == 1
        assert marker.read_bytes() == modified and stale.exists()
        assert skill_install.main([*command, "--force"]) == 0
        assert not stale.exists()
        assert skill_install.main([*command, "--check"]) == 0

    def test_identical_install_is_unchanged(self, tmp_path, capsys):
        command = ["--dest", str(tmp_path / "squidpy")]
        assert skill_install.main(command) == 0
        marker = tmp_path / "squidpy" / "SKILL.md"
        modified = marker.stat().st_mtime_ns
        assert skill_install.main(command) == 0
        assert marker.stat().st_mtime_ns == modified
        assert "already up to date" in capsys.readouterr().out

    def test_failed_refresh_preserves_existing_files(self, tmp_path, monkeypatch):
        destination = tmp_path / "squidpy"
        command = ["--dest", str(destination)]
        assert skill_install.main(command) == 0
        marker = destination / "SKILL.md"
        marker.write_text("local instructions")

        def fail_copy(*args, **kwargs):
            raise OSError("copy failed")

        monkeypatch.setattr(skill_install.shutil, "copytree", fail_copy)
        assert skill_install.main([*command, "--force"]) == 1
        assert marker.read_text() == "local instructions"

    def test_force_replaces_existing_file(self, tmp_path):
        destination = tmp_path / "squidpy"
        destination.write_text("old file")
        command = ["--dest", str(destination)]
        assert skill_install.main(command) == 1
        assert destination.read_text() == "old file"
        assert skill_install.main([*command, "--force"]) == 0
        assert skill_install.main([*command, "--check"]) == 0

    @pytest.mark.parametrize(("agent", "config"), [("codex", ".agents"), ("agents", ".agents"), ("claude", ".claude")])
    def test_agent_installation(self, skill_home, agent, config):
        assert skill_install.main(["--agent", agent]) == 0
        assert (skill_home / config / "skills" / "squidpy" / "SKILL.md").is_file()

    def test_custom_claude_configuration(self, skill_home, tmp_path, monkeypatch):
        configuration = tmp_path / "claude-config"
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(configuration))
        assert skill_install.main(["--agent", "claude"]) == 0
        assert (configuration / "skills" / "squidpy" / "SKILL.md").is_file()
        assert not (skill_home / ".claude").exists()

    def test_destination_overrides_agent(self, skill_home, tmp_path):
        destination = tmp_path / "project-skill"
        assert skill_install.main(["--agent", "claude-science", "--dest", str(destination)]) == 0
        assert (destination / "SKILL.md").is_file()
        assert not (skill_home / ".claude-science").exists()

    @pytest.mark.parametrize("organization", ["org-1234", None, "not-json", "../escape", "..\\escape", "..", []])
    def test_claude_science_organization(self, skill_home, organization):
        root = skill_home / ".claude-science"
        root.mkdir()
        if organization is not None:
            contents = "invalid json" if organization == "not-json" else json.dumps({"org_uuid": organization})
            (root / "active-org.json").write_text(contents)
        result = skill_install.main(["--agent", "claude-science"])
        assert result == (0 if organization == "org-1234" else 1)
        if result == 0:
            assert (root / "orgs" / "org-1234" / "skills" / "squidpy" / "SKILL.md").is_file()
        else:
            assert not (root / "orgs").exists()

    def test_explicit_target_required(self, capsys):
        assert skill_install.main([]) == 1
        assert "--agent or --dest" in capsys.readouterr().err

    def test_print_path_does_not_install(self, skill_home, capsys):
        assert skill_install.main(["--print-path"]) == 0
        assert Path(capsys.readouterr().out.strip()) == skill_install._source_dir()
        assert not list(skill_home.iterdir())

    def test_missing_bundle(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(skill_install.resources, "files", lambda _: tmp_path)
        assert skill_install.main(["--print-path"]) == 1
        assert "missing" in capsys.readouterr().err

    def test_force_preserves_symlink_target(self, tmp_path):
        original = tmp_path / "original"
        original.mkdir()
        marker = original / "keep.txt"
        marker.write_text("preserve this file")
        destination = tmp_path / "squidpy"
        destination.symlink_to(original, target_is_directory=True)
        assert skill_install.main(["--dest", str(destination), "--force"]) == 1
        assert destination.is_symlink() and marker.read_text() == "preserve this file"

    def test_destination_must_not_contain_bundle(self, tmp_path, monkeypatch):
        destination = tmp_path / "installation"
        source = destination / "data"
        source.mkdir(parents=True)
        marker = source / "SKILL.md"
        marker.write_text("bundled instructions")
        monkeypatch.setattr(skill_install, "_source_dir", lambda: source)
        assert skill_install.main(["--dest", str(destination), "--force"]) == 1
        assert marker.read_text() == "bundled instructions"
