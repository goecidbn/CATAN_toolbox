import pytest
from pathlib import Path

from catan.cli import main
from catan.cli.pipeline import (
    resolve_output_directory,
    resolve_session_paths,
)

def test_top_level_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_match_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["match", "--help"])

    assert exc_info.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_no_command_prints_help(capsys):
    exit_code = main([])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "usage:" in captured.out
    assert "match" in captured.out



def test_resolve_root_and_glob(tmp_path):
    session_1 = tmp_path / "Session01" / "results.hdf5"
    session_2 = tmp_path / "Session02" / "results.hdf5"

    session_1.parent.mkdir()
    session_2.parent.mkdir()

    session_1.touch()
    session_2.touch()

    paths = resolve_session_paths(
        explicit_paths=[],
        root=tmp_path,
        glob_pattern="Session*/results.hdf5",
        path_file=None,
    )

    assert paths == [
        session_1.resolve(),
        session_2.resolve(),
    ]


def test_root_is_default_output_directory(tmp_path):
    result = resolve_output_directory(
        paths=[],
        explicit_output=None,
        root=tmp_path,
    )

    assert result == tmp_path.resolve() / "matching"


def test_explicit_output_overrides_default(tmp_path):
    output = tmp_path / "custom-results"

    result = resolve_output_directory(
        paths=[],
        explicit_output=output,
        root=tmp_path / "mouse",
    )

    assert result == output.resolve()


def test_mixed_input_modes_are_rejected(tmp_path):
    session = tmp_path / "session.hdf5"
    session.touch()

    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_session_paths(
            explicit_paths=[session],
            root=tmp_path,
            glob_pattern="*.hdf5",
            path_file=None,
        )