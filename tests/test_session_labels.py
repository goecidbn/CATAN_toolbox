from types import SimpleNamespace

from catan.gui.utils.session_labels import session_label, session_labels, session_tag


def test_session_labels():
    sessions = [SimpleNamespace(name="20240721", id=0), SimpleNamespace(name="", id=1)]
    data = SimpleNamespace(sessions=sessions)

    assert session_tag(3) == "S3"
    assert session_label(data, 0) == "20240721 (S0)"
    assert session_label(sessions, 0, short=True) == "S0"
    assert session_label(sessions, 1) == "S1"  # no name
    assert session_label(sessions, 5) == "S5"  # out of range
    assert session_label(sessions[0]) == "20240721 (S0)"  # a session itself
    assert session_label(SimpleNamespace(name="new", id=-1)) == "new"  # unregistered
    assert session_labels(data, [0, 1]) == "20240721 (S0), S1"
