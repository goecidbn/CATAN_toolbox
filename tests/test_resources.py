from catan.gui.resources import load_stylesheet, read_bytes


def test_main_stylesheet_can_be_loaded():
    stylesheet = load_stylesheet("main.qss")

    assert isinstance(stylesheet, str)
    assert stylesheet.strip()



# def test_icon_is_packaged():
#     data = read_bytes("icons", "catan.svg")

#     assert data