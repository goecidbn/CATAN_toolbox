from PySide6.QtWidgets import QInputDialog, QMessageBox

class ConfigData():
    """Class to hold configuration data for the application."""

    def __init__(self, parent):

        super().__init__()
        self.parent = parent

        ## defines paths which can be defined (and reloaded) by GUI
        self.paths = {
            "root": {
                "mode": ["single", "tracking"],
                "type": "folder",
                "label": "Root folder",
                "root": None,
            },
            "model": {
                "mode": ["tracking"],
                "type": "file",
                "label": "Model file",
                "root": "root",
            },
            "registration": {
                "mode": ["tracking"],
                "type": "file",
                "label": "Registration file",
                "root": "root",
            },
        }