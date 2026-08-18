from PySide6.QtWidgets import QInputDialog, QMessageBox
from catan.core.structures import LoadConfig

class ConfigData(LoadConfig):
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



    def change_config_fields(self, group_key:str, label:str, get_field=None, method="edit", subpath="/estimates"):

        if method == "edit":
            """
                can correspond to 
                1. changing an existing entry (in static)
                2. adding a new entry (in dynamic)
            """
            if get_field is None:
                get_field = lambda: QInputDialog.getText(
                    self.parent,
                    "Change field path",
                    "Enter new path:",
                    text=label
                )
            field = get_field()
            if field is None:
                return
            
            if field in self.fields[group_key]["opts"].values():
                QMessageBox(None, "Adding field is not possible", f"Field {field} already exists in load options for {group_key}.")
                return

            if label is None:
                label = field
            
            self.fields[group_key]["opts"][label] = field
            

        elif method == "rename":
            """
                assigns a new name to an existing field, which is later used in the GUI
            """
            new_name, ok = QInputDialog.getText(
                self.parent,
                "Change field title",
                "Enter new title:",
                text=label,
            )

            if ok and new_name:
                self.fields[group_key]["opts"][new_name] = self.fields[group_key]["opts"].pop(label)
            
        elif method == "remove":
            del self.fields[group_key]["opts"][label]
