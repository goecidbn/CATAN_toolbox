from PySide6.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem

# import importlib
from catan.gui.plots import BasePlot

# importlib.reload(BasePlot)
# print("reloading SelectionDisplay")


class Display(QWidget):

    def __init__(self, parent, controls, config=None):
        super().__init__(parent)

        self.state = parent.state
        self.data = parent.data
        # self.tracking = parent.tracking

        self.controls = controls

        menu_layout = QVBoxLayout(self)
        menu_layout.setContentsMargins(0, 0, 0, 0)

        # Create statistics table
        self.stats_table = QTableWidget()

        # Define metric rows
        self.metrics = {
            "SNR": "SNR_comp",
            "r-value": "r_values",
            "CNN": "cnn_preds",
            # "Size": ,
            # "Spatial Footprint",
            # "Temporal Component",
        }
        self.stats_table.setColumnCount(
            2 + len(self.metrics)
        )  # one for ID, rest for metrics
        self.stats_table.setHorizontalHeaderLabels(
            ["(s,n)", "ID", *self.metrics.keys()]
        )
        self.stats_table.setRowCount(0)

        self.stats_table.resizeColumnsToContents()
        menu_layout.addWidget(self.stats_table)

        self.state.focused_component_changed.connect(self.highlight_display)

        self.state.data_changed.connect(self.update_display)

    def update_display(self):
        if self.state.selected_components is None:
            self.stats_table.setRowCount(0)
            return

        self.stats_table.setRowCount(len(self.state.selected_components))

        for n, component in enumerate(self.state.selected_components):
            footprint_id = self.state.get_footprint_from_component(component)

            self.stats_table.setItem(
                n,
                0,
                QTableWidgetItem(f"{component.id[0]},{int(component.id[1])}"),
            )
            if footprint_id is None or footprint_id < 0:
                self.stats_table.setItem(n, 1, QTableWidgetItem("--"))
                for i, metric in enumerate(self.metrics):
                    self.stats_table.setItem(n, i + 2, QTableWidgetItem("--"))
            else:
                self.stats_table.setItem(n, 1, QTableWidgetItem(f"{footprint_id}"))
                for i, metric in enumerate(self.metrics):
                    if (
                        self.metrics[metric]
                        not in self.data.sessions[component.session_id].quality
                    ):
                        self.stats_table.setItem(n, i + 2, QTableWidgetItem("--"))
                        continue
                    value = "{:.2f}".format(
                        self.data.sessions[component.session_id].quality[
                            self.metrics[metric]
                        ][footprint_id]
                    )
                    self.stats_table.setItem(n, i + 2, QTableWidgetItem(value))
        self.stats_table.resizeColumnsToContents()

    def highlight_display(self):
        # print("Highlighting display for selected neuron")
        if (
            self.state.selected_components is None
            or self.state.focused_component is None
        ):
            self.stats_table.clearSelection()
            return
        idx = self.state.selected_components.index(self.state.focused_component)
        self.stats_table.selectRow(idx)


class Controller(BasePlot.TableController):
    pass
