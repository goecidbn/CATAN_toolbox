"""Retained inspection results with remappable component identities."""

from catan.core.structures import NeuronComponent as Component
from catan.gui.data.curation_actions import EndpointRef


class RetainedResults:
    def __init__(self, result, data):
        self.snapshot = result
        self.root_id = result.root_result.group_id

        root = result.root_result.target_evidence
        self.remaining = (
            None
            if root is None
            else {
                target: set(sources)
                for target, sources in root.items()
                if target.neuron_id in result.neurons
            }
        )

        self.positions = {
            Component(n, s): Component(n, s)
            for n, row in enumerate(data.assignments.ids)
            for s in [None] + [sid for sid, fp in enumerate(row) if fp >= 0]
        }
        self._remember_data(data)

    def _remember_data(self, data):
        self.assignments = data.assignments
        self.sessions = tuple(data.sessions)
        self.ids = data.assignments.ids.copy()

    @property
    def neurons(self):
        endpoints = (
            self.remaining
            if self.remaining is not None
            else [Component(n, None) for n in self.snapshot.neurons]
        )
        return {
            component.neuron_id
            for old in endpoints
            if (component := self.positions.get(old)) is not None
        }

    def root_targets(self):
        if self.remaining is None:
            return {}

        mapping = {}
        for old_target, sources in self.remaining.items():
            target = self.positions.get(old_target)
            if target is not None:
                mapping.setdefault(target, set()).update(
                    component
                    for source in sources
                    if (component := self.positions.get(source)) is not None
                )
        return mapping

    def inspection_targets(self, neuron_id, source_type, source_id):
        if source_type == "group" and source_id == self.root_id:
            if self.remaining is None:
                return None
            return {
                target: sources
                for target, sources in self.root_targets().items()
                if target.neuron_id == neuron_id
            }

        snapshot = self.snapshot

        if source_type == "group":
            node = snapshot.group_results.get(source_id)
        else:
            condition = snapshot.condition_results.get(source_id)
            node = None if condition is None else condition.projected_result

        raw = snapshot.bound_inspections.get(source_id)
        if raw is None:
            raw = None if node is None else node.target_evidence
        if raw is None:
            return None

        mapping = {}
        for old_target, sources in raw.items():
            if old_target.neuron_id not in snapshot.neurons:
                continue

            if self.remaining is not None:
                if old_target not in self.remaining:
                    continue
                sources = sources & self.remaining[old_target]

            target = self.positions.get(old_target)
            if target is not None and target.neuron_id == neuron_id:
                mapping.setdefault(target, set()).update(
                    component
                    for source in sources
                    if (component := self.positions.get(source)) is not None
                )

        return mapping

    def _components(self, neuron_id, method, *args):
        if neuron_id not in self.neurons:
            return set()

        if self.remaining is not None:
            mapping = self.inspection_targets(neuron_id, *args)
            if mapping is not None:
                return set(mapping) | {
                    source for sources in mapping.values() for source in sources
                }

        components = set()
        for old_n in self.snapshot.neurons:
            owner = self.positions.get(Component(old_n, None))
            if owner is not None and owner.neuron_id == neuron_id:
                values = getattr(self.snapshot, method)(old_n, args[-1])
                components.update(
                    component
                    for old in values
                    if (component := self.positions.get(old)) is not None
                )

        return components

    def components_for_condition(self, neuron_id, condition_id):
        return self._components(
            neuron_id,
            "components_for_condition",
            "condition",
            condition_id,
        )

    def components_for_group(self, neuron_id, group_id):
        return self._components(
            neuron_id,
            "components_for_group",
            "group",
            group_id,
        )

    def applied(self, kind, sources, targets, prepared, data):
        # Consume the completed result before remapping assignment IDs.
        if self.remaining is not None:
            for old_target, old_sources in list(self.remaining.items()):
                target = self.positions.get(old_target)

                if kind == "split" and target in targets:
                    del self.remaining[old_target]
                    continue

                kept = {
                    old_source
                    for old_source in old_sources
                    if not (
                        (
                            target in targets
                            and self.positions.get(old_source) in sources
                        )
                        or (
                            kind == "neuron_merge"
                            and target in sources
                            and self.positions.get(old_source) in targets
                        )
                    )
                }

                if old_sources and not kept:
                    del self.remaining[old_target]
                else:
                    self.remaining[old_target] = kept

        for old, current in self.positions.items():
            if current is None:
                continue

            key = (current.neuron_id, current.session_id)

            if key in prepared.component_map:
                mapped = prepared.component_map[key]
                self.positions[old] = None if mapped is None else Component(*mapped)
            else:
                neuron_id = prepared.row_map.get(current.neuron_id)
                self.positions[old] = (
                    None
                    if neuron_id is None
                    else Component(neuron_id, current.session_id)
                )

        self._remember_data(data)

    def data_changed(self, data):
        # External changes: follow physical footprints.
        # Neuron endpoints require unchanged membership.
        same_data = (
            data.assignments is self.assignments
            and len(data.sessions) == len(self.sessions)
            and all(a is b for a, b in zip(data.sessions, self.sessions))
        )

        owners = {}
        if same_data:
            for n, row in enumerate(data.assignments.ids):
                for sid, fp in enumerate(row):
                    if fp >= 0 and data.sessions[sid].included[int(fp)]:
                        owners.setdefault((sid, int(fp)), []).append(n)

        for old, current in self.positions.items():
            replacement = None

            if same_data and current is not None:
                if current.session_id is None:
                    try:
                        replacement = EndpointRef.capture(self.ids, current).resolve(
                            data.assignments.ids
                        )
                    except ValueError:
                        pass
                else:
                    sid = current.session_id
                    fp = int(self.ids[current.neuron_id, sid])
                    rows = owners.get((sid, fp), [])

                    if len(rows) == 1:
                        replacement = Component(rows[0], sid)

            self.positions[old] = replacement

        self._remember_data(data)
