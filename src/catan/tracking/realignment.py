"""Prepare a realignment without mutating live sessions or assignments."""

from copy import deepcopy

import numpy as np
from scipy import sparse

from catan.core.data import center_of_mass
from catan.core.io import load_fields_from_sources


def build_realignment_update(
    tracking, session_id, *, background_template, remap, background_spec
):
    sessions = tuple(tracking.sessions)
    session = sessions[session_id]
    if not session.status["spatial_loaded"]:
        raise ValueError("Load spatial data before realigning this session.")
    if not remap.report.success:
        raise ValueError("Cannot commit an unsuccessful alignment proposal.")
    if any(tracking.alignment_is_stale(i) for i in range(session_id)):
        raise ValueError("Update earlier stale alignments first.")
    if session.source_config is None or session.path is None:
        raise ValueError("The original footprint source is not configured.")

    config = deepcopy(session.source_config)
    group = config.groups.get("spatial")
    if group is None or "footprints" not in group.fields:
        raise ValueError("No original footprint field is configured.")

    dims = tuple(session.dims)
    template = np.asarray(background_template, dtype=np.float32).copy()
    if template.shape != dims or not np.all(np.isfinite(template)):
        raise ValueError("Invalid proposed background template.")
    if tuple(remap.dims) != dims:
        raise ValueError("Proposed transform has incompatible dimensions.")

    included = np.asarray(session.included, dtype=bool).copy()
    synthetic = np.asarray(session.synthetic, dtype=bool).copy()
    if included.shape != (session.n_neurons,) or synthetic.shape != included.shape:
        raise ValueError("Component flags do not match the loaded footprints.")
    n_raw = int(np.count_nonzero(~synthetic))
    if np.any(synthetic[:n_raw]) or not np.all(synthetic[n_raw:]):
        raise ValueError("Original footprint IDs are not a contiguous prefix.")

    # Read ONLY original footprints. No traces, backgrounds or postprocessing.
    loaded = load_fields_from_sources(
        session.path, {"spatial": {"footprints": group.fields["footprints"]}}
    )
    raw = sparse.csc_matrix(loaded["spatial"]["footprints"])
    if raw.shape != (int(np.prod(dims)), n_raw):
        raise ValueError(
            "Original footprint dimensions/count differ from the loaded session. "
            "Realignment cannot safely preserve component IDs."
        )

    remap = deepcopy(remap)
    aligned = remap.apply_remap(raw, use_optical_flow=False)
    footprints = sparse.hstack([aligned, session.footprints[:, n_raw:]], format="csc")
    if background_spec is not None:
        group.fields["background"] = deepcopy(background_spec)

    # result component -> source component, across all assignment histories.
    # History entries describe direct copies in common coordinates.
    sources = {}
    for assignments in tracking._assignments.values():
        for record in assignments.manipulations.values():
            origins = record.get("sources", [])
            results = record.get("results", [])
            if len(origins) != len(results):
                raise ValueError("Inconsistent synthetic-component history.")
            for origin, result in zip(origins, results):
                target = (int(result["session_id"]), int(result["footprint_id"]))
                source = (int(origin["session_id"]), int(origin["footprint_id"]))
                if target in sources and sources[target] != source:
                    raise ValueError(
                        f"Conflicting synthetic provenance for component {target}."
                    )
                sources[target] = source

    # Without provenance we cannot know whether a copy depends on this session.
    for sid, item in enumerate(sessions):
        for fid in np.flatnonzero(item.synthetic):
            node = (sid, int(fid))
            if node not in sources:
                raise ValueError(
                    f"Synthetic component {node} has no source history. "
                    "Load its assignments/manipulation history before realigning."
                )

    def validate_node(node):
        sid, fid = node
        if not (0 <= sid < len(sessions)):
            raise ValueError(f"Synthetic source references missing session {sid}.")
        item = sessions[sid]
        if not (0 <= fid < item.n_neurons):
            raise ValueError(f"Synthetic source references missing component {node}.")

    depends_cache = {}
    visiting = set()

    def depends_on_change(node):
        if node in depends_cache:
            return depends_cache[node]
        validate_node(node)
        if node in visiting:
            raise ValueError("Cyclic synthetic-component history.")
        visiting.add(node)
        sid, fid = node
        if sessions[sid].synthetic[fid]:
            source = sources.get(node)
            if source is None:
                raise ValueError(f"Missing synthetic provenance for {node}.")
            result = depends_on_change(source)
        else:
            result = sid == session_id
        visiting.remove(node)
        depends_cache[node] = result
        return result

    replacements = {}
    resolved = {}

    def resolve_changed(node):
        if node in resolved:
            return resolved[node]
        sid, fid = node
        if sessions[sid].synthetic[fid]:
            column = resolve_changed(sources[node])
        else:
            column = footprints[:, fid]
        resolved[node] = column
        return column

    for sid, item in enumerate(sessions):
        for fid in np.flatnonzero(item.synthetic):
            node = (sid, int(fid))
            if not depends_on_change(node):
                continue
            if not item.status["spatial_loaded"]:
                raise ValueError(f"Load spatial data for dependent session {sid}.")
            if tuple(item.dims) != dims:
                raise ValueError("Dependent synthetic footprint dimensions differ.")
            replacements.setdefault(sid, {})[int(fid)] = resolve_changed(node)

    updates = {session_id: {"footprints": footprints}}
    for sid, columns in replacements.items():
        current = footprints if sid == session_id else sessions[sid].footprints
        updated = sparse.hstack(
            [
                columns[fid] if fid in columns else current[:, fid]
                for fid in range(current.shape[1])
            ],
            format="csc",
        )
        updates[sid] = {"footprints": updated}

    updates[session_id].update(
        background_template=template,
        background=remap.apply_remap(template, use_optical_flow=False),
        background_origin=(
            "loaded" if background_spec is not None else session.background_origin
        ),
        remap=remap,
        source_config=config,
        included=included,
        synthetic=synthetic,
        alignment_issue=None,
        alignment_metrics={
            "shift": remap.report.shift,
            "correlation": remap.report.correlation,
            "correlation_zscore": remap.report.correlation_zscore,
        },
    )
    for sid, values in updates.items():
        item = sessions[sid]
        values["centroids"] = center_of_mass(
            values["footprints"],
            *item.dims,
            convert=item.params.get("pxtomu", 1.0),
        )
        values["status"] = dict(item.status, registered_to_model=False)
        if sid == session_id:
            values["status"]["aligned"] = True

    return {"sessions": sessions, "updates": updates, "session_id": session_id}
