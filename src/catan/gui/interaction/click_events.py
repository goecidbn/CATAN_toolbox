import numpy as np
from typing import Optional
import vispy.scene as visuals

from catan.gui.structures import Data, AppState


def scene_to_data(scene, data):

    if data.ndim == 1:
        data = data[None, :]
    tr = scene.node_transform(scene)
    mapped = tr.map(data)

    if mapped.shape[-1] == 4:
        screen = mapped[:, :2] / mapped[:, 3:4]
    else:
        screen = mapped[:, :2]
    # print("screen min/max:", screen.min(axis=0), screen.max(axis=0))
    return screen


def visual_to_canvas(visual, data):
    tr = visual.get_transform(map_from="visual", map_to="canvas")
    mapped = tr.map(data)

    if mapped.shape[-1] == 4:
        screen = mapped[:, :2] / mapped[:, 3:4]
    else:
        screen = mapped[:, :2]
    # print("screen min/max:", screen.min(axis=0), screen.max(axis=0))
    return screen


def canvas_to_visual(visual, pos):
    """
    mostly for casting mouse event positions to data coords. pos should be (N, 2) array of screen coords.
    """
    pos = np.asarray(pos, dtype=np.float32)
    tr = visual.get_transform(map_from="canvas", map_to="visual")
    mapped = tr.map(pos)

    if mapped.shape[-1] == 4:
        screen = mapped[..., :2] / mapped[..., 3:4]
    else:
        screen = mapped[..., :2]
    return screen


def get_footprint_id_from_mouse_pos(canvas, pos, centroids) -> Optional[int]:
    # Transform screen -> canvas -> data coords
    data_pos = visual_to_canvas(canvas.h_background, centroids)
    tr = canvas.scene.node_transform(canvas.h_background)
    data_pos = tr.map(pos)
    x, y = float(data_pos[1]), float(data_pos[0])
    # print("position (x,y):",x,y)

    # find closest footprint
    distances = (centroids[:, 0] - x) ** 2 + (centroids[:, 1] - y) ** 2
    footprint_id = np.argmin(distances).astype(int)

    if np.sqrt(distances[footprint_id]) > 10.0:
        footprint_id = None
    return footprint_id


def print_debug(state: AppState, data: Data):

    print("Current session:", state.current_session_id)
    print("Sessions:", data.sessions)

    print("session neurons:", data.sessions[0].n_neurons)

    print("Model")
    print("Model details:", data.model)
    print("Model counts:", data.model.aggregate_counts())
    print("Model single counts", data.model.counts)

    # for session in data.sessions:
    #     print(f"remap {session.id}:", session.remap.report)

    # print("background:", data.sessions[0].background)
    # print("background:", np.percentile(data.sessions[0].background, [5, 50, 95]))
    # print("session included:", data.sessions[0].included.shape)

    # print("assignments info")
    # print("IDs:", data.assignments.ids.shape)
    # print("union n neurons:", data.assignments.union.n_neurons)
    # print("union included shape:", data.assignments.union.included.shape)
    # print("union synthetic shape:", data.assignments.union.synthetic.shape)
    # print("union footprints shape:", data.assignments.union.footprints.shape)
    # print(
    #     "match state:",
    #     data.assignments.matched_status.shape,
    #     data.assignments.matched_status,
    # )
    # print("review state:", data.assignments.review_status)
    # print("manipulations:", data.assignments.manipulations)

    # print("union included:", data.assignments.union.included)
    # print("union synthetic:", data.assignments.union.synthetic)

    # print("Manipulations:")
    # print(data.assignments.manipulations)

    # data.rebuild_union_neurons([215])
    # for s, fp_ids in enumerate(data.assignments.ids[215, :]):
    #     print(f"Session {s}, footprint IDs: {fp_ids}")
    #     print(data.sessions[s].footprints[:, fp_ids])
    # print(data.assignments.ids[215, :])
    # print(f"footprint 215:", data.assignments.union.footprints[:, 215])
    # print(f"footprint 477:", data.assignments.union.footprints[:, 477])

    # print(data.model.)
    # print("Status of first session:", data.sessions[0].status)
    # print("Footprints:", data.sessions[0].footprints.shape)
    # print("Footprints:", data.sessions[0].footprints)

    if print_remap := False:
        print("Remapping:")
        for session in data.sessions:
            print(f"\nSession {session.id} ({session.name})")
            if session.remap:
                print(f"Remap - transposed: {session.remap.transpose}")
                print(f"Remap - c_max: {session.remap.c_max}")
                print(f"Remap - c_zscored: {session.remap.c_zscored}")
                print(f"Remap - shift: {session.remap.shift}")
                print(f"Remap - total shift: {session.remap.total_shift}")

            session.evaluate_alignment_status()
            print("aligned:", session.status["aligned"])

    # q_params = data.sessions[0].quality.keys()
    # print(f"Session quality params: {q_params}")
    # for q in q_params:
    #     print(f"Session {data.sessions[0].id}, {q}: {data.sessions[0].quality[q][:10]}")

    # print("Assignments:", data.assignments.ids)
    # print("Assignments (state):", state.assignments)
    # print("Assignments:", data.assignments.ids.shape)
    # print("Assignments:", data.assignments.union.footprints)
    # print("tr")

    # print("session config:", data.sessions[0].source_config)

    # print("union data:", data.assignments.union)
    # print("union data:", data.assignments.union.footprints)
    # print("union data:", data.assignments.union.centroids)

    # print("Load config:", data.load_configs.current)
    # print("session traces:", data.sessions[0].traces.keys())
    # print("session traces:", data.sessions[0].traces)
    # print("session traces:", data.sessions[0].quality.keys())

    # print("")
    # print("Session data: ", data.sessions[0].name, data.sessions[0].path)
    # print("Session status:", data.sessions[0].status)

    # for session in data.sessions:
    #     print(f"Session {session.id} ({session.name}) status: {session.status}")

    # # print("config:", config.fields)
    # print("model reference_data:", data.reference_data.id, data.reference_data.name)
    # # print("counts:", data.counts)
    # print("counts sum:", data.counts["cross"][..., 0].sum())
    # print("model:", data.model)
    # print("evaluate f_same: ", data.model.f_same(1.0, 0.9))
    # print("ids:",data.session_order)
    # print("Session names:", [s.name for s in data.sessions])
    # print("Session order:", [(s.id,s.name) for s in data.sessions])
    # print("Session neuron numbers:", [s.n_neurons for s in data.sessions])
    # print("Selected components:", state.selected_components)
    # print("Session colors:", state._session_colors)
    # print("Sesssion:",data.current_session.name)
    # print("Sesssion:",data.current_session.path)

    # print(f"current session traces:", data.current_session.status["traces_loaded"])
    # print(f"current session traces:", data.current_session.traces)
    # print(f" union data:", data.union_data.A.shape)
    # print(f"union footprints:", data.union_data.A)
    # print(f"current session neurons:", data.sessions[1].traces_loaded)
    # print(f"current session neurons:", data.sessions[0].traces)

    # print(f"model counts:", data.counts["cross"].sum(axis=(0, 1)))
    # print(f"model:", data.model)
    # print(f"Data neurons:", len(data.neurons))
    # print(
    #     f"centroids shape:",
    #     data.centroids.shape if data.centroids is not None else None,
    # )
    # # print("footprint example:", data.neurons[0].footprints[:, 0].indices)
    # print(state.assignments.shape)


# def print_debug_primary(primary_display):
#     print("Primary display debug info:")
#     print("Current session:", primary_display.state.current_session_id)
#     print("roi handles:", primary_display.roi_handles)

#     # print("Data sessions:", sorted(primary_display.data.sessions.items()))
