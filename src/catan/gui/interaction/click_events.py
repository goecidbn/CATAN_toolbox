import numpy as np
from typing import Optional
import vispy.scene as visuals


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


def print_debug(state, data):

    # print("Current session:", state.current_session_id)
    print(f"current session neurons:", data.sessions[0].traces_loaded)
    print(f" union data:", data.union_data.A.shape)
    print(f"union footprints:", data.union_data.A)
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
