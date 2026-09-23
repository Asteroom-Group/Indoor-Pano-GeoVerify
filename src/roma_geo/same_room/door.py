# -*- coding: utf-8 -*-
"""某一側有八成以上落在門裡的共視整段拿掉。還留得下一段才判同房。"""
from .arcs import covered_length

CONTAIN = 0.8


def as_arcs(doors):
    """門寫成 (起始方位角, 寬度)。接受 (az0, width) 或含 az0、width_deg 的對應。"""
    arcs = []
    for door in doors or []:
        if isinstance(door, (tuple, list)):
            arcs.append((float(door[0]), float(door[1])))
            continue
        start = float(door["az0"])
        if door.get("width_deg") not in (None, ""):
            width = float(door["width_deg"])
        else:
            width = float(door["az1"]) - start
            if width < 0.0:
                width += 360.0
        arcs.append((start, width))
    return arcs


def side_in_door(az_range, door_arcs, contain):
    if not az_range or not door_arcs:
        return False
    start, end = float(az_range[0]), float(az_range[1])
    width = end - start
    if width < 0.0:
        width += 360.0
    if width <= 0.0:
        return False
    return covered_length(start, width, door_arcs) / width >= contain


def decides_same_room(regions, doors_a, doors_b, contain=CONTAIN):
    arcs_a = as_arcs(doors_a)
    arcs_b = as_arcs(doors_b)
    for region in regions or []:
        side_a = region.get("a") or {}
        side_b = region.get("b") or {}
        in_a = side_in_door(side_a.get("az_range"), arcs_a, contain)
        in_b = side_in_door(side_b.get("az_range"), arcs_b, contain)
        if not in_a and not in_b:
            return True
    return False
