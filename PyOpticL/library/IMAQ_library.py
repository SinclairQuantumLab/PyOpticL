"""Convenience component library for Sinclair Lab optical-board designs."""

from PyOpticL.layout import Component
from PyOpticL.library import Sinclair_library as sinclair


def mirror(label: str = "Mirror", thumbscrews: bool = True) -> Component:
    return Component(
        label=label,
        definition=sinclair.circular_mirror(
            mount_type=sinclair.mirror_mount_M05,
            mount_args={"thumbscrews": thumbscrews},
        ),
    )


def mirror_fix(label: str = "Fixed Mirror", thumbscrews: bool = False) -> Component:
    return Component(
        label=label,
        definition=sinclair.circular_mirror(
            mount_type=sinclair.mirror_mount_FMP05,
            mount_args={},
        ),
    )


def mirror_3knob(label: str = "Mirror 3-Knob", thumbscrews: bool = True) -> Component:
    return Component(
        label=label,
        definition=sinclair.circular_mirror(
            mount_type=sinclair.mirror_mount_M05X,
            mount_args={"thumbscrews": thumbscrews},
        ),
    )


def mirror_u(label: str = "Union Mirror", thumbscrews: bool = True) -> Component:
    return Component(
        label=label,
        definition=sinclair.circular_mirror_union_optic(
            mount_type=sinclair.mirror_mount_M05,
            mount_args={"thumbscrews": thumbscrews},
        ),
    )


def mirror_u_3knob(
    label: str = "Union Mirror 3-Knob",
    thumbscrews: bool = True,
) -> Component:
    return Component(
        label=label,
        definition=sinclair.circular_mirror_union_optic(
            mount_type=sinclair.mirror_mount_M05X,
            mount_args={"thumbscrews": thumbscrews},
        ),
    )


def lens_150(label: str = "Lens f=150 mm") -> Component:
    return Component(
        label=label,
        definition=sinclair.circular_lens(
            focal_length=150,
            mount_type=sinclair.lens_holder_l05g,
        ),
    )


def lens_50(label: str = "Lens f=50 mm") -> Component:
    return Component(
        label=label,
        definition=sinclair.circular_lens(
            focal_length=50,
            mount_type=sinclair.lens_holder_l05g,
        ),
    )


def hwp(label: str = "Half-Wave Plate", fast_axis_angle: float = 0.0) -> Component:
    return Component(
        label=label,
        definition=sinclair.waveplate(
            retardance=0.5,
            fast_axis_angle=fast_axis_angle,
            mount_type=sinclair.rotation_stage_rsp05,
        ),
    )


def qwp(label: str = "Quarter-Wave Plate", fast_axis_angle: float = 0.0) -> Component:
    return Component(
        label=label,
        definition=sinclair.waveplate(
            retardance=0.25,
            fast_axis_angle=fast_axis_angle,
            mount_type=sinclair.rotation_stage_rsp05,
        ),
    )


def cube_05(label: str = "0.5-inch Beamsplitter Cube") -> Component:
    return Component(
        label=label,
        definition=sinclair.cube_splitter(
            cube_size=12.7,
            invert=False,
            ref_ratio=0.5,
            mount_type=sinclair.cube_mount_halfinch,
        ),
    )


def cube_05_rot(label: str = "0.5-inch Beamsplitter Cube Rotated") -> Component:
    return Component(
        label=label,
        definition=sinclair.cube_splitter(
            cube_size=12.7,
            invert=True,
            ref_ratio=0.5,
            mount_type=sinclair.cube_mount_halfinch_rot90,
        ),
    )


def isolator1(label: str = "IO-3D-780-VLP Isolator") -> Component:
    return Component(label=label, definition=sinclair.isolator_780())


def isolator2(label: str = "IOT-5-780-MP Isolator") -> Component:
    return Component(label=label, definition=sinclair.isolator_780_mp())


def fiberport(
    label: str = "Fiberport",
    fiber_clamp: str | bool = "Standard",
    thumbscrews: bool = True,
) -> Component:
    return Component(
        label=label,
        definition=sinclair.fiberport_mount_KA05T(
            Fiber_Clamp=fiber_clamp,
            mount_args={"thumbscrews": thumbscrews},
        ),
    )


def ips_small(label: str = "IPS Laser Small Adapter") -> Component:
    return Component(
        label=label,
        definition=sinclair.ips_laser(adapter="small"),
    )


def ips_big(label: str = "IPS Laser Big Adapter") -> Component:
    return Component(
        label=label,
        definition=sinclair.ips_laser(adapter="big"),
    )


def toptica(label: str = "Toptica DFB Laser") -> Component:
    return Component(label=label, definition=sinclair.toptica_laser())


def ta(label: str = "Tapered Amplifier") -> Component:
    return Component(label=label, definition=sinclair.TA_butterfly())


def aom(
    label: str = "AOMO 3100-125",
    fiber_clamp: str | bool = False,
    rf_frequencies: float | list[float] = 0.0,
) -> Component:
    clamp_value = "None" if fiber_clamp is False else fiber_clamp
    if clamp_value is None:
        clamp_value = "None"
    if clamp_value is True:
        clamp_value = "Standard"
    return Component(
        label=label,
        definition=sinclair.AOMO_3100_125(
            Fiber_Clamp=clamp_value,
            rf_frequencies=rf_frequencies,
        ),
    )


def iris(label: str = "Iris") -> Component:
    return Component(label=label, definition=sinclair.pinhole_ida12())


def shutter(label: str = "Shutter") -> Component:
    return Component(label=label, definition=sinclair.shutter_sr475())


def vapor_cell(label: str = "Rubidium Vapor Cell") -> Component:
    return Component(label=label, definition=sinclair.Vapor_Ref_Cell())


def pd2(label: str = "PDB250A Photodetector") -> Component:
    return Component(label=label, definition=sinclair.photodetector_pdb250aa())
