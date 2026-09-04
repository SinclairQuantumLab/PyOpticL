# Sinclair Lab component library for PyOpticL v2
# It contains every directly referenced optomech class and every class-level
# subcomponent dependency required by those boards. Generic optical classes are
# retained temporarily for v1-board compatibility and can later be replaced by
# the stock PyOpticL.library.optics implementations.

from math import *

from pathlib import Path

import FreeCAD as App

import numpy as np

import Part

from PyOpticL import settings

from PyOpticL.beam_path import (
    AcoustoOptic,
    Interface,
    Lens,
    Reflection,
    Stop,
    Waveplate,
)

from PyOpticL.icons import optic_icon, thorlabs_icon

from PyOpticL.layout import Component, Subcomponent

from PyOpticL.library.optics import Cavity_Mirror

from PyOpticL.utils import Dimension as dim

from PyOpticL.utils import (
    bolt_shape,
    bolt_slot_shape,
    box_shape,
    cylinder_shape,
    default_bolt_length,
    import_model,
)

MODELS_DIRECTORY = Path(__file__).resolve().parent.parent / "models"

class LazyModel:
    """Load a CAD model only when the component mesh is first accessed."""

    def __init__(self, model_name, directory=MODELS_DIRECTORY):
        self.model_name = model_name
        self.directory = directory
        self._cached_mesh = None

    def __get__(self, instance, owner):
        if self._cached_mesh is None:
            self._cached_mesh = import_model(
                self.model_name,
                directory=self.directory,
            )
        return self._cached_mesh


INCH = dim(1, "in")

DRILL_DEPTH = dim(100, "mm")

bolt_4_40 = {
    "clear_dia": dim(0.120, "in"),
    "tap_dia": dim(0.089, "in"),
    "head_dia": dim(5.50, "mm"),
    "head_dz": dim(2.5, "mm"),
}

bolt_8_32 = {
    "clear_dia": dim(0.172, "in"),
    "tap_dia": dim(0.136, "in"),
    "head_dia": dim(7, "mm"),
    "head_dz": dim(4.4, "mm"),
}

bolt_14_20 = {
    "clear_dia": dim(0.260, "in"),
    "tap_dia": dim(0.201, "in"),
    "head_dia": dim(9.8, "mm"),
    "head_dz": dim(8, "mm"),
    "washer_dia": dim(9 / 16, "in"),
}

bolt_m2p5x4p5 = dict(bolt_14_20)

bolt_M2_5 = {
    "clear_dia": dim(2.7, "mm"),
    "tap_dia": dim(2.05, "mm"),
    "head_dia": dim(4.5, "mm"),
    "head_dz": dim(2.0, "mm"),
}

bolt_M4 = {
    "clear_dia": dim(4.3, "mm"),
    "tap_dia": dim(3.3, "mm"),
    "head_dia": dim(7.0, "mm"),
    "head_dz": dim(4.0, "mm"),
}

bolt_M6 = {
    "clear_dia": dim(6.6, "mm"),
    "tap_dia": dim(5.0, "mm"),
    "head_dia": dim(10.0, "mm"),
    "head_dz": dim(4.0, "mm"),
}

adapter_color = (0.6, 0.9, 0.6)

mount_color = (0.5, 0.5, 0.55)

glass_color = (0.5, 0.5, 0.8)

misc_color = (0.2, 0.2, 0.2)

def _custom_box(dx, dy, dz, x, y, z, fillet=0, dir=(0, 0, 1), fillet_dir=None):
    """Compatibility geometry helper implemented with the v2 shape API."""
    if fillet_dir is None:
        fillet_dir = tuple(np.abs(dir))
    return box_shape(
        dimensions=(dx, dy, dz),
        position=(x, y, z),
        center=tuple(-d for d in dir),
        fillet=fillet,
        fillet_direction=fillet_dir,
    )

def _custom_cylinder(
    dia,
    dz,
    x,
    y,
    z,
    head_dia=0,
    head_dz=0,
    dir=(0, 0, -1),
    countersink=False,
):
    """Create the v1 cylinder geometry without any FeaturePython dependency."""
    part = Part.makeCylinder(dia / 2, dz, App.Vector(0, 0, 0), App.Vector(*dir))
    if head_dia and head_dz:
        if countersink:
            head = Part.makeCone(head_dia / 2, dia / 2, head_dz, App.Vector(0, 0, 0), App.Vector(*dir))
        else:
            head = Part.makeCylinder(head_dia / 2, head_dz, App.Vector(0, 0, 0), App.Vector(*dir))
        part = part.fuse(head)
    part.translate(App.Vector(x, y, z))
    return part.removeSplitter()

def _fillet_all(part, fillet, dir=(0, 0, 1)):
    if not fillet:
        return part
    for edge in part.Edges:
        if edge.tangentAt(edge.FirstParameter) == App.Vector(*dir):
            try:
                part = part.makeFillet(fillet - 1e-3, [edge])
            except Exception:
                pass
    return part

def _bounding_box(
    body,
    tol,
    fillet,
    x_tol=True,
    y_tol=True,
    z_tol=False,
    min_offset=(0, 0, 0),
    max_offset=(0, 0, 0),
    plate_off=0,
):
    """Create the v1-style clearance envelope from a native v2 shape or mesh."""
    body = body.copy()
    body.Placement = App.Placement()
    bound = body.BoundBox
    x_min = bound.XMin - tol * x_tol + min_offset[0]
    x_max = bound.XMax + tol * x_tol + max_offset[0]
    y_min = bound.YMin - tol * y_tol + min_offset[1]
    y_max = bound.YMax + tol * y_tol + max_offset[1]
    z_min = min(bound.ZMin - tol * z_tol + min_offset[2], -INCH / 2 + plate_off)
    z_max = max(bound.ZMax + tol * z_tol + max_offset[2], -INCH / 2 + plate_off)
    return _custom_box(
        dx=x_max - x_min,
        dy=y_max - y_min,
        dz=z_max - z_min,
        x=x_min,
        y=y_min,
        z=z_min,
        dir=(1, 1, 1),
        fillet=fillet,
        fillet_dir=(0, 0, 1),
    )

def _cut_with_definition(part, definition):
    """Apply a directly supplied v2 definition's drill solid when available."""
    if definition is not None and hasattr(definition, "drill"):
        return part.cut(definition.drill())
    return part

class slide_mount:
    """Slide mount adapter for post-mounted parts

    Args:
        drill (bool) : Whether baseplate mounting for this part should be drilled
        slot_length (float) : The length of the slot used for mounting to the baseplate
        drill_offset (float) : The distance to offset the drill hole along the slot
        adapter_height (float) : The height of the suface adapter
        post_thickness (float) : The thickness of the post that mounts to the element
        outer_thickness (float) : The thickness of the walls around the bolt holes"""
    object_group = "mounts"
    object_icon = ""
    object_color = adapter_color

    def __init__(self, drill: bool = True, slot_length: dim = 10, drill_offset: dim = 0, adapter_height: dim = 8, post_thickness: dim = 4, outer_thickness: dim = 2, optical_height: dim = dim(0.5, "in")):
        self.drill_enabled = drill
        self.slot_length = slot_length
        self.drill_offset = drill_offset
        self.adapter_height = adapter_height
        self.post_thickness = post_thickness
        self.outer_thickness = outer_thickness
        self.optical_height = optical_height
        self.drill_enabled = drill
        self.slot_length = slot_length
        self.drill_offset = drill_offset
        self.adapter_height = adapter_height
        self.post_thickness = post_thickness
        self.outer_thickness = outer_thickness
        self.optical_height = optical_height

    def shape(self) -> Part.Shape:
        dx = bolt_8_32['head_dia'] + self.outer_thickness * 2
        dy = dx + self.slot_length + self.post_thickness
        dz = self.adapter_height
        part = _custom_box(dx=dx, dy=dy, dz=dz, x=0, y=-dy / 2, z=-self.optical_height, fillet=4)
        part = part.cut(_custom_box(dx=bolt_8_32['clear_dia'], dy=self.slot_length + bolt_8_32['clear_dia'], dz=dz, x=0, y=-dy / 2 - self.post_thickness / 2, z=-self.optical_height, fillet=bolt_8_32['clear_dia'] / 2))
        part = part.cut(_custom_box(dx=bolt_8_32['head_dia'], dy=self.slot_length + bolt_8_32['head_dia'], dz=bolt_8_32['head_dz'], x=0, y=-dy / 2 - self.post_thickness / 2, z=-self.optical_height + bolt_8_32['head_dz'], fillet=bolt_8_32['head_dia'] / 2))
        part = part.fuse(_custom_box(dx=dx, dy=self.post_thickness, dz=self.optical_height + bolt_8_32['head_dz'], x=0, y=-self.post_thickness / 2, z=-self.optical_height))
        part = part.cut(_custom_cylinder(dia=bolt_8_32['clear_dia'], dz=self.post_thickness, x=0, y=0, z=0, dir=(0, -1, 0)))
        return part

    def drill(self) -> Part.Shape:
        dx = bolt_8_32['head_dia'] + self.outer_thickness * 2
        dy = dx + self.slot_length + self.post_thickness
        dz = self.adapter_height
        part = _custom_box(dx=dx, dy=dy, dz=dz, x=0, y=-dy / 2, z=-self.optical_height, fillet=4)
        part = part.cut(_custom_box(dx=bolt_8_32['clear_dia'], dy=self.slot_length + bolt_8_32['clear_dia'], dz=dz, x=0, y=-dy / 2 - self.post_thickness / 2, z=-self.optical_height, fillet=bolt_8_32['clear_dia'] / 2))
        part = part.cut(_custom_box(dx=bolt_8_32['head_dia'], dy=self.slot_length + bolt_8_32['head_dia'], dz=bolt_8_32['head_dz'], x=0, y=-dy / 2 - self.post_thickness / 2, z=-self.optical_height + bolt_8_32['head_dz'], fillet=bolt_8_32['head_dia'] / 2))
        part = part.fuse(_custom_box(dx=dx, dy=self.post_thickness, dz=self.optical_height + bolt_8_32['head_dz'], x=0, y=-self.post_thickness / 2, z=-self.optical_height))
        part = part.cut(_custom_cylinder(dia=bolt_8_32['clear_dia'], dz=self.post_thickness, x=0, y=0, z=0, dir=(0, -1, 0)))
        shape = part
        part = _custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=0, y=-dy / 2 - self.post_thickness / 2 + self.drill_offset, z=-self.optical_height)
        drill_part = part
        return part

class rotation_stage_rsp05:
    """Rotation stage, model RSP05

    Args:
        invert (bool) : Whether the mount should be offset 90 degrees from the component
        mount_hole_dy (float) : The spacing between the two mount holes of it's adapter
        wave_plate_part_num (string) : The Thorlabs part number of the wave plate being used

    Sub-Parts:
        surface_adapter (adapter_args)"""
    object_group = "mounts"
    object_icon = thorlabs_icon
    object_color = misc_color
    model_source = {"stl": 'RSP05-Step.stl', "rotation": (90, -0, 90), "translation": (2.032, -0, 0), "scale": 1}
    mesh = LazyModel("rotation-stage-rsp05", directory=MODELS_DIRECTORY)

    def __init__(self, invert: bool = False, adapter_args: dict = None, adapter: bool = True):
        self.invert = invert
        self.adapter_args = {} if adapter_args is None else dict(adapter_args)
        self.adapter = adapter
        self.adapter_args.setdefault('mount_hole_dy', 25)
        self.invert = invert
        self.part_numbers = ['RSP05']
        self.transmission = True
        self.max_angle = 90
        self.max_width = INCH / 2

    def subcomponents(self) -> list[Subcomponent]:
        components = []
        adapter_args = dict(self.adapter_args)
        adapter_args.setdefault('mount_hole_dy', 25)
        if self.adapter:
            components.append(
                Subcomponent(
                    component=Component(label='Surface Adapter', definition=surface_adapter_rotation_stage_lip(**adapter_args)),
                    position=(1.397, 0, -13.97),
                    rotation=(0, 0, 90 * self.invert),
                )
            )
        return components

class surface_adapter_rotation_stage_lip:
    """Surface adapter for RSP05 with a lip"""
    object_group = "adapters"
    object_icon = thorlabs_icon
    object_color = adapter_color
    model_source = {"stl": 'Surface_Adapter_rsp05_lip.stl', "rotation": (0, 0, 0), "translation": [0, 0, 0], "scale": 1}
    mesh = LazyModel("surface-adapter-rotation-stage-lip", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = True, mount_hole_dy: dim = 36, adapter_height: dim = 8, outer_thickness: dim = 2, center_thread_depth: dim = 3):
        self.drill_enabled = drill
        self.mount_hole_dy = mount_hole_dy
        self.adapter_height = adapter_height
        self.outer_thickness = outer_thickness
        self.center_thread_depth = center_thread_depth
        self.drill_enabled = drill
        self.mount_hole_distance = mount_hole_dy
        self.adapter_height = adapter_height
        self.outer_thickness = outer_thickness
        self.center_thread_depth = center_thread_depth
        self.drill_tolerance = 1

    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()
        part = _bounding_box(shape, self.drill_tolerance, 0.125 * INCH)
        for i in [-1, 1]:
            part = part.fuse(_custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=0, y=i * self.mount_hole_distance / 2, z=0))
        drill_part = part
        return part

class surface_adapter_PD:
    """Surface adapter for RSP05 with a lip"""
    object_group = "adapters"
    object_icon = thorlabs_icon
    object_color = adapter_color
    model_source = {"stl": 'Surface_Adapter_PD.stl', "rotation": (0, 0, 0), "translation": [0, 0, 0], "scale": 1}
    mesh = LazyModel("surface-adapter-pd", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = True, mount_hole_dy: dim = 110, adapter_height: dim = 8, outer_thickness: dim = 2, center_thread_depth: dim = 3):
        self.drill_enabled = drill
        self.mount_hole_dy = mount_hole_dy
        self.adapter_height = adapter_height
        self.outer_thickness = outer_thickness
        self.center_thread_depth = center_thread_depth
        self.drill_enabled = drill
        self.mount_hole_distance = mount_hole_dy
        self.adapter_height = adapter_height
        self.outer_thickness = outer_thickness
        self.center_thread_depth = center_thread_depth
        self.drill_tolerance = 1

    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()
        part = _bounding_box(shape, self.drill_tolerance, 0.125 * INCH)
        for i in [-1, 1]:
            part = part.fuse(_custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=0, y=i * 55, z=16.5))
        drill_part = part
        return part

class cube_mount_halfinch:
    """Cube mount for 1/2" (12.7 mm) PBS."""
    object_group = "mounts"
    object_icon = thorlabs_icon
    object_color = adapter_color
    model_source = {"stl": 'Cube_Mount_Halfinch.stl', "rotation": (0, 0, 0), "translation": (0, 0, 0), "scale": 1}
    mesh = LazyModel("cube-mount-halfinch", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = True, bolt_length: dim = 15, mount_hole_dy: dim = 22.6):
        self.drill_enabled = drill
        self.bolt_length = bolt_length
        self.mount_hole_dy = mount_hole_dy
        self.drill_enabled = drill
        self.bolt_length = bolt_length
        self.mount_hole_distance = mount_hole_dy
        self.drill_tolerance = 1
        self.part_numbers = ['CUBE-MOUNT-1/2IN']

    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()
        part = _bounding_box(shape, self.drill_tolerance, 0.125 * INCH)
        for i in [-1, 1]:
            part = part.fuse(_custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=0, y=i * self.mount_hole_distance / 2, z=0))
        drill_part = part
        return part

class cube_mount_halfinch_rot90:
    """Cube mount for 1/2" (12.7 mm) PBS, rotated 90 degrees."""
    object_group = "mounts"
    object_icon = thorlabs_icon
    object_color = adapter_color
    model_source = {"stl": 'Cube_Mount_Halfinch.stl', "rotation": (0, 0, 90), "translation": (0, 0, 0), "scale": 1}
    mesh = LazyModel("cube-mount-halfinch-rot90", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = True, bolt_length: dim = 15, mount_hole_dy: dim = 22.6):
        self.drill_enabled = drill
        self.bolt_length = bolt_length
        self.mount_hole_dy = mount_hole_dy
        self.drill_enabled = drill
        self.bolt_length = bolt_length
        self.mount_hole_distance = mount_hole_dy
        self.drill_tolerance = 1
        self.part_numbers = ['CUBE-MOUNT-1/2IN']

    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()
        part = _bounding_box(shape, self.drill_tolerance, 0.125 * INCH)
        for i in [-1, 1]:
            part = part.fuse(_custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=i * self.mount_hole_distance / 2, y=0, z=0))
        drill_part = part
        return part

class mirror_mount_k05s1:
    """Mirror mount, model K05S1

    Args:
        drill (bool) : Whether baseplate mounting for this part should be drilled
        mirror (bool) : Whether to add a mirror component to the mount
        thumbscrews (bool): Whether or not to add two HKTS 5-64 adjusters"""
    object_group = "mounts"
    object_icon = thorlabs_icon
    object_color = mount_color
    model_source = {"stl": 'POLARIS-K05S1-Step.stl', "rotation": (90, 0, -90), "translation": (-4.514, 0.254, -0.254), "scale": 1}
    mesh = LazyModel("mirror-mount-k05s1", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = True, thumbscrews: bool = False):
        self.drill_enabled = drill
        self.thumbscrews = thumbscrews
        self.drill_enabled = drill
        self.thumb_screws = thumbscrews
        self.part_numbers = ['POLARIS-K05S1']

    def subcomponents(self) -> list[Subcomponent]:
        components = []
        if self.thumbscrews:
            components.append(
                Subcomponent(
                    component=Component(label='Upper Thumbscrew', definition=thumbscrew_hkts_5_64()),
                    position=(-11.22, 8.89, 8.89),
                    rotation=(0, 0, 0),
                )
            )
            components.append(
                Subcomponent(
                    component=Component(label='Lower Thumbscrew', definition=thumbscrew_hkts_5_64()),
                    position=(-11.22, -8.89, -8.89),
                    rotation=(0, 0, 0),
                )
            )
        return components

    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()
        part = _custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=-8.017, y=0, z=-INCH / 2)
        for i in [-1, 1]:
            part = part.fuse(_custom_cylinder(dia=2, dz=2.2, x=-8.017, y=i * 5, z=-INCH / 2))
        drill_part = part
        return part

class mirror_mount_M05:
    """Mirror mount, model M05

    Args:
        drill (bool) : Whether baseplate mounting for this part should be drilled
        mirror (bool) : Whether to add a mirror component to the mount
        thumbscrews (bool): Whether or not to add two HKTS 5-64 adjusters"""
    object_group = "mounts"
    object_icon = thorlabs_icon
    object_color = mount_color
    model_source = {"stl": 'Newport-M05.stl', "rotation": (0, 0, 0), "translation": (0, 0, 0), "scale": 1}
    mesh = LazyModel("mirror-mount-m05", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = True, thumbscrews: bool = False):
        self.drill_enabled = drill
        self.thumbscrews = thumbscrews
        self.drill_enabled = drill
        self.thumb_screws = thumbscrews
        self.part_numbers = ['Newport-M05']

    def subcomponents(self) -> list[Subcomponent]:
        components = []
        if self.thumbscrews:
            components.append(
                Subcomponent(
                    component=Component(label='Upper Thumbscrew', definition=thumbscrew_hkts_5_64()),
                    position=(-11.57956403, 9.144, 9.144),
                    rotation=(0, 0, 0),
                )
            )
            components.append(
                Subcomponent(
                    component=Component(label='Lower Thumbscrew', definition=thumbscrew_hkts_5_64()),
                    position=(-11.57956403, -9.144, -9.144),
                    rotation=(0, 0, 0),
                )
            )
        return components

    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()
        part = _custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=-0.274 * INCH, y=0, z=-INCH / 2)
        drill_part = part
        return part

class mirror_mount_M05X:
    """Mirror mount, model M05-X

    Args:
        drill (bool) : Whether baseplate mounting for this part should be drilled
        mirror (bool) : Whether to add a mirror component to the mount
        thumbscrews (bool): Whether or not to add two HKTS 5-64 adjusters"""
    object_group = "mounts"
    object_icon = thorlabs_icon
    object_color = mount_color
    model_source = {"stl": 'M05X.stl', "rotation": (0, 0, 0), "translation": (0, 0, 0), "scale": 1}
    mesh = LazyModel("mirror-mount-m05-x", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = True, thumbscrews: bool = False):
        self.drill_enabled = drill
        self.thumbscrews = thumbscrews
        self.drill_enabled = drill
        self.thumb_screws = thumbscrews
        self.part_numbers = ['M05X']

    def subcomponents(self) -> list[Subcomponent]:
        components = []
        if self.thumbscrews:
            components.append(
                Subcomponent(
                    component=Component(label='Upper Thumbscrew', definition=thumbscrew_hkts_5_64()),
                    position=(-11.57956403, 9.144, 9.144),
                    rotation=(0, 0, 0),
                )
            )
            components.append(
                Subcomponent(
                    component=Component(label='Lower Thumbscrew', definition=thumbscrew_hkts_5_64()),
                    position=(-11.57956403, -9.144, -9.144),
                    rotation=(0, 0, 0),
                )
            )
            components.append(
                Subcomponent(
                    component=Component(label='Position Thumbscrew', definition=thumbscrew_hkts_5_64()),
                    position=(-11.57956403, 9.144, -9.144),
                    rotation=(0, 0, 0),
                )
            )
        return components

    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()
        part = _custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=-0.274 * INCH, y=0, z=-INCH / 2)
        drill_part = part
        return part

class mirror_mount_km05:
    """Mirror mount, model KM05

    Args:
        drill (bool) : Whether baseplate mounting for this part should be drilled
        mirror (bool) : Whether to add a mirror component to the mount
        thumbscrews (bool): Whether or not to add two HKTS 5-64 adjusters
        bolt_length (float) : The length of the bolt used for mounting

    Sub-Parts:
        circular_mirror (mirror_args)"""
    object_group = "mounts"
    object_icon = thorlabs_icon
    object_color = mount_color
    model_source = {"stl": 'KM05-Step.stl', "rotation": (90, -0, 90), "translation": (2.084, -1.148, 0.498), "scale": 1}
    mesh = LazyModel("mirror-mount-km05", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = True, thumbscrews: bool = False, bolt_length: dim = 15):
        self.drill_enabled = drill
        self.thumbscrews = thumbscrews
        self.bolt_length = bolt_length
        self.drill_enabled = drill
        self.thumb_screws = thumbscrews
        self.bolt_length = bolt_length
        self.part_numbers = ['KM05']

    def subcomponents(self) -> list[Subcomponent]:
        components = []
        if self.thumbscrews:
            components.append(
                Subcomponent(
                    component=Component(label='Upper Thumbscrew', definition=thumbscrew_hkts_5_64()),
                    position=(-10.54, 9.906, 9.906),
                    rotation=(0, 0, 0),
                )
            )
            components.append(
                Subcomponent(
                    component=Component(label='Lower Thumbscrew', definition=thumbscrew_hkts_5_64()),
                    position=(-10.54, -9.906, -9.906),
                    rotation=(0, 0, 0),
                )
            )
        return components

    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()
        part = _bounding_box(shape, 2, 3, min_offset=(4.35, 0, 0))
        part = part.fuse(_bounding_box(shape, 2, 3, max_offset=(0, -20, 0)))
        part = _fillet_all(part, 3)
        part = part.fuse(_custom_cylinder(dia=bolt_8_32['clear_dia'], dz=INCH, head_dia=bolt_8_32['head_dia'], head_dz=0.92 * INCH - self.bolt_length, x=-7.29, y=0, z=-INCH * 3 / 2, dir=(0, 0, 1)))
        drill_part = part
        return part

class mirror_mount_KA05T:
    """Mirror mount, model KA05T

    Args:
        drill (bool) : Whether baseplate mounting for this part should be drilled
        mirror (bool) : Whether to add a mirror component to the mount
        thumbscrews (bool): Whether or not to add two HKTS 5-64 adjusters
        bolt_length (float) : The length of the bolt used for mounting

    Sub-Parts:
        circular_mirror (mirror_args)"""
    object_group = "mounts"
    object_icon = thorlabs_icon
    object_color = mount_color
    model_source = {"stl": 'KA05T.stl', "rotation": (0, 0, 0), "translation": (0, 0, 0), "scale": 1}
    mesh = LazyModel("mirror-mount-ka05-t", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = True, mount_hole_dy: dim = 36, thumbscrews: bool = False, bolt_length: dim = 15, Fiber_Clamp: str | bool = 'Standard'):
        self.drill_enabled = drill
        self.mount_hole_dy = mount_hole_dy
        self.thumbscrews = thumbscrews
        self.bolt_length = bolt_length
        self.fiber_clamp = Fiber_Clamp
        self.drill_enabled = drill
        self.thumb_screws = thumbscrews
        self.bolt_length = bolt_length
        self.mount_hole_distance = mount_hole_dy
        if isinstance(Fiber_Clamp, bool):
            Fiber_Clamp = 'Standard' if Fiber_Clamp else 'None'
        if Fiber_Clamp not in ('Standard', 'V1', 'None'):
            Fiber_Clamp = 'Standard'
        self.fiber_clamp = Fiber_Clamp
        self.part_numbers = ['KA05T']

    def subcomponents(self) -> list[Subcomponent]:
        components = []
        if isinstance(self.fiber_clamp, bool):
            Fiber_Clamp = 'Standard' if self.fiber_clamp else 'None'
        if self.fiber_clamp not in ('Standard', 'V1', 'None'):
            Fiber_Clamp = 'Standard'
        if self.thumbscrews:
            components.append(
                Subcomponent(
                    component=Component(label='Upper Thumbscrew', definition=thumbscrew_hkts_5_64()),
                    position=(-0.784 * INCH, 0.35 * INCH, 0.35 * INCH),
                    rotation=(0, 0, 0),
                )
            )
            components.append(
                Subcomponent(
                    component=Component(label='Lower Thumbscrew', definition=thumbscrew_hkts_5_64()),
                    position=(-0.784 * INCH, -0.35 * INCH, -0.35 * INCH),
                    rotation=(0, 0, 0),
                )
            )
        return components

    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()
        part = _bounding_box(shape, 2, 0.125 * INCH)
        fc = self.fiber_clamp
        if fc == 'Standard':
            for i in [-1, 1]:
                part = part.fuse(_custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=-2.5 * INCH, y=18 * i, z=0))
        elif fc == 'V1':
            part = part.fuse(_custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=-2.5 * INCH, y=18, z=0))
        part = part.fuse(_custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=-0.32264471 * INCH, y=0, z=-0.5 * INCH, dir=(0, 0, -1)))
        drill_part = part
        return part

class fiberport_mount_KA05T:
    """Mirror mount, model KA05T, adapted to use as fiberport mount

    Args:
        drill (bool) : Whether baseplate mounting for this part should be drilled

    Sub-Parts:
        mirror_mount_KA05T (mount_args)
        fiber_adapter_sm05fca2
        lens_tube_sm05l05
        lens_adapter_s05tm09
        mounted_lens_c220tmda"""
    object_group = "mounts"
    object_icon = ""
    object_color = misc_color

    def __init__(self, drill: bool = True, Fiber_Clamp: str | bool = 'Standard', mount_args: dict = None, adapter_args: dict = None):
        self.drill_enabled = drill
        self.fiber_clamp = Fiber_Clamp
        self.mount_args = {} if mount_args is None else dict(mount_args)
        self.adapter_args = {} if adapter_args is None else dict(adapter_args)
        self.drill_enabled = drill
        if isinstance(Fiber_Clamp, bool):
            Fiber_Clamp = 'Standard' if Fiber_Clamp else 'None'
        if Fiber_Clamp not in ('Standard', 'V1', 'None'):
            Fiber_Clamp = 'Standard'
        self.fiber_clamp = Fiber_Clamp

    def subcomponents(self) -> list[Subcomponent]:
        components = []
        mount_args = dict(self.mount_args)
        adapter_args = dict(self.adapter_args)
        if isinstance(self.fiber_clamp, bool):
            Fiber_Clamp = 'Standard' if self.fiber_clamp else 'None'
        if self.fiber_clamp not in ('Standard', 'V1', 'None'):
            Fiber_Clamp = 'Standard'
        mount_kw = dict(mount_args)
        mount_kw['Fiber_Clamp'] = self.fiber_clamp
        components.append(
            Subcomponent(
                component=Component(label='Mount', definition=mirror_mount_KA05T(**mount_kw)),
                position=(0, 0, 0),
                rotation=(0, 0, 0),
            )
        )
        components.append(
            Subcomponent(
                component=Component(label='Fiber Adapter', definition=fiber_adapter_sm05fca2()),
                position=(1.524, 0, 0),
                rotation=(0, 0, 0),
            )
        )
        components.append(
            Subcomponent(
                component=Component(label='Lens Tube', definition=lens_tube_sm05l05()),
                position=(1.524 + 3.812, 0, 0),
                rotation=(0, 0, 0),
            )
        )
        components.append(
            Subcomponent(
                component=Component(label='Lens Adapter', definition=lens_adapter_s05tm09()),
                position=(1.524 + 5, 0, 0),
                rotation=(0, 0, 0),
            )
        )
        components.append(
            Subcomponent(
                component=Component(label='Lens', definition=mounted_lens_c220tmda()),
                position=(1.524 + 3.167 + 5, 0, 0),
                rotation=(0, 0, 0),
            )
        )
        if self.fiber_clamp == 'Standard':
            components.append(
                Subcomponent(
                    component=Component(label='Fiber Clamp 3 Top', definition=fiber_clamp_3_top()),
                    position=(-2.5 * INCH, 0, -12.7),
                    rotation=(0, 0, -90),
                )
            )
            components.append(
                Subcomponent(
                    component=Component(label='Fiber Clamp 3 Bottom', definition=fiber_clamp_3_bottom()),
                    position=(-2.5 * INCH, 0, -12.7),
                    rotation=(0, 0, -90),
                )
            )
        else:
            if self.fiber_clamp == 'V1':
                components.append(
                    Subcomponent(
                        component=Component(label='Fiber Clamp 3 Top1', definition=fiber_clamp_3_top1()),
                        position=(-2.5 * INCH, 0, -12.7),
                        rotation=(0, 0, -90),
                    )
                )
                components.append(
                    Subcomponent(
                        component=Component(label='Fiber Clamp 3 Bottom1', definition=fiber_clamp_3_bottom1()),
                        position=(-2.5 * INCH, 0, -12.7),
                        rotation=(0, 0, -90),
                    )
                )
        return components


    def interfaces(self):
        """Terminate the free-space beam at the fiber-coupling plane."""
        return [
            Stop(
                position=(0, 0, 0),
                rotation=(0, 0, 0),
                diameter=dim(5, "mm"),
                max_angle=90,
            )
        ]

class lens_holder_l05g:
    """Lens Holder, Model L05G

    Args:
        drill (bool) : Whether baseplate mounting for this part should be drilled

    Sub-Parts:
        circular_lens (lens_args)"""
    object_group = "mounts"
    object_icon = thorlabs_icon
    object_color = mount_color
    model_source = {"stl": 'POLARIS-L05G-Step.stl', "rotation": (90, -0, 90), "translation": (-26.57, -13.29, -18.44), "scale": 1}
    mesh = LazyModel("lens-holder-l05g", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = True):
        self.drill_enabled = drill
        self.drill_enabled = drill
        self.part_numbers = ['POLARIS-L05G']

    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()
        part = _custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=-8, y=0, z=-INCH / 2)
        for i in [-1, 1]:
            part = part.fuse(_custom_box(dx=5, dy=2, dz=2.2, x=-8, y=i * 5, z=-INCH / 2, fillet=1, dir=(0, 0, -1)))
        drill_part = part
        return part

class pinhole_ida12:
    """Pinhole Iris, Model IDA12

    Args:
        drill (bool) : Whether baseplate mounting for this part should be drilled

    Sub-Parts:
        slide_mount (adapter_args)"""
    object_group = "components"
    object_icon = thorlabs_icon
    object_color = misc_color
    model_source = {"stl": 'IDA12-P5-Step.stl', "rotation": (90, 0, -90), "translation": (1.549, 0, -0), "scale": 1}
    mesh = LazyModel("pinhole-ida12", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = True, adapter_args: dict = None):
        self.drill_enabled = drill
        self.adapter_args = {} if adapter_args is None else dict(adapter_args)
        self.adapter_args.setdefault('slot_length', 10)
        self.drill_enabled = drill
        self.part_numbers = ['IDA12-P5']
        self.transmission = True
        self.max_angle = 90
        self.max_width = 1
        self.block_width = INCH / 2
        self.slot_length = self.adapter_args['slot_length']

    def subcomponents(self) -> list[Subcomponent]:
        components = []
        adapter_args = dict(self.adapter_args)
        adapter_args.setdefault('slot_length', 10)
        components.append(
            Subcomponent(
                component=Component(label='Slide Mount', definition=slide_mount(**adapter_args)),
                position=(1.956, -12.83, 0),
                rotation=(0, 0, 0),
            )
        )
        return components

    def interfaces(self):
        return [Interface(position=(0, 0, 0), rotation=(0, 0, 0), diameter=getattr(self, 'max_width', None), max_angle=getattr(self, 'max_angle', 90))]

    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()
        part = _custom_box(dx=6.5, dy=15 + self.slot_length, dz=1, x=1.956, y=0, z=-INCH / 2, fillet=0.125 * INCH, dir=(0, 0, -1))
        drill_part = part
        return part

class prism_mount_km100pm:
    """Kinematic Prism Mount, Model KM100PM

    Args:
        drill (bool) : Whether baseplate mounting for this part should be drilled"""
    object_group = "mounts"
    object_icon = thorlabs_icon
    object_color = mount_color
    model_source = {"stl": 'KM100PM.stl', "rotation": (90, -0, -90), "translation": (-8.877, 38.1, -6.731), "scale": 1}
    mesh = LazyModel("prism-mount-km100pm", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = True):
        self.drill_enabled = drill
        self.drill_enabled = drill
        self.part_numbers = ['KM100PM']

    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()
        part = _bounding_box(shape, 6, 0.125 * INCH, max_offset=(-18, -38, 0), z_tol=True)
        part = part.fuse(_bounding_box(shape, 3, 0.125 * INCH, min_offset=(17, 0, 0.63)))
        part = part.fuse(_custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=-14.02, y=12.63, z=17.5))
        drill_part = part
        return part

class surface_adapter:
    """Surface adapter for post-mounted parts

    Args:
        drill (bool) : Whether baseplate mounting for this part should be drilled
        mount_hole_dy (float) : The spacing between the two mount holes of the adapter
        adapter_height (float) : The height of the surface adapter
        outer_thickness (float) : The thickness of the walls around the bolt holes
        center_thread_depth (float) : The depth of the threaded portion in the center hole"""
    object_group = "adapters"
    object_icon = ""
    object_color = adapter_color

    def __init__(self, drill: bool = True, mount_hole_dy: dim = 36, adapter_height: dim = 8, outer_thickness: dim = 2, center_thread_depth: dim = 3, pd_cable: bool = False):
        self.drill_enabled = drill
        self.mount_hole_dy = mount_hole_dy
        self.adapter_height = adapter_height
        self.outer_thickness = outer_thickness
        self.center_thread_depth = center_thread_depth
        self.pd_cable = pd_cable
        self.drill_enabled = drill
        self.mount_hole_distance = mount_hole_dy
        self.adapter_height = adapter_height
        self.outer_thickness = outer_thickness
        self.center_thread_depth = center_thread_depth
        self.cable = pd_cable
        self.drill_tolerance = 1

    def shape(self) -> Part.Shape:
        dx = bolt_8_32['head_dia'] + self.outer_thickness * 2
        dy = dx + self.mount_hole_distance
        dz = self.adapter_height
        part = _custom_box(dx=dx, dy=dy, dz=dz, x=0, y=0, z=0, dir=(0, 0, -1), fillet=5)
        for i in [-1, 1]:
            part = part.cut(_custom_cylinder(dia=bolt_8_32['clear_dia'], dz=dz, head_dia=bolt_8_32['head_dia'] + self.outer_thickness, head_dz=bolt_8_32['head_dz'], x=0, y=i * self.mount_hole_distance / 2, z=0))
        part = part.cut(_custom_cylinder(dia=bolt_8_32['clear_dia'], dz=dz, head_dia=bolt_8_32['head_dia'] + self.outer_thickness, head_dz=bolt_8_32['head_dz'], x=0, y=0, z=-dz, dir=(0, 0, 1)))
        return part

    def drill(self) -> Part.Shape:
        dx = bolt_8_32['head_dia'] + self.outer_thickness * 2
        dy = dx + self.mount_hole_distance
        dz = self.adapter_height
        part = _custom_box(dx=dx, dy=dy, dz=dz, x=0, y=0, z=0, dir=(0, 0, -1), fillet=5)
        for i in [-1, 1]:
            part = part.cut(_custom_cylinder(dia=bolt_8_32['clear_dia'], dz=dz, head_dia=bolt_8_32['head_dia'] + self.outer_thickness, head_dz=bolt_8_32['head_dz'], x=0, y=i * self.mount_hole_distance / 2, z=0))
        part = part.cut(_custom_cylinder(dia=bolt_8_32['clear_dia'], dz=dz, head_dia=bolt_8_32['head_dia'] + self.outer_thickness, head_dz=bolt_8_32['head_dz'], x=0, y=0, z=-dz, dir=(0, 0, 1)))
        shape = part
        part = _bounding_box(shape, self.drill_tolerance, 6)
        for i in [-1, 1]:
            part = part.fuse(_custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=0, y=i * self.mount_hole_distance / 2, z=0))
        if self.cable:
            for i in [1, 2]:
                for j in [-3, -1, 1, 3]:
                    part = part.fuse(_custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=i * 10 - 55, y=j * 8.9, z=16.5))
        drill_part = part
        return part

class surface_adapter_isolator_lip:
    """Surface adapter for post-mounted parts

    Args:
        drill (bool) : Whether baseplate mounting for this part should be drilled
        mount_hole_dy (float) : The spacing between the two mount holes of the adapter
        adapter_height (float) : The height of the surface adapter
        outer_thickness (float) : The thickness of the walls around the bolt holes
        center_thread_depth (float) : The depth of the threaded portion in the center hole"""
    object_group = "adapters"
    object_icon = thorlabs_icon
    object_color = adapter_color
    model_source = {"stl": 'Surface_Adapter_isolator_lip.stl', "rotation": (0, 0, 0), "translation": [0, 0, 0], "scale": 1}
    mesh = LazyModel("surface-adapter-isolator-lip", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = True, mount_hole_dy: dim = 36, adapter_height: dim = 8, outer_thickness: dim = 2, center_thread_depth: dim = 3):
        self.drill_enabled = drill
        self.mount_hole_dy = mount_hole_dy
        self.adapter_height = adapter_height
        self.outer_thickness = outer_thickness
        self.center_thread_depth = center_thread_depth
        self.drill_enabled = drill
        self.mount_hole_distance = mount_hole_dy
        self.adapter_height = adapter_height
        self.outer_thickness = outer_thickness
        self.center_thread_depth = center_thread_depth
        self.drill_tolerance = 1

    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()
        part = _bounding_box(shape, self.drill_tolerance, 0.125 * INCH)
        for i in [-1, 1]:
            part = part.fuse(_custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=0, y=i * self.mount_hole_distance / 2, z=0))
        drill_part = part
        return part

class Surface_Adapter_isolator_1295:
    """Height-compensation adapter for installing IO-3D-780-VLP in a
    baseplate cutout originally designed for IOT-5-780-MP.

    This adapter is defined for later use and is not included automatically
    in either ``isolator_780`` or ``isolator_780_mp``.

    Parameters
    ----------
    drill
        Whether the adapter should generate mounting holes in the baseplate.

    mount_hole_dy
        Separation between the two baseplate mounting holes.

    drill_tolerance
        Clearance added around the imported adapter model.
    """

    object_group = "adapters"
    object_icon = thorlabs_icon
    object_color = adapter_color

    model_source = {
        "stl": "Surface_Adapter_isolator_1295.stl",
        "rotation": (0, 0, 0),
        "translation": (0, 0, 0),
        "scale": 1,
    }

    mesh = LazyModel(
        "surface-adapter-isolator-1295",
        directory=MODELS_DIRECTORY,
    )

    def __init__(
        self,
        drill: bool = True,
        mount_hole_dy: dim = 45,
        drill_tolerance: dim = 1,
    ):
        self.drill_enabled = drill
        self.mount_hole_distance = mount_hole_dy
        self.drill_tolerance = drill_tolerance
        self.part_numbers = [
            "Surface Adapter Isolator 1295"
        ]

    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()

        part = _bounding_box(
            shape,
            self.drill_tolerance,
            0.125 * INCH,
        )

        for direction in (-1, 1):
            part = part.fuse(
                _custom_cylinder(
                    dia=bolt_8_32["tap_dia"],
                    dz=DRILL_DEPTH,
                    x=0,
                    y=direction * self.mount_hole_distance / 2,
                    z=0,
                )
            )

        return part

class fiber_clamp_3_top:
    """Migrated v1 component: fiber_clamp_3_top."""
    object_group = "mounts"
    object_icon = thorlabs_icon
    object_color = misc_color
    model_source = {"stl": 'Fiber_Clamp_3_top.stl', "rotation": (0, 0, 0), "translation": (0, 0, 0), "scale": 1}
    mesh = LazyModel("fiber-clamp-3-top", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = False):
        self.drill_enabled = drill
        self.part_numbers = ['Fiber Clamp 3 Top']
        self.transmission = True

class fiber_clamp_3_top1:
    """Migrated v1 component: fiber_clamp_3_top1."""
    object_group = "mounts"
    object_icon = thorlabs_icon
    object_color = misc_color
    model_source = {"stl": 'Fiber_Clamp_3_top1.stl', "rotation": (0, 0, 0), "translation": (0, 0, 0), "scale": 1}
    mesh = LazyModel("fiber-clamp-3-top1", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = False):
        self.drill_enabled = drill
        self.part_numbers = ['Fiber Clamp 3 Top1']
        self.transmission = True

class fiber_clamp_3_bottom:
    """Migrated v1 component: fiber_clamp_3_bottom."""
    object_group = "mounts"
    object_icon = thorlabs_icon
    object_color = misc_color
    model_source = {"stl": 'Fiber_Clamp_3_bottom.stl', "rotation": (0, 0, 0), "translation": (0, 0, 0), "scale": 1}
    mesh = LazyModel("fiber-clamp-3-bottom", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = False):
        self.drill_enabled = drill
        self.part_numbers = ['Fiber Clamp 3 Bottom']
        self.transmission = True

    def drill(self) -> Part.Shape:
        part = None
        for x_position in (-18, 18):
            hole = _custom_cylinder(
                dia=bolt_8_32['tap_dia'],
                dz=DRILL_DEPTH,
                x=x_position,
                y=0,
                z=0,
            )
            part = hole if part is None else part.fuse(hole)
        return part

class fiber_clamp_3_bottom1:
    """Migrated v1 component: fiber_clamp_3_bottom1."""
    object_group = "mounts"
    object_icon = thorlabs_icon
    object_color = misc_color
    model_source = {"stl": 'Fiber_Clamp_3_bottom1.stl', "rotation": (0, 0, 0), "translation": (0, 0, 0), "scale": 1}
    mesh = LazyModel("fiber-clamp-3-bottom1", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = False):
        self.drill_enabled = drill
        self.part_numbers = ['Fiber Clamp 3 Bottom1']
        self.transmission = True

class Fiber_Clamp_Koheron:
    """Fiber Clamp for Koheron Controller"""
    object_group = "mounts"
    object_icon = thorlabs_icon
    object_color = misc_color
    model_source = {"stl": 'Fiber_Clamp_Koheron.stl', "rotation": (0, 0, 0), "translation": (0, 0, 0), "scale": 1}
    mesh = LazyModel("fiber-clamp-koheron", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = False):
        self.drill_enabled = drill


class AOMO_3100_125:
    """G&H AOMO 3100-125 AOM mounted on a KM100PM.

    The v2 AcoustoOptic interface computes diffraction from wavelength,
    RF frequency, acoustic velocity, and diffraction order.  With one RF tone
    and orders ``[0, ±1]``, an input beam ``0b10`` produces ``0b100`` and
    ``0b101``.

    ``diffraction_angle`` and ``backward_direction`` are retained only for
    compatibility with older board calls.  They do not override the v2
    AcoustoOptic calculation.
    """

    object_group = "components"
    object_icon = thorlabs_icon
    object_color = misc_color

    model_source = {
        "stl": "aomo_3100-125.stl",
        "rotation": (0, 0, -90),
        "translation": (0, -7.65, -7.1),
        "scale": 1,
    }

    mesh = LazyModel(
        "aomo-3100-125",
        directory=MODELS_DIRECTORY,
    )

    def __init__(
        self,
        drill: bool = True,
        diffraction_angle: float = 0.0,
        forward_direction: int = 1,
        backward_direction: int = 1,
        Fiber_Clamp: str | bool = "Standard",
        mount_args: dict | None = None,
        adapter_args: dict | None = None,
        rf_frequencies: float | list[float] = 100e6,
        sound_velocity: float = 4200,
        orders: list[int] | None = None,
        order_powers: list[float] | None = None,
        max_angle: float = 90,
    ):
        self.drill_enabled = drill
        self.diffraction_angle = diffraction_angle
        self.forward_direction = forward_direction
        self.backward_direction = backward_direction

        if forward_direction not in (-1, 1):
            raise ValueError("forward_direction must be -1 or 1.")

        if backward_direction not in (-1, 1):
            raise ValueError("backward_direction must be -1 or 1.")

        if isinstance(Fiber_Clamp, bool):
            Fiber_Clamp = "Standard" if Fiber_Clamp else "None"

        if Fiber_Clamp not in {"Standard", "V1", "None"}:
            raise ValueError(
                "Fiber_Clamp must be 'Standard', 'V1', 'None', True, or False."
            )

        self.fiber_clamp = Fiber_Clamp
        self.mount_args = {} if mount_args is None else dict(mount_args)
        self.adapter_args = {} if adapter_args is None else dict(adapter_args)

        if isinstance(rf_frequencies, (int, float)):
            rf_frequencies = [float(rf_frequencies)]

        self.rf_frequencies = list(rf_frequencies)
        if not self.rf_frequencies:
            raise ValueError("rf_frequencies must contain at least one RF tone.")

        # Preserve branch ordering: zero order first, diffracted order second.
        self.orders = (
            [0, forward_direction]
            if orders is None
            else list(orders)
        )
        if not self.orders:
            raise ValueError("orders must contain at least one diffraction order.")

        if order_powers is None:
            if len(self.orders) == 2:
                self.order_powers = [0.2, 0.8]
            else:
                self.order_powers = [1 / len(self.orders)] * len(self.orders)
        else:
            self.order_powers = list(order_powers)

        if len(self.orders) != len(self.order_powers):
            raise ValueError(
                "orders and order_powers must have the same length."
            )

        self.sound_velocity = sound_velocity
        self.max_angle = max_angle
        self.part_numbers = ["G&H AOMO 3100-125"]

    def subcomponents(self) -> list[Subcomponent]:
        components = [
            Subcomponent(
                component=Component(
                    label="KM100PM Mount",
                    definition=prism_mount_km100pm(**self.mount_args),
                ),
                position=(-30.3, -16.4, -24.5),
                rotation=(0, 0, 0),
            ),
            Subcomponent(
                component=Component(
                    label="AOM Adapter",
                    definition=aom_adapter(**self.adapter_args),
                ),
                position=(-17, -7.65, -17.1),
                rotation=(0, 0, -90),
            ),
        ]

        if self.fiber_clamp == "Standard":
            bottom_definition = fiber_clamp_3_bottom
            top_definition = fiber_clamp_3_top
        elif self.fiber_clamp == "V1":
            bottom_definition = fiber_clamp_3_bottom1
            top_definition = fiber_clamp_3_top1
        else:
            bottom_definition = None
            top_definition = None

        if bottom_definition is not None:
            components.extend(
                [
                    Subcomponent(
                        component=Component(
                            label="Fiber Clamp Bottom",
                            definition=bottom_definition(),
                        ),
                        position=(-0.4 * INCH, -2.2 * INCH, -12.7),
                        rotation=(0, 0, 0),
                    ),
                    Subcomponent(
                        component=Component(
                            label="Fiber Clamp Top",
                            definition=top_definition(),
                        ),
                        position=(-0.4 * INCH, -2.2 * INCH, -12.7),
                        rotation=(0, 0, 0),
                    ),
                ]
            )

        return components

    def interfaces(self):
        return [
            AcoustoOptic(
                position=(0, 0, 0),
                rotation=(0, 0, 0),
                sound_velocity=self.sound_velocity,
                rf_frequencies=self.rf_frequencies,
                orders=self.orders,
                order_powers=self.order_powers,
                diameter=dim(5, "mm"),
                max_angle=self.max_angle,
            )
        ]

    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()
        part = _bounding_box(
            shape,
            2,
            0.125 * INCH,
        )

        if self.fiber_clamp != "None":
            for direction in (-1, 1):
                part = part.fuse(
                    _custom_cylinder(
                        dia=bolt_8_32["tap_dia"],
                        dz=DRILL_DEPTH,
                        x=-0.4 * INCH + 18 * direction,
                        y=-2.2 * INCH,
                        z=0,
                    )
                )

        return part


class aom_adapter:
    """Adapter for AOMs on KM100PM Mount"""
    object_group = "adapters"
    object_icon = thorlabs_icon
    object_color = adapter_color
    model_source = {"stl": 'aom_adapter.stl', "rotation": (0, 0, 0), "translation": (0, 0, 0), "scale": 1}
    mesh = LazyModel("aom-adapter", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = True):
        self.drill_enabled = drill
        self.drill_enabled = drill
        self.part_numbers = ['AOM Adapter']
        self.transmission = True
        self.max_angle = 10
        self.max_width = 5


    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()
        part = _bounding_box(shape, 2, 0.125 * INCH)
        drill_part = part
        return part

class shutter_sr475:
    """shutter for SRS SR475"""
    object_group = "components"
    object_icon = thorlabs_icon
    object_color = misc_color
    model_source = {"stl": 'SR475.stl', "rotation": (0, 0, 0), "translation": (0, 0, 0), "scale": 1}
    mesh = LazyModel("shutter-sr475", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = True):
        self.drill_enabled = drill
        self.drill_enabled = drill
        self.part_numbers = ['SRS SR475 Shutter']
        self.transmission = True
        self.max_angle = 10
        self.max_width = 5

    def subcomponents(self) -> list[Subcomponent]:
        components = []
        components.append(
            Subcomponent(
                component=Component(label='Adapter', definition=shutter_adapter()),
                position=(0, -7.925, -4.763),
                rotation=(0, 0, 0),
            )
        )
        return components

    def interfaces(self):
        return [Interface(position=(0, 0, 0), rotation=(0, 0, 0), diameter=getattr(self, 'max_width', None), max_angle=getattr(self, 'max_angle', 90))]

class mirror_mount_FMP05:
    """Mirror mount, model FMP05

    Args:
        drill (bool) : Whether baseplate mounting for this part should be drilled
        mirror (bool) : Whether to add a mirror component to the mount"""
    object_group = "mounts"
    object_icon = thorlabs_icon
    object_color = mount_color
    model_source = {"stl": 'FMP05.stl', "rotation": (90, 0, 90), "translation": (3.1, 0, 0), "scale": 1}
    mesh = LazyModel("mirror-mount-fmp05", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = True):
        self.drill_enabled = drill
        self.drill_enabled = drill
        self.part_numbers = ['Thorlabs-FMP05']

    def subcomponents(self) -> list[Subcomponent]:
        components = []
        components.append(
            Subcomponent(
                component=Component(label='FMP05 Adapter', definition=adapter_FMP05()),
                position=(-6.9, 0, -24.25),
                rotation=(0, 0, -90),
            )
        )
        return components

class adapter_FMP05:
    """Adapter for mirror mount, model FMP05

    Args:
        drill (bool) : Whether baseplate mounting for this part should be drilled"""
    object_group = "adapters"
    object_icon = thorlabs_icon
    object_color = adapter_color
    model_source = {"stl": 'FMP05_Adapter.stl', "rotation": (0, 0, 0), "translation": (0, 0, 0), "scale": 1}
    mesh = LazyModel("adapter-fmp05", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = True):
        self.drill_enabled = drill
        self.drill_enabled = drill
        self.drill_tolerance = 1

    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()
        part = _bounding_box(shape, self.drill_tolerance, 0.125 * INCH)
        for i in [-1, 1]:
            part = part.fuse(_custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=i * 5, y=-3.5, z=0))
        drill_part = part
        return part

class shutter_adapter:
    """Adapter for SRS SR475 Shutter"""
    object_group = "adapters"
    object_icon = thorlabs_icon
    object_color = adapter_color
    model_source = {"stl": 'shutter_adapter.stl', "rotation": (0, 0, 0), "translation": (0, 0, 0), "scale": 1}
    mesh = LazyModel("shutter-adapter", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = True):
        self.drill_enabled = drill
        self.drill_enabled = drill
        self.part_numbers = ['Shutter SR475 Adapter']
        self.transmission = True
        self.max_angle = 10
        self.max_width = 5


    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()
        part = _bounding_box(shape, 1, 0.125 * INCH)
        for i in [-1, 1]:
            for j in [-1, 1]:
                part = part.fuse(_custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=j * 17, y= i * 22.86, z=0))
        drill_part = part
        return part

class isolator_850:
    """Isolator Optimized for 850nm, Model IOT-5-850-VLP

    Args:
        drill (bool) : Whether baseplate mounting for this part should be drilled

    Sub-Parts:
        surface_adapter (adapter_args)"""
    object_group = "components"
    object_icon = thorlabs_icon
    object_color = misc_color
    model_source = {"stl": 'IOT-5-850-VLP-Step.stl', "rotation": (90, 0, -90), "translation": (-19.05, -0, 0), "scale": 1}
    mesh = LazyModel("isolator-850", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = True, adapter_args: dict = None):
        self.drill_enabled = drill
        self.adapter_args = {} if adapter_args is None else dict(adapter_args)
        self.adapter_args.setdefault('mount_hole_dy', 45)
        self.drill_enabled = drill
        self.part_numbers = ['IOT-5-670-VLP']
        self.transmission = True
        self.max_angle = 10
        self.max_width = 5

    def subcomponents(self) -> list[Subcomponent]:
        components = []
        adapter_args = dict(self.adapter_args)
        adapter_args.setdefault('mount_hole_dy', 45)
        components.append(
            Subcomponent(
                component=Component(label='Surface Adapter', definition=surface_adapter_isolator_lip(**adapter_args)),
                position=(0, 0, -22.1),
                rotation=(0, 0, 0),
            )
        )
        return components

    def interfaces(self):
        return [Interface(position=(0, 0, 0), rotation=(0, 0, 0), diameter=getattr(self, 'max_width', None), max_angle=getattr(self, 'max_angle', 90))]

    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()
        part = _custom_box(dx=108, dy=35, dz=5, x=0, y=0, z=-INCH / 2, fillet=0.125 * INCH, dir=(0, 0, -1))
        drill_part = part
        return part

class isolator_780:
    """Isolator Optimized for 780nm, Model IO-3D-780-VLP

    Args:
        drill (bool) : Whether baseplate mounting for this part should be drilled

    Sub-Parts:
        surface_adapter (adapter_args)"""
    object_group = "components"
    object_icon = thorlabs_icon
    object_color = misc_color
    model_source = {"stl": 'IO-3D-780-VLP-Step.stl', "rotation": (90, 0, -90), "translation": (-15.66, -0, 0), "scale": 1}
    mesh = LazyModel("isolator-780", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = True, adapter_args: dict = None):
        self.drill_enabled = drill
        self.adapter_args = {} if adapter_args is None else dict(adapter_args)
        self.adapter_args.setdefault('mount_hole_dy', 45)
        self.drill_enabled = drill
        self.part_numbers = ['IO-3D-780-VLP']
        self.transmission = True
        self.max_angle = 10
        self.max_width = 5

    def subcomponents(self) -> list[Subcomponent]:
        components = []
        adapter_args = dict(self.adapter_args)
        adapter_args.setdefault('mount_hole_dy', 45)
        components.append(
            Subcomponent(
                component=Component(label='Surface Adapter', definition=surface_adapter_isolator_lip(**adapter_args)),
                position=(0, 0, -17.15),
                rotation=(0, 0, 0),
            )
        )
        return components

    def interfaces(self):
        return [Interface(position=(0, 0, 0), rotation=(0, 0, 0), diameter=getattr(self, 'max_width', None), max_angle=getattr(self, 'max_angle', 90))]

    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()
        part = _custom_box(dx=40, dy=25, dz=5, x=0, y=0, z=-INCH / 2, fillet=0.125 * INCH, dir=(0, 0, -1))
        drill_part = part
        return part

class isolator_780_mp:
    """Isolator Optimized for 780nm, Model IOT-5-780-MP

    Args:
        drill (bool) : Whether baseplate mounting for this part should be drilled

    Sub-Parts:
        surface_adapter (adapter_args)"""
    object_group = "components"
    object_icon = thorlabs_icon
    object_color = misc_color
    model_source = {"stl": 'IOT-5-780-MP.stl', "rotation": (90, 0, 90), "translation": (-46.482, -0, 0), "scale": 1}
    mesh = LazyModel("isolator-780-mp", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = True, adapter_args: dict = None):
        self.drill_enabled = drill
        self.adapter_args = {} if adapter_args is None else dict(adapter_args)
        self.adapter_args.setdefault('mount_hole_dy', 45)
        self.drill_enabled = drill
        self.part_numbers = ['IOT-5-670-VLP']
        self.transmission = True
        self.max_angle = 10
        self.max_width = 5

    def subcomponents(self) -> list[Subcomponent]:
        components = []
        adapter_args = dict(self.adapter_args)
        adapter_args.setdefault('mount_hole_dy', 45)
        components.append(
            Subcomponent(
                component=Component(label='Surface Adapter', definition=surface_adapter_isolator_lip(**adapter_args)),
                position=(0, 0, -22.1),
                rotation=(0, 0, 0),
            )
        )
        return components

    def interfaces(self):
        return [Interface(position=(0, 0, 0), rotation=(0, 0, 0), diameter=getattr(self, 'max_width', None), max_angle=getattr(self, 'max_angle', 90))]

    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()
        part = _custom_box(dx=120, dy=35, dz=5, x=0, y=0, z=-INCH / 2, fillet=0.125 * INCH, dir=(0, 0, -1))
        drill_part = part
        return part

class rb_cell:
    """Rubidium Cell Holder

    Args:
        drill (bool) : Whether baseplate mounting for this part should be drilled"""
    object_group = "optics"
    object_icon = optic_icon
    object_color = adapter_color
    model_source = {"stl": 'rb_cell_holder_middle.stl', "rotation": (0, 0, 0), "translation": [0, 5, 0], "scale": 1}
    mesh = LazyModel("rb-cell", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = True):
        self.drill_enabled = drill
        self.drill_enabled = drill
        self.transmission = True
        self.max_angle = 10
        self.max_width = 1

    def interfaces(self):
        return [Interface(position=(0, 0, 0), rotation=(0, 0, 0), diameter=getattr(self, 'max_width', None), max_angle=getattr(self, 'max_angle', 90))]

    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()
        part = _bounding_box(shape, 2, 0.125 * INCH)
        dx = 90
        for x, y in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
            part = part.fuse(_custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=x * dx / 2, y=y * 15.7, z=-INCH / 2))
        part = part.fuse(_custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=45, y=-15.7, z=-INCH / 2))
        for x in [1, -1]:
            part = part.fuse(_custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=x * dx / 2, y=25.7, z=-INCH / 2))
        drill_part = part
        return part

class photodetector_pdb250aa:
    """Photodetector, model PDB250A with cover plate

    Args:
        drill (bool) : Whether baseplate mounting for this part should be drilled

    Sub-Parts:
        surface_adapter (adapter_args)"""
    object_group = "detector"
    object_icon = thorlabs_icon
    object_color = misc_color
    model_source = {"stl": 'PDB250Aa.stl', "rotation": (0, 0, 0), "translation": (-5, 0, 0), "scale": 1}
    mesh = LazyModel("photodetector-pdb250aa", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = True, adapter_args: dict = None):
        self.drill_enabled = drill
        self.adapter_args = {} if adapter_args is None else dict(adapter_args)
        self.adapter_args.setdefault('mount_hole_dy', 60)
        self.drill_enabled = drill
        self.part_numbers = ['PDB250Aa']
        self.max_angle = 80
        self.max_width = 5

    def subcomponents(self) -> list[Subcomponent]:
        components = []
        adapter_args = dict(self.adapter_args)
        adapter_args.setdefault('mount_hole_dy', 60)
        components.append(
            Subcomponent(
                component=Component(label='Surface Adapter for PD', definition=surface_adapter_PD(**adapter_args)),
                position=(-17.75, 0, -16.6),
                rotation=(0, 0, 0),
            )
        )
        return components

    def interfaces(self):
        return [Stop(position=(0, 0, 0), rotation=(0, 0, 0), diameter=getattr(self, 'max_width', dim(1, 'mm')), max_angle=getattr(self, 'max_angle', 90))]

    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()
        part = _bounding_box(shape, 2, 0.25 * INCH)
        drill_part = part
        return part

class thumbscrew_hkts_5_64:
    """Thumbscrew for 5-64 hex adjusters, model HKTS 5-64

    Sub-Parts:
        slide_mount (adapter_args)"""
    object_group = "hardware"
    object_icon = thorlabs_icon
    object_color = misc_color
    model_source = {"stl": 'HKTS-5_64-Step.stl', "rotation": (90, 0, 90), "translation": (-11.31, -0.945, 0.568), "scale": 1}
    mesh = LazyModel("thumbscrew-hkts-5-64", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = True, adapter_args: dict = None):
        self.drill_enabled = drill
        self.adapter_args = {} if adapter_args is None else dict(adapter_args)
        self.adapter_args.setdefault('slot_length', 10)
        self.drill_enabled = drill
        self.part_numbers = ['HKTS-5/64(P4)']

    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()
        part = _bounding_box(shape, 2.75, 0.125 * INCH, z_tol=True, min_offset=(-6, 0, 0), max_offset=(-6, 0, 0))
        drill_part = part
        return part

class fiber_adapter_sm05fca2:
    """Fiber Adapter Plate, model SM05FCA2"""
    object_group = "adapters"
    object_icon = thorlabs_icon
    object_color = misc_color
    model_source = {"stl": 'SM05FCA2-Step.stl', "rotation": (0, 90, 0), "translation": (-2.334, -3.643, -0.435), "scale": 1}
    mesh = LazyModel("fiber-adapter-sm05fca2", directory=MODELS_DIRECTORY)

    def __init__(self):
        self.part_numbers = ['SM05FCA2']
        self.max_angle = 0
        self.max_width = 1

class lens_adapter_s05tm09:
    """SM05 to M9x0.5 Lens Cell Adapter, model S05TM09"""
    object_group = "mounts"
    object_icon = thorlabs_icon
    object_color = misc_color
    model_source = {"stl": 'S05TM09-Step.stl', "rotation": (90, 0, -90), "translation": (6.973, 0, -0), "scale": 1}
    mesh = LazyModel("lens-adapter-s05tm09", directory=MODELS_DIRECTORY)

    def __init__(self):
        self.part_numbers = ['S05TM09']

class lens_tube_sm05l05:
    """Lens Tube, model SM05L05"""
    object_group = "optics"
    object_icon = optic_icon
    object_color = misc_color
    model_source = {"stl": 'SM05L05-Step.stl', "rotation": (90, 0, -90), "translation": (0, 0, -0), "scale": 1}
    mesh = LazyModel("lens-tube-sm05l05", directory=MODELS_DIRECTORY)

    def __init__(self):
        self.part_numbers = ['SM05L05']

class mounted_lens_c220tmda:
    """Mounted Aspheric Lens, model C220TMD-A"""
    object_group = "mounts"
    object_icon = thorlabs_icon
    object_color = glass_color
    model_source = {"stl": 'C220TMD-A-Step.stl', "rotation": (-90, 0, -180), "translation": (0.419, 0, 0), "scale": 1}
    mesh = LazyModel("mounted-lens-c220tmda", directory=MODELS_DIRECTORY)

    def __init__(self):
        self.part_numbers = ['C220TMD-A']


class mirror_cube_ccm1_p01:
    """Thorlabs CCM1-P01 cage-cube-mounted turning mirror.

    The imported CAD model is already aligned to the PyOpticL optical
    coordinate system.  For a beam entering along local +X, the mirror
    redirects the beam toward local +Z.
    """

    object_group = "optics"
    object_icon = optic_icon
    object_color = mount_color

    model_source = {
        "stl": "mirror-cube-ccm1-p01.stl",
        "rotation": (0, 0, 0),
        "translation": (0, 0, 0),
        "scale": 1,
    }

    mesh = LazyModel(
        "mirror-cube-ccm1-p01",
        directory=MODELS_DIRECTORY,
    )

    def __init__(self):
        self.part_numbers = ["CCM1-P01"]
        self.max_angle = 90
        self.max_width = INCH

    def interfaces(self):
        return [
            Reflection(
                position=(0, 0, 0),
                rotation=(0, -135, 0),
                diameter=self.max_width,
                max_angle=self.max_angle,
            )
        ]

class pmot_lens_la1401:
    """pMOT LA1401 lens assembly.

    CAD assembly:
        LCP34 + SM2L05 + LA1401-C-ML

    The imported model is already aligned so that the optical center is at
    the local origin and the optical axis is local +X.
    """

    object_group = "optics"
    object_icon = optic_icon
    object_color = mount_color

    model_source = {
        "stl": "pmot-lens-la1401.stl",
        "rotation": (0, 0, 0),
        "translation": (0, 0, 0),
        "scale": 1,
    }

    mesh = LazyModel(
        "pmot-lens-la1401",
        directory=MODELS_DIRECTORY,
    )

    def __init__(self):
        self.part_numbers = [
            "LCP34",
            "SM2L05",
            "LA1401-C-ML",
        ]
        self.transmission = True
        self.focal_length = dim(60, "mm")
        self.max_angle = 90
        self.max_width = dim(2, "in")

    def interfaces(self):
        return [
            Lens(
                position=(0, 0, 0),
                rotation=(0, 0, 0),
                diameter=self.max_width,
                focal_length=self.focal_length,
            )
        ]


class pmot_lens_80:
    """pMOT LA1401 lens assembly, just assuming f=80mm instead.

    CAD assembly:
        LCP34 + SM2L05 + LA1401-C-ML

    The imported model is already aligned so that the optical center is at
    the local origin and the optical axis is local +X.
    """

    object_group = "optics"
    object_icon = optic_icon
    object_color = mount_color

    model_source = {
        "stl": "pmot-lens-la1401.stl",
        "rotation": (0, 0, 0),
        "translation": (0, 0, 0),
        "scale": 1,
    }

    mesh = LazyModel(
        "pmot-lens-la1401",
        directory=MODELS_DIRECTORY,
    )

    def __init__(self):
        self.part_numbers = [
            "LCP34",
            "SM2L05",
            "LA1401-C-ML",
        ]
        self.transmission = True
        self.focal_length = dim(80, "mm")
        self.max_angle = 90
        self.max_width = dim(2, "in")

    def interfaces(self):
        return [
            Lens(
                position=(0, 0, 0),
                rotation=(0, 0, 0),
                diameter=self.max_width,
                focal_length=self.focal_length,
            )
        ]


class pmot_glass_cell:
    """pMOT glass cell used only as a geometric reference.

    The CAD model does not introduce reflection, refraction, or focusing.
    A generic Interface is placed at the optical origin so that a BeamPath
    can pass through the component without changing direction.
    """

    object_group = "components"
    object_icon = optic_icon
    object_color = glass_color
    object_transparency = 70

    model_source = {
        "stl": "pmot-glass-cell.stl",
        "rotation": (0, 0, 0),
        "translation": (0, 0, 0),
        "scale": 1,
    }

    mesh = LazyModel(
        "pmot-glass-cell",
        directory=MODELS_DIRECTORY,
    )

    def __init__(self):
        self.part_numbers = ["pMOT Glass Cell"]
        self.transmission = True
        self.max_angle = 90
        self.max_width = dim(100, "mm")

    def interfaces(self):
        return [
            Interface(
                position=(0, 0, 0),
                rotation=(0, 0, 0),
                diameter=self.max_width,
                max_angle=self.max_angle,
            )
        ]


class mirror_mount_KA1:
    """Thorlabs Polaris KA1 mirror mount.

    Args:
        thumbscrews (bool): Whether or not to add two HKTS 5-64 adjusters
    """

    object_group = "mounts"
    object_icon = thorlabs_icon
    object_color = mount_color

    model_source = {
        "stl": "mirror-mount-ka1.stl",
        "rotation": (0, 0, 0),
        "translation": (0, 0, 0),
        "scale": 1,
    }

    mesh = LazyModel(
        "mirror-mount-ka1",
        directory=MODELS_DIRECTORY,
    )

    def __init__(self, thumbscrews: bool = False):
        self.thumbscrews = thumbscrews
        self.part_numbers = ["KA1"]

    def subcomponents(self) -> list[Subcomponent]:
        components = []
        if self.thumbscrews:
            components.append(
                Subcomponent(
                    component=Component(label='Upper Thumbscrew', definition=thumbscrew_hkts_5_64()),
                    position=(-1.0225 * INCH, 0.65 * INCH, 0.65 * INCH),
                    rotation=(0, 0, 0),
                )
            )
            components.append(
                Subcomponent(
                    component=Component(label='Lower Thumbscrew', definition=thumbscrew_hkts_5_64()),
                    position=(-1.0225 * INCH, -0.65 * INCH, -0.65 * INCH),
                    rotation=(0, 0, 0),
                )
            )
        return components

    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()
        part = _bounding_box(shape, 2, 0.125 * INCH)
        part = part.fuse(
            _custom_cylinder(
                dia=bolt_8_32['tap_dia'],
                dz=DRILL_DEPTH,
                x=-0.405 * INCH,
                y=0,
                z=-0.5 * INCH,
                dir=(0, 0, -1),
            )
        )
        drill_part = part
        return part


class mirror_mount_KA2T:
    """Thorlabs Polaris KA2T mirror mount."""

    object_group = "mounts"
    object_icon = thorlabs_icon
    object_color = mount_color

    model_source = {
        "stl": "mirror-mount-ka2t.stl",
        "rotation": (0, 0, 0),
        "translation": (0, 0, 0),
        "scale": 1,
    }

    mesh = LazyModel(
        "mirror-mount-ka2t",
        directory=MODELS_DIRECTORY,
    )

    def __init__(self):
        self.part_numbers = ["KA2T"]


class mounted_lens_AC254_040_C_ML:
    """Thorlabs AC254-040-C-ML mounted achromatic doublet."""

    object_group = "optics"
    object_icon = optic_icon
    object_color = glass_color

    model_source = {
        "stl": "lens-mounted-ac254-040-c-ml.stl",
        "rotation": (0, 0, 0),
        "translation": (0, 0, 0),
        "scale": 1,
    }

    mesh = LazyModel(
        "lens-mounted-ac254-040-c-ml",
        directory=MODELS_DIRECTORY,
    )

    def __init__(self):
        self.part_numbers = ["AC254-040-C-ML"]
        self.focal_length = dim(45, "mm")
        self.diameter = dim(1, "in")
        self.max_angle = 90
        self.max_width = self.diameter
        self.transmission = True

    def interfaces(self):
        return [
            Lens(
                position=(0, 0, 0),
                rotation=(0, 0, 0),
                diameter=self.diameter,
                focal_length=self.focal_length,
            )
        ]


class lens_mount_LCP34:
    """Thorlabs LCP34 60 mm cage plate / lens mount."""

    object_group = "mounts"
    object_icon = thorlabs_icon
    object_color = mount_color

    model_source = {
        "stl": "lens-mount-lcp34.stl",
        "rotation": (0, 0, 0),
        "translation": (0, 0, 0),
        "scale": 1,
    }

    mesh = LazyModel(
        "lens-mount-lcp34",
        directory=MODELS_DIRECTORY,
    )

    def __init__(self):
        self.part_numbers = ["LCP34"]


class DFB_butterfly_diode:
    """Toptica DFB butterfly laser diode, model eyP-BFW01-171218."""

    object_group = "components"
    object_icon = thorlabs_icon
    object_color = (1.0, 1.0, 0.0)

    model_source = {
        "stl": "eyP-BFW01-171218.stl",
        "rotation": (0, 0, 0),
        "translation": (0, 0, 0),
        "scale": 1,
    }

    mesh = LazyModel(
        "dfb-butterfly-diode",
        directory=MODELS_DIRECTORY,
    )

    def __init__(self):
        self.part_numbers = ["eyP-BFW01-171218"]

class IPS_butterfly_diode:
    """Diode Mount Adapter, model I0780.2SB0050PA-IS"""
    object_group = "components"
    object_icon = thorlabs_icon
    object_color = (1.0, 1.0, 0.0)
    model_source = {"stl": 'IPS_laser.stl', "rotation": (0, 0, 0), "translation": (0, 0, 0), "scale": 1}
    mesh = LazyModel("ips-butterfly-diode", directory=MODELS_DIRECTORY)

    def __init__(self):
        self.part_numbers = ['IPS-laser']

class DFB_adapter:
    """Koheron controller adapter for the Toptica DFB butterfly diode."""

    object_group = "adapters"
    object_icon = thorlabs_icon
    object_color = adapter_color

    model_source = {
        "stl": "DFB_adapter.stl",
        "rotation": (0, 0, 0),
        "translation": (0, 0, 0),
        "scale": 1,
    }

    mesh = LazyModel(
        "dfb-adapter",
        directory=MODELS_DIRECTORY,
    )

    def __init__(self, drill: bool = True):
        self.drill_enabled = drill
        self.part_numbers = ["DFB-adapter"]

    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()

        part = _bounding_box(
            shape,
            2,
            0.125 * INCH,
        )

        for y in (-37.5, 37.5):
            part = part.fuse(
                _custom_cylinder(
                    dia=bolt_8_32["tap_dia"],
                    dz=DRILL_DEPTH,
                    x=7,
                    y=y,
                    z=0,
                )
            )
            part = part.fuse(
                _custom_cylinder(
                    dia=bolt_8_32["tap_dia"],
                    dz=DRILL_DEPTH,
                    x=-68,
                    y=y,
                    z=0,
                )
            )

        return part

class IPS_adapter:
    """Koheron Laser adapter with IPS"""
    object_group = "adapters"
    object_icon = thorlabs_icon
    object_color = adapter_color
    model_source = {"stl": 'IPS_adapter.stl', "rotation": (0, 0, 0), "translation": (0, 0, 0), "scale": 1}
    mesh = LazyModel("ips-adapter", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = True):
        self.drill_enabled = drill
        self.drill_enabled = drill
        self.part_numbers = ['IPS-adapter']

    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()
        part = _bounding_box(shape, 2, 0.125 * INCH)
        part = part.fuse(_custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=9, y=37.5, z=0))
        part = part.fuse(_custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=9, y=-37.5, z=0))
        part = part.fuse(_custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=-66, y=-37.5, z=0))
        part = part.fuse(_custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=-66, y=37.5, z=0))
        drill_part = part
        return part

class Koheron_adapter:
    """Large/unified-height adapter for a Koheron laser assembly."""

    object_group = "adapters"
    object_icon = thorlabs_icon
    object_color = adapter_color

    model_source = {
        "stl": "Koheron_adapter.stl",
        "rotation": (0, 0, 0),
        "translation": (0, 0, 0),
        "scale": 1,
    }

    mesh = LazyModel(
        "koheron-adapter",
        directory=MODELS_DIRECTORY,
    )

    def __init__(
        self,
        drill: bool = True,
        bolt_length: dim = 15,
    ):
        self.drill_enabled = drill
        self.bolt_length = bolt_length
        self.part_numbers = ["Koheron_adapter"]

    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()

        part = _bounding_box(
            shape,
            2,
            0.125 * INCH,
        )

        for y in (-37.5, 37.5):
            part = part.fuse(
                _custom_cylinder(
                    dia=bolt_8_32["tap_dia"],
                    dz=DRILL_DEPTH,
                    x=9,
                    y=y,
                    z=0,
                )
            )
            part = part.fuse(
                _custom_cylinder(
                    dia=bolt_8_32["tap_dia"],
                    dz=DRILL_DEPTH,
                    x=-66,
                    y=y,
                    z=0,
                )
            )

        return part

class TA_adapter:
    """Adapter for TA board in unified size"""
    object_group = "adapters"
    object_icon = thorlabs_icon
    object_color = adapter_color
    model_source = {"stl": 'TA_adapter.stl', "rotation": (0, 0, 0), "translation": (0, 0, 0), "scale": 1}
    mesh = LazyModel("ta-adapter", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = True, bolt_length: dim = 15):
        self.drill_enabled = drill
        self.bolt_length = bolt_length
        self.drill_enabled = drill
        self.bolt_length = bolt_length
        self.part_numbers = ['TA_adapter']

    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()
        part = _bounding_box(shape, 2, 0.125 * INCH)
        part = part.fuse(_custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=9, y=37.5, z=0))
        part = part.fuse(_custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=9, y=-37.5, z=0))
        part = part.fuse(_custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=-66, y=-37.5, z=0))
        part = part.fuse(_custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=-66, y=37.5, z=0))
        drill_part = part
        return part

class Koheron_Controller:
    """Koheron Current + TEC Controller

    Args:
        drill (bool) : Whether baseplate mounting for this part should be drilled
        mirror (bool) : Whether to add a mirror component to the mount
        thumbscrews (bool): Whether or not to add two HKTS 5-64 adjusters"""
    object_group = "components"
    object_icon = thorlabs_icon
    object_color = mount_color
    model_source = {"stl": 'koheron-CTL200-V5.stl', "rotation": (0, 0, 0), "translation": (0, 0, 0), "scale": 1}
    mesh = LazyModel("koheron-controller", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = True):
        self.drill_enabled = drill
        self.drill_enabled = drill
        self.part_numbers = ['koheron-CTL200-V5']

    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()
        part = _custom_cylinder(dia=0.097 * INCH, dz=2.2, x=-2.511 * INCH, y=-0.9845 * INCH, z=-0.531 * INCH)
        part = part.fuse(_custom_cylinder(dia=0.097 * INCH, dz=2.2, x=-2.511 * INCH, y=+0.9845 * INCH, z=-0.531 * INCH))
        cutout = _bounding_box(shape, 2, 3, min_offset=(0, 0, -0.031 * INCH), max_offset=(0, 0, 0.0))
        cutout = _fillet_all(cutout, 1)
        part = part.fuse(cutout)
        part = part.fuse(_custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=-58.514, y=-73.5, z=-12.7))
        part = part.fuse(_custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=-13.406, y=-87.25, z=-12.7))
        drill_part = part
        return part



class ips_laser:
    """IPS butterfly diode mounted with a Koheron controller.

    Parameters
    ----------
    adapter
        ``"small"`` uses ``IPS_adapter``.
        ``"big"`` uses ``Koheron_adapter``.

    diode_args, adapter_args, controller_args, fiber_clamp_args
        Optional keyword arguments forwarded to the respective definitions.
    """

    object_group = "components"
    object_icon = ""
    object_color = misc_color

    def __init__(
        self,
        adapter: str = "small",
        fiber_clamp: bool = True,
        drill: bool = True,
        diode_args: dict | None = None,
        adapter_args: dict | None = None,
        controller_args: dict | None = None,
        fiber_clamp_args: dict | None = None,
    ):
        if adapter not in {"small", "big"}:
            raise ValueError(
                f"Unknown IPS adapter {adapter!r}. "
                "Valid choices are 'small' and 'big'."
            )

        self.adapter = adapter
        self.fiber_clamp = fiber_clamp
        self.drill_enabled = drill

        self.diode_args = (
            {} if diode_args is None else dict(diode_args)
        )
        self.adapter_args = (
            {} if adapter_args is None else dict(adapter_args)
        )
        self.controller_args = (
            {} if controller_args is None else dict(controller_args)
        )
        self.fiber_clamp_args = (
            {} if fiber_clamp_args is None
            else dict(fiber_clamp_args)
        )

    def subcomponents(self) -> list[Subcomponent]:
        adapter_definition = {
            "small": IPS_adapter,
            "big": Koheron_adapter,
        }[self.adapter]

        adapter_args = {
            "drill": self.drill_enabled,
            **self.adapter_args,
        }

        controller_args = {
            "drill": self.drill_enabled,
            **self.controller_args,
        }

        components = [
            Subcomponent(
                component=Component(
                    label="IPS Laser Diode",
                    definition=IPS_butterfly_diode(
                        **self.diode_args
                    ),
                ),
                position=(0, 0, 0),
                rotation=(0, 0, 0),
            ),
            Subcomponent(
                component=Component(
                    label="Koheron Controller",
                    definition=Koheron_Controller(
                        **controller_args
                    ),
                ),
                position=(0, 0, 0),
                rotation=(0, 0, 0),
            ),
            Subcomponent(
                component=Component(
                    label=(
                        "IPS Adapter Small"
                        if self.adapter == "small"
                        else "IPS Adapter Big"
                    ),
                    definition=adapter_definition(
                        **adapter_args
                    ),
                ),
                position=(0, 0, 0),
                rotation=(0, 0, 0),
            ),
        ]

        if self.fiber_clamp:
            components.append(
                Subcomponent(
                    component=Component(
                        label="Koheron Fiber Clamp",
                        definition=Fiber_Clamp_Koheron(
                            **self.fiber_clamp_args
                        ),
                    ),
                    position=(0, 0, 0),
                    rotation=(0, 0, 0),
                )
            )

        return components


class toptica_laser:
    """Toptica DFB butterfly diode mounted with a Koheron controller.

    The Koheron controller is displaced by ``(-1.4, 0, 1.4)`` mm relative
    to the Toptica diode. This preserves the controller position when
    exchanging the Toptica DFB and IPS laser assemblies.

    Parameters
    ----------

    diode_args, adapter_args, controller_args, fiber_clamp_args
        Optional keyword arguments forwarded to the respective definitions.
    """

    object_group = "components"
    object_icon = ""
    object_color = misc_color

    controller_position = (-1.4, 0, 1.4)

    def __init__(
        self,
        fiber_clamp: bool = False,
        drill: bool = True,
        diode_args: dict | None = None,
        adapter_args: dict | None = None,
        controller_args: dict | None = None,
        fiber_clamp_args: dict | None = None,
    ):
        self.fiber_clamp = fiber_clamp
        self.drill_enabled = drill

        self.diode_args = (
            {} if diode_args is None else dict(diode_args)
        )
        self.adapter_args = (
            {} if adapter_args is None else dict(adapter_args)
        )
        self.controller_args = (
            {} if controller_args is None else dict(controller_args)
        )
        self.fiber_clamp_args = (
            {} if fiber_clamp_args is None
            else dict(fiber_clamp_args)
        )

    def subcomponents(self) -> list[Subcomponent]:
        adapter_args = {
            "drill": self.drill_enabled,
            **self.adapter_args,
        }

        controller_args = {
            "drill": self.drill_enabled,
            **self.controller_args,
        }

        components = [
            Subcomponent(
                component=Component(
                    label="Toptica DFB Laser Diode",
                    definition=DFB_butterfly_diode(
                        **self.diode_args
                    ),
                ),
                position=(0, 0, 0),
                rotation=(0, 0, 0),
            ),
            Subcomponent(
                component=Component(
                    label="Koheron Controller",
                    definition=Koheron_Controller(
                        **controller_args
                    ),
                ),
                position=self.controller_position,
                rotation=(0, 0, 0),
            ),
            Subcomponent(
                component=Component(
                    label="Toptica Adapter",
                    definition=DFB_adapter(
                        **adapter_args
                    ),
                ),
                position=(0, 0, 0),
                rotation=(0, 0, 0),
            ),
        ]

        if self.fiber_clamp:
            components.append(
                Subcomponent(
                    component=Component(
                        label="Koheron Fiber Clamp",
                        definition=Fiber_Clamp_Koheron(
                            **self.fiber_clamp_args
                        ),
                    ),
                    position=(0, 0, 0),
                    rotation=(0, 0, 0),
                )
            )

        return components



class TA_butterfly:
    """Tapered Amplifier Evaluation board, model EYP-TPA-0785-0100-3006-CMT03

    Args:
        drill (bool) : Whether baseplate mounting for this part should be drilled"""
    object_group = "components"
    object_icon = thorlabs_icon
    object_color = mount_color
    model_source = {"stl": 'TAboard.stl', "rotation": (90, 0, 0), "translation": (0, 0, 0), "scale": 1}
    mesh = LazyModel("ta-butterfly", directory=MODELS_DIRECTORY)

    def __init__(self, drill: bool = True):
        self.drill_enabled = drill
        self.drill_enabled = drill
        self.part_numbers = ['TAboard']

    def subcomponents(self) -> list[Subcomponent]:
        components = []
        components.append(
            Subcomponent(
                component=Component(label='TA adapter', definition=TA_adapter()),
                position=(0, 0, 0),
                rotation=(0, 0, 0),
            )
        )
        components.append(
            Subcomponent(
                component=Component(label='Fiber Clamp 3 Bottom', definition=fiber_clamp_3_bottom()),
                position=(-18, 5 * INCH, -12.7),
                rotation=(0, 0, 0),
            )
        )
        components.append(
            Subcomponent(
                component=Component(label='Fiber Clamp 3 Top', definition=fiber_clamp_3_top()),
                position=(-18, 5 * INCH, -12.7),
                rotation=(0, 0, 0),
            )
        )
        components.append(
            Subcomponent(
                component=Component(label='Fiber Clamp 3 Bottom', definition=fiber_clamp_3_bottom()),
                position=(-54, 5.7 * INCH, -12.7),
                rotation=(0, 0, 0),
            )
        )
        components.append(
            Subcomponent(
                component=Component(label='Fiber Clamp 3 Top', definition=fiber_clamp_3_top()),
                position=(-54, 5.7 * INCH, -12.7),
                rotation=(0, 0, 0),
            )
        )
        return components

    def drill(self) -> Part.Shape:
        shape = self.mesh.copy()
        part = _custom_cylinder(dia=bolt_M2_5['tap_dia'], dz=DRILL_DEPTH / 12, y=15.875, x=-37.2, z=-13)
        part = part.fuse(_custom_cylinder(dia=bolt_M2_5['tap_dia'], dz=DRILL_DEPTH / 12, y=-15.875, x=-37.2, z=-13))
        part = part.fuse(_custom_cylinder(dia=bolt_M2_5['tap_dia'], dz=DRILL_DEPTH / 12, x=13.6, y=15.875, z=-13))
        part = part.fuse(_custom_cylinder(dia=bolt_M2_5['tap_dia'], dz=DRILL_DEPTH / 12, x=13.6, y=-15.875, z=-13))
        for i in [-1, 1]:
            part = part.fuse(_custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=-18 + 18 * i, y=5 * INCH, z=0))
        for i in [-1, 1]:
            part = part.fuse(_custom_cylinder(dia=bolt_8_32['tap_dia'], dz=DRILL_DEPTH, x=-54 + 18 * i, y=5.7 * INCH, z=0))
        drill_part = part
        return part

class rb_cell_holder_top:
    """importing the post mountable v-clamp
    version VBC2"""
    object_group = "mounts"
    object_icon = thorlabs_icon
    object_color = mount_color
    model_source = {"stl": 'rb_cell_holder_top.stl', "rotation": (0, 0, 0), "translation": (0, 0, 0), "scale": 1}
    mesh = LazyModel("rb-cell-holder-top", directory=MODELS_DIRECTORY)

    def __init__(self):
        self.part_numbers = ['VBC2']

class Vapor_Ref_Cell:
    """importing the vapor reference cell
    Vapor_Reference_Cell_version GC25075-RB

    Args:
        drill (bool) : Whether baseplate mounting for this part should be drilled
        mirror (bool) : Whether to add a mirror component to the mount
        thumbscrews (bool): Whether or not to add two HKTS 5-64 adjusters"""
    object_group = "optics"
    object_icon = optic_icon
    object_color = mount_color
    model_source = {"stl": 'GC25075-RB.stl', "rotation": (90, 90, 90), "translation": (0, 0, 0), "scale": 1}
    mesh = LazyModel("vapor-ref-cell", directory=MODELS_DIRECTORY)

    def __init__(self):
        self.part_numbers = ['GC25075-RB']

    def subcomponents(self) -> list[Subcomponent]:
        components = []
        components.append(
            Subcomponent(
                component=Component(label='rb cell holder middle', definition=rb_cell()),
                position=(0, 0, 0),
                rotation=(0, 0, 0),
            )
        )
        components.append(
            Subcomponent(
                component=Component(label='rb cell holder top', definition=rb_cell_holder_top()),
                position=(0, 5, 0),
                rotation=(0, 0, 0),
            )
        )
        return components

class cube_splitter:
    """Half-inch beam-splitter cube.

    The v2 Reflection interface produces two outputs when ``0 < ref_ratio < 1``:
    transmitted ``input_index << 1`` and reflected
    ``(input_index << 1) + 1``.

    Args:
        cube_size: Cube side length.
        invert: Reverse the reflected pick-off direction.
        cube_part_number: Part number string.
        mount_type: Optional mount definition class.
        mount_args: Arguments forwarded to ``mount_type``.
        ref_ratio: Reflected power fraction. Defaults to 0.5.
    """

    object_group = "optics"
    object_icon = optic_icon
    object_color = glass_color
    object_transparency = 50

    def __init__(
        self,
        cube_size: dim = dim(0.5, "in"),
        invert: bool = False,
        cube_part_number: str = "",
        mount_type: object = None,
        mount_args: dict = None,
        ref_ratio: float = 0.5,
    ):
        if not 0 < ref_ratio < 1:
            raise ValueError(
                "cube_splitter requires 0 < ref_ratio < 1 "
                "to generate both transmitted and reflected beams."
            )

        self.cube_size = cube_size
        self.invert = invert
        self.mount_type = mount_type
        self.mount_args = {} if mount_args is None else dict(mount_args)
        self.ref_ratio = ref_ratio
        self.part_numbers = [cube_part_number]

        # Match the stock PyOpticL.library.optics.Beamsplitter_Cube convention.
        # invert=False: reflected beam turns toward local +Y.
        # invert=True:  reflected beam turns toward local -Y.
        self.split_angle = 45 if invert else -45

    def subcomponents(self) -> list[Subcomponent]:
        if self.mount_type is None:
            return []

        return [
            Subcomponent(
                component=Component(
                    label="Mount",
                    definition=self.mount_type(**self.mount_args),
                ),
                position=(0, 0, -self.cube_size / 2),
                rotation=(0, 0, 0),
            )
        ]

    def interfaces(self):
        return [
            Reflection(
                position=(0, 0, 0),
                rotation=(0, 0, self.split_angle),
                width=self.cube_size * np.sqrt(2),
                height=self.cube_size * np.sqrt(2),
                ref_ratio=self.ref_ratio,
                max_angle=90,
            )
        ]

    def shape(self) -> Part.Shape:
        part = _custom_box(
            dx=self.cube_size,
            dy=self.cube_size,
            dz=self.cube_size,
            x=0,
            y=0,
            z=0,
            dir=(0, 0, 0),
        )
        diagonal = self.cube_size * np.sqrt(2)
        split_plane = _custom_box(
            dx=0.1,
            dy=diagonal,
            dz=diagonal,
            x=0,
            y=0,
            z=0,
            dir=(0, 0, 0),
        )
        split_plane.rotate(
            App.Vector(0, 0, 0),
            App.Vector(0, 0, 1),
            self.split_angle,
        )
        return part.cut(split_plane)

class circular_lens:
    """Circular Lens

    Args:
        drill (bool) : Whether baseplate mounting for this part should be drilled
        focal_length (float) : The focal length of the lens
        thickness (float) : The edge thickness of the lens
        diameter (float) : The width of the lens
        part_number (string) : The part number of the lens being used"""
    object_group = "optics"
    object_icon = optic_icon
    object_color = glass_color
    object_transparency = 50

    def __init__(self, drill: bool = True, focal_length: dim = 50, thickness: dim = 3, diameter: dim = INCH / 2, part_number: str = '', mount_type: object = None, mount_args: dict = None):
        self.drill_enabled = drill
        self.focal_length = focal_length
        self.thickness = thickness
        self.diameter = diameter
        self.part_number = part_number
        self.mount_type = mount_type
        self.mount_args = {} if mount_args is None else dict(mount_args)
        self.drill_enabled = drill
        self.focal_length = focal_length
        self.thickness = thickness
        self.diameter = diameter
        self.part_numbers = [part_number]
        self.transmission = True
        self.focal_length = self.focal_length
        self.max_angle = 90
        self.max_width = diameter

    def subcomponents(self) -> list[Subcomponent]:
        components = []
        mount_args = dict(self.mount_args)
        if self.mount_type != None:
            components.append(
                Subcomponent(
                    component=Component(label='Mount', definition=self.mount_type(**mount_args)),
                    position=(-self.thickness / 2, 0, 0),
                    rotation=(0, 0, 0),
                )
            )
        return components

    def interfaces(self):
        return [
            Lens(
                position=(0, 0, 0), rotation=(0, 0, 0),
                diameter=self.diameter, focal_length=self.focal_length,
            )
        ]

    def shape(self) -> Part.Shape:
        part = _custom_cylinder(dia=self.diameter, dz=self.thickness, x=-self.thickness / 2, y=0, z=0, dir=(1, 0, 0))
        return part

class waveplate:
    """Waveplate

    Args:
        drill (bool) : Whether baseplate mounting for this part should be drilled
        thickness (float) : The thickness of the waveplate
        diameter (float) : The width of the waveplate
        part_number (string) : The part number of the waveplate being used"""
    object_group = "optics"
    object_icon = optic_icon
    object_color = glass_color
    object_transparency = 50

    def __init__(self, drill: bool = True, thickness: dim = 1, diameter: dim = INCH / 2, part_number: str = '', mount_type: object = None, mount_args: dict = None, retardance: float = None, fast_axis_angle: float = 0.0):
        self.drill_enabled = drill
        self.thickness = thickness
        self.diameter = diameter
        self.part_number = part_number
        self.mount_type = mount_type
        self.mount_args = {} if mount_args is None else dict(mount_args)
        self.retardance = retardance
        self.fast_axis_angle = fast_axis_angle
        self.drill_enabled = drill
        self.thickness = thickness
        self.diameter = diameter
        self.part_numbers = [part_number]
        self.transmission = True
        self.max_angle = 90
        self.max_width = diameter
        self.retardance = retardance
        self.fast_axis_angle = fast_axis_angle

    def subcomponents(self) -> list[Subcomponent]:
        components = []
        mount_args = dict(self.mount_args)
        if self.mount_type != None:
            components.append(
                Subcomponent(
                    component=Component(label='Mount', definition=self.mount_type(**mount_args)),
                    position=(-self.thickness / 2, 0, 0),
                    rotation=(0, 0, 0),
                )
            )
        return components

    def interfaces(self):
        if self.retardance is None:
            return [Interface(position=(0, 0, 0), rotation=(0, 0, 0), diameter=getattr(self, 'diameter', getattr(self, 'max_width', None)), max_angle=getattr(self, 'max_angle', 90))]
        return [
            Waveplate(
                position=(0, 0, 0), rotation=(0, 0, 0),
                diameter=getattr(self, 'diameter', getattr(self, 'max_width', None)),
                retardance=self.retardance, fast_axis_angle=self.fast_axis_angle,
            )
        ]

    def shape(self) -> Part.Shape:
        part = _custom_cylinder(dia=self.diameter, dz=self.thickness, x=-self.thickness / 2, y=0, z=0, dir=(1, 0, 0))
        return part

class circular_mirror:
    """Circular Mirror

    Args:
        drill (bool) : Whether baseplate mounting for this part should be drilled
        thickness (float) : The thickness of the mirror
        diameter (float) : The width of the mirror
        part_number (string) : The part number of the mirror being used"""
    object_group = "optics"
    object_icon = optic_icon
    object_color = glass_color

    def __init__(self, drill: bool = True, thickness: dim = 6, diameter: dim = INCH / 2, part_number: str = '', mount_type: object = None, mount_args: dict = None, mount_height: dim = 0, adapter_type: object = None, adapter_args: dict = None):
        self.drill_enabled = drill
        self.thickness = thickness
        self.diameter = diameter
        self.part_number = part_number
        self.mount_type = mount_type
        self.mount_args = {} if mount_args is None else dict(mount_args)
        self.mount_height = mount_height
        self.adapter_type = adapter_type
        self.adapter_args = {} if adapter_args is None else dict(adapter_args)
        self.drill_enabled = drill
        self.thickness = thickness
        self.diameter = diameter
        self.part_numbers = [part_number]
        self.reflection_angle = 0
        self.max_angle = 90
        self.max_width = diameter

    def subcomponents(self) -> list[Subcomponent]:
        components = []
        mount_args = dict(self.mount_args)
        adapter_args = dict(self.adapter_args)
        if self.mount_type != None:
            components.append(
                Subcomponent(
                    component=Component(label='Mount', definition=self.mount_type(**mount_args)),
                    position=(-self.thickness, 0, 0),
                    rotation=(0, 0, 0),
                )
            )
        return components

    def interfaces(self):
        return [Reflection(position=(0, 0, 0), rotation=(0, 0, 0), diameter=self.diameter, max_angle=getattr(self, 'max_angle', 90))]

    def shape(self) -> Part.Shape:
        part = _custom_cylinder(dia=self.diameter, dz=self.thickness, x=0, y=0, z=0, dir=(-1, 0, 0))
        return part

class cavity_mirror_M05(Cavity_Mirror):
    """Cavity mirror mounted in an M05."""

    def __init__(
        self,
        drill: bool = True,
        thickness: dim = 6,
        diameter: dim = INCH / 2,
        ref_ratio: float = 0.5,
        input: bool = True,
        thumbscrews: bool = True,
        part_number: str = "",
    ):
        super().__init__(
            diameter=diameter,
            thickness=thickness,
            ref_ratio=ref_ratio,
            input=input,
            mount_definition=mirror_mount_M05(
                drill=drill,
                thumbscrews=thumbscrews,
            ),
            part_number=part_number,
        )

        self.drill_enabled = drill


class circular_mirror_union_optic:
    """Circular Mirror

    Args:
        drill (bool) : Whether baseplate mounting for this part should be drilled
        thickness (float) : The thickness of the mirror
        diameter (float) : The width of the mirror
        part_number (string) : The part number of the mirror being used"""
    object_group = "optics"
    object_icon = optic_icon
    object_color = glass_color

    def __init__(self, drill: bool = True, thickness: dim = 3, diameter: dim = INCH / 2, part_number: str = '', mount_type: object = None, mount_args: dict = None, mount_height: dim = 0, adapter_type: object = None, adapter_args: dict = None):
        self.drill_enabled = drill
        self.thickness = thickness
        self.diameter = diameter
        self.part_number = part_number
        self.mount_type = mount_type
        self.mount_args = {} if mount_args is None else dict(mount_args)
        self.mount_height = mount_height
        self.adapter_type = adapter_type
        self.adapter_args = {} if adapter_args is None else dict(adapter_args)
        self.drill_enabled = drill
        self.thickness = thickness
        self.diameter = diameter
        self.part_numbers = [part_number]
        self.reflection_angle = 0
        self.max_angle = 90
        self.max_width = diameter

    def subcomponents(self) -> list[Subcomponent]:
        components = []
        mount_args = dict(self.mount_args)
        adapter_args = dict(self.adapter_args)
        if self.mount_type != None:
            components.append(
                Subcomponent(
                    component=Component(label='Mount', definition=self.mount_type(**mount_args)),
                    position=(-self.thickness, 0, 0),
                    rotation=(0, 0, 0),
                )
            )
        return components

    def interfaces(self):
        return [Reflection(position=(0, 0, 0), rotation=(0, 0, 0), diameter=self.diameter, max_angle=getattr(self, 'max_angle', 90))]

    def shape(self) -> Part.Shape:
        part = _custom_cylinder(dia=self.diameter, dz=self.thickness, x=0, y=0, z=0, dir=(-1, 0, 0))
        return part

surface_adapter_4_40 = surface_adapter

surface_adapter_lying_down = surface_adapter

