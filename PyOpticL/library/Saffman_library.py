"""Saffman Lab component library for PyOpticL v2."""

from pathlib import Path

import FreeCAD as App
import numpy as np
import Part

from PyOpticL.beam_path import Interface, Reflection, Stop
from PyOpticL.icons import optic_icon, thorlabs_icon
from PyOpticL.layout import Component, Subcomponent
from PyOpticL.library import optics
from PyOpticL.library import Sinclair_library as sinclair
from PyOpticL.utils import Dimension as dim
from PyOpticL.utils import box_shape, import_model

MODELS_DIRECTORY = Path(__file__).resolve().parent.parent / "models"

INCH = dim(1, "in")

# The Saffman board is designed around a 1-inch optical height.
# The collision cutout helper below extends each imported-model bounding box
# down/up to this baseplate surface so the cutout always intersects the board.
SAFFMAN_OPTICAL_HEIGHT = INCH
BASEPLATE_SURFACE_Z = -SAFFMAN_OPTICAL_HEIGHT

DRILL_DEPTH = sinclair.DRILL_DEPTH

bolt_4_40 = sinclair.bolt_4_40
bolt_8_32 = sinclair.bolt_8_32
bolt_14_20 = sinclair.bolt_14_20
bolt_m2p5x4p5 = sinclair.bolt_m2p5x4p5
bolt_M2_5 = sinclair.bolt_M2_5
bolt_M4 = sinclair.bolt_M4
bolt_M6 = sinclair.bolt_M6

CUTOUT_TOLERANCE = dim(2, "mm")
CUTOUT_FILLET = dim(0.125, "in")

mount_color = (0.5, 0.5, 0.55)
glass_color = (0.5, 0.5, 0.8)
misc_color = (0.2, 0.2, 0.2)


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


def _bounding_box_cutout(
    body,
    tolerance=CUTOUT_TOLERANCE,
    fillet=CUTOUT_FILLET,
    x_tolerance=True,
    y_tolerance=True,
    z_tolerance=False,
    min_offset=(0, 0, 0),
    max_offset=(0, 0, 0),
    plate_z=BASEPLATE_SURFACE_Z,
    extend_to_plate=True,
):
    """Create a Sinclair-style clearance envelope around an imported model."""

    body = body.copy()
    body.Placement = App.Placement()
    bound = body.BoundBox

    x_min = bound.XMin - tolerance * x_tolerance + min_offset[0]
    x_max = bound.XMax + tolerance * x_tolerance + max_offset[0]
    y_min = bound.YMin - tolerance * y_tolerance + min_offset[1]
    y_max = bound.YMax + tolerance * y_tolerance + max_offset[1]

    z_min_actual = bound.ZMin - tolerance * z_tolerance + min_offset[2]
    z_max_actual = bound.ZMax + tolerance * z_tolerance + max_offset[2]
    if extend_to_plate:
        z_min = min(z_min_actual, plate_z)
        z_max = max(z_max_actual, plate_z)
    else:
        z_min = z_min_actual
        z_max = z_max_actual

    return box_shape(
        dimensions=(
            x_max - x_min,
            y_max - y_min,
            z_max - z_min,
        ),
        position=(x_min, y_min, z_min),
        center=(-1, -1, -1),
        fillet=fillet,
        fillet_direction=(0, 0, 1),
    )


def _no_cutout_shape():
    """Return a valid solid that cannot intersect the baseplate."""
    return box_shape(
        dimensions=(0.01, 0.01, 0.01),
        position=(1.0e6, 1.0e6, 1.0e6),
        center=(-1, -1, -1),
    )


def _xyz(point):
    """Return (x, y, z) from a FreeCAD vector or coordinate tuple."""
    if hasattr(point, "x"):
        return float(point.x), float(point.y), float(point.z)
    return float(point[0]), float(point[1]), float(point[2])


def _mesh_facets(body):
    """Return mesh facets for either Mesh.MeshObject or a wrapped mesh."""
    if hasattr(body, "Facets"):
        return body.Facets
    if hasattr(body, "Mesh") and hasattr(body.Mesh, "Facets"):
        return body.Mesh.Facets
    return []


def _x_range_cutout(
    body,
    x_min,
    x_max,
    xy_clearance=dim(2, "mm"),
    z_clearance=dim(0, "mm"),
    negative_x_clearance=dim(0, "mm"),
    positive_x_clearance=dim(0, "mm"),
    fillet=CUTOUT_FILLET,
    plate_z=BASEPLATE_SURFACE_Z,
):
    """Create a cutout only around mesh geometry occupying a given X range.

    The STL imported by PyOpticL is a Mesh object, not a Part.Shape, so this
    helper deliberately avoids Part boolean operations such as ``common()``.

    Facets whose X-span overlaps [x_min, x_max] are selected.  Their Y/Z
    extent determines the local pocket; the user-supplied X boundaries remain
    the exact feature boundaries.  Machining clearance is then added around
    that local region.
    """

    body = body.copy()
    body.Placement = App.Placement()

    selected_points = []

    for facet in _mesh_facets(body):
        points = [_xyz(point) for point in facet.Points]
        facet_x_min = min(point[0] for point in points)
        facet_x_max = max(point[0] for point in points)

        if facet_x_max >= x_min and facet_x_min <= x_max:
            selected_points.extend(points)

    if not selected_points:
        return _no_cutout_shape()

    y_min_actual = min(point[1] for point in selected_points)
    y_max_actual = max(point[1] for point in selected_points)
    z_min_actual = min(point[2] for point in selected_points)

    # The feature only needs a pocket when it actually enters the board.
    if z_min_actual >= plate_z:
        return _no_cutout_shape()

    pocket_x_min = x_min - xy_clearance - negative_x_clearance
    pocket_x_max = x_max + xy_clearance + positive_x_clearance
    pocket_y_min = y_min_actual - xy_clearance
    pocket_y_max = y_max_actual + xy_clearance
    pocket_z_min = z_min_actual - z_clearance
    pocket_z_max = plate_z

    dx = pocket_x_max - pocket_x_min
    dy = pocket_y_max - pocket_y_min
    dz = pocket_z_max - pocket_z_min

    # Keep the requested 1/8-inch fillet unless the local pocket is too narrow.
    local_fillet = min(
        fillet,
        max(dim(0, "mm"), dx / 2 - dim(0.01, "mm")),
        max(dim(0, "mm"), dy / 2 - dim(0.01, "mm")),
    )

    return box_shape(
        dimensions=(dx, dy, dz),
        position=(pocket_x_min, pocket_y_min, pocket_z_min),
        center=(-1, -1, -1),
        fillet=local_fillet,
        fillet_direction=(0, 0, 1),
    )


class _FiveAngleReflection(Reflection):
    """Reflection interface whose output angle is five times incidence."""

    angle_multiplier = 5

    def get_output_beams(self, incident_beam):
        output_beams = super().get_output_beams(incident_beam)
        if len(output_beams) != 1:
            return output_beams

        beam_direction = incident_beam.get_global_direction()
        normal = self.get_global_normal()
        if np.dot(normal, beam_direction) > 0:
            normal = -normal

        incident_angle = np.arccos(
            np.clip(np.dot(-beam_direction, normal), -1, 1)
        )
        tangent = beam_direction + normal * np.cos(incident_angle)
        tangent_norm = np.linalg.norm(tangent)

        if np.isclose(tangent_norm, 0):
            return output_beams

        tangent /= tangent_norm
        reflected_angle = self.angle_multiplier * incident_angle
        direction = (
            normal * np.cos(reflected_angle)
            + tangent * np.sin(reflected_angle)
        )
        output_beams[0].direction = tuple(
            incident_beam.get_relative_direction(direction)
        )
        return output_beams


class mirror_mount_K1E:
    """Thorlabs Polaris K1E mirror mount."""

    object_group = "mounts"
    object_icon = thorlabs_icon
    object_color = mount_color

    model_source = {
        "stl": "mirror-mount-k1e.stl",
        "rotation": (0, 0, 0),
        "translation": (0, 0, 0),
        "scale": 1,
    }

    mesh = LazyModel(
        "mirror-mount-k1e",
        directory=MODELS_DIRECTORY,
    )

    def __init__(self):
        self.part_numbers = ["K1E"]

    def drill(self):
        part = _x_range_cutout(
            self.mesh,
            x_min=dim(-1.248, "in"),
            x_max=dim(-0.873, "in"),
            xy_clearance=dim(2, "mm"),
            z_clearance=dim(2, "mm"),
            negative_x_clearance=dim(10, "mm"),
            fillet=dim(0.125, "in"),
        )
        # Actual threaded mount hole locations for the K1E mirror mount.
        part = part.fuse(
            sinclair._custom_cylinder(
                dia=bolt_8_32["tap_dia"],
                dz=dim(100, "mm"),
                x=dim(-0.415, "in"),
                y=0,
                z=0,
                dir=(0, 0, -1),
            )
        )
        return part


class vbg_K1E:
    """Volume Bragg Grating filter mounted on a K1E."""

    object_group = "optics"
    object_icon = optic_icon
    object_color = glass_color

    model_source = {
        "stl": "vbg-k1e.stl",
        "rotation": (0, 0, 0),
        "translation": (0, 0, 0),
        "scale": 1,
    }

    mesh = LazyModel(
        "vbg-k1e",
        directory=MODELS_DIRECTORY,
    )

    def __init__(self):
        self.part_numbers = ["VBG", "K1E"]
        self.max_angle = 18
        self.max_width = INCH

    def interfaces(self):
        return [
            _FiveAngleReflection(
                position=(0, 0, 0),
                rotation=(0, 0, 0),
                diameter=self.max_width,
                max_angle=self.max_angle,
            )
        ]

    def drill(self):
        part = _x_range_cutout(
            self.mesh,
            x_min=dim(0.873, "in"),
            x_max=dim(1.248, "in"),
            xy_clearance=dim(2, "mm"),
            z_clearance=dim(2, "mm"),
            positive_x_clearance=dim(10, "mm"),
            fillet=dim(0.125, "in"),
        )
        # Actual threaded mount hole location for the VBG/K1E assembly.
        part = part.fuse(
            sinclair._custom_cylinder(
                dia=bolt_8_32["tap_dia"],
                dz=dim(100, "mm"),
                x=dim(0.415, "in"),
                y=0,
                z=0,
                dir=(0, 0, -1),
            )
        )
        return part


class Surface_Adapter_CP13:
    """CP13 surface adapter with two 8-32 baseplate mounting holes."""

    object_group = "adapters"
    object_icon = thorlabs_icon
    object_color = mount_color

    model_source = {
        "step": "Surface_Adapter_CP13.step",
        "rotation": (0, 0, 0),
        "translation": (0, 0, 0),
        "scale": 1,
    }

    mesh = LazyModel(
        "Surface_Adapter_CP13",
        directory=MODELS_DIRECTORY,
    )

    def __init__(self):
        self.part_numbers = ["CP13 Surface Adapter"]

    def drill(self):
        part = _bounding_box_cutout(
            self.mesh,
            tolerance=dim(1, "mm"),
            x_tolerance=True,
            y_tolerance=True,
            z_tolerance=False,
            fillet=dim(0.125, "in"),
            extend_to_plate=False,
        )
        part = part.fuse(
            sinclair._custom_cylinder(
                dia=bolt_8_32["tap_dia"],
                dz=dim(100, "mm"),
                x=0,
                y=dim(-1, "in"),
                z=dim(-4.4, "mm"),
                dir=(0, 0, -1),
            )
        )
        part = part.fuse(
            sinclair._custom_cylinder(
                dia=bolt_8_32["tap_dia"],
                dz=dim(100, "mm"),
                x=0,
                y=dim(1, "in"),
                z=dim(-4.4, "mm"),
                dir=(0, 0, -1),
            )
        )
        return part


class Vertical_Cube_10mm:
    """Plastic vertical cube with a separate 10 mm beam splitter at its origin."""

    object_group = "misc"
    object_icon = optic_icon
    object_color = misc_color

    model_source = {
        "step": "vertical-cube-10mm.step",
        "rotation": (0, 0, 0),
        "translation": (0, 0, 0),
        "scale": 1,
    }

    mesh = LazyModel(
        "vertical-cube-10mm",
        directory=MODELS_DIRECTORY,
    )

    def __init__(self):
        self.part_numbers = [
            "Vertical Cube 10 mm",
            "10 mm Beam Splitter",
            "CP13 Surface Adapter",
        ]

    def subcomponents(self):
        return [
            Subcomponent(
                component=Component(
                    label="10 mm Beam Splitter",
                    definition=optics.Beamsplitter_Cube(
                        side_length=dim(10, "mm"),
                        ref_ratio=0.5,
                        part_number="10 mm Beam Splitter",
                    ),
                ),
                position=(0, 0, 0),
                rotation=(0, 0, 0),
            ),
            Subcomponent(
                component=Component(
                    label="CP13 Surface Adapter",
                    definition=Surface_Adapter_CP13(),
                ),
                position=(dim(-9.44, "mm"), 0, dim(-20.32, "mm")),
                rotation=(0, 0, 0),
            ),
        ]

    def interfaces(self):
        return []



class lens_tube_SM1L03:
    """Thorlabs SM1L03 lens tube."""

    object_group = "mounts"
    object_icon = thorlabs_icon
    object_color = mount_color

    model_source = {
        "stl": "lens-tube-sm1-l03.stl",
        "rotation": (90, 0, 0),
        "translation": (8.382, 0, 0),
        "scale": 1,
    }

    mesh = LazyModel(
        "lens-tube-sm1-l03",
        directory=MODELS_DIRECTORY,
    )

    def __init__(self):
        self.part_numbers = ["SM1L03"]
        self.max_angle = 90
        self.max_width = INCH

    def interfaces(self):
        return [
            Interface(
                position=(0, 0, 0),
                rotation=(0, 0, 0),
                diameter=self.max_width,
                max_angle=self.max_angle,
            )
        ]

    def drill(self):
        return _bounding_box_cutout(self.mesh)


class lens_mount_HPT1:
    """Thorlabs HPT1 post-mounted lens holder.

    A transparent placement interface is included at the optic center so a
    BeamPath can pass through the clear aperture.
    """

    object_group = "mounts"
    object_icon = thorlabs_icon
    object_color = mount_color

    model_source = {
        "stl": "lens-mount-hpt1.stl",
        "rotation": (0, 0, 0),
        "translation": (0, 0, 0),
        "scale": 1,
    }

    mesh = LazyModel(
        "lens-mount-hpt1",
        directory=MODELS_DIRECTORY,
    )

    def __init__(self):
        self.part_numbers = ["HPT1"]
        self.max_angle = 90
        self.max_width = INCH

    def subcomponents(self):
        return [
            Subcomponent(
                component=Component(
                    label="CP13 Surface Adapter",
                    definition=Surface_Adapter_CP13(),
                ),
                position=(dim(-16.193, "mm"), 0, dim(-20.319, "mm")),
                rotation=(0, 0, 0),
            )
        ]

    def interfaces(self):
        return [
            Interface(
                position=(0, 0, 0),
                rotation=(0, 0, 0),
                diameter=self.max_width,
                max_angle=self.max_angle,
            )
        ]

    def drill(self):
        return _bounding_box_cutout(self.mesh)


class fiber_paddle_MPC320:
    """Thorlabs MPC320 motorized fiber polarization controller paddle."""

    object_group = "misc"
    object_icon = thorlabs_icon
    object_color = misc_color

    model_source = {
        "stl": "fiber-paddle-mpc320.stl",
        "rotation": (0, 0, 0),
        "translation": (0, 0, 0),
        "scale": 1,
    }

    mesh = LazyModel(
        "fiber-paddle-mpc320",
        directory=MODELS_DIRECTORY,
    )

    def __init__(self):
        self.part_numbers = ["MPC320"]

    def drill(self):
        part = _bounding_box_cutout(self.mesh)
        for x, y in (
            (dim(-11.633, "mm"), dim(41.189, "mm")),
            (dim(-11.633, "mm"), dim(91.189, "mm")),
            (dim(-86.433, "mm"), dim(41.189, "mm")),
            (dim(-86.433, "mm"), dim(91.189, "mm")),
        ):
            part = part.fuse(
                sinclair._custom_cylinder(
                    dia=bolt_14_20["tap_dia"],
                    dz=dim(100, "mm"),
                    x=x,
                    y=y,
                    z=0,
                    dir=(0, 0, -1),
                )
            )
        return part

class rotation_stage_ELL14:
    """Thorlabs ELL14 rotation stage.

    A transparent placement interface is included at the optic center so a
    BeamPath can pass through the clear aperture.
    """

    object_group = "mounts"
    object_icon = thorlabs_icon
    object_color = mount_color

    model_source = {
        "stl": "rotation-stage-ell14.stl",
        "rotation": (0, 0, 0),
        "translation": (0, 0, 0),
        "scale": 1,
    }

    mesh = LazyModel(
        "rotation-stage-ell14",
        directory=MODELS_DIRECTORY,
    )

    def __init__(self):
        self.part_numbers = ["ELL14"]
        self.max_angle = 90
        self.max_width = INCH

    def subcomponents(self):
        return [
            Subcomponent(
                component=Component(
                    label="CP13 Surface Adapter",
                    definition=Surface_Adapter_CP13(),
                ),
                position=(dim(20.44, "mm"), 0, dim(-20.32, "mm")),
                rotation=(0, 0, 0),
            )
        ]

    def interfaces(self):
        return [
            Interface(
                position=(0, 0, 0),
                rotation=(0, 0, 0),
                diameter=self.max_width,
                max_angle=self.max_angle,
            )
        ]

    def drill(self):
        return _x_range_cutout(
            self.mesh,
            x_min=dim(-7.6, "mm"),
            x_max=dim(7.2, "mm"),
            xy_clearance=dim(2, "mm"),
            z_clearance=dim(2, "mm"),
            fillet=dim(0.125, "in"),
        )


class Polarimeter_Adapter:
    """Polarimeter surface adapter with two 8-32 tapped mounting holes."""

    object_group = "adapters"
    object_icon = thorlabs_icon
    object_color = mount_color

    model_source = {
        "step": "polarimeter-adapter.step",
        "rotation": (0, 0, 0),
        "translation": (0, 0, 0),
        "scale": 1,
    }

    mesh = LazyModel(
        "polarimeter-adapter",
        directory=MODELS_DIRECTORY,
    )

    def __init__(self):
        self.part_numbers = ["Polarimeter Adapter"]

    def drill(self):
        part = _bounding_box_cutout(
            self.mesh,
            tolerance=dim(1, "mm"),
            x_tolerance=True,
            y_tolerance=True,
            z_tolerance=False,
            fillet=dim(0.125, "in"),
            extend_to_plate=False,
        )
        part = part.fuse(
            sinclair._custom_cylinder(
                dia=bolt_8_32["tap_dia"],
                dz=dim(100, "mm"),
                x=0,
                y=dim(30, "mm"),
                z=0,
                dir=(0, 0, -1),
            )
        )
        part = part.fuse(
            sinclair._custom_cylinder(
                dia=bolt_8_32["tap_dia"],
                dz=dim(100, "mm"),
                x=0,
                y=dim(-30, "mm"),
                z=0,
                dir=(0, 0, -1),
            )
        )
        return part


class polarimeter_PAX1000IR1:
    """Thorlabs PAX1000IR1 polarimeter."""

    object_group = "detector"
    object_icon = thorlabs_icon
    object_color = misc_color

    model_source = {
        "stl": "polarimeter-pax1000ir1.stl",
        "rotation": (0, 0, 0),
        "translation": (0, 0, 0),
        "scale": 1,
    }

    mesh = LazyModel(
        "polarimeter-pax1000ir1",
        directory=MODELS_DIRECTORY,
    )

    def __init__(self):
        self.part_numbers = ["PAX1000IR1", "Polarimeter Adapter"]
        self.max_angle = 80
        self.max_width = dim(10, "mm")

    def subcomponents(self):
        return [
            Subcomponent(
                component=Component(
                    label="Polarimeter Adapter",
                    definition=Polarimeter_Adapter(),
                ),
                position=(dim(-28.532, "mm"), 0, dim(-25, "mm")),
                rotation=(0, 0, 0),
            )
        ]

    def interfaces(self):
        return [
            Stop(
                position=(0, 0, 0),
                rotation=(0, 0, 0),
                diameter=self.max_width,
                max_angle=self.max_angle,
            )
        ]

    def drill(self):
        return _bounding_box_cutout(
            self.mesh,
            fillet=dim(0.25, "in"),
        )


class fiberport_SM1ZA:
    """Thorlabs SM1ZA fiber collimator.

    The imported CAD model is aligned to the PyOpticL optical coordinate
    system with the fiber-coupling plane at the local origin.
    """

    object_group = "mounts"
    object_icon = thorlabs_icon
    object_color = mount_color

    model_source = {
        "stl": "fiberport-sm1za.stl",
        "rotation": (0, 0, 0),
        "translation": (0, 0, 0),
        "scale": 1,
    }

    mesh = LazyModel(
        "fiberport-sm1za",
        directory=MODELS_DIRECTORY,
    )

    def __init__(self):
        self.part_numbers = ["SM1ZA", "CP13 Surface Adapter"]

    def subcomponents(self):
        return [
            Subcomponent(
                component=Component(
                    label="CP13 Surface Adapter",
                    definition=Surface_Adapter_CP13(),
                ),
                position=(dim(-0.7, "in"), 0, dim(-0.8, "in")),
                rotation=(0, 0, 0),
            )
        ]

    def interfaces(self):
        return [
            Stop(
                position=(0, 0, 0),
                rotation=(0, 0, 0),
                diameter=dim(25, "mm"),
                single_sided=True,
            )
        ]

    def drill(self):
        return _bounding_box_cutout(self.mesh)


class fiberport_SM1ZA_flipped:
    """Flipped Thorlabs SM1ZA fiber collimator."""

    object_group = "mounts"
    object_icon = thorlabs_icon
    object_color = mount_color

    model_source = {
        "stl": "fiberport-sm1za-flipped.stl",
        "rotation": (0, 0, 0),
        "translation": (0, 0, 0),
        "scale": 1,
    }

    mesh = LazyModel(
        "fiberport-sm1za-flipped",
        directory=MODELS_DIRECTORY,
    )

    def __init__(self):
        self.part_numbers = ["SM1ZA_Flipped", "CP13 Surface Adapter"]

    def subcomponents(self):
        return [
            Subcomponent(
                component=Component(
                    label="CP13 Surface Adapter",
                    definition=Surface_Adapter_CP13(),
                ),
                position=(dim(-0.7, "in"), 0, dim(-0.8, "in")),
                rotation=(0, 0, 0),
            )
        ]

    def interfaces(self):
        return [
            Stop(
                position=(0, 0, 0),
                rotation=(0, 0, 0),
                diameter=dim(25, "mm"),
                single_sided=True,
            )
        ]

    def drill(self):
        return _bounding_box_cutout(self.mesh)


class etalon_adapter_model:
    """Etalon box adapter with two baseplate mounting holes."""

    object_group = "adapters"
    object_icon = thorlabs_icon
    object_color = mount_color

    model_source = {
        "step": "etalon-adapter.step",
        "rotation": (0, 0, 0),
        "translation": (0, 0, 0),
        "scale": 1,
    }

    mesh = LazyModel(
        "etalon-adapter",
        directory=MODELS_DIRECTORY,
    )

    def __init__(self):
        self.part_numbers = ["Etalon Adapter"]

    def drill(self):
        part = _bounding_box_cutout(
            self.mesh,
            tolerance=dim(2, "mm"),
            min_offset=(dim(-3, "mm"), 0, 0),
            max_offset=(dim(3, "mm"), 0, 0),
            extend_to_plate=False,
        )
        for y_position in (dim(-72.5, "mm"), dim(72.5, "mm")):
            part = part.fuse(
                sinclair._custom_cylinder(
                    dia=bolt_8_32["tap_dia"],
                    dz=dim(100, "mm"),
                    x=0,
                    y=y_position,
                    z=0,
                    dir=(0, 0, -1),
                )
            )
        return part


class etalon_box_model:
    """Etalon box enclosure with a transparent beam path interface."""

    object_group = "misc"
    object_icon = thorlabs_icon
    object_color = misc_color

    model_source = {
        "stl": "etalon-box.stl",
        "rotation": (0, 0, 0),
        "translation": (0, 0, 0),
        "scale": 1,
    }

    mesh = LazyModel(
        "etalon-box",
        directory=MODELS_DIRECTORY,
    )

    def __init__(self):
        self.part_numbers = ["Etalon Box"]
        self.max_angle = 90
        self.max_width = INCH

    def subcomponents(self):
        return [
            Subcomponent(
                component=Component(
                    label="Etalon Adapter",
                    definition=etalon_adapter_model(),
                ),
                position=(dim(-68.66, "mm"), dim(12.7, "mm"), dim(-39.7, "mm")),
                rotation=(0, 0, 0),
            ),
            Subcomponent(
                component=Component(
                    label="Etalon Adapter",
                    definition=etalon_adapter_model(),
                ),
                position=(dim(68.66, "mm"), dim(12.7, "mm"), dim(-39.7, "mm")),
                rotation=(0, 0, 0),
            ),
        ]

    def interfaces(self):
        return [
            Interface(
                position=(0, 0, 0),
                rotation=(0, 0, 0),
                diameter=self.max_width,
                max_angle=self.max_angle,
            )
        ]

    def drill(self):
        return _bounding_box_cutout(
            self.mesh,
            tolerance=dim(5, "mm"),
            min_offset=(0, 0, dim(-0.5, "mm")),
            max_offset=(0, 0, dim(0.5, "mm")),
        )


def mirror_mount(label: str = "K1E Mirror Mount") -> Component:
    return Component(
        label=label,
        definition=mirror_mount_K1E(),
    )


def mirror(label: str = "Mirror") -> Component:
    return Component(
        label=label,
        definition=optics.Circular_Mirror(
            diameter=dim(1, "in"),
            thickness=dim(8, "mm"),
            mount_definition=mirror_mount_K1E(),
            part_number="New Focus 5102",
        ),
    )


def vbg(label: str = "Volume Bragg Grating") -> Component:
    return Component(
        label=label,
        definition=vbg_K1E(),
    )


def bs_10(label: str = "Vertical Cube 10 mm") -> Component:
    return Component(
        label=label,
        definition=Vertical_Cube_10mm(),
    )


def surface_adapter_cp13(label: str = "CP13 Surface Adapter") -> Component:
    return Component(
        label=label,
        definition=Surface_Adapter_CP13(),
    )


def polarimeter_adapter(label: str = "Polarimeter Adapter") -> Component:
    return Component(
        label=label,
        definition=Polarimeter_Adapter(),
    )


def fiberport(label: str = "Fiberport SM1ZA") -> Component:
    return Component(
        label=label,
        definition=fiberport_SM1ZA(),
    )

def fiberport_flipped(label: str = "Fiberport SM1ZA Flipped") -> Component:
    return Component(
        label=label,
        definition=fiberport_SM1ZA_flipped(),
    )


def fiberport_flipped(label: str = "Fiberport SM1ZA Flipped") -> Component:
    return Component(
        label=label,
        definition=fiberport_SM1ZA_flipped(),
    )


def lens_mount(label: str = "HPT1 Lens Mount") -> Component:
    return Component(
        label=label,
        definition=lens_mount_HPT1(),
    )


def lens_tube_sm1l03(label: str = "SM1L03 Lens Tube") -> Component:
    return Component(
        label=label,
        definition=lens_tube_SM1L03(),
    )


def fiber_paddle(label: str = "MPC320 Fiber Paddle") -> Component:
    return Component(
        label=label,
        definition=fiber_paddle_MPC320(),
    )


def rotation_stage(label: str = "ELL14 Rotation Stage") -> Component:
    return Component(
        label=label,
        definition=rotation_stage_ELL14(),
    )


def etalon_box(label: str = "Etalon Box") -> Component:
    return Component(
        label=label,
        definition=etalon_box_model(),
    )


def etalon_adapter(label: str = "Etalon Adapter") -> Component:
    return Component(
        label=label,
        definition=etalon_adapter_model(),
    )


def polarimeter(label: str = "PAX1000IR1 Polarimeter") -> Component:
    return Component(
        label=label,
        definition=polarimeter_PAX1000IR1(),
    )
