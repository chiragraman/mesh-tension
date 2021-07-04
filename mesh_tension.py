"""
Compute tension at mesh vertices and store in vertex colours and vertex groups.

"""
bl_info = {
    "name": "Mesh Tension",
    "author": "Chirag Raman, based on Steve Miller's original implementation",
    "version": (1, 5, 3),
    "blender": (2, 93, 0),
    "location": "Properties > Mesh Data",
    "description": "Store mesh tension in vertex colours and vertex groups",
    "category": "Mesh",
}

import logging
from functools import reduce
from typing import Callable, Mapping, Sequence, Set, Tuple

import bpy
import bmesh
from bmesh.types import BMesh, BMVert
from bpy.app.handlers import persistent
from bpy.props import CollectionProperty
from bpy.types import (
    Context, Depsgraph, LoopColors, Mesh, Modifier, Object, Operator, Panel,
    PropertyGroup, Scene, VertexGroups
)
from idprop.types import IDPropertyArray


## Module constants

VERTEX_COLORS_LAYER_NAME = "tension_map"
VERTEX_GROUP_COMPRESS_NAME = "tension_compress"
VERTEX_GROUP_STRETCH_NAME = "tension_stretch"
TENSION_MASK_NAME = "tension_mask"
TENSION_PROPAGATION_THRESHOLD = 1e-3
GENERATIVE_MODIFIERS = [
    "ARRAY", "BEVEL", "BOOLEAN", "BUILD", "DECIMATE", "EDGE_SPLIT", "MASK",
    "MIRROR", "MULTIRES", "REMESH", "SCREW", "SKIN", "SOLIDIFY", "SUBSURF",
    "TRIANGULATE", "WELD", "WIREFRAME"
]


## Module variables

logger = logging.getLogger(__name__)
pristine_meshes = dict()
rendering = False
skip = False
skip_depsgraph_pre = False
skip_depsgraph_post = False


## Type Aliases

VertexTension = Mapping[BMVert, float]


## Property Callbacks

def call_refresh_mask(self, context: Context) -> None:
    """ Refresh the tension mask for the object """
    refresh_mask(context.object)


def call_refresh_scene_tension_objects(self, context: Context) -> None:
    """ Refresh the list of objects in the scene that need tension computation """
    refresh_scene_tension_objects(context.scene)


## Class definitions

class TensionItem(PropertyGroup):
    """ Encapsulate an item that is relvant for tension computation.

    The item is meant to be stored in a collection property. Current use-cases
    are to refer to an object in a scene requiring tension computation or a
    modifier that needs to be disabled for tension computation.

    """
    name: bpy.props.StringProperty(name="Name")
    viewport: bpy.props.BoolProperty(name="Show Viewport")
    render: bpy.props.BoolProperty(name="Show Render")


class TensionMeshProps(PropertyGroup):
    """ Encapsulate mesh tension properties """
    enabled: bpy.props.BoolProperty(
        name = "Enable", default=False, update=call_refresh_scene_tension_objects
    )
    strength: bpy.props.FloatProperty(name = "Strength", default=1.0)
    bias: bpy.props.FloatProperty(name = "Bias", default=0.0)
    stretch_iterations: bpy.props.IntProperty(
        name = "Stretch Propagation Iterations",
        default = 0, soft_min=-4, soft_max=4
    )
    compress_iterations: bpy.props.IntProperty(
        name = "Compression Propagation Iterations",
        default = 0, soft_min=-4, soft_max=4
    )
    mask: bpy.props.StringProperty(
        name = "Vertex Mask", update=call_refresh_mask
    )
    suspended_modifiers: bpy.props.CollectionProperty(type=TensionItem)
    always_update: bpy.props.BoolProperty(
        name = "Always Update", default = False,
        description=("Update even when animation not playing "
                     "(may impact general viewport performance).")
    )


class TensionSceneProps(PropertyGroup):
    """ Encapsulate scene tension properties """
    enabled: bpy.props.BoolProperty(
        name = "Enable", default=True,
        update=call_refresh_scene_tension_objects
    )
    objects: bpy.props.CollectionProperty(type=TensionItem)


class MaskRefreshOperator(Operator):
    """ Refresh the tension mask """
    bl_idname="id.refresh_mask"
    bl_label="Refresh Mask"

    @classmethod
    def poll(cls, context: Context) -> bool:
        """ Test if the operator can be called or not """
        return True

    def execute(self, context : Context) -> Set:
        """ Perform the mask refreshing """
        refresh_mask(context.object)
        return {'FINISHED'}


## UI panels

class TensionMeshPanel(Panel):
    """ Encapsulate the panel for the mesh tension properties """
    bl_label = "Tension Maps"
    bl_idname = "MESH_PT_tension"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "data"

    @classmethod
    def poll(cls, context: Context) -> bool:
        """ Check if the panel should be drawn """
        return context.object is not None and context.object.type == "MESH"

    def draw_header(self, context: Context) -> None:
        """ Draw UI elements into the panel’s header UI layout """
        if not context.scene.render.use_lock_interface:
            self.layout.enabled = False
        self.layout.prop(context.object.data.tension_props, "enabled", text="")

    def draw(self, context: Context) -> None:
        """ Draw UI elements into the panel UI layout """
        global generative_modifiers
        if context.scene.render.use_lock_interface:
            self.layout.use_property_split = True
            ob = context.object
            self.layout.prop(ob.data.tension_props, "strength")
            self.layout.prop(ob.data.tension_props, "bias")
            self.layout.prop(ob.data.tension_props, "stretch_iterations")
            self.layout.prop(ob.data.tension_props, "compress_iterations")
            row = self.layout.row()
            row.prop_search(ob.data.tension_props, "mask", ob, "vertex_groups")
            row.operator("id.refresh_mask", text="", icon="FILE_REFRESH")
            self.layout.prop(ob.data.tension_props, "always_update")
        else:
            self.layout.label(text="Enable 'Render > Lock Interface' to use")


class TensionScenePanel(Panel):
    """ Encapsulate the panel for the scene tension properties

    This serves as a convenience for turning off tension at the scene level.

    """
    bl_label = "Tension Maps"
    bl_idname = "SCENE_PT_tension"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "scene"

    def draw_header(self, context: Context) -> None:
        """ Draw UI elements into the panel’s header UI layout """
        if not context.scene.render.use_lock_interface:
            self.layout.enabled = False
        self.layout.prop(context.scene.tension_props, "enabled", text="")

    def draw(self, context: Context) -> None:
        """ Draw UI elements into the panel UI layout """
        if not context.scene.render.use_lock_interface:
            self.layout.label(text="Enable 'Render > Lock Interface' to use")


## Utilities

def clear_pristine_meshes() -> None:
    """ Clear the pristine bmesh data """
    global pristine_meshes
    for bm in pristine_meshes.values():
        bm.free()
    pristine_meshes.clear()


def init_tension_bmesh(object: Object) -> BMesh:
    """ Initialize a bmesh for an object for which to compute tension """
    bm = bmesh.new()
    bm.from_mesh(object.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    return bm


def needs_suspension(modifier: Modifier) -> bool:
    """ Check if modifier needs to be suspended for tension computation """
    should_suspend = (
        (hasattr(modifier, "vertex_group")
         and modifier.vertex_group in [VERTEX_GROUP_COMPRESS_NAME, VERTEX_GROUP_STRETCH_NAME])
        or modifier.type in GENERATIVE_MODIFIERS
        or modifier.name[-11:].upper() == "SKIPTENSION"
    )
    return should_suspend


def suspend_modifier(modifier: Modifier, suspended_modifiers: CollectionProperty):
    """ Disable modifiers prior to tension computation.

    Modifiers that are generative or dependant on tension vertex groups are
    disabled. Store their state for restoring
    """
    dm = suspended_modifiers.add()
    dm.name = modifier.name
    dm.viewport = modifier.show_viewport
    dm.render = modifier.show_render
    modifier.show_viewport = False
    modifier.show_render = False


def restore_modifiers(object: Object) -> None:
    """ Restore modifier state after tension computation """
    supended_modifiers = object.data.tension_props.suspended_modifiers
    for modifier in supended_modifiers:
        if modifier.name in object.modifiers:
            object.modifiers[modifier.name].show_viewport = modifier.viewport
            object.modifiers[modifier.name].show_render = modifier.render


def should_always_update(scene: Scene) -> bool:
    """ Check if interactive tension updates should be made """
    update_condition = (
        scene.render.use_lock_interface # safety measure (see "Note on altering data", app handlers bpy docs")
        and scene.tension_props.enabled # scene level flag
        and not rendering
        and not bpy.context.screen.is_animation_playing # Frame change causing updates already
    )
    return update_condition


def refresh_scene_tension_objects(scene: Scene) -> None:
    """ Refresh the list of objects in the scene for which to compute tension

    The objects are stored as a collection property of the scene.

    """
    global pristine_meshes
    clear_pristine_meshes()

    # Clear the objects in the scene property
    scene_tension_props = scene.tension_props
    scene_tension_props.objects.clear()

    # Iterate through the scene objects and setup or clear tension properties
    for ob in scene.objects:
        if ob.type == "MESH":
            if ob.data.tension_props.enabled:
                # Ignore duplicates TODO: potentially record duplicates too.
                is_duplicate = False
                for prop_obj in scene_tension_props.objects:
                    prop_obj = scene.objects[prop_obj.name]
                    if prop_obj.data.name == ob.data.name:
                        is_duplicate = True
                if is_duplicate:
                    continue

                # Add object to scene collection property
                new_ob = scene_tension_props.objects.add()
                new_ob.name = ob.name
                # Setup tension mask, vertex groups, and vertex colors
                setup_object_tension(ob)
                # Track the pristine bmesh to compute tension
                bm = init_tension_bmesh(ob)
                pristine_meshes[ob.name] = bm
            else:
                # Cleanup tension mask, vertex groups, and vertex colors
                clear_object_tension(ob)


def ensure_vertex_colors(
        object: Object, names: Sequence[str], overwrite: bool = False
    ) -> None:
    """ Create a vertex color layer for the object with the given name if it doesn't exist

    Args:
        object      --  The blender object for which to get vertex color layer
        name        --  The name of the vertex color layer
        overwrite   --  Overwrite existing vertext groups if True

    """
    for name in names:
        create = True
        if name in object.data.vertex_colors:
            # Handle existing vertex colors
            if overwrite:
                object.data.vertex_colors.remove(object.data.vertex_colors[name])
            else:
                create = False
                logger.debug(f"Found existing vertex colors {name}, and not overwriting.")
        # Create the new colors
        if create:
            object.data.vertex_colors.new(name=name)


def ensure_vertex_groups(
        object: Object, names: Sequence[str], overwrite: bool = False
    ) -> None:
    """ Create a vertex group with the given name for the object if it doesn't exist.

    Args:
        object      --  The blender object for which to get vertex group
        names       --  The names of the vertex groups to create
        overwrite   --  Overwrite existing vertext groups if True

    """
    for name in names:
        create = True
        if name in object.vertex_groups:
            # Handle existing vertex group
            if overwrite:
                object.vertex_groups.remove(object.vertex_groups[name])
            else:
                create = False
                logger.debug(f"Found existing vertex group {name}, and not overwriting.")
        # Create the new group if required
        if create:
            object.vertex_groups.new(name=name)


def refresh_mask(object: Object) -> None:
    """ Refresh the tension mask, vertex colors, and vertex groups """
    # Check if a custom vertex mask is provided for tension computations
    mask_index = object.vertex_groups.find(object.data.tension_props.mask)
    if mask_index == -1:
        # Compute tension for all vertices
        object.data[TENSION_MASK_NAME] = [v.index for v in object.data.vertices]
    else:
        # Compute tension for vertices in the mask
        object.data[TENSION_MASK_NAME] = [
            v.index
            for v in object.data.vertices
            if mask_index in [vg.group for vg in v.groups]
        ]


def refresh_tension_map(object: Object) -> None:
    """ Refresh the tension vertex colors """
    tension_map = object.data.vertex_colors.get(VERTEX_COLORS_LAYER_NAME)
    for index in tension_map.data:
        index.color = (0, 0, 0, 0)


def setup_object_tension(object: Object) -> None:
    """ Initialize the tension mask and vertex colors and groups """
    refresh_mask(object)
    ensure_vertex_groups(
        object, [VERTEX_GROUP_STRETCH_NAME, VERTEX_GROUP_COMPRESS_NAME]
    )
    ensure_vertex_colors(object, [VERTEX_COLORS_LAYER_NAME])
    refresh_tension_map(object)


def clear_object_tension(object: Object) -> None:
    """ Remove the tension mask and vertex colors and groups """
    # Tension mask
    if object.data.get(TENSION_MASK_NAME):
        del object.data[TENSION_MASK_NAME]

    # Vertex Groups
    stretch_group = object.vertex_groups.get(VERTEX_GROUP_STRETCH_NAME)
    if stretch_group:
        object.vertex_groups.remove(stretch_group)
    compress_group = object.vertex_groups.get(VERTEX_GROUP_COMPRESS_NAME)
    if compress_group:
        object.vertex_groups.remove(compress_group)

    # Vertex colors
    tension_map = object.data.vertex_colors.get(VERTEX_COLORS_LAYER_NAME)
    if tension_map:
        object.data.vertex_colors.remove(tension_map)

    # Clear list of suspended modifiers after restoring
    restore_modifiers(object)
    object.data.tension_props.suspended_modifiers.clear()


def prepare_tension_operations(object: Object) -> None:
    """ Perform operations required before tension computation.

    This involves suspending generative modifiers or those dependent on tension
    vertex groups.

    Args:
        object  --  The object for which to compute tension

    """
    suspended_modifiers = object.data.tension_props.suspended_modifiers
    suspended_modifiers.clear()
    for modifier in object.modifiers:
        if needs_suspension(modifier):
            suspend_modifier(modifier, suspended_modifiers)


def reducer(accumulator: VertexTension, element: VertexTension) -> VertexTension:
    """ Accumulate tension mappings by adding tension at common vertices """
    for key, value in element.items():
        accumulator[key] = accumulator.get(key, 0) + value
    return accumulator


def compute_tension(
        pristine: BMesh, evaluated: Mesh, vertex_mask: IDPropertyArray
    ) -> Tuple[VertexTension]:
    """ Compute tension at vertices from pristine and evluated meshes.

    Positive tension indicates compression, negative stretching.

    Args:
        pristine    --  The BMesh representing the pristine mesh
        evaluated   --  The mesh of the object evaluated from the depsgraph
        vertex_mask --  Mask indicating vertices that need tension computation

    Returns two dictionaries of vertices in the pristine bmesh and corresponding
    tension measures, one for stretching, and another for compression.

    """
    # Dictionaries for storing vertices and corresponding tenstion measures
    stretched = dict()
    compressed = dict()

    # Estimate initial tension before propagaion iterations
    for v_index in vertex_mask:
        pristine_v = pristine.verts[v_index]
        tension = 0
        if not pristine_v.link_edges:
            continue
        for edge in pristine_v.link_edges:
            eval_edge = evaluated.edges[edge.index]
            eval_edge_length = (
                evaluated.vertices[eval_edge.vertices[0]].co
                - evaluated.vertices[eval_edge.vertices[1]].co
            ).length
            edge_tension = eval_edge_length / edge.calc_length()
            tension += edge_tension
        tension /= len(pristine_v.link_edges) # normalize over edge count
        tension = (1 - tension) # 0 if unchanged, -ve if stretched, +ve if compressed

        # Store vertices with tension that needs to be propagated,
        # along with the corresponding tensions
        if abs(tension) > TENSION_PROPAGATION_THRESHOLD:
            mask = stretched if tension < 0 else compressed
            mask[pristine_v] = tension

    return stretched, compressed


def propagate_tension(
        vertex_tension: VertexTension, iterations: int,
        comparator: Callable[[float, float], float],
        vertex_mask: IDPropertyArray
    ) -> None:
    """ Propagate tension to adjacent vertices.

    The `vertex_tension` mapping contains updated vertices and tension measures.

    Args:
        vertex_tension  --  Mapping from vertex to tension measure
        iterations      --  Number of iterations to propagate tension
        comparator      --  Expected to be `min` or `max` depending on whether
                            tension is being eroded or dilated
        vertex_mask     --  Mask indicating vertices that need tension computation

    """
    for _ in range(iterations):
        previous_tension = vertex_tension.copy()
        for v, t in previous_tension.items():
            for e in v.link_edges:
                v2 = e.other_vert(v)
                if not v2.index in vertex_mask:
                    continue
                if not v2 in vertex_tension:
                    vertex_tension[v2] = 0
                vertex_tension[v2] = comparator(t, vertex_tension[v2])
        previous_tension.clear()


def set_tension_groups_and_map(
        object: Object, pristine_mesh: BMesh, vertex_tension: VertexTension
    ) -> None:
    """ Add vertices to tension vertex groups and update vertex colors

    The object is expected to be initialized for tension computation and
    possess the vertex groups and color layers.

    """
    # Get the vertex groups and vertex colors layer
    stretch_group = object.vertex_groups.get(VERTEX_GROUP_STRETCH_NAME)
    compress_group = object.vertex_groups.get(VERTEX_GROUP_COMPRESS_NAME)
    tension_map = object.data.vertex_colors.get(VERTEX_COLORS_LAYER_NAME)
    strength = object.data.tension_props.strength
    bias = object.data.tension_props.bias

    for v_index in object.data[TENSION_MASK_NAME]:
        tension = 0
        vertex = pristine_mesh.verts[v_index]
        if vertex in vertex_tension:
            tension = vertex_tension[vertex] * strength
        stretch_group.add([vertex.index], sorted((0, -tension + bias, 1))[1], "REPLACE")
        compress_group.add([vertex.index], sorted((0, tension - bias, 1))[1], "REPLACE")
        for loop in vertex.link_loops:
            if tension < 0:
                color = (0, -tension + bias, 0, 1.0) # stretched
            else:
                color =  (tension - bias, 0, 0, 1.0) # compressed
            tension_map.data[loop.index].color = color


def perform_tension_operations(object: Object, depsgraph: Depsgraph) -> None:
    """ Compute and propagate tension in the object mesh, and restore modifiers.

    The object is assumed to be configured for tension at this point, and
    the `pristine_meshes` dictionary is expected to contain the pristine mesh
    for the object.

    Args:
        object      --  The object for which to compute tension
        depsgraph   --  The dependency graph for accessing the evaluated object

    """
    global pristine_meshes
    # Get evaluated object and mesh from the depsgraph with modifiers and shape
    # keys applied
    evaluated_obj = object.evaluated_get(depsgraph)
    evaluated_mesh = evaluated_obj.data
    # The objects should be configured for tension computation by this point.
    pristine_bmesh = pristine_meshes.get(object.name)

    # Estimate initial tension at vertices
    stretched, compressed = compute_tension(pristine_bmesh, evaluated_mesh,
                                            object.data[TENSION_MASK_NAME])

    # Propagate stretch
    stretch_its = object.data.tension_props.stretch_iterations
    comparator = min if stretch_its > 0 else max
    propagate_tension(stretched, abs(stretch_its), comparator,
                      object.data[TENSION_MASK_NAME])

    # Propagate compression
    compress_its = object.data.tension_props.compress_iterations
    comparator = max if compress_its > 0 else min
    propagate_tension(compressed, abs(compress_its), comparator,
                      object.data[TENSION_MASK_NAME])

    # Merge stretch and compression mappings. Add tension values for vertices
    # that have both stretch and compression measures after tension propagation
    vertex_tension = reduce(reducer, [stretched, compressed], dict())

    # Set the tension vertex groups and vertex colors
    set_tension_groups_and_map(object, pristine_bmesh, vertex_tension)

    # Restore the modifiers that were suspended for tension computation
    restore_modifiers(object)


## Handlers

@persistent
def load_post(scene: Scene):
    """ Generate the collection of scene objects that need tension computation """
    refresh_scene_tension_objects(scene)


@persistent
def render_pre(scene: Scene) -> None:
    """ Track rendering state """
    global rendering
    rendering = True


@persistent
def render_post(scene: Scene) -> None:
    """ Track rendering state """
    global rendering
    rendering = False


def handle_common_pre(scene: Scene, check_always_update: bool = False) -> None:
    """ Refresh collection of scene tension objects and suspend modifiers """
    # Pre-emptively refresh tension objects
    refresh_scene_tension_objects(scene)
    for object in scene.tension_props.objects:
        ob = scene.objects.get(object.name)
        if ob.mode == "OBJECT":
            if not (check_always_update and not ob.data.tension_props.always_update):
                prepare_tension_operations(ob)


def handle_common_post(
        scene: Scene, depsgraph: Depsgraph, check_always_update: bool = False
    ) -> None:
    """ Compute tension for objects that need tension computation """
    refresh_scene_tension_objects(scene)
    for i, object in enumerate(scene.tension_props.objects):
        ob = scene.objects.get(object.name)
        if ob.mode == "OBJECT":
            if not (check_always_update and not ob.data.tension_props.always_update):
                perform_tension_operations(ob, depsgraph)


def should_handle_frame_change(scene: Scene) -> bool:
    """ Check if the frame change handlers should be executed """
    global skip
    skip_condition = (skip
                      or not scene.tension_props.enabled
                      or not scene.render.use_lock_interface)
    return skip_condition


@persistent
def frame_change_pre(scene: Scene):
    """ Handle frame change, pre data evaluation """
    # Check if handler should be skipped
    if not should_handle_frame_change(scene):
        return
    handle_common_pre(scene)


@persistent
def frame_change_post(scene: Scene, depsgraph: Depsgraph) -> None:
    """ Handle frame change, post data evaluation """
    global skip, rendering
    if not should_handle_frame_change(scene):
        return
    handle_common_post(scene, depsgraph)
    if rendering:
        # Rendering needs this update
        skip = True
        scene.frame_set(scene.frame_current)
        skip = False


def should_handle_depsgraph_update(scene: Scene) -> bool:
    """ Check if the depsgraph update handlers should be executed """
    global skip_depsgraph_pre, skip_depsgraph_post
    return (not (skip_depsgraph_pre or skip_depsgraph_post)
            and should_always_update(scene))


@persistent
def depsgraph_update_pre(scene: Scene) -> None:
    """ Handle depsgraph updates (pre) """
    global skip_depsgraph_pre
    if not should_handle_depsgraph_update(scene):
        return

    skip_depsgraph_pre = True # prevent recursion
    handle_common_pre(scene, check_always_update=True)
    skip_depsgraph_pre = False


@persistent
def depsgraph_update_post(scene: Scene, depsgraph: Depsgraph) -> None:
    """ Handle depsgraph updates (post) """
    global skip_depsgraph_post
    if not should_handle_depsgraph_update(scene):
        return

    skip_depsgraph_post = True
    handle_common_post(scene, depsgraph, check_always_update=True)
    skip_depsgraph_post = False


## Register, Unregister classes!

def register():
    """ Register classes and properties """
    # Classes
    bpy.utils.register_class(TensionItem)
    bpy.utils.register_class(TensionMeshProps)
    bpy.utils.register_class(TensionSceneProps)

    # Tension Props
    bpy.types.Mesh.tension_props = bpy.props.PointerProperty(type=TensionMeshProps)
    bpy.types.Scene.tension_props = bpy.props.PointerProperty(type=TensionSceneProps)

    # Operators and Panels
    bpy.utils.register_class(MaskRefreshOperator)
    bpy.utils.register_class(TensionMeshPanel)
    bpy.utils.register_class(TensionScenePanel)

    # App handlers
    bpy.app.handlers.load_post.append(load_post)
    bpy.app.handlers.frame_change_pre.append(frame_change_pre)
    bpy.app.handlers.frame_change_post.append(frame_change_post)
    bpy.app.handlers.render_pre.append(render_pre)
    bpy.app.handlers.render_post.append(render_post)
    bpy.app.handlers.depsgraph_update_pre.append(depsgraph_update_pre)
    bpy.app.handlers.depsgraph_update_post.append(depsgraph_update_post)


def unregister():
    """ Unregister classes and properties """
    # Classes
    bpy.utils.unregister_class(TensionItem)
    bpy.utils.unregister_class(TensionMeshProps)
    bpy.utils.unregister_class(TensionSceneProps)

    # Tension Props
    del bpy.types.Mesh.tension_props
    del bpy.types.Scene.tension_props

    # Operators and Panels
    bpy.utils.unregister_class(MaskRefreshOperator)
    bpy.utils.unregister_class(TensionMeshPanel)
    bpy.utils.unregister_class(TensionScenePanel)

    # App handlers
    bpy.app.handlers.load_post.remove(load_post)
    bpy.app.handlers.frame_change_pre.remove(frame_change_pre)
    bpy.app.handlers.frame_change_post.remove(frame_change_post)
    bpy.app.handlers.render_pre.remove(render_pre)
    bpy.app.handlers.render_post.remove(render_post)
    bpy.app.handlers.depsgraph_update_pre.remove(depsgraph_update_pre)
    bpy.app.handlers.depsgraph_update_post.remove(depsgraph_update_post)


# This allows you to run the script directly from Blender's Text editor
# to test the add-on without having to install it.
if __name__ == "__main__":
    register()
