
from glob import glob
import numpy as np
import os
import bpy
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=10000, type=int)
    parser.add_argument('--out_dir', default='./data/sphere')
    parser.add_argument('--env_dir', default=None)
    parser.add_argument('--camera_distance', default=4.0, type=float)
    parser.add_argument('--focal_length', default=50, type=float)
    parser.add_argument('--resolution', default=256, type=int)
    parser.add_argument('--n_rot', default=5+1+1, type=int)
    opts = parser.parse_args()
    os.makedirs(opts.out_dir, exist_ok=True)

    for item in bpy.data.objects:
        bpy.data.objects.remove(item)
    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat)
    for item in bpy.data.meshes:
        bpy.data.meshes.remove(item)
    for item in bpy.data.worlds:
        bpy.data.worlds.remove(item)

    cycles_preferences = bpy.context.preferences.addons['cycles'].preferences
    cycles_preferences.refresh_devices()
    cuda_devices = cycles_preferences.devices
    cycles_preferences.compute_device_type = 'OPTIX'
    bpy.context.preferences.view.render_display_type = 'NONE'

    for i in range(len(cuda_devices)):
        cuda_devices[i].use = True
    

    # Camera settings
    bpy.ops.object.camera_add()
    bpy.context.scene.camera = bpy.context.object
    camera = bpy.data.objects['Camera']
    camera.data.type = 'PERSP'
    bpy.context.object.data.lens = opts.focal_length
    camera.location = (0,-opts.camera_distance,0)
    camera_rotation_euler_z = 0
    camera.rotation_euler = (np.pi/2,0,camera_rotation_euler_z)



    # Generate a sphere
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
    bpy.context.active_object.name = 'sphere'
    bpy.data.objects['sphere'].scale = (1.5,1.5,1.5)
    mat = bpy.data.materials.new(name="Shader")
    bpy.data.objects['sphere'].active_material = mat
    mat.use_nodes = True
    node_tree_sphere = mat.node_tree
    mat_output = node_tree_sphere.nodes.get('Material Output')
    bpy.data.objects['sphere'].active_material = mat
    bpy.data.objects['sphere'].modifiers.new("subsurf", type='SUBSURF')
    bpy.data.objects['sphere'].modifiers["subsurf"].levels = 6
    bpy.data.objects['sphere'].modifiers["subsurf"].render_levels = 6
    bpy.ops.object.modifier_apply(modifier="subsurf")
    bpy.ops.object.editmode_toggle()
    bpy.ops.transform.tosphere(value=1, mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
    bpy.ops.object.editmode_toggle()
    bpy.ops.object.shade_smooth()
    bpy.data.objects['sphere'].hide_render = True



    # Create a sphere to cast shadows
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
    bpy.context.active_object.name = 'sphere_for_shadow'
    bpy.data.objects['sphere_for_shadow'].scale = (0.6,0.6,0.6)
    mat = bpy.data.materials.new(name="Shader")
    bpy.data.objects['sphere_for_shadow'].active_material = mat
    mat.use_nodes = True
    node_tree_sphere = mat.node_tree
    mat_output = node_tree_sphere.nodes.get('Material Output')
    bpy.data.objects['sphere_for_shadow'].modifiers.new("subsurf", type='SUBSURF')
    bpy.data.objects['sphere_for_shadow'].modifiers["subsurf"].levels = 6
    bpy.data.objects['sphere_for_shadow'].modifiers["subsurf"].render_levels = 6
    bpy.ops.object.modifier_apply(modifier="subsurf")
    bpy.ops.object.editmode_toggle()
    bpy.ops.transform.tosphere(value=1, mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
    bpy.ops.object.editmode_toggle()
    bpy.ops.object.shade_smooth()


    # Create a plane to cast shadows
    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD')
    bpy.context.active_object.name = 'plane_for_shadow'
    bpy.data.objects['plane_for_shadow'].location = (0,0,0)
    bpy.data.objects['plane_for_shadow'].scale = (1.441,1.441,1.441)
    bpy.data.objects['plane_for_shadow'].rotation_euler[0] = np.pi/2

    # Join the sphere and the plane
    bpy.data.objects['sphere'].select_set(False)
    bpy.data.objects['sphere_for_shadow'].select_set(True)
    bpy.data.objects['plane_for_shadow'].select_set(True)
    bpy.ops.object.join()
    bpy.context.active_object.name = 'sphere_w_plane'
    bpy.data.objects['sphere_w_plane'].hide_render = True


    # Render settings
    bpy.context.scene.render.resolution_x = opts.resolution
    bpy.context.scene.render.resolution_y = opts.resolution
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.max_bounces = 0
    bpy.context.scene.cycles.diffuse_bounces = 0
    bpy.context.scene.cycles.glossy_bounces = 0
    bpy.context.scene.cycles.transmission_bounces = 0
    bpy.context.scene.cycles.volume_bounces = 0
    bpy.context.scene.cycles.transparent_max_bounces = 0
    bpy.context.scene.cycles.sample_clamp_direct = 0
    bpy.context.scene.cycles.sample_clamp_indirect = 0
    bpy.context.scene.display_settings.display_device = 'None'
    bpy.context.scene.sequencer_colorspace_settings.name = 'Linear'
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.use_denoising = False
    bpy.context.scene.render.film_transparent = True



    # Node settings for normal map
    mat_normal = bpy.data.materials.new(name="mat_normal")
    mat_normal.use_nodes = True
    node_tree = mat_normal.node_tree
    geometry = node_tree.nodes.new('ShaderNodeNewGeometry')
    vector_transform = node_tree.nodes.new('ShaderNodeVectorTransform')
    vector_transform.vector_type = 'VECTOR'
    vector_transform.convert_from = 'WORLD'
    vector_transform.convert_to = 'CAMERA'
    multiply = node_tree.nodes.new('ShaderNodeVectorMath')
    multiply.operation = 'MULTIPLY'
    multiply.inputs[1].default_value = [0.5,0.5,-1]
    add = node_tree.nodes.new('ShaderNodeVectorMath')
    add.operation = 'ADD'
    add.inputs[1].default_value = [0.5,0.5,0]
    node_tree.links.new(geometry.outputs[1], vector_transform.inputs[0])
    node_tree.links.new(vector_transform.outputs[0], multiply.inputs[0])
    node_tree.links.new(multiply.outputs[0], add.inputs[0])
    node_tree.links.new(add.outputs[0], node_tree.nodes['Material Output'].inputs['Surface'])
    bpy.data.objects['sphere_w_plane'].active_material = mat_normal
    bpy.data.objects['sphere'].active_material = mat_normal

    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGB'
    bpy.context.scene.render.image_settings.color_depth = '16'
    bpy.context.scene.cycles.samples = 1024
    bpy.context.scene.cycles.filter_width = 1.5
    
    bpy.data.objects['sphere'].hide_render = False
    bpy.ops.render.render(write_still=True)
    bpy.data.images['Render Result'].save_render(filepath = '%s/normal_sphere.png' % (opts.out_dir)) 
    bpy.data.objects['sphere'].hide_render = True

    bpy.data.objects['sphere_w_plane'].hide_render = False
    bpy.ops.render.render(write_still=True)
    bpy.data.images['Render Result'].save_render(filepath = '%s/normal_sphere_and_plane.png' % (opts.out_dir)) 
    bpy.data.objects['sphere_w_plane'].hide_render = True




    # Node settings for depth map
    mat_depth = bpy.data.materials.new('mat_depth')
    mat_depth.use_nodes = True
    node_tree = mat_depth.node_tree
    camera_data = node_tree.nodes.new('ShaderNodeCameraData')
    node_tree.links.new(camera_data.outputs[1], node_tree.nodes['Material Output'].inputs['Surface'])
    bpy.data.objects['sphere_w_plane'].active_material = mat_depth

    bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
    bpy.context.scene.render.image_settings.color_mode = 'RGB'
    bpy.context.scene.render.image_settings.color_depth = '32'
    bpy.context.scene.cycles.samples = 1
    bpy.context.scene.cycles.filter_width = 0.01
    bpy.data.objects['sphere_w_plane'].hide_render = False
    bpy.ops.render.render(write_still=True)
    bpy.data.images['Render Result'].save_render(filepath = '%s/depth_sphere_and_plane.exr' % (opts.out_dir)) 
    bpy.data.objects['sphere_w_plane'].hide_render = True




    # Node settings for mask
    mat_mask = bpy.data.materials.new('mat_mask')
    mat_mask.use_nodes = True
    node_tree = mat_mask.node_tree
    value = node_tree.nodes.new('ShaderNodeValue')
    value.outputs[0].default_value = 1
    node_tree.links.new(value.outputs['Value'], node_tree.nodes['Material Output'].inputs['Surface'])
    bpy.data.objects['sphere'].active_material = mat_mask

    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'BW'
    bpy.context.scene.render.image_settings.color_depth = '16'
    bpy.context.scene.cycles.samples = 1
    bpy.context.scene.cycles.filter_width = 0.01
    bpy.data.objects['sphere'].hide_render = False
    bpy.ops.render.render(write_still=True)
    bpy.data.images['Render Result'].save_render(filepath = '%s/mask_sphere.png' % (opts.out_dir)) 
    bpy.data.objects['sphere'].hide_render = True




    # Node settings for diffuse shading
    mat_diffuse = bpy.data.materials.new(name="mat_diffuse")
    mat_diffuse.use_nodes = True
    node_tree = mat_diffuse.node_tree
    bsdfdiffuse = node_tree.nodes.new('ShaderNodeBsdfDiffuse')
    bsdfdiffuse.inputs[0].default_value = [1,1,1,1]
    bsdfdiffuse.inputs[1].default_value = 0
    node_tree.links.new(bsdfdiffuse.outputs[0], node_tree.nodes['Material Output'].inputs['Surface'])
    bpy.data.objects['sphere_w_plane'].active_material = mat_diffuse




    # Node settings for specular shading
    mat_specular = bpy.data.materials.new(name="mat_specular")
    mat_specular.use_nodes = True
    node_tree = mat_specular.node_tree
    bsdfspecular = node_tree.nodes.new('ShaderNodeBsdfPrincipled')
    bsdfspecular.inputs['Base Color'].default_value = [0,0,0,1]
    bsdfspecular.inputs['Specular'].default_value = 0.5
    bsdfspecular.inputs['Roughness'].default_value = 0.5
    node_tree.links.new(bsdfspecular.outputs[0], node_tree.nodes['Material Output'].inputs['Surface'])




    # Node settings for environment map
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    node_tree = world.node_tree
    for node in node_tree.nodes:
        node_tree.nodes.remove(node)
    texcoord = node_tree.nodes.new("ShaderNodeTexCoord")
    mapping = node_tree.nodes.new("ShaderNodeMapping")
    texenv = node_tree.nodes.new("ShaderNodeTexEnvironment")
    background_env = node_tree.nodes.new("ShaderNodeBackground")
    world_output = node_tree.nodes.new("ShaderNodeOutputWorld")
    node_tree.links.new(texcoord.outputs[0], mapping.inputs[0])
    node_tree.links.new(mapping.outputs[0], texenv.inputs[0])
    node_tree.links.new(texenv.outputs[0], background_env.inputs[0])
    node_tree.links.new(background_env.outputs[0], world_output.inputs[0])



    bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
    bpy.context.scene.render.image_settings.color_mode = 'RGB'
    bpy.context.scene.render.image_settings.color_depth = '32'
    bpy.context.scene.cycles.samples = 256
    bpy.context.scene.cycles.filter_width = 1.5

    env_paths = sorted(glob('%s/*/*.hdr' % (opts.env_dir)))[opts.start:opts.end]
    for env_path in env_paths:
        env_name = os.path.basename(env_path)[:-len('.hdr')]
        out_dir = '%s/%s' % (opts.out_dir,env_name)
        os.makedirs(out_dir,exist_ok=True)

        # load environment map
        texenv.image = bpy.data.images.load(env_path) ###

        for id in ['sphere','sphere_w_plane']:
            if id == 'sphere':
                bpy.data.objects['sphere_w_plane'].hide_render = True
                bpy.data.objects['sphere'].hide_render = False
            elif id == 'sphere_w_plane':
                bpy.data.objects['sphere_w_plane'].hide_render = False
                bpy.data.objects['sphere'].hide_render = True

            for i_rot in range(opts.n_rot):
                # rotate the environment map
                if i_rot < (opts.n_rot-2):
                    theta_rot = (i_rot/(opts.n_rot-2))*2*np.pi
                    mapping.inputs[2].default_value = (0,0,theta_rot)
                elif i_rot == (opts.n_rot-2): #up
                    theta_rot = -np.pi/2
                    mapping.inputs[2].default_value = (theta_rot,0,0)
                elif i_rot == (opts.n_rot-1): #bottom
                    theta_rot = np.pi/2
                    mapping.inputs[2].default_value = (theta_rot,0,0)
            

                
                if id == 'sphere':
                    # render diffuse shading
                    bpy.data.objects['sphere'].active_material = mat_diffuse
                    bpy.data.objects['sphere'].visible_shadow = False
                    bpy.data.objects['sphere'].visible_diffuse = False
                    bpy.data.objects['sphere'].visible_glossy = False
                    bpy.data.objects['sphere'].visible_transmission = False
                    bpy.data.objects['sphere'].visible_volume_scatter = False
                    bpy.ops.render.render(write_still=True)
                    if i_rot < (opts.n_rot-2): 
                        bpy.data.images['Render Result'].save_render(filepath = '%s/diffuse_shading_wo_shadow_sphere__%.4f.exr' % (out_dir,theta_rot)) 
                    elif i_rot == (opts.n_rot-2): #up
                        bpy.data.images['Render Result'].save_render(filepath = '%s/diffuse_shading_wo_shadow_sphere__up.exr' % (out_dir)) 
                    elif i_rot == (opts.n_rot-1): #bottom
                        bpy.data.images['Render Result'].save_render(filepath = '%s/diffuse_shading_wo_shadow_sphere__bottom.exr' % (out_dir)) 

                    # render specular shading
                    bpy.data.objects['sphere'].active_material = mat_specular
                    bpy.data.objects['sphere'].visible_shadow = False
                    bpy.data.objects['sphere'].visible_diffuse = False
                    bpy.data.objects['sphere'].visible_glossy = False
                    bpy.data.objects['sphere'].visible_transmission = False
                    bpy.data.objects['sphere'].visible_volume_scatter = False
                    bpy.ops.render.render(write_still=True)
                    if i_rot < (opts.n_rot-2): 
                        bpy.data.images['Render Result'].save_render(filepath = '%s/specular_shading_wo_shadow_sphere__%.4f.exr' % (out_dir,theta_rot)) 
                    elif i_rot == (opts.n_rot-2): #up
                        bpy.data.images['Render Result'].save_render(filepath = '%s/specular_shading_wo_shadow_sphere__up.exr' % (out_dir)) 
                    elif i_rot == (opts.n_rot-1): #bottom
                        bpy.data.images['Render Result'].save_render(filepath = '%s/specular_shading_wo_shadow_sphere__bottom.exr' % (out_dir)) 


                elif id == 'sphere_w_plane':
                    # render diffuse shading without shadow
                    bpy.data.objects['sphere_w_plane'].visible_shadow = False
                    bpy.data.objects['sphere_w_plane'].visible_diffuse = False
                    bpy.data.objects['sphere_w_plane'].visible_glossy = False
                    bpy.data.objects['sphere_w_plane'].visible_transmission = False
                    bpy.data.objects['sphere_w_plane'].visible_volume_scatter = False
                    bpy.ops.render.render(write_still=True)
                    if i_rot < (opts.n_rot-2): 
                        bpy.data.images['Render Result'].save_render(filepath = '%s/diffuse_shading_wo_shadow_sphere_and_plane__%.4f.exr' % (out_dir,theta_rot)) 
                    elif i_rot == (opts.n_rot-2): #up
                        bpy.data.images['Render Result'].save_render(filepath = '%s/diffuse_shading_wo_shadow_sphere_and_plane__up.exr' % (out_dir)) 
                    elif i_rot == (opts.n_rot-1): #bottom
                        bpy.data.images['Render Result'].save_render(filepath = '%s/diffuse_shading_wo_shadow_sphere_and_plane__bottom.exr' % (out_dir)) 

                    # render diffuse shading with shadow
                    bpy.data.objects['sphere_w_plane'].visible_shadow = True
                    bpy.data.objects['sphere_w_plane'].visible_diffuse = True
                    bpy.data.objects['sphere_w_plane'].visible_glossy = True
                    bpy.data.objects['sphere_w_plane'].visible_transmission = True
                    bpy.data.objects['sphere_w_plane'].visible_volume_scatter = True
                    bpy.ops.render.render(write_still=True)
                    if i_rot < (opts.n_rot-2):
                        bpy.data.images['Render Result'].save_render(filepath = '%s/diffuse_shading_w_shadow_sphere_and_plane__%.4f.exr' % (out_dir,theta_rot)) 
                    elif i_rot == (opts.n_rot-2): #up
                        bpy.data.images['Render Result'].save_render(filepath = '%s/diffuse_shading_w_shadow_sphere_and_plane__up.exr' % (out_dir)) 
                    elif i_rot == (opts.n_rot-1): #bottom
                        bpy.data.images['Render Result'].save_render(filepath = '%s/diffuse_shading_w_shadow_sphere_and_plane__bottom.exr' % (out_dir)) 

if __name__ == '__main__':
    main()