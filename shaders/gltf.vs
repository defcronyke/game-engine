#version 450

layout(set = 0, binding = 0) uniform InstanceParams {
    mat4 world_to_framebuffer;
} u_instance_params;

layout(location = 0) in vec3 i_position;
layout(location = 1) in vec3 i_normal;
layout(location = 2) in vec4 i_tangent;
layout(location = 3) in vec2 i_texcoord_0;
layout(location = 4) in vec2 i_texcoord_1;
layout(location = 5) in vec4 i_color_0;
layout(location = 6) in vec4 i_joints_0;
layout(location = 7) in vec4 i_weights_0;

layout(location = 0) out vec3 v_position;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec2 v_texcoord_0;
layout(location = 3) out vec2 v_texcoord_1;

void main() {
    v_position = i_position;
    v_normal = i_normal;
    v_texcoord_0 = i_texcoord_0;
    v_texcoord_1 = i_texcoord_1;

    gl_Position = u_instance_params.world_to_framebuffer * vec4(i_position, 1.0);
}
