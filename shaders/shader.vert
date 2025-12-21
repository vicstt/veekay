#version 450

layout(location = 0) in vec3 v_position;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_uv;

layout(location = 0) out vec3 f_position;
layout(location = 1) out vec3 f_normal;
layout(location = 2) out vec2 f_uv;
layout(location = 3) out vec4 f_shadow_position;

layout(binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    mat4 shadow_projection;
    vec3 view_position;
    float _pad0;
    float time;
    float _pad5[3];
} scene;

layout(binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;   float _pad0;
    vec3 specular_color; float _pad1;
    float shininess;     vec3 _pad2;
} model;

void main() {
    vec4 world_pos = model.model * vec4(v_position, 1.0);
    
    gl_Position = scene.view_projection * world_pos;
    f_shadow_position = scene.shadow_projection * world_pos;
    
    f_position = world_pos.xyz;
    f_normal = mat3(model.model) * v_normal;
    f_uv = v_uv;
}