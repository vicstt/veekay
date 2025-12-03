#version 450

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv;

layout (location = 0) out vec3 f_position; 
layout (location = 1) out vec3 f_normal;
layout (location = 2) out vec2 f_uv; 

layout (binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    vec3 view_position; 
    float _pad0;
    vec3 ambient_light_intensity;
    float _pad1;
    vec3 sun_light_direction;
    float _pad2;
    vec3 sun_light_color;
    float _pad3;
    uint point_light_count; 
    float time; 
    float _pad4;
    float _pad5;
    float _pad6;
};

layout (binding = 1, std140) uniform ModelUniforms {
    mat4 model;
} model;

void main() {
    vec4 worldPos = model.model * vec4(v_position, 1.0);

    mat3 normal_matrix = transpose(inverse(mat3(model.model)));
    vec3 worldNormal = normalize(normal_matrix * v_normal);

    f_position = worldPos.xyz;
    f_normal = worldNormal;
    f_uv = v_uv; 

    gl_Position = view_projection * worldPos;
}