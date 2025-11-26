#version 450

layout(location = 0) in vec3 f_position;
layout(location = 1) in vec3 f_normal;  
layout(location = 2) in vec2 f_uv;

layout(location = 0) out vec4 final_color;

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
    vec3 albedo_color;
    float _pad0;
    vec3 specular_color;
    float shininess;
} material;

struct PointLight {
    vec3 position;
    float intensity;
    vec3 color;
    float _pad0; // padding
};

struct SpotLight {
    vec3 position;
    float _pad0; // padding
    vec3 direction;
    float _pad1; // padding
    vec3 color;
    float _pad2; // padding
    float inner_cutOff_cos; 
    float outer_cutOff_cos;
    float _pad3; // padding
    float _pad4; // padding
};

layout(std430, binding = 2) readonly buffer LightBuffer {
    vec3 camera_position; 
    float _p0; // padding
    vec3 ambient_color;
    float _p1; // padding
    vec3 directional_dir;
    float _p2; // padding
    vec3 directional_color;
    float _p3; // padding

    SpotLight spot_lights[10];
    uint spot_light_count;
    float _pad1;
    float _pad2;
    float _pad3;

    PointLight point_lights[10];
    uint point_light_count;
    float _pad4;
    float _pad5;
    float _pad6;
} lb;

vec3 calculate_blinn_phong(vec3 N, vec3 V, vec3 L, vec3 light_color, float attenuation, float shininess, vec3 albedo_color, vec3 specular_color) {
    float diff = max(dot(N, L), 0.0);
    vec3 H = normalize(V + L);
    float spec = pow(max(dot(N, H), 0.0), shininess);
    return light_color * attenuation * diff * (albedo_color + specular_color * spec);
}

void main() {
    vec3 N = normalize(f_normal);
    vec3 V = normalize(lb.camera_position - f_position);

    vec3 color = ambient_light_intensity * material.albedo_color;

    if (lb.point_light_count > 0) {
        for (uint i = 0; i < lb.point_light_count && i < 10; ++i) {
            PointLight light = lb.point_lights[i];

            vec3 light_vec = light.position - f_position;
            float dist = length(light_vec);
            vec3 L = normalize(light_vec);

            float attenuation = light.intensity / (dist * dist + 1);

            color += calculate_blinn_phong(N, V, L, light.color, attenuation, material.shininess, material.albedo_color, material.specular_color);
        }
    }

    if (lb.spot_light_count > 0) {
        for (uint i = 0; i < lb.spot_light_count && i < 10; ++i) {
            SpotLight light = lb.spot_lights[i];

            vec3 light_vec = light.position - f_position;
            float dist = length(light_vec);
            vec3 L = normalize(light_vec);

            float constant = 1.0;
            float linear = 0.14;
            float quadratic = 0.07;
            float attenuation = 1.0 / (constant + linear * dist + quadratic * dist * dist);

            float theta = dot(-L, normalize(light.direction));
            float inner_cos = light.inner_cutOff_cos;
            float outer_cos = light.outer_cutOff_cos;
            float epsilon = inner_cos - outer_cos;

            float spot_intensity = 0.0;
            if (theta > outer_cos) {
                if (epsilon > 0.0001) {
                    spot_intensity = clamp((theta - outer_cos) / epsilon, 0.0, 1.0);
                } else {
                    spot_intensity = (theta >= inner_cos) ? 1.0 : 0.0;
                }
            }

            float total_attenuation = attenuation * spot_intensity;

            color += calculate_blinn_phong(N, V, L, light.color, total_attenuation, material.shininess, material.albedo_color, material.specular_color);
        }
    }

    final_color = vec4(color, 1.0);
}