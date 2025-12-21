#version 450

struct PointLight {
    vec3 position;
    float intensity;
    vec3 color;
    float _pad0;
};

struct SpotLight {
    vec3 position;
    float _pad0;
    vec3 direction;
    float _pad1;
    vec3 color;
    float _pad2;
    float inner_cutOff_cos;
    float outer_cutOff_cos;
    float _pad3;
    float _pad4;
};

struct LightData {
    vec3 camera_position;
    float _p0;
    vec3 ambient_color;
    float _p1;
    vec3 directional_dir;
    float _p2;
    vec3 directional_color;
    float _p3;

    SpotLight spot_lights[16];
    uint spot_light_count;
    float _pad1;
    float _pad2;
    float _pad3;

    PointLight point_lights[16];
    uint point_light_count;
    float _pad4;
    float _pad5;
    float _pad6;
};

layout(location = 0) in vec3 f_position;
layout(location = 1) in vec3 f_normal;
layout(location = 2) in vec2 f_uv;
layout(location = 3) in vec4 f_shadow_position;

layout(location = 0) out vec4 final_color;

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

layout(binding = 2, std430) readonly buffer Lights {
    LightData light_data;
};

layout(binding = 4) uniform sampler2D albedo_texture;
layout(binding = 5) uniform sampler2D specular_texture;
layout(binding = 6) uniform sampler2D emissive_texture;
layout(binding = 7) uniform sampler2DShadow shadow_texture;

float noise(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

vec3 calculate_blinn_phong(vec3 N, vec3 V, vec3 L, vec3 light_color, float attenuation, vec4 albedo_texel, vec4 spec_texel) {
    float diff = max(dot(N, L), 0.0);
    vec3 H = normalize(V + L);
    float spec = pow(max(dot(N, H), 0.0), model.shininess);
    return light_color * attenuation * (diff * albedo_texel.xyz * model.albedo_color + spec * model.specular_color * spec_texel.xyz);
}

void main() {
    vec3 N = normalize(f_normal);
    vec3 V = normalize(light_data.camera_position - f_position);

    float time_factor = scene.time * 2.0;
    float noise_factor = noise(f_position.xz * 0.1); 
    vec2 animated_uv = f_uv + vec2(
        sin(time_factor + f_position.x * 2.0) * 0.01 + noise_factor * 0.005,
        cos(time_factor + f_position.z * 2.0) * 0.01 + noise_factor * 0.005
    );

    vec4 albedo_texel = texture(albedo_texture, animated_uv);
    vec4 spec_texel = texture(specular_texture, animated_uv);
    vec4 emissive_texel = texture(emissive_texture, animated_uv);

    vec3 shadow_coord = f_shadow_position.xyz / f_shadow_position.w;
    shadow_coord.xy = shadow_coord.xy * 0.5 + 0.5;
    shadow_coord.z = shadow_coord.z;
    float shadow_factor = texture(shadow_texture, shadow_coord);

    vec3 color = albedo_texel.xyz * model.albedo_color * light_data.ambient_color;

    // Направленный свет с тенями
    vec3 L_dir = normalize(-light_data.directional_dir);
    float diff_dir = max(dot(N, L_dir), 0.0);
    vec3 H_dir = normalize(V + L_dir);
    float spec_dir = pow(max(dot(N, H_dir), 0.0), model.shininess);
    
    vec3 dir_light_diffuse = light_data.directional_color * shadow_factor * diff_dir * albedo_texel.xyz * model.albedo_color;
    vec3 dir_light_specular = light_data.directional_color * shadow_factor * spec_dir * model.specular_color * spec_texel.xyz;
    color += dir_light_diffuse + dir_light_specular;

    // Точечные источники
    for (uint i = 0u; i < light_data.point_light_count; i++) {
        PointLight light = light_data.point_lights[i];
        vec3 light_vec = light.position - f_position;
        float dist = length(light_vec);
        
        vec3 L = normalize(light_vec);
        float ndotl = dot(N, L);
        if (ndotl <= 0.0) continue;
        
        float attenuation = light.intensity / (dist * dist + 1.0);
        
        color += calculate_blinn_phong(N, V, L, light.color, attenuation, 
                                    albedo_texel, spec_texel);
    }

    // Прожекторы 
    for (uint i = 0u; i < light_data.spot_light_count; i++) {
        SpotLight light = light_data.spot_lights[i];
        
        vec3 light_vec = light.position - f_position;
        float dist = length(light_vec);
        vec3 L = normalize(light_vec);
        
        float constant = 1.0;
        float linear = 0.09;
        float quadratic = 0.032;
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
        
        if (spot_intensity > 0.0) {
            float ndotl = dot(N, L);
            if (ndotl > 0.0) {
                color += calculate_blinn_phong(N, V, L, light.color, total_attenuation, 
                                            albedo_texel, spec_texel);
            }
        }
    }

    // Emissive цвет
    color += emissive_texel.xyz;
    
    final_color = vec4(color, 1.0);
}