#version 450

layout(location=0) in vec3 f_position; 
layout(location=1) in vec3 f_normal;
layout(location=2) in vec2 f_uv;

layout(set=0, binding=1) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    float shininess;
    vec3 specular_color;
    float _pad;
} model;

layout(push_constant) uniform PushConstants {
    vec3 camera_position;
    float _p0;
    vec3 ambient_color;
    float _p1;
    vec3 directional_dir;
    float _p2;
    vec3 directional_color;
    float _p3;

    vec3 spot_pos;
    float _s_p0;
    vec3 spot_dir;
    float _s_p1;
    vec3 spot_col;
    float _s_p2;
    float inner_cutOff_cos; 
    float outer_cutOff_cos; 
    float _s_p3;
    float _s_p4;
} pc;

layout(location=0) out vec4 final_color;

vec3 calculate_blinn_phong(vec3 N, vec3 V, vec3 L, vec3 light_color, float attenuation) {
    float diff = max(dot(N, L), 0.0); 
    vec3 H = normalize(V + L); 
    float spec = pow(max(dot(N, H), 0.0), model.shininess);
    return light_color * attenuation * (model.albedo_color * diff + model.specular_color * spec);
}

vec3 calculate_spot_light(vec3 N, vec3 V) {
    vec3 light_vec = pc.spot_pos - f_position;
    float dist = length(light_vec);
    
    if (dist == 0.0) return vec3(0.0);
    
    
    vec3 L = light_vec / dist;
    
    vec3 D = pc.spot_dir; 
    
    float theta = dot(-L, D);
    
    float constant = 1.0;
    float linear = 0.14; 
    float quadratic = 0.07;
    float attenuation = 1.0 / (constant + linear * dist + quadratic * dist * dist);
    
    float inner_cos = pc.inner_cutOff_cos;
    float outer_cos = pc.outer_cutOff_cos;
    float epsilon = inner_cos - outer_cos; 
    
    float intensity = 0.0;
    
    if (theta > outer_cos) {
        if (epsilon > 0.0001) {
            intensity = clamp((theta - outer_cos) / epsilon, 0.0, 1.0);
        } else {
            intensity = (theta >= inner_cos) ? 1.0 : 0.0;
        }
    }

    return calculate_blinn_phong(N, V, L, pc.spot_col, attenuation * intensity);
}

void main() {
    vec3 N = normalize(f_normal);
    vec3 V = normalize(pc.camera_position - f_position);
    vec3 color = pc.ambient_color * model.albedo_color;


    vec3 dir_light_dir = normalize(-pc.directional_dir);
    color += calculate_blinn_phong(N, V, dir_light_dir, pc.directional_color, 1.0); 

    color += calculate_spot_light(N, V); 
    final_color = vec4(color, 1.0);
}
