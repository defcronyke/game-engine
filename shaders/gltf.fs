#version 450

layout(set = 1, binding = 0) uniform MaterialParams {
    vec4 base_color_factor;
    int base_color_texture_tex_coord;
    float metallic_factor;
    float roughness_factor;
    int metallic_roughness_texture_tex_coord;
    float normal_texture_scale;
    int normal_texture_tex_coord;
    int occlusion_texture_tex_coord;
    float occlusion_texture_strength;
    int emissive_texture_tex_coord;
    vec3 emissive_factor;
} u_material_params;

layout(set = 1, binding = 1) uniform sampler2D u_base_color;
layout(set = 1, binding = 2) uniform sampler2D u_metallic_roughness;
layout(set = 1, binding = 3) uniform sampler2D u_normal_texture;
layout(set = 1, binding = 4) uniform sampler2D u_occlusion_texture;
layout(set = 1, binding = 5) uniform sampler2D u_emissive_texture;

layout(location = 0) in vec3 v_position;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_texcoord_0;
layout(location = 3) in vec2 v_texcoord_1;

layout(location = 0) out vec4 f_color;

const float M_PI = 3.141592653589793;

float SmithG1_var2(float n_dot_v, float r) {
    float tanSquared = (1.0 - n_dot_v * n_dot_v) / max((n_dot_v * n_dot_v), 0.00001);
    return 2.0 / (1.0 + sqrt(1.0 + r * r * tanSquared));
}

void main() {
    // Load the metallic and roughness properties values.
    float metallic = 1.0;
    float perceptual_roughness = 1.0;
    if (u_material_params.base_color_texture_tex_coord == 0) {
        vec2 v = texture(u_metallic_roughness, v_texcoord_0).rg;
        metallic = v.r;
        perceptual_roughness = v.g;
    } else if (u_material_params.base_color_texture_tex_coord == 1) {
        vec2 v = texture(u_metallic_roughness, v_texcoord_1).rg;
        metallic = v.r;
        perceptual_roughness = v.g;
    }
    metallic *= u_material_params.metallic_factor;
    perceptual_roughness *= u_material_params.roughness_factor;

    // Load the base color of the material.
    vec4 base_color = vec4(0.0);
    if (u_material_params.base_color_texture_tex_coord == 0) {
        base_color.rgb = texture(u_base_color, v_texcoord_0).rgb;
    } else if (u_material_params.base_color_texture_tex_coord == 1) {
        base_color.rgb = texture(u_base_color, v_texcoord_1).rgb;
    }
    base_color *= u_material_params.base_color_factor;


    // TODO: temp ; move to uniform buffer
    vec3 u_LightColor = vec3(1.0);
    vec3 u_Camera = vec3(0.0, 0.0, 300.0);
    vec3 u_LightDirection = vec3(-0.4, 0.7, 0.2);


    // Complex maths here.

    vec3 n = v_normal;      // TODO:

    vec3 v = normalize(u_Camera - v_position);
    vec3 l = normalize(u_LightDirection);
    vec3 h = normalize(l + v);
    //vec3 reflection = -normalize(reflect(v, n));

    float n_dot_l = clamp(dot(n, l), 0.001, 1.0);
    float n_dot_v = abs(dot(n, v)) + 0.001;
    float n_dot_h = clamp(dot(n, h), 0.0, 1.0);
    float l_dot_h = clamp(dot(l, h), 0.0, 1.0);
    float v_dot_h = clamp(dot(v, h), 0.0, 1.0);

    vec3 diffuse_color = mix(base_color.rgb * (1 - 0.04), vec3(0.0), metallic);
    vec3 specular_color = mix(vec3(0.04), base_color.rgb, metallic);

    float reflectance = max(max(specular_color.r, specular_color.g), specular_color.b);
    vec3 specular_environment_r90 = vec3(1.0, 1.0, 1.0) * clamp(reflectance * 25.0, 0.0, 1.0);
    float alpha_roughness = perceptual_roughness * perceptual_roughness;

    vec3 fresnel_schlick_2 = specular_color + (specular_environment_r90 - specular_color) * pow(clamp(1.0 - v_dot_h, 0.0, 1.0), 5.0);
    float geometric_occlusion_smith_ggx = SmithG1_var2(n_dot_l, alpha_roughness) * SmithG1_var2(n_dot_v, alpha_roughness);
    float ggx;
    {
        float roughness_sq = alpha_roughness * alpha_roughness;
        float f = (n_dot_h * roughness_sq - n_dot_h) * n_dot_h + 1.0;
        ggx = roughness_sq / (M_PI * f * f);
    }

    vec3 diffuse_contrib = (1.0 - fresnel_schlick_2) * base_color.rgb / M_PI;

    vec3 spec_contrib = fresnel_schlick_2 * geometric_occlusion_smith_ggx * ggx / (4.0 * n_dot_l * n_dot_v);

    f_color.rgb = n_dot_l * u_LightColor * (diffuse_contrib + spec_contrib);
    f_color.a = base_color.a;


    // Add ambient occlusion.
    {
        float ao = 1.0;
        if (u_material_params.occlusion_texture_tex_coord == 0) {
            ao = texture(u_occlusion_texture, v_texcoord_0).x;
        } else if (u_material_params.occlusion_texture_tex_coord == 1) {
            ao = texture(u_occlusion_texture, v_texcoord_1).x;
        }
        f_color.rgb = mix(f_color.rgb, f_color.rgb * ao,
                        u_material_params.occlusion_texture_strength);
    }

    // Add the emissive color.
    {
        vec4 emissive = vec4(0.0);
        if (u_material_params.emissive_texture_tex_coord == 0) {
            emissive.rgb = texture(u_emissive_texture, v_texcoord_0).rgb;
            emissive.a = 1.0;
        } else if (u_material_params.emissive_texture_tex_coord == 1) {
            emissive.rgb = texture(u_emissive_texture, v_texcoord_1).rgb;
            emissive.a = 1.0;
        }
        f_color.rgb += emissive.rgb * emissive.a;
    }


    /*f_color.rgb = mix(f_color.rgb, fresnel_schlick_2, u_ScaleFGDSpec.x);
    f_color.rgb = mix(f_color.rgb, vec3(geometric_occlusion_smith_ggx), u_ScaleFGDSpec.y);
    f_color.rgb = mix(f_color.rgb, vec3(ggx), u_ScaleFGDSpec.z);
    f_color.rgb = mix(f_color.rgb, specContrib, u_ScaleFGDSpec.w);

    f_color.rgb = mix(f_color.rgb, diffuseContrib, u_ScaleDiffBaseMR.x);
    f_color.rgb = mix(f_color.rgb, baseColor.rgb, u_ScaleDiffBaseMR.y);
    f_color.rgb = mix(f_color.rgb, vec3(metallic), u_ScaleDiffBaseMR.z);
    f_color.rgb = mix(f_color.rgb, vec3(perceptualRoughness), u_ScaleDiffBaseMR.w);*/
}
