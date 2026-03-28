struct SharedFrame {
    mvp: mat4x4<f32>,
    camera_pos: vec3<f32>,
    elapsed_time: f32,
}

struct SceneUniform {
    reflection_view_proj: mat4x4<f32>,
    reflection_plane_y: f32,
    render_mode: f32,
}

struct SharedFog {
    // .xyz = fog RGB (same as framebuffer clear), .w = fog_near world distance
    fog_color_near: vec4<f32>,
    fog_far: f32,
    // u32: matches CPU b32; bool is not host-shareable in uniform buffers (naga/wgpu).
    fog_enabled: u32,
    _pad0: vec2<f32>,
} 

@group(0) @binding(0) var<uniform> frame: SharedFrame;
@group(0) @binding(1) var<uniform> scene: SceneUniform;
@group(0) @binding(2) var<uniform> fog: SharedFog;
@group(0) @binding(3) var atlas_tex: texture_2d<f32>;
@group(0) @binding(4) var atlas_sampler: sampler;
@group(0) @binding(5) var reflection_tex: texture_2d<f32>;
@group(0) @binding(6) var reflection_sampler: sampler;

struct Vertex_Output {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) ao: f32,
    @location(3) @interpolate(flat) material_marker: f32,
    @location(4) world_pos: vec3<f32>,
    @location(5) voxel_light: f32,
}

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) ao: f32,
    @location(4) material_marker: f32,
    @location(5) voxel_light: f32,
) -> Vertex_Output {
    var world_pos = pos;
    var world_normal = normal;
    let is_water = material_marker > 0.5 && material_marker < 1.5;
    let is_flower = material_marker > 1.5;
    if (is_flower) {
        let local_offset = normal;
        let bend_weight = clamp(local_offset.y / 0.75, 0.0, 1.0);
        // Phase from world-space position keeps nearby flowers related but not perfectly synced.
        let phase = pos.x * FLOWER_WOBBLE_FREQ_X + pos.z * FLOWER_WOBBLE_FREQ_Z;
        let wobble_a = sin(frame.elapsed_time * FLOWER_WOBBLE_SPEED_A + phase);
        let wobble_b = sin(frame.elapsed_time * FLOWER_WOBBLE_SPEED_B + phase * 1.37);
        let sway = (wobble_a + wobble_b * 0.5) * FLOWER_WOBBLE_AMPLITUDE * bend_weight;
        let depth_drift = (wobble_b - wobble_a * 0.35) * FLOWER_WOBBLE_DEPTH_AMPLITUDE * bend_weight;
        let vertical_bob = sin(frame.elapsed_time * 2.9 + phase * 0.61) * FLOWER_WOBBLE_BOB_AMPLITUDE * bend_weight;
        // Flower vertices store center in `pos` and billboard-local offset in `normal`.
        // Rotate only around world-up so sprites stay upright even when looking up/down.
        let to_cam_xz = frame.camera_pos.xz - pos.xz;
        let to_cam_len = length(to_cam_xz);
        let facing_xz = select(vec2<f32>(0.0, 1.0), to_cam_xz / to_cam_len, to_cam_len > 1e-4);
        let billboard_right = vec3<f32>(facing_xz.y, 0.0, -facing_xz.x);
        let billboard_forward = vec3<f32>(facing_xz.x, 0.0, facing_xz.y);
        world_pos =
            pos +
            billboard_right * (local_offset.x + sway) +
            billboard_forward * depth_drift +
            vec3<f32>(0.0, 1.0, 0.0) * (local_offset.y + vertical_bob);
        // Keep lighting stable while the sprite rotates toward the camera.
        world_normal = vec3<f32>(0.0, 1.0, 0.0);
    } else if (is_water && abs(normal.y) > 0.9) {
        // Geometric wobble adds subtle water surface lift inspired by classic block water.
        let wave_a = sin(world_pos.x * WATER_GEOM_WOBBLE_FREQ_X + frame.elapsed_time * WATER_GEOM_WOBBLE_SPEED_A);
        let wave_b = sin(world_pos.z * WATER_GEOM_WOBBLE_FREQ_Z + frame.elapsed_time * WATER_GEOM_WOBBLE_SPEED_B);
        world_pos.y += (wave_a + wave_b) * WATER_GEOM_WOBBLE_AMPLITUDE;
    }
    var out: Vertex_Output;
    out.clip_pos = frame.mvp * vec4<f32>(world_pos, 1.0);
    // Translation-only chunk model: normals stay in world space.
    out.world_normal = world_normal;
    out.uv = uv;
    out.ao = ao;
    out.material_marker = material_marker;
    out.world_pos = world_pos;
    out.voxel_light = voxel_light;
    return out;
}

const WATER_TILE_WIDTH: f32 = 16.0 / 512.0;
const WATER_TILE_HEIGHT: f32 = 16.0 / 256.0;
const WATER_TILE_INSET_U: f32 = 0.25 / 512.0;
const WATER_TILE_INSET_V: f32 = 0.25 / 256.0;
const WATER_SPAN_U: f32 = WATER_TILE_WIDTH - 2.0 * WATER_TILE_INSET_U;
const WATER_SPAN_V: f32 = WATER_TILE_HEIGHT - 2.0 * WATER_TILE_INSET_V;
const WATER_BASE_TILE: vec2<f32> = vec2<f32>(2.0 * WATER_TILE_WIDTH, 0.0 * WATER_TILE_HEIGHT);
const WATER_BASE_UV0: vec2<f32> = WATER_BASE_TILE + vec2<f32>(WATER_TILE_INSET_U, WATER_TILE_INSET_V);
const WATER_TILE20_UV0: vec2<f32> = vec2<f32>(2.0 * WATER_TILE_WIDTH + WATER_TILE_INSET_U, 0.0 * WATER_TILE_HEIGHT + WATER_TILE_INSET_V);
const WATER_TILE30_UV0: vec2<f32> = vec2<f32>(3.0 * WATER_TILE_WIDTH + WATER_TILE_INSET_U, 0.0 * WATER_TILE_HEIGHT + WATER_TILE_INSET_V);
const WATER_TILE31_UV0: vec2<f32> = vec2<f32>(3.0 * WATER_TILE_WIDTH + WATER_TILE_INSET_U, 1.0 * WATER_TILE_HEIGHT + WATER_TILE_INSET_V);
const WATER_TILE21_UV0: vec2<f32> = vec2<f32>(2.0 * WATER_TILE_WIDTH + WATER_TILE_INSET_U, 1.0 * WATER_TILE_HEIGHT + WATER_TILE_INSET_V);
const WATER_FRAME_RATE: f32 = 1.25;
const WATER_UV_WOBBLE_STRENGTH: f32 = 0.08;
const WATER_UV_WOBBLE_RATE_A: f32 = 1.7;
const WATER_UV_WOBBLE_RATE_B: f32 = 2.3;
const WATER_GEOM_WOBBLE_AMPLITUDE: f32 = 0.04;
const WATER_GEOM_WOBBLE_FREQ_X: f32 = 1.5707963;
const WATER_GEOM_WOBBLE_FREQ_Z: f32 = 1.5707963;
const WATER_GEOM_WOBBLE_SPEED_A: f32 = 1.0;
const WATER_GEOM_WOBBLE_SPEED_B: f32 = 1.5;
// Keep stems rooted while letting petals sway with layered motion.
const FLOWER_WOBBLE_AMPLITUDE: f32 = 0.11;
const FLOWER_WOBBLE_FREQ_X: f32 = 0.9;
const FLOWER_WOBBLE_FREQ_Z: f32 = 0.7;
const FLOWER_WOBBLE_SPEED_A: f32 = 1.55;
const FLOWER_WOBBLE_SPEED_B: f32 = 2.35;
const FLOWER_WOBBLE_DEPTH_AMPLITUDE: f32 = 0.045;
const FLOWER_WOBBLE_BOB_AMPLITUDE: f32 = 0.03;
const WATER_FRESNEL_POWER: f32 = 4.5;
const WATER_FRESNEL_STRENGTH: f32 = 0.28;
const WATER_SPECULAR_EXPONENT: f32 = 48.0;
const WATER_SPECULAR_STRENGTH: f32 = 0.22;
// Flower billboards use simplified lighting; this keeps them from reading brighter than terrain.
const FLOWER_BRIGHTNESS_MUL: f32 = 0.5;
// const WATER_SKY_TINT: vec3<f32> = vec3<f32>(0.20, 0.32, 0.44);
// Planar reflection is mixed in by Fresnel; this scales how strong that mix can get.
const WATER_PLANAR_REFLECTION_STRENGTH: f32 = 2.35;
// Keep geometry reflections mostly intact; only sky-like reflection color gets a stronger lift.
const WATER_REFLECTION_BASE_BRIGHTNESS_BOOST: f32 = 1.04;
const WATER_REFLECTION_SKY_BRIGHTNESS_BOOST: f32 = 1.22;
// Caps blend toward pure reflection texture (never fully replace water at normal incidence if < 1).
// const PLANAR_MIX_MAX: f32 = 0.92;
const PLANAR_MIX_MAX: f32 = 1;
// Underside of open water (seen from below): softer than the air-side surface.
const WATER_UNDERFACE_ALPHA_MUL: f32 = 0.5;
const GHOST_PREVIEW_ALPHA_BASE: f32 = 0.42;
const GHOST_PREVIEW_ALPHA_PULSE_AMPLITUDE: f32 = 0.14;
const GHOST_PREVIEW_ALPHA_PULSE_SPEED: f32 = 2.8;
const ARMED_TNT_FLASH_SPEED: f32 = 14.0;
const ARMED_TNT_FLASH_MIN_BRIGHTNESS: f32 = 0.7;
const ARMED_TNT_FLASH_MAX_BRIGHTNESS: f32 = 1.7;
// Nearby geometry would otherwise ignore atmosphere; ties voxels to clear/fog color.
const SKY_AMBIENT_MIX: f32 = 0.25;
const WATER_FRAME_ORIGINS: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    WATER_TILE20_UV0,
    WATER_TILE30_UV0,
    WATER_TILE21_UV0,
    WATER_TILE31_UV0,
);

@fragment
fn fs_main(
    @location(0) world_normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) ao: f32,
    @location(3) @interpolate(flat) material_marker: f32,
    @location(4) world_pos: vec3<f32>,
    @location(5) voxel_light: f32,
) -> @location(0) vec4<f32> {
    let n = normalize(world_normal);
    let sun = normalize(vec3<f32>(0.35, 0.85, 0.4));
    let is_water = material_marker > 0.5 && material_marker < 1.5;
    let is_flower = material_marker > 1.5;
    let is_armed_tnt = material_marker < -1.5;
    let is_ghost = material_marker > -1.5 && material_marker < -0.5;
    if (scene.render_mode > 0.5 && scene.render_mode < 1.5 && world_pos.y < scene.reflection_plane_y) {
        discard;
    }
    // Keep unlit faces readable; harsh contrast hurts voxel shape readability.
    let lambert = max(dot(n, sun), 0.22);
    let ao_term = max(clamp(ao, 0.0, 1.0), 0.35);
    var albedo: vec4<f32>;
    if (is_water) {
        // Water mesh is baked with water_0 UVs, so reconstruct tile-local UV from that
        // exact inset/stride mapping before applying flow motion.
        let local_uv = vec2<f32>(
            clamp((uv.x - WATER_BASE_UV0.x) / WATER_SPAN_U, 0.0, 1.0),
            clamp((uv.y - WATER_BASE_UV0.y) / WATER_SPAN_V, 0.0, 1.0),
        );
        // Use a continuous world-space phase field so adjacent water faces don't
        // show hard seams at voxel boundaries.
        let phase_field = sin(dot(world_pos.xz, vec2<f32>(0.41, 0.27)));
        let phase_offset = (phase_field * 0.5 + 0.5) * 4.0;
        let anim = frame.elapsed_time * WATER_FRAME_RATE + phase_offset;
        let phase = phase_offset * 6.2831853;
        // Non-directional wobble keeps water visibly alive even when atlas frames are identical.
        let uv_wobble = vec2<f32>(
            sin(frame.elapsed_time * WATER_UV_WOBBLE_RATE_A + phase),
            cos(frame.elapsed_time * WATER_UV_WOBBLE_RATE_B + phase * 1.31),
        ) * WATER_UV_WOBBLE_STRENGTH;
        let animated_local_uv = fract(local_uv + uv_wobble);
        let frame0 = i32(floor(anim)) & 3;
        let frame1 = (frame0 + 1) & 3;
        let blend = fract(anim);
        let uv0 = WATER_FRAME_ORIGINS[frame0] + animated_local_uv * vec2<f32>(WATER_SPAN_U, WATER_SPAN_V);
        let uv1 = WATER_FRAME_ORIGINS[frame1] + animated_local_uv * vec2<f32>(WATER_SPAN_U, WATER_SPAN_V);
        albedo = mix(
            textureSampleLevel(atlas_tex, atlas_sampler, uv0, 0.0),
            textureSampleLevel(atlas_tex, atlas_sampler, uv1, 0.0),
            blend,
        );
    } else {
        albedo = textureSampleLevel(atlas_tex, atlas_sampler, uv, 0.0);
    }
    let base = albedo.rgb;
    var rgb: vec3<f32>;
    if (is_water) {
        let lit_base = base * lambert * ao_term;
        let view_dir = normalize(frame.camera_pos - world_pos);
        let ndv = max(dot(n, view_dir), 0.0);
        let fresnel = pow(1.0 - ndv, WATER_FRESNEL_POWER);
        let reflect_dir = reflect(-sun, n);
        let spec = pow(max(dot(reflect_dir, view_dir), 0.0), WATER_SPECULAR_EXPONENT);
        let specular_term =
            vec3<f32>(fresnel * WATER_FRESNEL_STRENGTH) + vec3<f32>(spec * WATER_SPECULAR_STRENGTH);
        var surface = min(lit_base + specular_term, vec3<f32>(1.0));
        if (scene.render_mode > 1.5) {
            let is_top_face = n.y > 0.9;
            let is_sea_level_surface = abs(world_pos.y - scene.reflection_plane_y) < 0.35;
            if (is_top_face && is_sea_level_surface) {
                let reflection_clip = scene.reflection_view_proj * vec4<f32>(world_pos, 1.0);
                let safe_w = max(abs(reflection_clip.w), 0.0001);
                let ndc = reflection_clip.xy / safe_w;
                let reflection_uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
                let in_bounds =
                    reflection_uv.x >= 0.0 && reflection_uv.x <= 1.0 &&
                    reflection_uv.y >= 0.0 && reflection_uv.y <= 1.0;
                if (in_bounds) {
                    // Unlit scene color from reflection RT - not multiplied by lambert/ao/albedo.
                    // Reflection sample sits in per-pixel conditional flow; explicit LOD avoids
                    // derivative/uniform-flow constraints from implicit sampling.
                    let planar_raw = textureSampleLevel(reflection_tex, reflection_sampler, reflection_uv, 0.0).rgb;
                    // Use closeness to fog/clear color as a sky proxy so sky reflection brightens
                    // more than reflected meshes.
                    let sky_match =
                        1.0 - clamp(length(planar_raw - fog.fog_color_near.xyz) * 1.6, 0.0, 1.0);
                    let planar_boost = mix(
                        WATER_REFLECTION_BASE_BRIGHTNESS_BOOST,
                        WATER_REFLECTION_SKY_BRIGHTNESS_BOOST,
                        sky_match,
                    );
                    let planar = min(
                        planar_raw * planar_boost,
                        vec3<f32>(1.0),
                    );
                    let mix_w = clamp(fresnel * WATER_PLANAR_REFLECTION_STRENGTH * PLANAR_MIX_MAX, 0.0, 1.0);
                    surface = mix(surface, planar, mix_w);
                }
            }
        }
        rgb = surface;
    } else {
        rgb = base * lambert * ao_term;
        if (is_flower) {
            // Flowers don't get per-voxel AO, so apply a gentle artistic bias for parity with cubes.
            rgb *= FLOWER_BRIGHTNESS_MUL;
        }
    }
    rgb *= clamp(voxel_light, 0.0, 1.0);
    if (is_armed_tnt) {
        let tnt_flash = 0.5 + 0.5 * sin(frame.elapsed_time * ARMED_TNT_FLASH_SPEED);
        let tnt_brightness = mix(ARMED_TNT_FLASH_MIN_BRIGHTNESS, ARMED_TNT_FLASH_MAX_BRIGHTNESS, tnt_flash);
        rgb *= tnt_brightness;
    }
    let fog_far = fog.fog_far;
    let fog_near = fog.fog_color_near.w;
    let fog_rgb = fog.fog_color_near.xyz;
    let fog_enabled = fog.fog_enabled != 0u;
    if (fog_enabled) {
        rgb = mix(rgb, fog_rgb, SKY_AMBIENT_MIX);
        let fog_dist = length(world_pos - frame.camera_pos);
        let fog_t = smoothstep(fog_near, max(fog_far, fog_near + 1e-4), fog_dist);
        rgb = mix(rgb, fog_rgb, fog_t);
    }
    // material_marker: 0 solid voxel, 1 water, 2 flower billboard, -1 placement ghost, -2 armed TNT overlay.
    var alpha = select(1.0, max(albedo.a, 0.72), is_water);
    if (is_flower) {
        alpha = albedo.a;
        // Skip near-zero alpha texels so sprite cutouts stay crisp.
        if (alpha < 0.02) {
            discard;
        }
    } else if (is_ghost) {
        // Pulse opacity so the preview reads as interactive placement feedback.
        let ghost_alpha_pulse = 0.5 + 0.5 * sin(frame.elapsed_time * GHOST_PREVIEW_ALPHA_PULSE_SPEED);
        let ghost_alpha = GHOST_PREVIEW_ALPHA_BASE + (ghost_alpha_pulse - 0.5) * 2.0 * GHOST_PREVIEW_ALPHA_PULSE_AMPLITUDE;
        // Clamp against source alpha to keep transparent atlas texels from becoming opaque.
        alpha = min(alpha, ghost_alpha);
    }
    // Underface quads are the duplicate open-water top with normal (0,-1,0); air-side top is (0,1,0).
    // Threshold leaves slack for float / interpolation (compare to is_top_face n.y > 0.9 elsewhere).

    // NOTE: Water top underface (visible from below)
    // NOTE: Unfortunately, I think it affects how water is seen from BELOW WATER and not just from INSIDE water.
    // if (material_marker > 0.5 && n.y < -0.9) {
    //     alpha = alpha * WATER_UNDERFACE_ALPHA_MUL;
    // }

    return vec4<f32>(rgb, alpha);
}
