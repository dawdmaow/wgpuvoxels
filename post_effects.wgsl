// Fullscreen composite: scene color + depth -> swapchain. Underwater: wobble, blue tint, distance murk.

struct PostUniform {
    // .xyz = camera, .w = elapsed_time
    camera_elapsed: vec4<f32>,
    underwater_strength: f32,
    bloom_strength: f32,
    resolution: vec2<f32>,
    inv_view_proj: mat4x4<f32>,
    fxaa_enabled: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> post: PostUniform;
@group(0) @binding(1) var scene_tex: texture_2d<f32>;
@group(0) @binding(2) var scene_sampler: sampler;
@group(0) @binding(3) var depth_tex: texture_depth_2d;
@group(0) @binding(4) var depth_sampler: sampler;
@group(0) @binding(5) var bloom_tex: texture_2d<f32>;

const UNDERWATER_TINT: vec3<f32> = vec3<f32>(0.025, 0.09, 0.2);
const TINT_MIX: f32 = 0.15;
const FOG_DENSITY: f32 = 0.15;
// UV-space offset; clamp-to-edge hides seams - ~10x prior so refraction reads clearly.
const DISTORT_STRENGTH: f32 = 0.0038;
const FOG_COLOR: vec3<f32> = vec3<f32>(0.015, 0.06, 0.14);
const BLOOM_PRE_DARKEN: f32 = 0.5;
const FXAA_CONTRAST_MIN: f32 = 0.045;
const FXAA_CONTRAST_SCALE: f32 = 0.17;
const FXAA_BLEND_STRENGTH: f32 = 0.75;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let p = pos[vid];
    var out: VertexOutput;
    out.clip_pos = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return out;
}

fn world_pos_from_depth(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc_x = uv.x * 2.0 - 1.0;
    let ndc_y = 1.0 - uv.y * 2.0;
    let clip = vec4<f32>(ndc_x, ndc_y, depth, 1.0);
    var h = post.inv_view_proj * clip;
    return h.xyz / max(h.w, 1e-5);
}

fn sample_depth(uv: vec2<f32>) -> f32 {
    let dims = textureDimensions(depth_tex);
    // Clamp away from the right/bottom edge so integer conversion stays in-range.
    let uv_safe = clamp(uv, vec2<f32>(0.0), vec2<f32>(0.99999994));
    let coord = vec2<i32>(uv_safe * vec2<f32>(dims));
    return textureLoad(depth_tex, coord, 0);
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

fn sample_scene_with_fxaa(uv: vec2<f32>) -> vec3<f32> {
    if (post.fxaa_enabled < 0.5) {
        return textureSample(scene_tex, scene_sampler, uv).rgb;
    }

    let res = max(post.resolution, vec2<f32>(1.0, 1.0));
    let px = 1.0 / res;
    let c = textureSample(scene_tex, scene_sampler, uv).rgb;
    let n = textureSample(scene_tex, scene_sampler, uv + vec2<f32>(0.0, -px.y)).rgb;
    let s = textureSample(scene_tex, scene_sampler, uv + vec2<f32>(0.0, px.y)).rgb;
    let e = textureSample(scene_tex, scene_sampler, uv + vec2<f32>(px.x, 0.0)).rgb;
    let w = textureSample(scene_tex, scene_sampler, uv + vec2<f32>(-px.x, 0.0)).rgb;

    let lc = luma(c);
    let ln = luma(n);
    let ls = luma(s);
    let le = luma(e);
    let lw = luma(w);

    let l_min = min(lc, min(min(ln, ls), min(le, lw)));
    let l_max = max(lc, max(max(ln, ls), max(le, lw)));
    let l_range = l_max - l_min;
    let edge_threshold = max(FXAA_CONTRAST_MIN, l_max * FXAA_CONTRAST_SCALE);

    if (l_range < edge_threshold) {
        return c;
    }

    // Blend along the stronger local gradient so we smooth pixel stairs, not flat regions.
    let grad_h = abs(le - lw);
    let grad_v = abs(ln - ls);
    let pair = select(n + s, e + w, grad_h > grad_v) * 0.5;
    return mix(c, pair, FXAA_BLEND_STRENGTH);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let w = max(post.underwater_strength, 0.0);
    var sample_uv = in.uv;

    if (w > 1e-3) {
        let t = post.camera_elapsed.w * 10;
        let ax = sample_uv.y * 42.0 + t * 1.1;
        let ay = sample_uv.x * 38.0 + t * 0.95;
        sample_uv.x += sin(ax) * DISTORT_STRENGTH * w;
        sample_uv.y += cos(ay) * DISTORT_STRENGTH * w;
        let bx = sample_uv.x * 18.0 - t * 0.7;
        sample_uv.x += sin(bx + sample_uv.y * 12.0) * DISTORT_STRENGTH * 0.95 * w;
        sample_uv.y += cos(sample_uv.x * 31.0 + t * 1.4) * DISTORT_STRENGTH * 0.55 * w;
    }

    let scene = sample_scene_with_fxaa(sample_uv);
    let bloom = textureSample(bloom_tex, scene_sampler, in.uv).rgb;
    let bloom_s = max(post.bloom_strength, 0.0);
    // Darken the base scene only when bloom is active so glow reads brighter by contrast.
    let scene_pre_bloom = select(scene, scene * BLOOM_PRE_DARKEN, bloom_s > 1e-3);
    var rgb = scene_pre_bloom + bloom * bloom_s;
    let depth = sample_depth(in.uv);
    let wp = world_pos_from_depth(in.uv, depth);
    let dist = length(wp - post.camera_elapsed.xyz);

    if (w < 1e-3) {
        return vec4<f32>(rgb, 1.0);
    }
    rgb = mix(rgb, UNDERWATER_TINT, TINT_MIX * w);

    let fog_t = (1.0 - exp(-FOG_DENSITY * dist)) * w;
    rgb = mix(rgb, FOG_COLOR, clamp(fog_t, 0.0, 0.92));

    return vec4<f32>(rgb, 1.0);
}
