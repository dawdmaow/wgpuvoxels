struct PostUniform {
    camera_elapsed: vec4<f32>,
    underwater_strength: f32,
    bloom_strength: f32,
    resolution: vec2<f32>,
    inv_view_proj: mat4x4<f32>,
    fxaa_enabled: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    // World-space start/end of blur ramp and blend strength [0,1].
    dof_start: f32,
    dof_end: f32,
    dof_strength: f32,
    _pad3: f32,
}

@group(0) @binding(0) var<uniform> post: PostUniform;
@group(0) @binding(1) var sharp_tex: texture_2d<f32>;
@group(0) @binding(2) var sharp_sampler: sampler;
@group(0) @binding(3) var blur_tex: texture_2d<f32>;
@group(0) @binding(4) var depth_tex: texture_depth_2d;
@group(0) @binding(5) var depth_sampler: sampler;

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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sharp_rgb = textureSample(sharp_tex, sharp_sampler, in.uv).rgb;
    let dof_strength = clamp(post.dof_strength, 0.0, 1.0);
    if (dof_strength < 1e-3) {
        return vec4<f32>(sharp_rgb, 1.0);
    }

    let depth = sample_depth(in.uv);
    let wp = world_pos_from_depth(in.uv, depth);
    let dist = length(wp - post.camera_elapsed.xyz);
    let dof_start = max(post.dof_start, 0.0);
    let dof_end = max(post.dof_end, dof_start + 1e-3);
    let dof_t = clamp((dist - dof_start) / max(dof_end - dof_start, 1e-3), 0.0, 1.0);
    let blur_rgb = textureSample(blur_tex, sharp_sampler, in.uv).rgb;
    let rgb = mix(sharp_rgb, blur_rgb, dof_t * dof_strength);
    return vec4<f32>(rgb, 1.0);
}
