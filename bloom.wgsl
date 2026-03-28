// Half-res bright extract + separable Gaussian blur for LDR bloom.

struct BloomUniform {
    // .x = threshold, .y = knee (soft knee range), .zw unused
    threshold_knee: vec4<f32>,
    // .xy = texel size in half-res UV space, .zw = blur axis (1,0) or (0,1)
    texel_dir: vec4<f32>,
}

// Slight boosts here keep bloom visibly stronger without retuning CPU-side uniforms.
const BLOOM_EXTRACT_GAIN: f32 = 1.22;
const BLOOM_BLUR_GAIN: f32 = 1.12;
// const BLOOM_INPUT_DARKEN: f32 = 1.0; // NOTE: Doesn't do anything

@group(0) @binding(0) var<uniform> bloom: BloomUniform;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var src_sampler: sampler;

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

// LDR bloom: threshold on luminance (max channel was starving mid-tones at threshold 0.7+).
fn prefilter_color(color: vec3<f32>, threshold: f32, knee: f32) -> vec3<f32> {
    let luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    let soft_clamped = clamp(luma - threshold + knee, 0.0, 2.0 * knee);
    let soft_sq = soft_clamped * soft_clamped / (4.0 * knee + 1e-5);
    let contrib = max(luma - threshold, soft_sq);
    return color * (contrib / max(luma, 1e-5));
}

@fragment
fn fs_extract(in: VertexOutput) -> @location(0) vec4<f32> {
    // Bias bloom response toward only the strongest highlights by dimming
    // the sampled source before thresholding.
    // let c = textureSample(src_tex, src_sampler, in.uv).rgb * BLOOM_INPUT_DARKEN; // NOTE: Doesn't do anything
    let c = textureSample(src_tex, src_sampler, in.uv).rgb;
    let t = bloom.threshold_knee.x;
    let k = bloom.threshold_knee.y;
    let rgb = prefilter_color(c, t, k) * BLOOM_EXTRACT_GAIN;
    return vec4<f32>(rgb, 1.0);
}

@fragment
fn fs_copy(in: VertexOutput) -> @location(0) vec4<f32> {
    let c = textureSample(src_tex, src_sampler, in.uv).rgb;
    return vec4<f32>(c, 1.0);
}

// 7-tap separable Gaussian with a widened radius so bloom reads clearly in motion.
@fragment
fn fs_blur(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let t = bloom.texel_dir.xy;
    let d = bloom.texel_dir.zw;
    let radius = 2.1;

    // Normalized weights for offsets 0..3.
    let w0 = 0.19648255;
    let w1 = 0.29690696;
    let w2 = 0.09447040;
    let w3 = 0.01038136;

    let o1 = d * t * radius;
    let o2 = d * t * (2.0 * radius);
    let o3 = d * t * (3.0 * radius);
    var c = textureSample(src_tex, src_sampler, uv) * w0;
    c += (textureSample(src_tex, src_sampler, uv + o1) + textureSample(src_tex, src_sampler, uv - o1)) * w1;
    c += (textureSample(src_tex, src_sampler, uv + o2) + textureSample(src_tex, src_sampler, uv - o2)) * w2;
    c += (textureSample(src_tex, src_sampler, uv + o3) + textureSample(src_tex, src_sampler, uv - o3)) * w3;
    return vec4<f32>(c.rgb * BLOOM_BLUR_GAIN, 1.0);
}
