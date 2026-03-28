// 2.5D volumetric-style clouds: stacked XZ slices (instanced quads) with fBm noise.
// Intentionally does NOT apply terrain distance fog - only a fixed sky-color tint so
// clouds stay readable at the same apparent density regardless of world fog.

struct SharedFrame {
    mvp: mat4x4<f32>,
    camera_pos: vec3<f32>,
    elapsed_time: f32,
}

struct CloudUniform {
    cloud_y_bounds: vec2<f32>,
    cloud_sorted_y_lo: vec4<f32>,
    cloud_sorted_y_hi: vec4<f32>,
}

struct SharedFog {
    fog_color_near: vec4<f32>,
}

@group(0) @binding(0) var<uniform> frame: SharedFrame;
@group(0) @binding(1) var<uniform> cloud: CloudUniform;
@group(0) @binding(2) var<uniform> fog: SharedFog;

struct Vertex_Output {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) world_y: f32,
    @location(1) world_xz: vec2<f32>,
}

fn layer_y_from_instance(iid: u32) -> f32 {
    // Matches CPU-packed vec4 (sort_cloud_layers in main.odin): fixed ascending world Y per instance.
    switch iid {
        case 0u: {
            return cloud.cloud_sorted_y_lo.x;
        }
        case 1u: {
            return cloud.cloud_sorted_y_lo.y;
        }
        case 2u: {
            return cloud.cloud_sorted_y_lo.z;
        }
        case 3u: {
            return cloud.cloud_sorted_y_lo.w;
        }
        case 4u: {
            return cloud.cloud_sorted_y_hi.x;
        }
        case 5u: {
            return cloud.cloud_sorted_y_hi.y;
        }
        case 6u: {
            return cloud.cloud_sorted_y_hi.z;
        }
        case 7u: {
            return cloud.cloud_sorted_y_hi.w;
        }
        default: {
            return cloud.cloud_sorted_y_hi.w;
        }
    }
}

const CLOUD_QUAD_HALF: f32 = 148.0;
const WIND: vec2<f32> = vec2<f32>(0.11, 0.07);
// Radial fade on each slice so the square billboard doesn't cut off in a hard rectangle.
const CARD_FADE_START_R: f32 = 74.0;
const CARD_FADE_END_R: f32 = 128.0;

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> Vertex_Output {
    let y = layer_y_from_instance(iid);
    var lx: f32;
    var lz: f32;
    switch vid {
        case 0u: {
            lx = -1.0;
            lz = -1.0;
        }
        case 1u: {
            lx = 1.0;
            lz = -1.0;
        }
        case 2u: {
            lx = -1.0;
            lz = 1.0;
        }
        case 3u: {
            lx = 1.0;
            lz = -1.0;
        }
        case 4u: {
            lx = 1.0;
            lz = 1.0;
        }
        default: {
            lx = -1.0;
            lz = 1.0;
        }
    }
    let wx = frame.camera_pos.x + lx * CLOUD_QUAD_HALF;
    let wz = frame.camera_pos.z + lz * CLOUD_QUAD_HALF;
    var out: Vertex_Output;
    out.clip_pos = frame.mvp * vec4<f32>(wx, y, wz, 1.0);
    out.world_y = y;
    out.world_xz = vec2<f32>(wx, wz);
    return out;
}

fn hash21(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn noise2(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Two octaves only - main cost was three fBm chains per pixel; one cheap fBm for bulk density.
fn fbm2(p: vec2<f32>) -> f32 {
    var v: f32 = 0.0;
    var a: f32 = 0.5;
    var x = p;
    for (var i: i32 = 0; i < 2; i = i + 1) {
        v += a * noise2(x);
        x = x * 2.02 + vec2<f32>(17.0, 9.0);
        a *= 0.5;
    }
    return v;
}

@fragment
fn fs_main(
    @location(0) @interpolate(flat) world_y: f32,
    @location(1) world_xz: vec2<f32>,
) -> @location(0) vec4<f32> {
    // Cheap masks first: looking up fills the screen with cloud quads - skip fBm off the card
    // and outside the vertical envelope (big win at corners / under the stack).
    let cam_xz = vec2<f32>(frame.camera_pos.x, frame.camera_pos.z);
    let r = length(world_xz - cam_xz);
    let card_fade = 1.0 - smoothstep(CARD_FADE_START_R, CARD_FADE_END_R, r);
    // Vertical shaping anchored in world-space layer heights; camera Y changes should not
    // change cloud opacity (jumping used to "lift" the cloud stack visually).
    let cloud_min_y = cloud.cloud_y_bounds.x;
    let cloud_max_y = cloud.cloud_y_bounds.y;
    let lower_fade = smoothstep(cloud_min_y - 8.0, cloud_min_y + 2.0, world_y);
    let upper_fade = 1.0 - smoothstep(cloud_max_y + 2.0, cloud_max_y + 14.0, world_y);
    let vert_env = lower_fade * upper_fade;
    let geo_mask = card_fade * vert_env;
    if (geo_mask < 0.0001) {
        discard;
    }

    let scroll = WIND * frame.elapsed_time;
    // Without a Y term, every slice used the same (xz) noise - stacked identical billboards.
    let y_shear = vec2<f32>(world_y * 0.009, world_y * -0.006);
    let p = world_xz * 0.0062 + scroll + y_shear;
    // Single noise warp + 2-oct fBm + single noise wispy - same look family, ~3× less ALU than triple fBm.
    let warp = noise2(p * 0.35 + vec2<f32>(3.1, 1.7)) * 7.2;
    let d = fbm2(p + vec2<f32>(warp, warp * 0.7));
    // Wider smoothstep band + slightly lower floor = visibly more cloud mass vs sky.
    let cover = smoothstep(0.22, 0.88, d);
    let wispy = smoothstep(0.04, 0.99, noise2(p * 1.85 + vec2<f32>(warp * 0.2, 0.0)));
    var a = cover * wispy * 0.22 * geo_mask;
    let sky = fog.fog_color_near.xyz;
    let sun = normalize(vec3<f32>(0.35, 0.85, 0.4));
    let lit = pow(max(dot(sun, vec3<f32>(0.0, 1.0, 0.0)), 0.0), 0.35);
    let base = mix(vec3<f32>(0.92, 0.95, 1.0), sky, 0.42);
    let rgb = mix(base, vec3<f32>(1.0, 1.0, 1.0), lit * 0.18);
    return vec4<f32>(rgb, a);
}
