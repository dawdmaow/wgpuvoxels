struct SharedFrame {
    mvp: mat4x4<f32>,
    camera_pos: vec3<f32>,
    elapsed_time: f32,
}

@group(0) @binding(0) var<uniform> frame: SharedFrame;

struct Vertex_Output {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) along: f32,
    @location(1) across: f32,
}

fn hash_u32(n: u32) -> u32 {
    var x = n * 747796405u + 2891336453u;
    x = ((x >> 16u) ^ x) * 2246822519u;
    x = ((x >> 13u) ^ x) * 3266489917u;
    x = (x >> 16u) ^ x;
    return x;
}

fn u01(a: u32) -> f32 {
    return f32(a & 0xffffffu) / f32(0xffffffu);
}

fn u01_stream(seed: u32, stream: u32) -> f32 {
    return u01(hash_u32(seed ^ (stream * 0x9e3779b9u)));
}

const RAIN_XZ_HALF: f32 = 26.0;
const STREAK_LEN: f32 = 3.2;
const FALL_CYCLE_MIN: f32 = 22.0;
const FALL_CYCLE_MAX: f32 = 38.0;
const FALL_SPEED_MIN: f32 = 0.18;
const FALL_SPEED_MAX: f32 = 1.65;
const Y_ANCHOR_SPREAD: f32 = 14.0;
const Y_ANCHOR_BIAS: f32 = 6.0;

// World-space half-width of each streak quad (total thickness ~= 2x this).
const STREAK_HALF_W_MIN: f32 = 0.01;
const STREAK_HALF_W_MAX: f32 = 0.1;

const WIND_BASE: vec2<f32> = vec2<f32>(0.38, 0.14);

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> Vertex_Output {
    let hx = u01_stream(iid, 0u);
    let hz = u01_stream(iid, 1u);
    let h_speed = u01_stream(iid, 2u);
    let h_phase = u01_stream(iid, 3u);
    let h_anchor = u01_stream(iid, 4u);
    let h_cycle = u01_stream(iid, 5u);
    let h_wind = u01_stream(iid, 6u);
    let h_thick = u01_stream(iid, 7u);

    let ox = (hx - 0.5) * 2.0 * RAIN_XZ_HALF;
    let oz = (hz - 0.5) * 2.0 * RAIN_XZ_HALF;

    let fall_speed = mix(FALL_SPEED_MIN, FALL_SPEED_MAX, h_speed);
    let phase = h_phase * 400.0 + h_phase * h_phase * 97.0;
    let fall_cycle = mix(FALL_CYCLE_MIN, FALL_CYCLE_MAX, h_cycle);

    let cycle_pos = fract(frame.elapsed_time * fall_speed + phase);
    let y_drop = cycle_pos * fall_cycle;

    let y_anchor = frame.camera_pos.y + Y_ANCHOR_BIAS + (h_anchor - 0.5) * 2.0 * Y_ANCHOR_SPREAD;
    let y_mid = y_anchor - y_drop;

    let y_bot = y_mid - STREAK_LEN * 0.5;
    let y_top = y_mid + STREAK_LEN * 0.5;

    let wind = WIND_BASE * (0.65 + 0.7 * h_wind);
    let dy_bot = y_bot - frame.camera_pos.y;
    let dy_top = y_top - frame.camera_pos.y;
    let bx = frame.camera_pos.x + ox + wind.x * dy_bot * 0.1;
    let bz = frame.camera_pos.z + oz + wind.y * dy_bot * 0.1;
    let tx = frame.camera_pos.x + ox + wind.x * dy_top * 0.1;
    let tz = frame.camera_pos.z + oz + wind.y * dy_top * 0.1;

    let B = vec3<f32>(bx, y_bot, bz);
    let T = vec3<f32>(tx, y_top, tz);
    let streak_dir = T - B;
    let sl = length(streak_dir);
    let dir = streak_dir / max(sl, 1e-5);
    let mid = (B + T) * 0.5;
    var to_cam = frame.camera_pos - mid;
    let tcl = length(to_cam);
    to_cam = to_cam / max(tcl, 1e-5);
    // Billboard: width vector in plane of streak, facing the camera.
    var right = cross(dir, to_cam);
    let rl = length(right);
    right = select(normalize(cross(dir, vec3<f32>(0.0, 1.0, 0.0))), right / rl, rl > 1e-4);
    let half_w = mix(STREAK_HALF_W_MIN, STREAK_HALF_W_MAX, h_thick);

    // Per-corner world position (two triangles, shared vertices duplicated).
    var world_pos: vec3<f32>;
    var along: f32;
    var across: f32;
    switch vid {
        case 0u: {
            world_pos = B - right * half_w;
            along = 0.0;
            across = -1.0;
        }
        case 1u: {
            world_pos = B + right * half_w;
            along = 0.0;
            across = 1.0;
        }
        case 2u: {
            world_pos = T - right * half_w;
            along = 1.0;
            across = -1.0;
        }
        case 3u: {
            world_pos = B + right * half_w;
            along = 0.0;
            across = 1.0;
        }
        case 4u: {
            world_pos = T + right * half_w;
            along = 1.0;
            across = 1.0;
        }
        default: {
            // case 5u
            world_pos = T - right * half_w;
            along = 1.0;
            across = -1.0;
        }
    }

    var out: Vertex_Output;
    out.clip_pos = frame.mvp * vec4<f32>(world_pos, 1.0);
    out.along = along;
    out.across = across;
    return out;
}

const RAIN_RGB_CONSTANT: f32 = 0.5;
const RAIN_RGB: vec3<f32> = vec3<f32>(0.3, 0.4, 0.7) * RAIN_RGB_CONSTANT;
const RAIN_ALPHA: f32 = 0.05;

@fragment
fn fs_main(@location(0) along: f32, @location(1) across: f32) -> @location(0) vec4<f32> {
    let cap = 0.14;
    let a_len = smoothstep(0.0, cap, along) * smoothstep(1.0, 1.0 - cap, along);
    // Feather left/right edges of the quad (|across| is 1 at sides).
    let a_wide = smoothstep(1.0, 0.72, abs(across));
    let alpha = RAIN_ALPHA * (0.35 + 0.65 * a_len) * (0.55 + 0.45 * a_wide);
    return vec4<f32>(RAIN_RGB, alpha);
}
