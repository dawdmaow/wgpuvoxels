struct SharedFrame {
    mvp: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> frame: SharedFrame;

struct Vertex_Output {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1) color: vec4<f32>,
) -> Vertex_Output {
    var out: Vertex_Output;
    out.clip_pos = frame.mvp * vec4<f32>(pos, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
    return color;
}
