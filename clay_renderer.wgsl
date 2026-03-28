struct ScreenUniform {
    screen_size: vec2<f32>,
    _pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> u_screen: ScreenUniform;
@group(0) @binding(1) var u_tex: texture_2d<f32>;
@group(0) @binding(2) var u_samp: sampler;

struct VsIn {
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) use_tex: f32,
    @location(4) rect_size: vec2<f32>,
    @location(5) corner_radius: f32,
    @location(6) uv_shape: vec2<f32>,
    @location(7) border: vec4<f32>,
    @location(8) border_only: f32,
};

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) use_tex: f32,
    @location(3) rect_size: vec2<f32>,
    @location(4) corner_radius: f32,
    @location(5) uv_shape: vec2<f32>,
    @location(6) border: vec4<f32>,
    @location(7) border_only: f32,
};

fn sd_rounded_box_centered(p: vec2<f32>, size: vec2<f32>, radius: f32) -> f32 {
    let half_size = size * 0.5;
    let r = min(radius, min(half_size.x, half_size.y));
    let q = abs(p) - (half_size - vec2<f32>(r, r));
    return length(max(q, vec2<f32>(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - r;
}

@vertex
fn vs_main(input: VsIn) -> VsOut {
    var out: VsOut;
    let ndc_x = (input.pos.x / u_screen.screen_size.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (input.pos.y / u_screen.screen_size.y) * 2.0;
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.uv = input.uv;
    out.color = input.color;
    out.use_tex = input.use_tex;
    out.rect_size = input.rect_size;
    out.corner_radius = input.corner_radius;
    out.uv_shape = input.uv_shape;
    out.border = input.border;
    out.border_only = input.border_only;
    return out;
}

@fragment
fn fs_main(input: VsOut) -> @location(0) vec4<f32> {
    if (input.corner_radius > 0.0) {
        // Rounded mask works in shape UV-space so textures can still use atlas UVs.
        let p = input.uv_shape * input.rect_size - input.rect_size * 0.5;
        let outside = sd_rounded_box_centered(p, input.rect_size, input.corner_radius);
        if (outside > 0.0) {
            discard;
        }
    }

    if (input.border_only > 0.5) {
        let border_left = max(input.border.x, 0.0);
        let border_right = max(input.border.y, 0.0);
        let border_top = max(input.border.z, 0.0);
        let border_bottom = max(input.border.w, 0.0);

        let inner_w = input.rect_size.x - border_left - border_right;
        let inner_h = input.rect_size.y - border_top - border_bottom;
        if (inner_w <= 0.0 || inner_h <= 0.0) {
            // No inner space remains: keep full outer shape as border.
        } else {
            // Inner rounded cutout approximates CSS border behavior for uniform radii.
            let inner_radius = max(
                0.0,
                input.corner_radius - max(max(border_left, border_right), max(border_top, border_bottom)),
            );
            let inner_center = vec2<f32>(
                (border_left + inner_w * 0.5) - input.rect_size.x * 0.5,
                (border_top + inner_h * 0.5) - input.rect_size.y * 0.5,
            );
            let p = input.uv_shape * input.rect_size - input.rect_size * 0.5;
            let inside_inner = sd_rounded_box_centered(p - inner_center, vec2<f32>(inner_w, inner_h), inner_radius);
            if (inside_inner <= 0.0) {
                discard;
            }
        }
    }

    let texel = textureSample(u_tex, u_samp, input.uv);
    let sampled = texel * input.color;
    let solid = input.color;
    let choose = select(0.0, 1.0, input.use_tex > 0.5);
    return solid * (1.0 - choose) + sampled * choose;
}
