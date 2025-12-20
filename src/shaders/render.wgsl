struct FloatMetadata {
    grav_constant: f32,
    delta_time: f32,
    epsilon_multiplier: f32,
    _pad: f32,
    cam_center: vec2<f32>,
    cam_half_size: vec2<f32>,
    viewport: vec2<f32>,
}

struct UintMetadata {
    num_bodies: u32,
}

struct VSOut {
    @builtin(position) pos: vec4<f32>,
};

// metadata buffers
@group(0) @binding(0) var<uniform> float_metadata: FloatMetadata;
@group(0) @binding(1) var<uniform> uint_metadata: UintMetadata;

// data buffers
@group(0) @binding(2) var<storage, read> pos_buf: array<vec2<f32>>;

fn world_to_ndc(p: vec2<f32>) -> vec2<f32> {
    // maps to [-1,1]
    return (p - float_metadata.cam_center) / float_metadata.cam_half_size;
}

@vertex
fn vertex_main(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VSOut {
    // 6 vertices = 2 triangles making a quad
    let quad = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0)
    );

    var out: VSOut;

    if iid >= uint_metadata.num_bodies {
        out.pos = vec4<f32>(2.0, 2.0, 0.0, 1.0); // off-screen
        return out;
    }

    let world_pos = pos_buf[iid];
    let center_ndc = world_to_ndc(world_pos);

    let half_px_ndc = 1 * vec2<f32>(1.0, 1.0) / float_metadata.viewport;

    let offset = quad[vid] * half_px_ndc;

    let p_ndc = center_ndc + offset;

    out.pos = vec4<f32>(p_ndc, 0.0, 1.0);
    return out;
}

@fragment
fn fragment_main() -> @location(0) vec4<f32> {
    // return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    return vec4<f32>(0.3, 0.3, 0.3, 1.0);
}