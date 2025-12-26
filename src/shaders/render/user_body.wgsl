// STRUCTS

struct Metadata {
    cam_center: vec2<f32>,
    cam_half_size: vec2<f32>,
    viewport: vec2<f32>,
    user_body_pos: vec2<f32>,
    user_body_mass: f32,
    num_bodies: u32,
}

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>, // quad local coords in [-1,1]
    @location(1) @interpolate(flat) radius_px: f32,
};


// BINDINGS AND BUFFERS

// metadata buffers
@group(0) @binding(0) var<uniform> metadata: Metadata;


// HELPER FUNCTIONS

fn world_to_ndc(p: vec2<f32>) -> vec2<f32> {
    var ndc = (p - metadata.cam_center) / metadata.cam_half_size;
    ndc.y = -ndc.y; // y axis has to be flipped for some reason?
    return ndc;
}


// MAIN SHADERS

@vertex
fn vertex_main(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VSOut {
    let quad = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0)
    );

    var out: VSOut;

    let world_pos = metadata.user_body_pos;
    let center_ndc = world_to_ndc(world_pos);

    var radius_px = 8.0;
    let px_to_ndc = 2.0 / metadata.viewport;
    let radius_ndc = radius_px * px_to_ndc;

    let uv = quad[vid];
    out.pos = vec4<f32>(center_ndc + uv * radius_ndc, 0.0, 1.0);
    out.uv = uv;
    out.radius_px = radius_px;

    return out;
}

@fragment
fn fragment_main(in: VSOut) -> @location(0) vec4<f32> {
    // soft kernel (gaussian-like)
    let r = length(in.uv);

    if r > 1.0 {
        discard;
    }

    let color = vec3<f32>(0.95, 0.25, 0.25);

    return vec4<f32>(color, 1.0);
}