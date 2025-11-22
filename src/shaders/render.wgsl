struct UintMetadata {
    width: u32,
    height: u32,
    num_iters: u32,
}

@group(0) @binding(0) var<uniform> uint_metadata : UintMetadata;
@group(0) @binding(1) var<storage, read> dye : array<f32>;
@group(0) @binding(2) var<storage, read> obstacles : array<u32>;

struct VSOut {
    @builtin(position) pos : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@vertex
fn vertex_main(@builtin(vertex_index) vi : u32) -> VSOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );

    var out: VSOut;
    let pos = positions[vi];
    out.pos = vec4<f32>(pos, 0.0, 1.0);
    out.uv = pos * 0.5 + vec2<f32>(0.5, 0.5);
    return out;
}

@fragment
fn fragment_main(in : VSOut) -> @location(0) vec4<f32> {
    // Convert UV back to pixel coords (clamp to edge)
    let px = clamp(i32(in.uv.x * f32(uint_metadata.width)),  0, i32(uint_metadata.width)  - 1);
    let py = clamp(i32(in.uv.y * f32(uint_metadata.height)), 0, i32(uint_metadata.height) - 1);

    let idx = u32(py) * uint_metadata.width + u32(px);
    let is_solid = obstacles[idx] == 1u;

    if (is_solid) {
        // render obstacles as black
        return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    } else {
        let dye_amount = dye[idx];
        return vec4<f32>(dye_amount, dye_amount, dye_amount, 1.0);
    }
}
