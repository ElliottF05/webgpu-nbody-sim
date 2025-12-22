// METADATA STRUCTS

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


// BINDINGS AND BUFFERS

// metadata buffers
@group(0) @binding(0) var<uniform> float_metadata: FloatMetadata;
@group(0) @binding(1) var<uniform> uint_metadata: UintMetadata;

// data buffers
@group(0) @binding(2) var<storage, read> pos_buf: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> morton_codes: array<u32>;
@group(0) @binding(4) var<storage, read_write> indices: array<u32>;


// HELPER FUNCTIONS
// none for now



@compute @workgroup_size(64)
fn compute_morton_codes_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= uint_metadata.num_bodies {
        return;
    }

    let scale = 10000.0; // side length of the square region we map to
    let half = 0.5 * scale;

    let pos = pos_buf[i];
    var uv = (pos + vec2<f32>(half)) / vec2<f32>(scale); // map from [-half, half] to [0,1]
    uv = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)); // clamp to [0, 1]

    let x = min(u32(round(uv.x * 65535.0)), 65535u); // 16 bits
    let y = min(u32(round(uv.y * 65535.0)), 65535u); // 16 bits

    var morton_code = 0u;
    for (var bit = 0u; bit < 16u; bit += 1u) {
        let mask = 1u << bit;
        let bit_x = ((x & mask) >> bit) << (2 * bit);
        let bit_y = ((y & mask) >> bit) << (2 * bit + 1);
        morton_code = morton_code | bit_x | bit_y;
    }

    morton_codes[i] = morton_code;
    indices[i] = i;
}