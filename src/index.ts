// Minimal compute -> graphics demo
// - compute: operates on a 1D storage buffer shaped (width * height) of f32 values
//   and writes an RGBA8 storage texture where R=G=B=value, A=1
// - graphics: samples that texture and draws it to the canvas as grayscale

const adapter = (await navigator.gpu.requestAdapter())!;
const device = await adapter.requestDevice();

const canvas = document.getElementById("webgpu-canvas") as HTMLCanvasElement;
const context = canvas.getContext("webgpu") as GPUCanvasContext;
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device, format: canvasFormat, alphaMode: "premultiplied" });

// devicePixelRatio to size the GPU buffers/textures in device pixels
const dpr = window.devicePixelRatio || 1;
const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
console.log("canvas size (px):", width, height, "dpr:", dpr);

// -------------------------------
// Seed buffer: one f32 per pixel (row-major y*width + x)
// Leftmost column = 1.0, all other entries = 0.0
// -------------------------------
const pixelCount = width * height;
const seed = new Float32Array(pixelCount);
for (let y = 0; y < height; ++y) {
    seed[y * width + 0] = 1.0; // leftmost column
}

// Input storage buffer (read-only in the compute shader)
const inputBuffer = device.createBuffer({
    size: seed.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(inputBuffer, 0, seed.buffer, seed.byteOffset, seed.byteLength);

// Uniform buffer: width and height as u32s for the shader
const sizeU32 = new Uint32Array([width, height]);
const sizeBuffer = device.createBuffer({
    size: sizeU32.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(sizeBuffer, 0, sizeU32.buffer, sizeU32.byteOffset, sizeU32.byteLength);

// Output storage texture: compute will write RGBA8 values; fragment shader will sample.
const outputTexture = device.createTexture({
    size: { width, height, depthOrArrayLayers: 1 },
    format: "rgba8unorm",
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
});

// Sampler for sampling the texture in the fragment shader
const sampler = device.createSampler({ magFilter: "nearest", minFilter: "nearest" });

// -------------------------------
// WGSL: compute shader (inline)
// - binding 0: input storage buffer (read-only array<f32>)
// - binding 1: output storage texture (write-only)
// - binding 2: uniform vec2<u32> for width/height
// Behavior: for each (x,y) write the value of the pixel to its left into the output texture.
// Leftmost column keeps the seed value (so it remains 1.0); others copy from (x-1,y).
// -------------------------------
const computeShaderCode = `
@group(0) @binding(0) var<storage, read> inputBuf: array<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> uSize: vec2<u32>;

@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x: u32 = gid.x;
    let y: u32 = gid.y;
    if (x >= uSize.x || y >= uSize.y) { return; }

    let idx: u32 = y * uSize.x + x;
    var val: f32 = 0.0;
    if (x == 0u) {
        // leftmost column: keep seed value
        val = inputBuf[idx];
    } else {
        // copy the value of the pixel to the left
        let leftIdx: u32 = y * uSize.x + (x - 1u);
        val = inputBuf[leftIdx];
    }

    // write to storage texture (R=G=B=val, A=1)
    textureStore(outputTex, vec2<i32>(i32(x), i32(y)), vec4<f32>(val, val, val, 1.0));
}
`;

// -------------------------------
// WGSL: graphics shader (inline)
// - binding 0: sampled texture (texture_2d<f32>)
// - binding 1: sampler
// Vertex: fullscreen triangle; Fragment: sample texture and output color
// -------------------------------
const graphicsShaderCode = `
@group(0) @binding(0) var myTex: texture_2d<f32>;
@group(0) @binding(1) var mySampler: sampler;

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>
};

@vertex
fn vertex_main(@builtin(vertex_index) vi: u32) -> VSOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    var out: VSOut;
    out.pos = vec4<f32>(positions[vi], 0.0, 1.0);
    out.uv = positions[vi] * 0.5 + vec2<f32>(0.5, 0.5);
    return out;
}

@fragment
fn fragment_main(in: VSOut) -> @location(0) vec4<f32> {
    // sample the texture written by the compute shader
    let c = textureSample(myTex, mySampler, in.uv);
    return c;
}
`;

// -------------------------------
// Create shader modules
// -------------------------------
const computeModule = device.createShaderModule({ code: computeShaderCode });
const graphicsModule = device.createShaderModule({ code: graphicsShaderCode });

// -------------------------------
// Compute pipeline + bind group
// -------------------------------
const computeBindGroupLayout = device.createBindGroupLayout({
    entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba8unorm" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ]
});

const computePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
    compute: { module: computeModule, entryPoint: "main" }
});

const computeBindGroup = device.createBindGroup({
    layout: computeBindGroupLayout,
    entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: outputTexture.createView() },
        { binding: 2, resource: { buffer: sizeBuffer } },
    ]
});

// -------------------------------
// Graphics pipeline + bind group
// -------------------------------
const graphicsBindGroupLayout = device.createBindGroupLayout({
    entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
    ]
});

const renderPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [graphicsBindGroupLayout] }),
    vertex: { module: graphicsModule, entryPoint: "vertex_main", buffers: [] },
    fragment: { module: graphicsModule, entryPoint: "fragment_main", targets: [{ format: canvasFormat }] },
    primitive: { topology: "triangle-list" }
});

const graphicsBindGroup = device.createBindGroup({
    layout: graphicsBindGroupLayout,
    entries: [
        { binding: 0, resource: outputTexture.createView() },
        { binding: 1, resource: sampler },
    ]
});

// -------------------------------
// Run compute pass (single step)
// -------------------------------
{
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(computePipeline);
    pass.setBindGroup(0, computeBindGroup);

    const wgX = 16;
    const wgY = 16;
    const dispatchX = Math.ceil(width / wgX);
    const dispatchY = Math.ceil(height / wgY);
    pass.dispatchWorkgroups(dispatchX, dispatchY);
    pass.end();

    device.queue.submit([encoder.finish()]);
}

// -------------------------------
// Render the texture to the canvas
// -------------------------------
{
    const encoder = device.createCommandEncoder();
    const view = context.getCurrentTexture().createView();
    const rpd: GPURenderPassDescriptor = {
        colorAttachments: [
            {
                view,
                loadOp: "clear",
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                storeOp: "store",
            }
        ]
    };

    const pass = encoder.beginRenderPass(rpd);
    pass.setPipeline(renderPipeline);
    pass.setBindGroup(0, graphicsBindGroup);
    pass.draw(3);
    pass.end();

    device.queue.submit([encoder.finish()]);
}

console.log("Compute+render submitted. Left column seeded=1, others copied from left.");