export class GPUManager {
    private adapter: GPUAdapter;
    private device: GPUDevice;

    private constructor(adapter: GPUAdapter, device: GPUDevice) {
        this.adapter = adapter;
        this.device = device;
    }

    public static async initializeGpu(): Promise<GPUManager> {
        const adapter = (await navigator.gpu.requestAdapter())!;
        const device = await adapter.requestDevice();
        return new GPUManager(adapter, device);
    }

    public getDevice(): GPUDevice {
        return this.device;
    }
    public getAdapter(): GPUAdapter {
        return this.adapter;
    }
}

//// ------------------------
//// GPU initialization
//// ------------------------
// Request an adapter and device via a helper (GPUManager).
// This returns an object that wraps navigator.gpu.requestAdapter() and device creation.
// The returned gpuManager exposes getDevice() to access the GPUDevice and its queue.
const gpuManager = await GPUManager.initializeGpu();

//// ------------------------
//// RENDER SHADER (WGSL)
//// ------------------------
// WGSL shader for a simple colored triangle.
// Key WGSL bits:
// - @vertex / @fragment: marks entry points for the pipeline stages.
// - @location(n): user-defined vertex/fragment input/outputs that map to vertex attributes
//   and pipeline target slots.
// - @builtin(position): the special builtin that the vertex shader must write with clip-space position.
const shaderCode = `
struct VertexOut {
  @builtin(position) position : vec4f,   // clip-space position (x, y, z, w)
  @location(0) color : vec4f             // color passed to the fragment shader
}

@vertex
fn vertex_main(@location(0) position: vec4f,  // first vertex attribute (vec4)
               @location(1) color: vec4f)    // second vertex attribute (vec4)
               -> VertexOut
{
  var output : VertexOut;
  output.position = position;  // supply clip-space position
  output.color = color;        // pass color through to fragment stage
  return output;
}

@fragment
fn fragment_main(fragData: VertexOut) -> @location(0) vec4f
{
  // return the interpolated color to the first color target
  return fragData.color;
}`;

const shaderModule = gpuManager.getDevice().createShaderModule({
    code: shaderCode
})

//// ------------------------
//// CANVAS CONFIGURATION
//// ------------------------
const canvas = document.getElementById("webgpu-canvas") as HTMLCanvasElement;
const context = canvas.getContext("webgpu") as GPUCanvasContext;

// navigator.gpu.getPreferredCanvasFormat() returns the swap chain format the platform prefers.
// Important: the same format must be used for the pipeline's fragment target and the canvas.
context.configure({
    device: gpuManager.getDevice(),
    format: navigator.gpu.getPreferredCanvasFormat(),
    alphaMode: "premultiplied" // controls how the canvas alpha is blended (common choice)
});

//// ------------------------
//// VERTEX DATA SETUP
//// ------------------------
// Layout per-vertex: vec4 position (4 floats), vec4 color (4 floats)
// Each vertex therefore has 8 floats (8 * 4 = 32 bytes).
const vertices = new Float32Array([
  // vertex 1: position(x,y,z,w), color(r,g,b,a)
  0.0, 0.6, 0, 1,   1, 0, 0, 1,
  // vertex 2
  -0.5, -0.6, 0, 1,  0, 1, 0, 1,
  // vertex 3
  0.5, -0.6, 0, 1,   0, 0, 1, 1,
]);

// allocate a GPUBuffer sized to hold the vertex array.
// size must be in bytes; use vertices.byteLength to avoid truncation.
const vertexBuffer = gpuManager.getDevice().createBuffer({
    size: vertices.byteLength,
    // VERTEX: used in setVertexBuffer; COPY_DST: allows queue.writeBuffer to copy data into it.
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});

// queue.writeBuffer parameters:
// - destination buffer
// - destination offset (bytes) in the GPU buffer
// - source ArrayBuffer (the underlying buffer of the TypedArray)
// - source offset (bytes) inside the ArrayBuffer
// - size (bytes) to copy
// Passing vertices.byteLength ensures we copy the full contents (not vertices.length).
gpuManager.getDevice().queue.writeBuffer(
    vertexBuffer,           // destination GPUBuffer
    0,                      // destination offset
    vertices.buffer,        // source ArrayBuffer
    0,                      // source offset
    vertices.byteLength     // number of bytes to copy
);

//// ------------------------
//// VERTEX LAYOUT (Pipeline)
//// ------------------------
// Describe how vertex attributes are laid out in the vertex buffer so the GPU can
// map buffer memory to shader @location inputs.
// - shaderLocation: matches @location(N) in WGSL
// - offset: byte offset of the attribute inside a single vertex
// - format: scalar/vec format in bytes (float32x4 = 4 * 4 bytes = 16 bytes)
// - arrayStride: total size of one vertex in bytes (here 32 bytes: vec4 + vec4)
// - stepMode: "vertex" means advance per-vertex (alternatively "instance")
const vertexBuffers: GPUVertexBufferLayout[] = [
    {
        attributes: [
            {
                shaderLocation: 0, // maps to @location(0) position in vertex shader
                offset: 0,         // position starts at byte 0
                format: "float32x4"
            },
            {
                shaderLocation: 1, // maps to @location(1) color in vertex shader
                offset: 16,        // color starts after the position (16 bytes)
                format: "float32x4"
            }
        ],
        arrayStride: 32, // 8 floats * 4 bytes per float = 32 bytes per vertex
        stepMode: "vertex"
    }
];

//// ------------------------
//// RENDER PIPELINE SETUP
//// ------------------------
// The pipeline ties together the shader stages and fixed-function state.
// layout: "auto" lets the implementation derive a pipeline layout from the shader module(s).
const pipelineDescriptor: GPURenderPipelineDescriptor = {
    vertex: {
        module: shaderModule,
        entryPoint: "vertex_main",
        buffers: vertexBuffers
    },
    fragment: {
        module: shaderModule,
        entryPoint: "fragment_main",
        targets: [
            {
                format: navigator.gpu.getPreferredCanvasFormat() // must match context.configure format
            }
        ]
    },
    primitive: {
        topology: "triangle-list" // interpreting vertices as independent triangles
    },
    layout: "auto"
}

const renderPipeline = gpuManager.getDevice().createRenderPipeline(pipelineDescriptor);

//// ------------------------
//// ENCODE A RENDER PASS
//// ------------------------
// create a command encoder, begin a render pass, set pipeline and buffers, submit.
const commandEncoder = gpuManager.getDevice().createCommandEncoder();

const clearColor: GPUColor = { r: 0.5, g: 0.5, b: 0.8, a: 1.0 };
const renderPassDescriptor: GPURenderPassDescriptor = {
    colorAttachments: [
        {
            // clearValue: used when loadOp is "clear"
            clearValue: clearColor,
            loadOp: "clear",            // clear the attachment at the start of the pass
            storeOp: "store",           // store the result after the pass completes
            view: context.getCurrentTexture().createView() // the current swap-chain texture view
        }
    ]
};

const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
passEncoder.setPipeline(renderPipeline);
passEncoder.setVertexBuffer(0, vertexBuffer);
// draw(vertexCount, instanceCount?, firstVertex?, firstInstance?)
// Here: draw 3 vertices (one triangle), firstVertex defaults to 0.
passEncoder.draw(3);

passEncoder.end();
const commandBuffer = commandEncoder.finish();
gpuManager.getDevice().queue.submit([commandBuffer]);
console.log("Submitted draw commands to GPU", commandBuffer);

//// ------------------------
//// COMPUTE SHADER SECTION
//// ------------------------
// Simple compute example that writes values into a storage buffer for readback.

// Number of elements and bytes-per-element (f32 = 4 bytes)
const NUM_ELEMENTS = 1000;
const BUFFER_SIZE = NUM_ELEMENTS * 4; // buffer size in bytes

// WGSL compute shader explanation:
// - @group(0) @binding(0): bind the storage buffer at group=0 binding=0 (matches the bind group created later)
// - var<storage, read_write> output: array<f32>: a GPU-visible storage buffer we can write into.
// - @compute @workgroup_size(64): sets the workgroup size. Each workgroup contains 64 local threads.
// - @builtin(global_invocation_id): the absolute thread index across all workgroups.
// - @builtin(local_invocation_id): the index within the workgroup (0..workgroup_size-1).
const computeShaderCode = `
@group(0) @binding(0)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(
  @builtin(global_invocation_id)
  global_id : vec3u,

  @builtin(local_invocation_id)
  local_id : vec3u,
) {
  // Guard against out-of-bounds writes when the number of dispatched threads exceeds NUM_ELEMENTS
  if (global_id.x >= ${NUM_ELEMENTS}u) {
    return;
  }

  // store a value dependent on the global index and the local id
  output[global_id.x] =
    f32(global_id.x) * 1000. + f32(local_id.x);
}
`;

const computeShaderModule = gpuManager.getDevice().createShaderModule({
    code: computeShaderCode
});

//// ------------------------
//// CREATE BUFFERS FOR COMPUTE
//// ------------------------
// output buffer is a STORAGE buffer (GPU-side) and we'll copy it to CPU-visible staging buffer.
// COPY_SRC on the output buffer lets us copy from it into another buffer.
const output = gpuManager.getDevice().createBuffer({
    size: BUFFER_SIZE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
});

// stagingBuffer is CPU-readable: COPY_DST to receive copied data and MAP_READ to map it for reading.
const stagingBuffer = gpuManager.getDevice().createBuffer({
    size: BUFFER_SIZE,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
});

//// ------------------------
//// BIND GROUP (binding resources to the compute shader)
//// ------------------------
// BindGroupLayout entries must match the shader's @group/@binding annotations.
const bindGroupLayout = gpuManager.getDevice().createBindGroupLayout({
    entries: [
        {
            binding: 0,                          // matches @binding(0) in WGSL
            visibility: GPUShaderStage.COMPUTE,  // shader stages that can see this binding
            buffer: {
                type: "storage"                  // matches var<storage, read_write> in WGSL
            }
        }
    ]
});

const bindGroup = gpuManager.getDevice().createBindGroup({
    layout: bindGroupLayout, // this layout corresponds to WGSL @group(0)
    entries: [
        {
            binding: 0,               // must match the layout.binding & WGSL @binding(0)
            resource: {
                buffer: output        // the GPUBuffer that backs the WGSL 'output' variable
            }
        }
    ]
});

//// ------------------------
//// COMPUTE PIPELINE
//// ------------------------
const computePipeline = gpuManager.getDevice().createComputePipeline({
    layout: gpuManager.getDevice().createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
    }),
    compute: {
        module: computeShaderModule,
        entryPoint: "main"
    }
});

//// ------------------------
//// RUN THE COMPUTE PASS
//// ------------------------
const newCommandEncoder = gpuManager.getDevice().createCommandEncoder();
const computePassEncoder = newCommandEncoder.beginComputePass();

computePassEncoder.setPipeline(computePipeline);
computePassEncoder.setBindGroup(0, bindGroup); // matches @group(0) in WGSL

// dispatchWorkgroups: number of workgroups to launch.
// workgroupSize (64) * numWorkgroups should cover NUM_ELEMENTS.
// Math.ceil ensures we have enough groups; the shader guards out-of-bounds writes.
const workgroupSize = 64;
const numWorkgroups = Math.ceil(NUM_ELEMENTS / workgroupSize);
computePassEncoder.dispatchWorkgroups(numWorkgroups);

computePassEncoder.end();

//// ------------------------
//// READ BACK RESULTS
//// ------------------------
// Copy the GPU output buffer into a CPU-readable staging buffer.
newCommandEncoder.copyBufferToBuffer(
    output,
    0, // source offset (bytes) in the output buffer
    stagingBuffer,
    0, // destination offset (bytes) in the staging buffer
    BUFFER_SIZE // number of bytes to copy
);

// submit compute + copy commands
gpuManager.getDevice().queue.submit([newCommandEncoder.finish()]);

// Map the staging buffer for reading on the CPU.
await stagingBuffer.mapAsync(
    GPUMapMode.READ,
    0, // offset
    BUFFER_SIZE // size in bytes
);

// getMappedRange returns an ArrayBuffer view of the mapped region.
// Copy the data out (slice) before unmapping.
const copyArrayBuffer = stagingBuffer.getMappedRange(0, BUFFER_SIZE);
const data = copyArrayBuffer.slice();
stagingBuffer.unmap();

// Interpret the returned bytes as 32-bit floats on the CPU.
console.log(new Float32Array(data));