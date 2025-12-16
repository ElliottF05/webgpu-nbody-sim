import computeShaderCode from "./shaders/compute.wgsl?raw";
import renderShaderCode from "./shaders/render.wgsl?raw";

// Initialize WebGPU
const adapter = (await navigator.gpu.requestAdapter())!;
const device = await adapter.requestDevice();

// Setup canvas context
const canvas = document.getElementById("webgpu-canvas") as HTMLCanvasElement;
const context = canvas.getContext("webgpu") as GPUCanvasContext;
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device, format: canvasFormat, alphaMode: "premultiplied" });

// Canvas dimensions
const width = canvas.width;
const height = canvas.height;

// Simulation parameters
const numBodies = 2000;
const gravConstant = 1.0;
const maxMass = 1.0;
const minStepsPerOrbit = 50;
const substeps = 10;
const initCamCenter = [0.0, 0.0];
const initCamHalfSize = [10.0, 10.0];
const viewPort = [width, height]

// Derived values
const deltaTime = 0.1 * 1.0 / (60.0 * substeps)
const epsilon = 1.0 * Math.pow(gravConstant * 2 * maxMass * (minStepsPerOrbit * deltaTime / (2 * Math.PI)) * (minStepsPerOrbit * deltaTime / (2 * Math.PI)), 1.0 / 3.0);

// Dynamic values
let camCenter = initCamCenter;
let camHalfSize = initCamHalfSize;

// ----- Initialize GPU buffers -----

function createBuffer(initialData: Uint32Array | Float32Array): GPUBuffer {
  const buffer = device.createBuffer({
    size: initialData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(buffer, 0, initialData.buffer, initialData.byteOffset, initialData.byteLength);
  return buffer;
}
function createMetadataBuffer(initialData: Uint32Array | Float32Array): GPUBuffer {
  const buffer = device.createBuffer({
    size: initialData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, initialData.buffer, initialData.byteOffset, initialData.byteLength);
  return buffer;
}

// Metadata buffers
const uintMetadata = new Uint32Array([numBodies]); // num_bodies
const floatMetadata = new Float32Array([gravConstant, deltaTime, epsilon, 0.0, ...camCenter, ...camHalfSize, ...viewPort]);

const uintMetadataBuffer = createMetadataBuffer(uintMetadata);
const floatMetadataBuffer = createMetadataBuffer(floatMetadata);

function syncFloatMetadata() {
  floatMetadata[4] = camCenter[0];
  floatMetadata[5] = camCenter[1];
  floatMetadata[6] = camHalfSize[0];
  floatMetadata[7] = camHalfSize[1];

  device.queue.writeBuffer(
    floatMetadataBuffer,
    0,
    floatMetadata.buffer,
    floatMetadata.byteOffset,
    floatMetadata.byteLength
  )
}

// Mass buffer
const mass = new Float32Array(numBodies).fill(1.0);
const massBuffer = createBuffer(mass);


// Pos buffers
const initPos = new Float32Array(2 * numBodies).fill(0.0);
for (let i = 0; i < numBodies; i++) {
  const x = (Math.random() - 0.5) * 15;
  const y = (Math.random() - 0.5) * 15;
  initPos[2*i] = x;
  initPos[2*i + 1] = y;
}
const posBuffer = createBuffer(initPos);


// Velocity buffers
const initVel = new Float32Array(2 * numBodies).fill(0.0);
for (let i = 0; i < numBodies; i++) {
  const x = (Math.random() - 0.5) * 1;
  const y = (Math.random() - 0.5) * 1;
  initVel[2*i] = x;
  initVel[2*i + 1] = y;
}
const velBuffer = createBuffer(initVel);

// ----- Create compute/graphics pipelines -----
const computeShaderModule = device.createShaderModule({ code: computeShaderCode });
const renderShaderModule = device.createShaderModule({ code: renderShaderCode });

// // metadata buffers
// @group(0) @binding(0) var<uniform> float_metadata: FloatMetadata;
// @group(0) @binding(1) var<uniform> uint_metadata: UintMetadata;

// // data buffers
// @group(0) @binding(2) var<storage, read_write> mass_buf: array<f32>;
// @group(0) @binding(3) var<storage, read_write> pos_buf: array<vec2<f32>>;
// @group(0) @binding(4) var<storage, read_write> vel_buf: array<vec2<f32>>;

// Compute pipelines
const halfVelocityStepPipeline = device.createComputePipeline({
  layout: "auto",
  compute: {
    module: computeShaderModule,
    entryPoint: "half_vel_step_main",
  },
})
const halfVelocityStepBindGroup = device.createBindGroup({
  layout: halfVelocityStepPipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: floatMetadataBuffer } },
    { binding: 1, resource: { buffer: uintMetadataBuffer } },
    { binding: 2, resource: { buffer: massBuffer } },
    { binding: 3, resource: { buffer: posBuffer } },
    { binding: 4, resource: { buffer: velBuffer } },
  ]
})

const posStepPipeline = device.createComputePipeline({
  layout: "auto",
  compute: {
    module: computeShaderModule,
    entryPoint: "pos_step_main",
  },
})
const posStepBindGroup = device.createBindGroup({
  layout: posStepPipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: floatMetadataBuffer } },
    { binding: 1, resource: { buffer: uintMetadataBuffer } },
    // { binding: 2, resource: { buffer: massBuffer } },
    { binding: 3, resource: { buffer: posBuffer } },
    { binding: 4, resource: { buffer: velBuffer } },
  ]
})


// Render pipeline
const renderPipeline = device.createRenderPipeline({
  layout: "auto",
  vertex: {
    module: renderShaderModule,
    entryPoint: "vertex_main",
  },
  fragment: {
    module: renderShaderModule,
    entryPoint: "fragment_main",
    targets: [{ format: canvasFormat }],
  },
  primitive: {
    topology: "triangle-list",
  },
});

const renderBindGroup = device.createBindGroup({
  layout: renderPipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: floatMetadataBuffer } },
    { binding: 1, resource: { buffer: uintMetadataBuffer } },
    { binding: 2, resource: { buffer: posBuffer } },
  ],
});


// ----- Interactivity -----
function canvasPixelToWorldPos(clientX: number, clientY: number) {
  const rect = canvas.getBoundingClientRect();

  // pixel coords relative to canvas
  const px = clientX - rect.left;
  const py = clientY - rect.top;

  // normalized to [-1,1]
  const u = (2 * px / rect.width) - 1;
  const v = (2 * py / rect.height) - 1;

  // to world pos
  const worldX = u * camHalfSize[0] + camCenter[0];
  const worldY = v * camHalfSize[1] + camCenter[1];

  return [worldX, worldY];
}

// Scrolling for zooming
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();

  // position before zoom
  const [x1, y1] = canvasPixelToWorldPos(e.clientX, e.clientY);

  const zoomSpeed = 0.0015;
  const zoomFactor = Math.exp(e.deltaY * zoomSpeed);

  // clamp
  const minHalfSize = 0.01;
  const maxHalfSize = 1e4;
  camHalfSize[0] = Math.min(maxHalfSize, Math.max(minHalfSize, camHalfSize[0] * zoomFactor));

  // keep aspect ratio
  const aspect = viewPort[0] / viewPort[1];
  camHalfSize[1] = camHalfSize[0] / aspect;

  // get position after zoom
  const [x2, y2] = canvasPixelToWorldPos(e.clientX, e.clientY);

  // shift camera
  camCenter[0] += x1 - x2;
  camCenter[1] += y2 - y1;
}, {passive: false})


// Click and drag for panning
let isPanning = false;
let lastClientPos = [0,0];

canvas.addEventListener("pointerdown", (e) => {
  if (e.button !== 0) {
    return; // must be left moust button
  }
  isPanning = true;
  lastClientPos = [e.clientX, e.clientY];
  canvas.setPointerCapture(e.pointerId);
})

canvas.addEventListener("pointermove", (e) => {
  if (!isPanning) {
    return;
  }

  const rect = canvas.getBoundingClientRect();
  const dxPixels = e.clientX - lastClientPos[0];
  const dyPixels = e.clientY - lastClientPos[1];
  lastClientPos = [e.clientX, e.clientY];

  const worldPerPixelX = (2 * camHalfSize[0]) / rect.width;
  const worldPerPixelY = (2 * camHalfSize[1]) / rect.height;

  camCenter[0] -= dxPixels * worldPerPixelX;
  camCenter[1] += dyPixels * worldPerPixelY
})

function endPan(e: PointerEvent) {
  if (!isPanning) {
    return;
  }
  isPanning = false;
  try {
    canvas.releasePointerCapture(e.pointerId);
  } catch {
    // do nothing
  }
}

canvas.addEventListener("pointerup", endPan);
canvas.addEventListener("pointercancel", endPan);
canvas.addEventListener("pointerleave", (_e) => { isPanning = false })


// ----- Main simulation loop -----
const workgroupX = 16;
const dispatchX = Math.ceil(numBodies / workgroupX);

function frame() {
  const commandEncoder = device.createCommandEncoder();

  // Compute pass
  const computePass = commandEncoder.beginComputePass();
  for (let i = 0; i < substeps; i++) {
    computePass.setPipeline(halfVelocityStepPipeline);
    computePass.setBindGroup(0, halfVelocityStepBindGroup);
    computePass.dispatchWorkgroups(dispatchX);

    computePass.setPipeline(posStepPipeline);
    computePass.setBindGroup(0, posStepBindGroup);
    computePass.dispatchWorkgroups(dispatchX);

    computePass.setPipeline(halfVelocityStepPipeline);
    computePass.setBindGroup(0, halfVelocityStepBindGroup);
    computePass.dispatchWorkgroups(dispatchX);
  }
  
  computePass.end();

  // Render pass
  const textureView = context.getCurrentTexture().createView();
  const renderPass = commandEncoder.beginRenderPass({
    colorAttachments: [
      {
        view: textureView,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  });

  renderPass.setPipeline(renderPipeline);
  renderPass.setBindGroup(0, renderBindGroup);
  renderPass.draw(6, numBodies, 0, 0);
  renderPass.end();

  device.queue.submit([commandEncoder.finish()]);

  // sync metadata at the end of each frame to get updated camera info
  syncFloatMetadata();

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);