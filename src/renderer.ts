import { createRenderBindGroups, type RenderBindGroups } from "./bindGroups";
import { createRenderBuffers, type RenderBuffers, type SimBuffers } from "./buffers";
import type { SimConfig } from "./config";
import type { GPUCommandSource } from "./main";
import { createRenderPipelines, type RenderPipelines } from "./pipelines";


export class Renderer implements GPUCommandSource {
    // gpu device and canvas context
    private readonly device: GPUDevice;
    private readonly context: GPUCanvasContext;

    // sim config
    private readonly config: SimConfig;

    // GPU buffers, pipelines, and bind groups
    private buffers: RenderBuffers;
    private pipelines: RenderPipelines;
    private bindGroups: RenderBindGroups;

    public constructor(device: GPUDevice, context: GPUCanvasContext, simConfig: SimConfig,simBuffers: SimBuffers, canvasFormat: GPUTextureFormat) {
        this.device = device;
        this.context = context;

        this.config = simConfig;

        // GPU buffers, pipelines, and bind groups
        this.buffers = createRenderBuffers(simBuffers);
        this.pipelines = createRenderPipelines(device, canvasFormat);
        this.bindGroups = createRenderBindGroups(device, this.buffers, this.pipelines);
    }

    public getCommands(): GPUCommandBuffer {
        const commandEncoder = this.device.createCommandEncoder();
        const textureView = this.context.getCurrentTexture().createView();
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

        renderPass.setPipeline(this.pipelines.render);
        renderPass.setBindGroup(0, this.bindGroups.render);
        renderPass.draw(6, this.config.numBodies, 0, 0);
        renderPass.end();

        return commandEncoder.finish();
    }
}