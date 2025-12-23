import { readBufferData } from "./buffers";
import { initDeviceAndContext } from "./gpu";
import { InteractionController } from "./interaction";
import { Renderer } from "./renderer";
import { Simulation } from "./simulation";

export interface GPUCommandSource {
    getCommands(): GPUCommandBuffer;
}

async function main() {
    const { device, context, canvasFormat, canvas } = await initDeviceAndContext("webgpu-canvas");

    const sim = new Simulation(device, canvas);
    const renderer = new Renderer(device, context, sim.getConfig(),sim.getBuffers(), canvasFormat);
    const interaction = new InteractionController(device, canvasFormat, canvas, context, sim);  

    // let lastTime = performance.now();
    function frame() {
        device.queue.submit([
            sim.getCommands(),
            renderer.getCommands(),
        ]);
        interaction.sendUpdateToGPU();
        requestAnimationFrame(frame);

        // const currentTime = performance.now();
        // const deltaTime = currentTime - lastTime;
        // lastTime = currentTime;
        // if (deltaTime > 100) {
        //     // debug, remove later
        //     readBufferData(device, sim.getBuffers().mortonCodes, 4 * sim.getConfig().numBodies, Uint32Array).then((data) => {
        //         console.log("Morton Codes:", data);
        //     });

        //     readBufferData(device, sim.getBuffers().nodeData, 12 * 4 * (2 * sim.getConfig().numBodies - 1), Uint32Array).then((data) => {
        //         for (let i = 0; i < 2 * sim.getConfig().numBodies - 1; i++) {
        //             const baseIdx = i * 12;
        //             const leftChild = data[baseIdx + 8];
        //             const rightChild = data[baseIdx + 9];
        //             const parent = data[baseIdx + 10];
        //             console.log(`Node ${i}: leftChild=${leftChild}, rightChild=${rightChild}, parent=${parent}`);
        //         }
        //     });
        // } else {
        //     requestAnimationFrame(frame);
        // }
        
    }
    requestAnimationFrame(frame);
}

await main();