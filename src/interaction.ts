import type { Renderer } from "./renderer";
import type { Simulation } from "./simulation";


export class InteractionController {
    private readonly canvas: HTMLCanvasElement;
    private readonly sim: Simulation;
    private readonly renderer: Renderer;

    private interactionMode: "camera" | "body" = "camera";
    private userBodySliderValue: number = 0.0;
    private isMouseDown: boolean = false;
    private lastMouseCanvasPos: [number, number] = [0, 0];
    
    public constructor(canvas: HTMLCanvasElement, sim: Simulation, renderer: Renderer) {
        this.canvas = canvas;
        this.sim = sim;
        this.renderer = renderer;

        // set up event listeners
        this.addScrollListener();
        this.addDragListener();
        this.addResizeListener();
        this.addUserMassListeners();
        this.addNumBodiesListeners();
    }

    private clientToCanvasCoords(clientX: number, clientY: number): [number, number] {
        const rect = this.canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const canvasX = (clientX - rect.left) * dpr;
        const canvasY = (clientY - rect.top) * dpr;
        return [canvasX, canvasY];
    }

    private addScrollListener() {
        // Scrolling for zooming
        this.canvas.addEventListener("wheel", (e) => {
        e.preventDefault();

        const zoomSpeed = 0.0015;
        const zoomFactor = Math.exp(e.deltaY * zoomSpeed);

        const [canvasX, canvasY] = this.clientToCanvasCoords(e.clientX, e.clientY);
        this.renderer.zoomCamera(zoomFactor, canvasX, canvasY);
        }, {passive: false})
    }

    private addDragListener() {
        this.canvas.addEventListener("pointerdown", (e) => {
            if (e.button !== 0) {
                return; // must be left mouse button
            }
            this.isMouseDown = true;
            this.lastMouseCanvasPos = this.clientToCanvasCoords(e.clientX, e.clientY);
            console.log("Pointer down at ", this.lastMouseCanvasPos[0], this.lastMouseCanvasPos[1]);
            console.log("World pos: ", this.renderer.canvasPxToWorld(this.lastMouseCanvasPos[0], this.lastMouseCanvasPos[1]));
            this.canvas.setPointerCapture(e.pointerId);
        });

        this.canvas.addEventListener("pointermove", (e) => {
            if (!this.isMouseDown) {
                return;
            }
            const [canvasX, canvasY] = this.clientToCanvasCoords(e.clientX, e.clientY);
            const deltaX = canvasX - this.lastMouseCanvasPos[0];
            const deltaY = canvasY - this.lastMouseCanvasPos[1];
            this.lastMouseCanvasPos = [canvasX, canvasY];

            if (this.interactionMode === "camera") {
                this.renderer.panCamera(deltaX, deltaY);
            } else {
                const [worldX, worldY] = this.renderer.canvasPxToWorld(canvasX, canvasY);
                this.sim.setUserBodyPos(worldX, worldY);
                this.renderer.updateMetadataBuffer();
            }
        });

        const endPan = (e: PointerEvent) => {
            if (!this.isMouseDown) {
                return;
            }
            this.isMouseDown = false;
            try {
                this.canvas.releasePointerCapture(e.pointerId);
            } catch {
                // do nothing
            }
        }

        this.canvas.addEventListener("pointerup", endPan);
        this.canvas.addEventListener("pointercancel", endPan);
    }

    private addUserMassListeners() {
        const modeRadios = document.getElementsByName("mode") as NodeListOf<HTMLInputElement>;
        modeRadios.forEach(radio => {
            radio.addEventListener("change", (_e) => {
                if (radio.checked) {
                    if (radio.value === "camera") {
                        this.interactionMode = "camera";
                        this.sim.setUserBodyMass(0.0);
                    } else {
                        this.interactionMode = "body";
                        this.sim.setUserBodyMass(this.userBodySliderValue);
                    }
                }
            });
        });

        const massSlider = document.getElementById("massSlider") as HTMLInputElement;
        const massValue = document.getElementById("massValue") as HTMLSpanElement;

        massSlider.addEventListener("input", (_e) => {
            const sliderVal = parseFloat(massSlider.value);
            const a = 10; // larger => steeper ends, flatter center
            const value = Math.sinh(a * sliderVal) / Math.sinh(a) * 100_000;
            massValue.textContent = value.toFixed(1);
            this.userBodySliderValue = value;
            if (this.interactionMode === "body") {
                this.sim.setUserBodyMass(this.userBodySliderValue);
            }
        });

        window.addEventListener("keydown", (e) => {
            if (e.key === "v") {
                if (this.interactionMode === "body") {
                    this.interactionMode = "camera";
                    this.sim.setUserBodyMass(0.0);
                    modeRadios.forEach(radio => {
                        if (radio.value === "camera") {
                            radio.checked = true;
                        }
                    });
                } else {
                    this.interactionMode = "body";
                    this.sim.setUserBodyMass(this.userBodySliderValue);
                    modeRadios.forEach(radio => {
                        if (radio.value === "body") {
                            radio.checked = true;
                        }
                    });
                }
            }
        });
    }

    private addNumBodiesListeners() {
        const numBodiesSelect = document.getElementById("numBodiesSelect") as HTMLInputElement;
        numBodiesSelect.addEventListener("change", () => {
            const numBodies = parseInt(numBodiesSelect.value, 10);
            if (Number.isNaN(numBodies) || numBodies <= 0) {
                return;
            }
            this.sim.setNumBodies(numBodies);
        });
    }

    private addResizeListener() {
        window.addEventListener("resize", () => {
            this.renderer.resizeCanvasToDisplaySize();
        });
    }
}