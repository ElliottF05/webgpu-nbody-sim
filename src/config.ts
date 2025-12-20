export type SimConfig = Readonly<{
    numBodies: number;
    gravConstant: number;
    maxMass: number;
    minStepsPerOrbit: number;
    substeps: number;
    viewPort: [number, number];
    deltaTime: number;
    epsilonMultiplier: number;
}>;

export function getDefaultSimConfig(canvas: HTMLCanvasElement): SimConfig {
    const numBodies = 2000;
    const gravConstant = 1.0;
    const maxMass = 1.0;
    const minStepsPerOrbit = 50;
    const substeps = 10;
    const viewPort: [number, number] = [canvas.width, canvas.height];
    const deltaTime = 0.1 * 1.0 / (60.0 * substeps);
    const epsilonMultiplier = 1.0;

    return {
        numBodies,
        gravConstant,
        maxMass,
        minStepsPerOrbit,
        substeps,
        viewPort,
        deltaTime,
        epsilonMultiplier,
    };
}