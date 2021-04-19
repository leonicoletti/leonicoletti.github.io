// main.js

function setup() {

    let sizeRange = document.getElementById("size-range");
    let sizeLabel = document.getElementById("size-label");
    let viscRange = document.getElementById("visc-range");
    let viscLabel = document.getElementById("visc-label");
    let desktopControls = document.getElementById("desktopcontrols");
    let mobileControls = document.getElementById("mobilecontrols");
    let renderWrapper = document.getElementById("render-wrapper");

    let mobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

    this.simulationTime = new Int32Array(60);
    this.renderTime = new Int32Array(60);
    this.frame = 0;
    this.lastFrameTime = Date.now();
    this.stepsToSimulate = 0;
    this.speedSound = 2; // (m/s) Assume simulation width = 1 meter

    this.deltaU = 0.0001;
    this.interactMode = document.querySelector('input[name="interactmode-radio"]:checked').value;
    this.drawing = false;
    this.mPos = new PIXI.Point(0, 0);
    this.mPosPrev = new PIXI.Point(0, 0);

    this.colorscale = 1.0 / this.deltaU;
    this.rendermode = document.querySelector('input[name="rendermode-radio"]:checked').value;


    this.fluidGrid = new GasSim(Number(sizeRange.value), Number(sizeRange.value), Number(viscRange.value));
    this.fluidGraphics = new PIXI.Graphics();

    // Hide certain controls
    if (mobile) {
        desktopControls.style.display = "none";
    } else {
        mobileControls.style.display = "none";
    }

    // Resize renderer on window resize
    let resize = () => {
        let width = renderWrapper.clientWidth;
        let height = renderWrapper.clientHeight;

        let lesser = Math.max(Math.min(width, height), 1);
        app.renderer.resize(lesser, lesser);

        this.gridscale = lesser / Math.max(this.fluidGrid.width, this.fluidGrid.height);
        this.fluidGraphics.x = 0;
        this.fluidGraphics.y = this.fluidGrid.height * this.gridscale;
        this.fluidGraphics.scale.x = this.gridscale;
        this.fluidGraphics.scale.y = -this.gridscale;
    };
    resize();
    window.addEventListener('resize', resize);
    window.addEventListener('fullscreen', resize);

    // Radio button controls
    document.getElementsByName("interactmode-radio").forEach((element) => {
        element.onclick = element.ontouchstart = () => {
            this.interactMode = element.value;
        }
    });

    document.getElementsByName("rendermode-radio").forEach((element) => {
        element.onclick = element.ontouchstart = () => {
            this.rendermode = element.value;
        }
    });

    // Slider controls
    viscRange.oninput = () => {
        let value = viscRange.value;
        this.fluidGrid.setViscosity(Number(value));
        viscLabel.innerText = "Viscosity (" + parseFloat(value).toFixed(3) + ")";
    };
    sizeRange.oninput = () => {
        let value = sizeRange.value;
        sizeLabel.innerText = "Grid size (" + value + "x" + value + ")";
        this.fluidGrid = new GasSim(Number(value), Number(value), this.fluidGrid.viscosity());
        resize();
    };

    // Simulation interactivity
    this.fluidGraphics.interactive = true;
    let trackPosition = (e) => {
        let pos = e.data.getLocalPosition(this.fluidGraphics);
        this.mPos = {x: pos.x, y: pos.y};
    };
    let updateInteractMode = (e) => {
        if (e.data.buttons === 2) {
            let mx = Math.floor(this.mPos.x);
            let my = Math.floor(this.mPos.y);
            this.interactMode = this.fluidGrid.obst(mx, my) === 0 ? "draw" : "erase";
            this.drawing = true;
        } else if (e.data.buttons === 1) {
            this.interactMode = "drag";
            this.drawing = true;
        }
    };
    this.fluidGraphics.on('pointermove', trackPosition);
    this.fluidGraphics.on('pointerdown', trackPosition);
    this.fluidGraphics.on('pointerdown', () => { this.drawing = true; });
    this.fluidGraphics.on('mousedown', updateInteractMode);
    this.fluidGraphics.on('rightdown', updateInteractMode);
    this.fluidGraphics.on('pointerup', () => { this.drawing = false; });
    this.fluidGraphics.on('pointerupoutside', () => { this.drawing = false; });

    app.stage.addChild(this.fluidGraphics);
}

// Lattice vectors TODO: find a better place for these
let lv = [[0, 0], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];

function update() {
    let now = Date.now();
    let delta = now - this.lastFrameTime;

    // User interaction
    let w = 0.1;
    this.mPosPrev = {
        x: this.mPosPrev.x * (1 - w) + this.mPos.x * w,
        y: this.mPosPrev.y * (1 - w) + this.mPos.y * w
    };

    let mx = Math.floor(this.mPos.x);
    let my = Math.floor(this.mPos.y);
    if (this.drawing && !(mx < 0 || mx >= this.fluidGrid.width || my < 0 || my >= this.fluidGrid.height)) {
        switch (this.interactMode) {
            case "draw":
                this.fluidGrid.setObst(mx, my, 1);
                break;
            case "erase":
                this.fluidGrid.setObst(mx, my, 0);
                break;
            case "drag":
                // We may want to replace this with a function inside the simulator
                // It's weird to need to directly modify DFs
                let dMPos = [(this.mPos.x - this.mPosPrev.x) / this.fluidGrid.width, (this.mPos.y - this.mPosPrev.y) / this.fluidGrid.width];
                let d = this.fluidGrid.rho(mx, my) * this.deltaU * (Math.sqrt(dot(dMPos, dMPos)) + 0.1) * this.speedSound * (this.fluidGrid.width / 50.);
                let dr = scale(norm(dMPos), d);
                let flow = 0;
                for (let i = 1; i < 9; i++) {
                    let val;
                    if (dot(dMPos, dMPos) > 0.0005)
                        val = Math.max(dot(dr, lv[i]) / dot(lv[i], lv[i]), 0);
                    else val = d;
                    this.fluidGrid.df[(mx + my * this.fluidGrid.width) * 9 + i] += val;
                    flow += val;
                }
                this.fluidGrid.df[(mx + my * this.fluidGrid.width) * 9] -= flow;
                break;
            default:
                break;
        }
    }

    // Run simulation
    let startTime = Date.now();
    this.stepsToSimulate += this.speedSound * this.fluidGrid.width * (delta / 1000.0);
    let steps = Math.min(Math.floor(this.stepsToSimulate), 3);
    if (steps > 0) {
        this.fluidGrid.simulate(steps);
        this.stepsToSimulate -= steps;
    }
    this.simulationTime[this.frame % 60] = Date.now() - startTime;

    // Draw graphics
    startTime = Date.now();
    if (steps > 0)
        drawSimulation();
    this.renderTime[this.frame % 60] = Date.now() - startTime;

    // Performance stats
    if (this.frame % 300 === 0) {
        let simMS = avg(this.simulationTime);
        let renMS = avg(this.renderTime);
        document.getElementById("simulationTime").innerText = "" + simMS.toFixed(1) + " ms/" + renMS.toFixed(1) + "ms";
    }
    this.lastFrameTime = now;
    this.frame++;

    // Request next frame
    requestAnimationFrame(update);
}

function drawSimulation() {
    this.fluidGraphics.clear();

    // Draw background
    this.fluidGraphics.beginFill(0x111111);
    this.fluidGraphics.drawRect(0, 0, this.fluidGrid.width, this.fluidGrid.height);
    this.fluidGraphics.endFill();

    let getValue;
    switch (this.rendermode) {
        default:
        case "density":
            getValue = (x, y) => {return (this.fluidGrid.rho(x, y) - 1) * this.colorscale};
            break;
        case "curl":
            getValue = (x, y) => {return this.fluidGrid.curl(x, y) * this.colorscale};
            break;
        case "speed":
            getValue = (x, y) => {
                let u = this.fluidGrid.u(x, y);
                return (Math.sqrt(u[0] * u[0] + u[1] * u[1])) * this.colorscale - 0.5;
            };
            break;
    }

    // Draw obstacles and fluid densities
    for (let x = 0; x < this.fluidGrid.width; x++) {
        for (let y = 0; y < this.fluidGrid.height; y++) {
            this.fluidGraphics.lineStyle(0);
            if (this.fluidGrid.obst(x, y)) {
                this.fluidGraphics.beginFill(0xeeeeee);
                this.fluidGraphics.drawRect(x, y, 1, 1);
                this.fluidGraphics.endFill();
            } else {
                let value = getValue(x, y);
                let bright = 0.9;

                let color = rgb2hex([0, value * (1 + bright) + bright, -value * (1 + bright) + bright]);
                this.fluidGraphics.beginFill(color);
                this.fluidGraphics.drawRect(x, y, 1, 1);
                this.fluidGraphics.endFill();
            }
        }
    }
}

setup();
update();