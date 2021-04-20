
function greenBlueCM(v, bright = 0.9) {
    return rgb2hex([0, v * (1 + bright) + bright, -v * (1 + bright) + bright]);
}

function rainbowCM(v) {
    v = clamp(v, 0, 1);
    return hsl2hex(0.66 * (1 - v), 1, 0.5);
}

COLOR_MAP = rainbowCM
BC_COLOR = PIXI.utils.rgb2hex([1, 1, 1]);

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

    this.interactMode = document.querySelector('input[name="interactmode-radio"]:checked').value;
    this.rendermode = document.querySelector('input[name="rendermode-radio"]:checked').value;
    this.drawing = false; // Interactive mode.
    this.mPos = new PIXI.Point(0, 0);
    this.mPosPrev = new PIXI.Point(0, 0);

    // Simulation parameters.
    this.SoS = 2; // Speed of sound, in m/s, with width = 1 meter.
    this.forcedVelocity = 0.0001; // Forced velocity.
    this.colorScale = 1.0 / this.forcedVelocity;
    this.sim = new GasSimulator(Number(sizeRange.value), Number(sizeRange.value), Number(viscRange.value));
    this.gfx = new PIXI.Graphics();

    if (mobile) desktopControls.style.display = "none";
    else mobileControls.style.display = "none";

    let resize = () => {
        let width = renderWrapper.clientWidth;
        let height = renderWrapper.clientHeight;

        let lesser = Math.max(Math.min(width, height), 1);
        app.renderer.resize(lesser, lesser);

        this.gridscale = lesser / Math.max(this.sim.width, this.sim.height);
        this.gfx.x = 0;
        this.gfx.y = this.sim.height * this.gridscale;
        this.gfx.scale.x = this.gridscale;
        this.gfx.scale.y = -this.gridscale;
    };
    resize();
    window.addEventListener('resize', resize);
    window.addEventListener('fullscreen', resize);

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

    // Slider controls.
    viscRange.oninput = () => {
        let value = viscRange.value;
        this.sim.setViscosity(Number(value));
        viscLabel.innerText = "Viscosity: " + parseFloat(value).toFixed(3);
    };
    sizeRange.oninput = () => {
        let value = sizeRange.value;
        sizeLabel.innerText = "Grid: " + value + "x" + value;
        this.sim = new GasSimulator(Number(value), Number(value), this.sim.viscosity());
        resize();
    };

    // Simulation interactivity.
    this.gfx.interactive = true;
    let trackPosition = (e) => {
        let pos = e.data.getLocalPosition(this.gfx);
        this.mPos = {x: pos.x, y: pos.y}
    };
    let updateInteractMode = (e) => {
        if (e.data.buttons === 2) {
            let mx = Math.floor(this.mPos.x);
            let my = Math.floor(this.mPos.y);
            this.interactMode = this.sim.bc(mx, my) === 0 ? "draw" : "erase";
            this.drawing = true;
        } else if (e.data.buttons === 1) {
            this.interactMode = "drag";
            this.drawing = true;
        }
    };
    this.gfx.on('pointermove', trackPosition);
    this.gfx.on('pointerdown', trackPosition);
    this.gfx.on('pointerdown', () => { this.drawing = true; });
    this.gfx.on('mousedown', updateInteractMode);
    this.gfx.on('rightdown', updateInteractMode);
    this.gfx.on('pointerup', () => { this.drawing = false; });
    this.gfx.on('pointerupoutside', () => { this.drawing = false; });

    app.stage.addChild(this.gfx);
}

function update() {
    let now = Date.now();
    let dt = now - this.lastFrameTime;
    let w = 0.1;
    this.mPosPrev = {
        x: this.mPosPrev.x * (1 - w) + this.mPos.x * w,
        y: this.mPosPrev.y * (1 - w) + this.mPos.y * w};

    let mx = Math.floor(this.mPos.x);
    let my = Math.floor(this.mPos.y);
    if (this.drawing && !(mx < 0 || mx >= this.sim.width || my < 0 || my >= this.sim.height)) {
        switch (this.interactMode) {
            case "draw":
                this.sim.setBc(mx, my, 1);
                break;
            case "erase":
                this.sim.setBc(mx, my, 0);
                break;
            case "drag":
                this.sim.push(this.mPosPrev, this.mPos, mx, my, this.SoS, forcedVelocity)
                break;
            default:
                break;
        }
    }

    // Run simulation.
    let startTime = Date.now();
    this.stepsToSimulate += this.SoS * this.sim.width * (dt / 1000.0);
    let steps = Math.min(Math.floor(this.stepsToSimulate), 3);
    if (steps > 0) {
        this.sim.simulate(steps);
        this.stepsToSimulate -= steps;
    }
    this.simulationTime[this.frame % 60] = Date.now() - startTime;

    // Draw graphics.
    startTime = Date.now();
    if (steps > 0) drawSimulation();
    this.renderTime[this.frame % 60] = Date.now() - startTime;

    // Performance stats.
    if (this.frame % 300 === 0) {
        let simMS = avg(this.simulationTime);
        let renMS = avg(this.renderTime);
        document.getElementById("stats").innerText = "Simulation: " + simMS.toFixed(1) + "ms, rendering: " + renMS.toFixed(1) + "ms";
    }
    this.lastFrameTime = now;
    this.frame++;
    requestAnimationFrame(update); // Request next frame.
}

function drawSimulation() {
    this.gfx.clear();
    this.gfx.beginFill(0x111111);
    this.gfx.drawRect(0, 0, this.sim.width, this.sim.height);
    this.gfx.endFill();

    let getValue;
    switch (this.rendermode) {
        default:
        case "density":
            getValue = (x, y) => {return (this.sim.rho(x, y) - 1) * this.colorScale};
            break;
        case "curl":
            getValue = (x, y) => {return this.sim.curl(x, y) * this.colorScale};
            break;
        case "pressure":
            getValue = (x, y) => {return this.sim.q(x, y) * this.colorScale * this.colorScale};
            break;
        case "speed":
            getValue = (x, y) => {
                let u = this.sim.u(x, y);
                return (Math.sqrt(u[0] * u[0] + u[1] * u[1])) * this.colorScale - 0.5;
            };
            break;
    }

    for (let x = 0; x < this.sim.width; x++) {
        for (let y = 0; y < this.sim.height; y++) {
            this.gfx.lineStyle(0);
            if (this.sim.bc(x, y)) {
                this.gfx.beginFill(BC_COLOR);
                this.gfx.drawRect(x, y, 1, 1);
                this.gfx.endFill();
            } else {
                let v = getValue(x, y);
                let c = COLOR_MAP(v);
                this.gfx.beginFill(c);
                this.gfx.drawRect(x, y, 1, 1);
                this.gfx.endFill();
            }
        }
    }
}

setup();
update();
