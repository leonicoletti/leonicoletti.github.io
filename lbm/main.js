
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

    this.grid = new GasSim(Number(sizeRange.value), Number(sizeRange.value), Number(viscRange.value));
    this.gfx = new PIXI.Graphics();

    if (mobile) {
        desktopControls.style.display = "none";
    } else {
        mobileControls.style.display = "none";
    }

    let resize = () => {
        let width = renderWrapper.clientWidth;
        let height = renderWrapper.clientHeight;

        let lesser = Math.max(Math.min(width, height), 1);
        app.renderer.resize(lesser, lesser);

        this.gridscale = lesser / Math.max(this.grid.width, this.grid.height);
        this.gfx.x = 0;
        this.gfx.y = this.grid.height * this.gridscale;
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

    // Slider controls
    viscRange.oninput = () => {
        let value = viscRange.value;
        this.grid.setViscosity(Number(value));
        viscLabel.innerText = "Viscosity: " + parseFloat(value).toFixed(3);
    };
    sizeRange.oninput = () => {
        let value = sizeRange.value;
        sizeLabel.innerText = "Grid: " + value + "x" + value;
        this.grid = new GasSim(Number(value), Number(value), this.grid.viscosity());
        resize();
    };

    // Simulation interactivity
    this.gfx.interactive = true;
    let trackPosition = (e) => {
        let pos = e.data.getLocalPosition(this.gfx);
        this.mPos = {x: pos.x, y: pos.y};
    };
    let updateInteractMode = (e) => {
        if (e.data.buttons === 2) {
            let mx = Math.floor(this.mPos.x);
            let my = Math.floor(this.mPos.y);
            this.interactMode = this.grid.obst(mx, my) === 0 ? "draw" : "erase";
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

// Lattice vectors TODO: find a better place for these
let lv = [[0, 0], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];

function update() {
    let now = Date.now();
    let delta = now - this.lastFrameTime;
    let w = 0.1;
    this.mPosPrev = {
        x: this.mPosPrev.x * (1 - w) + this.mPos.x * w,
        y: this.mPosPrev.y * (1 - w) + this.mPos.y * w
    };

    let mx = Math.floor(this.mPos.x);
    let my = Math.floor(this.mPos.y);
    if (this.drawing && !(mx < 0 || mx >= this.grid.width || my < 0 || my >= this.grid.height)) {
        switch (this.interactMode) {
            case "draw":
                this.grid.setObst(mx, my, 1);
                break;
            case "erase":
                this.grid.setObst(mx, my, 0);
                break;
            case "drag":
                // We may want to replace this with a function inside the simulator
                // It's weird to need to directly modify DFs
                let dMPos = [(this.mPos.x - this.mPosPrev.x) / this.grid.width, (this.mPos.y - this.mPosPrev.y) / this.grid.width];
                let d = this.grid.rho(mx, my) * this.deltaU * (Math.sqrt(dot(dMPos, dMPos)) + 0.1) * this.speedSound * (this.grid.width / 50.);
                let dr = scale(norm(dMPos), d);
                let flow = 0;
                for (let i = 1; i < 9; i++) {
                    let val;
                    if (dot(dMPos, dMPos) > 0.0005)
                        val = Math.max(dot(dr, lv[i]) / dot(lv[i], lv[i]), 0);
                    else val = d;
                    this.grid.df[(mx + my * this.grid.width) * 9 + i] += val;
                    flow += val;
                }
                this.grid.df[(mx + my * this.grid.width) * 9] -= flow;
                break;
            default:
                break;
        }
    }

    // Run simulation.
    let startTime = Date.now();
    this.stepsToSimulate += this.speedSound * this.grid.width * (delta / 1000.0);
    let steps = Math.min(Math.floor(this.stepsToSimulate), 3);
    if (steps > 0) {
        this.grid.simulate(steps);
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
        document.getElementById("times").innerText = "Simulation: " + simMS.toFixed(1) + "ms, rendering: " + renMS.toFixed(1) + "ms";
    }
    this.lastFrameTime = now;
    this.frame++;

    // Request next frame.
    requestAnimationFrame(update);
}

function drawSimulation() {
    this.gfx.clear();
    this.gfx.beginFill(0x111111);
    this.gfx.drawRect(0, 0, this.grid.width, this.grid.height);
    this.gfx.endFill();

    let getValue;
    switch (this.rendermode) {
        default:
        case "density":
            getValue = (x, y) => {return (this.grid.rho(x, y) - 1) * this.colorscale};
            break;
        case "curl":
            getValue = (x, y) => {return this.grid.curl(x, y) * this.colorscale};
            break;
        case "ke":
            getValue = (x, y) => {
                let ke = this.grid.ke(x, y);
                if (getRandomInt(300) < 2) console.log(ke, this.colorscale)
                return ke * this.colorscale;
            };
            break;
        case "speed":
            getValue = (x, y) => {
                let u = this.grid.u(x, y);
                return (Math.sqrt(u[0] * u[0] + u[1] * u[1])) * this.colorscale - 0.5;
            };
            break;
    }

    for (let x = 0; x < this.grid.width; x++) {
        for (let y = 0; y < this.grid.height; y++) {
            this.gfx.lineStyle(0);
            if (this.grid.obst(x, y)) {
                this.gfx.beginFill(0xeeeeee);
                this.gfx.drawRect(x, y, 1, 1);
                this.gfx.endFill();
            } else {
                let value = getValue(x, y);
                let bright = 0.9;
                let color = rgb2hex([0, value * (1 + bright) + bright, -value * (1 + bright) + bright]);
                this.gfx.beginFill(color);
                this.gfx.drawRect(x, y, 1, 1);
                this.gfx.endFill();
            }
        }
    }
}

setup();
update();
