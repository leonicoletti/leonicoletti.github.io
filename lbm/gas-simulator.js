
// Math utils.

function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1];
}

function sum(arr) {
    let sum = 0;
    for (let i = 0; i < arr.length; i++)
        sum += arr[i];
    return sum;
}

function avg(arr) {
    return sum(arr) / arr.length;
}

function scale(vec, scale) {
    return [vec[0] * scale, vec[1] * scale];
}

function norm(vec) {
    let len = Math.sqrt(dot(vec, vec));
    if (len === 0) return vec;
    return [vec[0] / len, vec[1] / len];
}

function clamp(value, low, high) {
    if (value < low) return low;
    if (value > high) return high;
    return value;
}

function rgb2hex(rgb) {
    return PIXI.utils.rgb2hex([clamp(rgb[0], 0, 1), clamp(rgb[1], 0, 1), clamp(rgb[2], 0, 1)]);
}

// LBM D2Q9 scheme.
let lv = [[0, 0], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];

function GasSimulator(width, height, viscosity, density=1) {

    this.width = width;
    this.height = height;

    const df = new Float32Array(width * height * 9).fill(density / 9.0);
    const rho = new Float32Array(width * height).fill(0);
    const ux = new Float32Array(width * height).fill(0);
    const uy = new Float32Array(width * height).fill(0);
    const bc = new Int8Array(width * height).fill(0); // Boundary mask.

    this.df = df;

    // Hard borders.
    for (let x = 0; x < width; x++) {
        bc[x] = 1;
        bc[x + (height - 1) * width] = 1;
    }
    for (let y = 0; y < height; y++) {
        bc[y * width] = 1;
        bc[width - 1 + y * width] = 1;
    }

    const four9th = 4.0 / 9.0;
    const one9th = 1.0 / 9.0;
    const one36th = 1.0 / 36.0;

    const collide = function() {
        let omega = 1.0 / (3.0 * viscosity + 0.5);
        for (let y = 1; y < height - 1; y++) {
            let yw = y * width;
            for (let x = 1; x < width - 1; x++) {
                let i = x + yw;
                let i9 = i * 9;

                let thisrho = df[i9] + df[i9 + 1] + df[i9 + 2] + df[i9 + 3] + df[i9 + 4] + df[i9 + 5] + df[i9 + 6] + df[i9 + 7] + df[i9 + 8];
                let thisux = df[i9 + 1] + df[i9 + 2] + df[i9 + 8] - df[i9 + 4] - df[i9 + 5] - df[i9 + 6];
                let thisuy = df[i9 + 2] + df[i9 + 3] + df[i9 + 4] - df[i9 + 6] - df[i9 + 7] - df[i9 + 8];
                rho[i] = thisrho;
                ux[i] = thisux;
                uy[i] = thisuy;

                let one9thrho = one9th * thisrho;
                let one36thrho = one36th * thisrho;
                let ux3 = 3 * thisux;
                let uy3 = 3 * thisuy;
                let ux2 = thisux * thisux;
                let uy2 = thisuy * thisuy;
                let uxuy2 = 2 * thisux * thisuy;
                let u2 = ux2 + uy2;
                let u215 = 1.5 * u2;

                df[i9] += omega * (four9th * thisrho * (1 - u215) - df[i9]);
                df[i9 + 1] += omega * (one9thrho * (1 + ux3 + 4.5 * ux2 - u215) - df[i9 + 1]);
                df[i9 + 5] += omega * (one9thrho * (1 - ux3 + 4.5 * ux2 - u215) - df[i9 + 5]);
                df[i9 + 3] += omega * (one9thrho * (1 + uy3 + 4.5 * uy2 - u215) - df[i9 + 3]);
                df[i9 + 7] += omega * (one9thrho * (1 - uy3 + 4.5 * uy2 - u215) - df[i9 + 7]);
                df[i9 + 2] += omega * (one36thrho * (1 + ux3 + uy3 + 4.5 * (u2 + uxuy2)) - df[i9 + 2]);
                df[i9 + 8] += omega * (one36thrho * (1 + ux3 - uy3 + 4.5 * (u2 - uxuy2)) - df[i9 + 8]);
                df[i9 + 4] += omega * (one36thrho * (1 - ux3 + uy3 + 4.5 * (u2 - uxuy2)) - df[i9 + 4]);
                df[i9 + 6] += omega * (one36thrho * (1 - ux3 - uy3 + 4.5 * (u2 + uxuy2)) - df[i9 + 6]);
            }
        }
    };

    const stream = function() {
        let width9 = width * 9;
        for (let y = height - 1; y > 0; y--) {
            let yw = y * width;
            for (let x = 0; x < width - 1; x++) {
                let i9 = (x + yw) * 9;
                df[i9 + 3] = df[i9 - width9 + 3]; // y-1
                df[i9 + 4] = df[i9 + 9 - width9 + 4]; // x+1 y-1
            }
        }
        for (let y = height - 1; y > 0; y--) {
            let yw = y * width;
            for (let x = width - 1; x > 0; x--) {
                let i9 = (x + yw) * 9;
                df[i9 + 1] = df[i9 - 9 + 1]; // x-1
                df[i9 + 2] = df[i9 - 9 - width9 + 2]; // x-1 y-1
            }
        }
        for (let y = 0; y < height - 1; y++) {
            let yw = y * width;
            for (let x = width - 1; x > 0; x--) {
                let i9 = (x + yw) * 9;
                df[i9 + 7] = df[i9 + width9 + 7]; // y+1
                df[i9 + 8] = df[i9 - 9 + width9 + 8]; // x-1 y+1
            }
        }
        for (let y = 0; y < height - 1; y++) {
            let yw = y * width;
            for (let x = 0; x < width - 1; x++) {
                let i9 = (x + yw) * 9;
                df[i9 + 5] = df[i9 + 9 + 5]; // x+1
                df[i9 + 6] = df[i9 + 9 + width9 + 6]; // x+1 y+1
            }
        }
    };

    const bounce = function() {
        let width9 = width * 9;
        for (let y = 1; y < height - 1; y++) {
            let yw = y * width;
            for (let x = 1; x < width - 1; x++) {
                // TODO implement moving bcacles
                if (bc[x + yw] !== 0) {
                    let i9 = (x + yw) * 9;
                    df[i9 + 9 + 1] = df[i9 + 5];
                    df[i9 - 9 + 5] = df[i9 + 1];
                    df[i9 + width9 + 3] = df[i9 + 7];
                    df[i9 - width9 + 7] = df[i9 + 3];
                    df[i9 + 9 + width9 + 2] = df[i9 + 6];
                    df[i9 - 9 + width9 + 4] = df[i9 + 8];
                    df[i9 + 9 - width9 + 8] = df[i9 + 4];
                    df[i9 - 9 - width9 + 6] = df[i9 + 2];
                }
            }
        }
        for (let y = 1; y < height - 1; y++) {
            let yw = y * width;
            let x = 0;
            if (bc[x + yw] !== 0) {
                let i9 = (x + yw) * 9;
                df[i9 + 9 + 1] = df[i9 + 5];
                df[i9 + width9 + 3] = df[i9 + 7];
                df[i9 - width9 + 7] = df[i9 + 3];
                df[i9 + 9 + width9 + 2] = df[i9 + 6];
                df[i9 + 9 - width9 + 8] = df[i9 + 4];
            }
            x = width - 1;
            if (bc[x + yw] !== 0) {
                let i9 = (x + yw) * 9;
                df[i9 - 9 + 5] = df[i9 + 1];
                df[i9 + width9 + 3] = df[i9 + 7];
                df[i9 - width9 + 7] = df[i9 + 3];
                df[i9 - 9 + width9 + 4] = df[i9 + 8];
                df[i9 - 9 - width9 + 6] = df[i9 + 2];
            }
        }
        for (let x = 1; x < width - 1; x++) {
            let yw = 0;
            if (bc[x + yw] !== 0) {
                let i9 = (x + yw) * 9;
                df[i9 + 9 + 1] = df[i9 + 5];
                df[i9 + width9 + 3] = df[i9 + 7];
                df[i9 + 9 + width9 + 2] = df[i9 + 6];
                df[i9 - 9 + width9 + 4] = df[i9 + 8];
                df[i9 + 9 - width9 + 8] = df[i9 + 4];
            }
            yw = (height - 1) * width;
            if (bc[x + yw] !== 0) {
                let i9 = (x + yw) * 9;
                df[i9 + 9 + 1] = df[i9 + 5];
                df[i9 - 9 + 5] = df[i9 + 1];
                df[i9 - width9 + 7] = df[i9 + 3];
                df[i9 + 9 - width9 + 8] = df[i9 + 4];
                df[i9 - 9 - width9 + 6] = df[i9 + 2];
            }
        }

        // Corners.
        let x = 0;
        let yw = 0;
        if (bc[x + yw] !== 0) {
            let i9 = (x + yw) * 9;
            df[i9 + 9 + 1] = df[i9 + 5];
            df[i9 + width9 + 3] = df[i9 + 7];
            df[i9 + 9 + width9 + 2] = df[i9 + 6];
        }
        x = width - 1;
        yw = 0;
        if (bc[x + yw] !== 0) {
            let i9 = (x + yw) * 9;
            df[i9 - 9 + 5] = df[i9 + 1];
            df[i9 + width9 + 3] = df[i9 + 7];
            df[i9 - 9 + width9 + 4] = df[i9 + 8];
        }
        x = width - 1;
        yw = (height - 1) * width;
        if (bc[x + yw] !== 0) {
            let i9 = (x + yw) * 9;
            df[i9 - 9 + 5] = df[i9 + 1];
            df[i9 - width9 + 7] = df[i9 + 3];
            df[i9 - 9 - width9 + 6] = df[i9 + 2];
        }
        x = 0;
        yw = (height - 1) * width;
        if (bc[x + yw] !== 0) {
            let i9 = (x + yw) * 9;
            df[i9 + 9 + 1] = df[i9 + 5];
            df[i9 - width9 + 7] = df[i9 + 3];
            df[i9 + 9 - width9 + 8] = df[i9 + 4];
        }
    };

    this.simulate = function(steps=1) {
        for (let step = 0; step < steps; step++) {
            collide();
            stream();
            bounce();
        }
    };

    this.push = function(from, to, mx, my, speedSound, deltaU) {
        let dMPos = [(to.x - from.x) / width, (to.y - from.y) / width];
        let d = this.rho(mx, my) * deltaU * (Math.sqrt(dot(dMPos, dMPos)) + 0.1) * speedSound * (width / 50.);
        let dr = scale(norm(dMPos), d);
        let flow = 0;
        for (let i = 1; i < 9; i++) {
            let val;
            if (dot(dMPos, dMPos) > 0.0005) val = Math.max(dot(dr, lv[i]) / dot(lv[i], lv[i]), 0);
            else val = d;
            df[(mx + my * width) * 9 + i] += val;
            flow += val;
        }
        df[(mx + my * width) * 9] -= flow;
    };

    // Physical fields.

    const outside = function(x, y, b=1) {
        return (x < b - 1 || x > width - b || y < b - 1 || y > height - b)
    };

    this.rho = function(x, y) {
        if (outside(x, y)) return 0;
        return rho[x + y * width];
    };

    this.u = function(x, y) {
        if (outside(x, y)) return 0;
        return [ux[x + y * width], uy[x + y * width]];
    };

    this.ke = function(x, y) {
        if (outside(x, y)) return 0;
        return rho[x + y * width] * (Math.pow(ux[x + y * width], 2) + Math.pow(uy[x + y * width], 2));
    };

    this.curl = function(x, y) {
        if (outside(x, y, 2)) return 0;
        return uy[x + 1 + y * width] - uy[x - 1 + y * width] - ux[x + (y + 1) * width] + ux[x + (y - 1) * width];
    };

    this.bc = function(x, y) {
        if (outside(x, y)) return 0;
        return bc[x + y * width];
    };

    this.setBc = function(x, y, value) {
        if (outside(x, y, 2)) return;
        return bc[x + y * width] = value;
    };

    this.viscosity = function() { return viscosity; };
    this.setViscosity = function(value) { viscosity = value; };
}
