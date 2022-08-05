#include <stdio.h>
#include <math.h>

float cbf(float* f, float* r, float* box, size_t n_bonds, int* a, int* b, float* r0, float* k) {
    size_t aa, bb;
    float df, rab_norm, rab[3], fa[3], energy=0.0;

    for (size_t i = 0; i < n_bonds; i++) {
        aa = 3 * a[i];
        bb = 3 * b[i];

        rab[0] = r[bb] - r[aa];
        rab[1] = r[bb + 1] - r[aa + 1];
        rab[2] = r[bb + 2] - r[aa + 2];

        rab[0] -= box[0] * round(rab[0]/box[0]);
        rab[1] -= box[1] * round(rab[1]/box[1]);
        rab[2] -= box[2] * round(rab[2]/box[2]);

        rab_norm = sqrt(rab[0] * rab[0] + rab[1] * rab[1] + rab[2] * rab[2]);

        df = k[i] * (rab_norm - r0[i]);
        fa[0] = -df * rab[0] / rab_norm;
        fa[1] = -df * rab[1] / rab_norm;
        fa[2] = -df * rab[2] / rab_norm;

        f[aa] -= fa[0];
        f[aa + 1] -= fa[1];
        f[aa + 2] -= fa[2];

        f[bb] += fa[0];
        f[bb + 1] += fa[1];
        f[bb + 2] += fa[2];

        energy += 0.5 * k[i] * pow(rab_norm - r0[i], 2);
    }
    return energy;
}
