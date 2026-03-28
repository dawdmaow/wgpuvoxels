// Must match Cube_Kind in game.odin:
// None=0, Grass=1, Dirt=2, Stone=3, Bedrock=4, Water=5, Sand=6, Flower1..Flower12=7..18,
// Wood=19, Cobblestone=20, Pumpkin=21, Brick=22, TNT=23, Ore_Diamond=24, Ore_Gold=25,
// Ore_Iron=26, Ore_Green=27, Ore_Red=28, Ore_Blue=29, Ore_Coal=30
const KIND_NONE: u32 = 0u;
const KIND_GRASS: u32 = 1u;
const KIND_DIRT: u32 = 2u;
const KIND_STONE: u32 = 3u;
const KIND_BEDROCK: u32 = 4u;
const KIND_WATER: u32 = 5u;
const KIND_SAND: u32 = 6u;
const KIND_FLOWER_FIRST: u32 = 7u;
const FLOWER_KIND_COUNT: u32 = 12u;
const KIND_PUMPKIN: u32 = 21u;
const KIND_ORE_DIAMOND: u32 = 24u;
const KIND_ORE_GOLD: u32 = 25u;
const KIND_ORE_IRON: u32 = 26u;
const KIND_ORE_GREEN: u32 = 27u;
const KIND_ORE_RED: u32 = 28u;
const KIND_ORE_BLUE: u32 = 29u;
const KIND_ORE_COAL: u32 = 30u;
// Slow FBM drives vertical range and ±shift so large regions read as different “biomes”.
const BIOME_MACRO_SCALE: f32 = 0.0041;
// Lower boost/shift so non-plains terrain is still hilly, not endless sharp peaks (chunk is only 48 tall).
const HEIGHT_BOOST_MIN: f32 = 1.04;
const HEIGHT_BOOST_MAX: f32 = 1.42;
const BIOME_VERTICAL_SHIFT: i32 = 2;
// Separate slow FBM from biome_macro so “plains” patches are large and don’t track hills/mountains.
const PLAINS_MACRO_SCALE: f32 = 0.0018;
// FBM spends most of its mass ~0.35–0.65; a tight smoothstep here left plains≈0 almost everywhere.
const PLAINS_SMOOTH_LO: f32 = 0.30;
const PLAINS_SMOOTH_HI: f32 = 0.62;
const FLAT_BASE_ABOVE_SEA: i32 = 6;
const FLAT_ROLL_SCALE: f32 = 0.09;
const FLAT_ROLL_AMP: f32 = 2.5;
// River: narrow bands where `fbm` crosses 0.5 — carve down and cap to sea level so water fills.
const RIVER_NOISE_SCALE: f32 = 0.0027;
const RIVER_BANK_HALF: f32 = 0.026;
const RIVER_MAX_CARVE: i32 = 7;
// Steeper base terrain (before ravines/rivers) → thinner dirt; integer slope = max step to a neighbor.
const SLOPE_DIRT_SHALLOW: i32 = 2;
const SLOPE_DIRT_STEEP: i32 = 4;
const SEA_LEVEL: i32 = 12;
const FLAT_MAX_Y: i32 = SEA_LEVEL + 12;
const BEACH_BAND: i32 = 2;
const CAVE_MIN_Y: i32 = 3;
const CAVE_SURFACE_MARGIN: i32 = 5;
const CAVE_SCALE: f32 = 0.075;
// Lower threshold carves a larger share of the same field, yielding roomier underworld caverns.
const CAVE_THRESHOLD: f32 = 0.72;
const CAVE_VERTICAL_SQUASH: f32 = 0.7;

// Rare bedrock inclusions above the floor row (ly==0 is always a full bedrock layer).
const BEDROCK_SCATTER_MAX_Y: i32 = 2;
const BEDROCK_SCATTER_THRESHOLD: f32 = 0.5;

// Underground tunnels: same 3D noise as blob caves but sampled with one horizontal axis stretched
// so voids read as corridors; several headings cross like ravines.
const TUNNEL_ALONG_SCALE: f32 = 0.031;
// Lower cross-axis frequency thickens corridor radius without changing long-run tunnel directionality.
const TUNNEL_CROSS_SCALE: f32 = 0.11;
const TUNNEL_Y_SCALE: f32 = 0.11;
const TUNNEL_THRESHOLD: f32 = 0.69;

// Elongated surface cuts: strongly anisotropic FBM so valleys read as gorges rather than round dimples.
// `smoothstep` must have edge0 < edge1 (otherwise results are undefined on GPU); use `1.0 - smoothstep` to invert.
const RAVINE_MAX_DEPTH: i32 = 9;
const RAVINE_FLOOR_MIN: i32 = 5;
const RAVINE_LOW: f32 = 0.10;
const RAVINE_HIGH: f32 = 0.34;

// Ore veins: higher `ORE_*_SCALE` = finer noise = smaller pockets; higher thresholds = fewer pockets.
// Colored ores: one field + `hash3` variant. Each `ORE_*_THRESHOLD` is `base * ORE_THRESHOLD_MULTIPLIER`.
const ORE_THRESHOLD_MULTIPLIER: f32 = 1.05;

const ORE_DIAMOND_MIN_Y: i32 = 3;
const ORE_DIAMOND_MAX_Y: i32 = 14;
const ORE_DIAMOND_SCALE: f32 = 0.186;
const ORE_DIAMOND_THRESHOLD: f32 = 0.77 * ORE_THRESHOLD_MULTIPLIER;

const ORE_GOLD_MIN_Y: i32 = 4;
const ORE_GOLD_MAX_Y: i32 = 20;
const ORE_GOLD_SCALE: f32 = 0.167;
const ORE_GOLD_THRESHOLD: f32 = 0.67 * ORE_THRESHOLD_MULTIPLIER;

const ORE_IRON_MIN_Y: i32 = 4;
const ORE_IRON_MAX_Y: i32 = 22;
const ORE_IRON_SCALE: f32 = 0.158;
const ORE_IRON_THRESHOLD: f32 = 0.70 * ORE_THRESHOLD_MULTIPLIER;

const ORE_COLORED_MIN_Y: i32 = 4;
const ORE_COLORED_MAX_Y: i32 = 18;
const ORE_COLORED_SCALE: f32 = 0.163;
const ORE_COLORED_THRESHOLD: f32 = 0.67 * ORE_THRESHOLD_MULTIPLIER;

const ORE_COAL_MIN_Y: i32 = 4;
const ORE_COAL_MAX_Y: i32 = 26;
const ORE_COAL_SCALE: f32 = 0.147;
const ORE_COAL_THRESHOLD: f32 = 0.63 * ORE_THRESHOLD_MULTIPLIER;

// Per-type XZ rotation + world shift + slight Y scale so vein fields don’t line up in the same places.
const ORE_D_YAW: f32 = 0.91;
const ORE_D_Y_SCALE: f32 = 1.08;
const ORE_D_SHIFT: vec3<f32> = vec3<f32>(1021.0, 307.0, 419.0);
const ORE_G_YAW: f32 = 2.05;
const ORE_G_Y_SCALE: f32 = 0.93;
const ORE_G_SHIFT: vec3<f32> = vec3<f32>(-773.0, 211.0, 887.0);
const ORE_I_YAW: f32 = 3.17;
const ORE_I_Y_SCALE: f32 = 1.05;
const ORE_I_SHIFT: vec3<f32> = vec3<f32>(641.0, -509.0, -301.0);
const ORE_CLR_YAW: f32 = 4.22;
const ORE_CLR_Y_SCALE: f32 = 0.98;
const ORE_CLR_SHIFT: vec3<f32> = vec3<f32>(-401.0, 673.0, -919.0);
const ORE_COAL_YAW: f32 = 5.31;
const ORE_COAL_Y_SCALE: f32 = 1.0;
const ORE_COAL_SHIFT: vec3<f32> = vec3<f32>(823.0, -127.0, 503.0);

const CHUNK_W: u32 = 16u;
const CHUNK_H: u32 = 48u;
const VOXELS_PER_CHUNK: u32 = CHUNK_W * CHUNK_H * CHUNK_W;

struct TerrainBatch {
    chunk_count: u32,
    noise_scale: f32,
    seed: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> batch: TerrainBatch;
@group(0) @binding(1) var<storage, read> chunk_coords: array<vec2<i32>>;
@group(0) @binding(2) var<storage, read_write> voxels: array<u32>;

fn hash2(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn noise2(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash2(i);
    let b = hash2(i + vec2<f32>(1.0, 0.0));
    let c = hash2(i + vec2<f32>(0.0, 1.0));
    let d = hash2(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn hash3(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
}

fn noise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let n000 = hash3(i);
    let n100 = hash3(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash3(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash3(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash3(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash3(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash3(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash3(i + vec3<f32>(1.0, 1.0, 1.0));

    let nx00 = mix(n000, n100, u.x);
    let nx10 = mix(n010, n110, u.x);
    let nx01 = mix(n001, n101, u.x);
    let nx11 = mix(n011, n111, u.x);
    let nxy0 = mix(nx00, nx10, u.y);
    let nxy1 = mix(nx01, nx11, u.y);
    return mix(nxy0, nxy1, u.z);
}

fn fbm(p: vec2<f32>) -> f32 {
    var v = 0.0;
    var a = 0.5;
    var f = 1.0;
    for (var i = 0; i < 4; i++) {
        v += a * noise2(p * f);
        f *= 2.0;
        a *= 0.5;
    }
    return v;
}

fn plains_blend(wx: i32, wz: i32, seed: f32) -> f32 {
    let plains_p = vec2<f32>(f32(wx), f32(wz)) * PLAINS_MACRO_SCALE + vec2<f32>(seed * 47.0, seed * -23.0);
    let plains_raw = fbm(plains_p);
    return smoothstep(PLAINS_SMOOTH_LO, PLAINS_SMOOTH_HI, plains_raw);
}

fn ravine_depth(wx: i32, wz: i32, seed: f32) -> i32 {
    let fx = f32(wx);
    let fz = f32(wz);
    // Stretched axes → long straight-ish gorges; two orientations + diagonal so they cross naturally.
    let p0 = vec2<f32>(fx * 0.027, fz * 0.0033) + vec2<f32>(seed * 19.0, seed * -13.0);
    let p1 = vec2<f32>(fx * 0.0033, fz * 0.027) + vec2<f32>(seed * -7.0, seed * 23.0);
    let px = fx * 0.70710678 + fz * 0.70710678;
    let pz = -fx * 0.70710678 + fz * 0.70710678;
    let p2 = vec2<f32>(px * 0.024, pz * 0.003) + vec2<f32>(seed * 41.0, seed * -17.0);
    let v0 = fbm(p0);
    let v1 = fbm(p1);
    let v2 = fbm(p2);
    // Deep only in the bottom of each stretched FBM — typical values sit ~0.45–0.55, so this stays sparse.
    let t0 = 1.0 - smoothstep(RAVINE_LOW, RAVINE_HIGH, v0);
    let t1 = 1.0 - smoothstep(RAVINE_LOW, RAVINE_HIGH, v1);
    let t2 = 1.0 - smoothstep(RAVINE_LOW, RAVINE_HIGH, v2);
    let t = max(max(t0, t1), t2);
    return i32(t * f32(RAVINE_MAX_DEPTH));
}

fn cave_fbm(p: vec3<f32>) -> f32 {
    // Mixing frequencies reduces obvious grid-aligned cavities.
    let n0 = noise3(p);
    let n1 = noise3(p * 2.03 + vec3<f32>(37.2, 19.8, 11.4));
    let n2 = noise3(p * 4.11 + vec3<f32>(8.3, 42.7, 27.1));
    return n0 * 0.6 + n1 * 0.28 + n2 * 0.12;
}

fn tunnel_cave_value(wx: i32, wy: i32, wz: i32, seed: f32) -> f32 {
    let fx = f32(wx);
    let fy = f32(wy);
    let fz = f32(wz);
    let sx = seed * 19.3;
    let sy = seed * -11.7;
    let sz = seed * 7.1;
    // Along +X and +Z at different seeds so grids don’t coincide; diagonals for variety.
    let p_x = vec3<f32>(
        fx * TUNNEL_ALONG_SCALE + sx,
        fy * TUNNEL_Y_SCALE + sy,
        fz * TUNNEL_CROSS_SCALE + sz,
    );
    let p_z = vec3<f32>(
        fx * TUNNEL_CROSS_SCALE - sx,
        fy * TUNNEL_Y_SCALE - sy,
        fz * TUNNEL_ALONG_SCALE + sz,
    );
    let px = fx * 0.70710678 + fz * 0.70710678;
    let pz = -fx * 0.70710678 + fz * 0.70710678;
    let p_d1 = vec3<f32>(
        px * TUNNEL_ALONG_SCALE + sx * 0.7,
        fy * TUNNEL_Y_SCALE + sy * 1.1,
        pz * TUNNEL_CROSS_SCALE - sz * 0.5,
    );
    let px2 = fx * 0.70710678 - fz * 0.70710678;
    let pz2 = fx * 0.70710678 + fz * 0.70710678;
    let p_d2 = vec3<f32>(
        px2 * TUNNEL_CROSS_SCALE - sx * 0.4,
        fy * TUNNEL_Y_SCALE + sy * 0.6,
        pz2 * TUNNEL_ALONG_SCALE + sz * 1.2,
    );
    let v0 = cave_fbm(p_x);
    let v1 = cave_fbm(p_z);
    let v2 = cave_fbm(p_d1);
    let v3 = cave_fbm(p_d2);
    return max(max(v0, v1), max(v2, v3));
}

fn ore_field(p: vec3<f32>, scale: f32, seed_offset: vec3<f32>) -> f32 {
    // World-space multi-octave field; heavier high-frequency weight → smaller blobs above threshold.
    let base = p * scale + seed_offset;
    let n0 = noise3(base);
    let n1 = noise3(base * 2.11 + vec3<f32>(17.3, 9.7, 23.5));
    let n2 = noise3(base * 4.35 + vec3<f32>(41.0, 2.0, 19.0));
    return n0 * 0.40 + n1 * 0.33 + n2 * 0.27;
}

fn ore_vein_sample(p: vec3<f32>, yaw: f32, y_scale: f32, shift: vec3<f32>) -> vec3<f32> {
    let c = cos(yaw);
    let s = sin(yaw);
    let x = p.x * c - p.z * s;
    let z = p.x * s + p.z * c;
    return vec3<f32>(x, p.y * y_scale, z) + shift;
}

// Pre-ravine height: detail FBM + macro FBM for boost/shift (chunk-stable, world-space).
fn base_surface_height(wx: i32, wz: i32, seed: f32, noise_scale: f32) -> i32 {
    let p = vec2<f32>(f32(wx), f32(wz)) * noise_scale + vec2<f32>(seed * 17.13, seed * 9.7);
    let n = fbm(p);
    let biome_macro = fbm(
        vec2<f32>(f32(wx), f32(wz)) * BIOME_MACRO_SCALE + vec2<f32>(seed * 31.0, seed * -19.0),
    );
    let height_boost = mix(HEIGHT_BOOST_MIN, HEIGHT_BOOST_MAX, biome_macro);
    let boosted_n = clamp((n - 0.5) * height_boost + 0.5, 0.0, 1.0);
    let max_h = i32(CHUNK_H) - 4;
    let min_h = 4;
    var mountain_surface = min_h + i32(boosted_n * f32(max_h - min_h));
    mountain_surface = clamp(mountain_surface, 1, i32(CHUNK_H) - 2);
    let vertical_shift = i32((biome_macro - 0.5) * f32(BIOME_VERTICAL_SHIFT * 2));
    mountain_surface = mountain_surface + vertical_shift;
    mountain_surface = clamp(mountain_surface, 1, i32(CHUNK_H) - 2);

    let roll = fbm(
        vec2<f32>(f32(wx), f32(wz)) * FLAT_ROLL_SCALE + vec2<f32>(seed * 61.0, seed * 29.0),
    );
    let flat_base = SEA_LEVEL + FLAT_BASE_ABOVE_SEA;
    var flat_surface = flat_base + i32((roll - 0.5) * 2.0 * FLAT_ROLL_AMP);
    flat_surface = clamp(flat_surface, SEA_LEVEL + 1, min(FLAT_MAX_Y, i32(CHUNK_H) - 2));

    let plains = plains_blend(wx, wz, seed);
    var surface = i32(mix(f32(mountain_surface), f32(flat_surface), plains));
    surface = clamp(surface, 1, i32(CHUNK_H) - 2);
    return surface;
}

// Carve narrow channels toward sea level so low strips read as rivers/lakeshore.
fn river_carve_surface(wx: i32, wz: i32, seed: f32, surface: i32) -> i32 {
    let p = vec2<f32>(f32(wx), f32(wz)) * RIVER_NOISE_SCALE + vec2<f32>(seed * 53.0, seed * -37.0);
    let v = fbm(p);
    let d = abs(v - 0.5);
    if (d >= RIVER_BANK_HALF) {
        return surface;
    }
    let t = 1.0 - d / RIVER_BANK_HALF;
    let carve = i32(t * f32(RIVER_MAX_CARVE));
    var s = surface - carve;
    s = min(s, SEA_LEVEL);
    s = max(s, RAVINE_FLOOR_MIN);
    s = clamp(s, 1, i32(CHUNK_H) - 2);
    return s;
}

// Dispatch: (2, 6, 4 * chunk_count); chunk_i = gid.z / CHUNK_W, local_z = gid.z % CHUNK_W
@compute @workgroup_size(8, 8, 4)
fn terrain_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let chunk_i = gid.z / CHUNK_W;
    let local_z = gid.z % CHUNK_W;

    if (chunk_i >= batch.chunk_count) {
        return;
    }
    if (gid.x >= CHUNK_W || gid.y >= CHUNK_H || local_z >= CHUNK_W) {
        return;
    }

    let cc = chunk_coords[chunk_i];
    let wx = cc.x * i32(CHUNK_W) + i32(gid.x);
    let wz = cc.y * i32(CHUNK_W) + i32(local_z);
    let ly = i32(gid.y);

    let h0 = base_surface_height(wx, wz, batch.seed, batch.noise_scale);
    let h_px = base_surface_height(wx + 1, wz, batch.seed, batch.noise_scale);
    let h_mx = base_surface_height(wx - 1, wz, batch.seed, batch.noise_scale);
    let h_pz = base_surface_height(wx, wz + 1, batch.seed, batch.noise_scale);
    let h_mz = base_surface_height(wx, wz - 1, batch.seed, batch.noise_scale);
    let slope = max(
        max(abs(h0 - h_px), abs(h0 - h_mx)),
        max(abs(h0 - h_pz), abs(h0 - h_mz)),
    );
    var dirt_depth: i32 = 2;
    if (slope >= SLOPE_DIRT_STEEP) {
        dirt_depth = 0;
    } else if (slope >= SLOPE_DIRT_SHALLOW) {
        dirt_depth = 1;
    }

    var surface = h0;
    let plains = plains_blend(wx, wz, batch.seed);
    let rd = ravine_depth(wx, wz, batch.seed);
    let rd_atten = i32(f32(rd) * (1.0 - plains));
    surface = surface - rd_atten;
    surface = max(surface, RAVINE_FLOOR_MIN);
    surface = clamp(surface, 1, i32(CHUNK_H) - 2);
    surface = river_carve_surface(wx, wz, batch.seed, surface);

    let grass_biome = surface > SEA_LEVEL + BEACH_BAND;
    let can_place_flower = grass_biome && ly == surface + 1;
    let can_place_pumpkin = grass_biome && ly == surface + 1;
    let flower_patch_noise = fbm(
        vec2<f32>(
            f32(wx) * 0.08 + batch.seed * 2.1,
            f32(wz) * 0.08 - batch.seed * 1.7,
        ),
    );
    let flower_noise = hash2(
        vec2<f32>(
            f32(wx) * 3.17 + batch.seed * 11.0,
            f32(wz) * 5.13 - batch.seed * 7.0,
        ),
    );
    let flower_kind_noise = hash2(
        vec2<f32>(
            f32(wx) * 13.37 - batch.seed * 3.0,
            f32(wz) * 17.41 + batch.seed * 19.0,
        ),
    );
    let pumpkin_noise = hash2(
        vec2<f32>(
            f32(wx) * 7.73 + batch.seed * 23.0,
            f32(wz) * 9.91 - batch.seed * 41.0,
        ),
    );

    var kind: u32 = KIND_NONE;
    if (ly == 0) {
        kind = KIND_BEDROCK;
    }
    else if (can_place_pumpkin && pumpkin_noise > 0.99) {
        kind = KIND_PUMPKIN;
    }
    else if (can_place_flower && flower_patch_noise > 0.6 && flower_noise > 0.9) {
        // Keep both placement and variant stable in world space so chunk reloads never reshuffle flowers.
        let flower_kind_offset = min(
            u32(floor(flower_kind_noise * f32(FLOWER_KIND_COUNT))),
            FLOWER_KIND_COUNT - 1u,
        );
        kind = KIND_FLOWER_FIRST + flower_kind_offset;
    } else if (ly > surface) {
        if (ly <= SEA_LEVEL) {
            kind = KIND_WATER;
        } else {
            kind = KIND_NONE;
        }
    } else if (ly == surface) {
        if (surface <= SEA_LEVEL + BEACH_BAND) {
            kind = KIND_SAND;
        } else {
            kind = KIND_GRASS;
        }
    } else if (dirt_depth > 0 && ly < surface && ly >= surface - dirt_depth) {
        if (surface <= SEA_LEVEL + BEACH_BAND) {
            kind = KIND_SAND;
        } else {
            kind = KIND_DIRT;
        }
    } else {
        kind = KIND_STONE;
    }

    // Ore replacement: only carve veins into solid stone well below the surface band so beaches/topsoil stay clean.
    if (kind == KIND_STONE && ly < surface - dirt_depth) {
        let wp = vec3<f32>(f32(wx), f32(ly), f32(wz));
        let seed_jitter = vec3<f32>(batch.seed * 13.0, batch.seed * -9.0, batch.seed * 5.0);

        // Rarer ores first (diamond → gold → iron → colored → coal). Colored: one vein test, hash picks R/G/B.
        if (ly >= ORE_DIAMOND_MIN_Y && ly <= ORE_DIAMOND_MAX_Y) {
            let p_d = ore_vein_sample(wp, ORE_D_YAW, ORE_D_Y_SCALE, ORE_D_SHIFT + seed_jitter);
            let f = ore_field(
                p_d,
                ORE_DIAMOND_SCALE,
                vec3<f32>(batch.seed * 13.3, batch.seed * -7.1, batch.seed * 19.7),
            );
            if (f > ORE_DIAMOND_THRESHOLD) {
                kind = KIND_ORE_DIAMOND;
            }
        }

        if (kind == KIND_STONE && ly >= ORE_GOLD_MIN_Y && ly <= ORE_GOLD_MAX_Y) {
            let p_g = ore_vein_sample(wp, ORE_G_YAW, ORE_G_Y_SCALE, ORE_G_SHIFT + seed_jitter);
            let f = ore_field(
                p_g,
                ORE_GOLD_SCALE,
                vec3<f32>(batch.seed * -5.9, batch.seed * 21.3, batch.seed * 9.1),
            );
            if (f > ORE_GOLD_THRESHOLD) {
                kind = KIND_ORE_GOLD;
            }
        }

        if (kind == KIND_STONE && ly >= ORE_IRON_MIN_Y && ly <= ORE_IRON_MAX_Y) {
            let p_i = ore_vein_sample(wp, ORE_I_YAW, ORE_I_Y_SCALE, ORE_I_SHIFT + seed_jitter);
            let f = ore_field(
                p_i,
                ORE_IRON_SCALE,
                vec3<f32>(batch.seed * 7.7, batch.seed * -3.5, batch.seed * 15.9),
            );
            if (f > ORE_IRON_THRESHOLD) {
                kind = KIND_ORE_IRON;
            }
        }

        if (kind == KIND_STONE && ly >= ORE_COLORED_MIN_Y && ly <= ORE_COLORED_MAX_Y) {
            let p_c = ore_vein_sample(wp, ORE_CLR_YAW, ORE_CLR_Y_SCALE, ORE_CLR_SHIFT + seed_jitter);
            let f = ore_field(
                p_c,
                ORE_COLORED_SCALE,
                vec3<f32>(batch.seed * 1.7, batch.seed * -9.3, batch.seed * 6.5),
            );
            if (f > ORE_COLORED_THRESHOLD) {
                let h = hash3(wp * 0.31 + vec3<f32>(batch.seed * 44.1, 91.0 + batch.seed, -batch.seed * 12.0));
                if (h < 0.33333334) {
                    kind = KIND_ORE_RED;
                } else if (h < 0.66666669) {
                    kind = KIND_ORE_BLUE;
                } else {
                    kind = KIND_ORE_GREEN;
                }
            }
        }

        if (kind == KIND_STONE && ly >= ORE_COAL_MIN_Y && ly <= ORE_COAL_MAX_Y) {
            let p_coal = ore_vein_sample(wp, ORE_COAL_YAW, ORE_COAL_Y_SCALE, ORE_COAL_SHIFT + seed_jitter);
            let f = ore_field(
                p_coal,
                ORE_COAL_SCALE,
                vec3<f32>(batch.seed * -2.3, batch.seed * 6.5, batch.seed * -10.7),
            );
            if (f > ORE_COAL_THRESHOLD) {
                kind = KIND_ORE_COAL;
            }
        }
    }

    // Overflow hook: keep water static for now. A future pass can use this marker
    // to spread water into neighboring voxels/chunks when dynamic fluids are added.
    if (false && kind == KIND_WATER) {
        kind = KIND_WATER;
    }

    let is_solid_underground =
        kind == KIND_STONE ||
        kind == KIND_DIRT ||
        kind == KIND_SAND ||
        kind == KIND_ORE_COAL ||
        kind == KIND_ORE_IRON ||
        kind == KIND_ORE_GOLD ||
        kind == KIND_ORE_RED ||
        kind == KIND_ORE_BLUE ||
        kind == KIND_ORE_GREEN ||
        kind == KIND_ORE_DIAMOND;
    let cave_ceiling = surface - CAVE_SURFACE_MARGIN;
    if (is_solid_underground && ly >= CAVE_MIN_Y && ly < cave_ceiling) {
        // Keep cave field in world space so chunk streaming never reshuffles cave layouts.
        let cave_p = vec3<f32>(
            f32(wx) * CAVE_SCALE + batch.seed * 3.7,
            f32(ly) * CAVE_SCALE * CAVE_VERTICAL_SQUASH - batch.seed * 2.9,
            f32(wz) * CAVE_SCALE + batch.seed * 5.1,
        );
        let cave_value = cave_fbm(cave_p);
        let tunnel_value = tunnel_cave_value(wx, ly, wz, batch.seed);
        if (cave_value > CAVE_THRESHOLD || tunnel_value > TUNNEL_THRESHOLD) {
            kind = KIND_NONE;
        }
    }

    // Break up the flat stone-over-bedrock look with sparse bedrock shards (only replaces solid stone).
    if (kind == KIND_STONE && ly >= 1 && ly <= BEDROCK_SCATTER_MAX_Y) {
        let h = hash3(
            vec3<f32>(
                f32(wx) * 0.131 + batch.seed * 41.0,
                f32(ly) * 0.173 + batch.seed * -19.0,
                f32(wz) * 0.137 + batch.seed * 23.0,
            ),
        );
        if (h > BEDROCK_SCATTER_THRESHOLD) {
            kind = KIND_BEDROCK;
        }
    }

    let base = chunk_i * VOXELS_PER_CHUNK;
    let idx = base + gid.x + gid.y * CHUNK_W + local_z * CHUNK_W * CHUNK_H;
    voxels[idx] = kind;
}
