#!/usr/bin/env node
// CI bench ratio table: interleaved rust-vs-cpp rounds -> per-op median ratio.
// Informational only: cross-toolchain wall-clock ratios are a ledger snapshot,
// not a gate (backend versions differ by stable-channel design).
//
// Usage: node bench_compare.mjs --cpp c1.json [c2.json ...] --rust r1.json [r2.json ...]
import { readFileSync } from "node:fs";

function parseArgs(argv) {
    const out = { cpp: [], rust: [] };
    let cur = null;
    for (const a of argv) {
        if (a === "--cpp" || a === "--rust") {
            cur = a.slice(2);
            continue;
        }
        if (cur === null) {
            console.error(`unexpected argument: ${a} (expected --cpp/--rust first)`);
            process.exit(1);
        }
        out[cur].push(a);
    }
    if (out.cpp.length === 0 || out.rust.length === 0) {
        console.error("need at least one --cpp and one --rust results file");
        process.exit(1);
    }
    return out;
}

function median(xs) {
    const s = [...xs].sort((a, b) => a - b);
    return s[s.length >> 1];
}

// op -> median of per-round medians (round-median before op-median keeps the
// interleaving symmetric between the two sides)
function load(files) {
    const perOp = new Map();
    for (const f of files) {
        let j;
        try {
            j = JSON.parse(readFileSync(f, "utf8"));
        } catch (e) {
            console.error(`cannot parse ${f}: ${e.message}`);
            process.exit(1);
        }
        for (const r of j.results) {
            if (!perOp.has(r.op)) perOp.set(r.op, []);
            perOp.get(r.op).push(r.median_ms);
        }
    }
    const med = new Map();
    for (const [op, xs] of perOp) med.set(op, median(xs));
    return med;
}

const args = parseArgs(process.argv.slice(2));
const cpp = load(args.cpp);
const rust = load(args.rust);
const ops = [...new Set([...cpp.keys(), ...rust.keys()])].sort();

console.log(`rounds: cpp=${args.cpp.length} rust=${args.rust.length}`);
console.log("op                 cpp_ms    rust_ms   ratio");
let worst = 0;
let worstOp = "-";
for (const op of ops) {
    const c = cpp.get(op);
    const r = rust.get(op);
    if (c === undefined || r === undefined) {
        console.log(`${op.padEnd(18)}  MISSING ON ONE SIDE`);
        continue;
    }
    const ratio = r / c;
    if (ratio > worst) {
        worst = ratio;
        worstOp = op;
    }
    console.log(
        `${op.padEnd(18)}  ${c.toFixed(4).padStart(7)}   ${r.toFixed(4).padStart(7)}   ${ratio.toFixed(3)}x`,
    );
}
console.log(`\nworst: ${worstOp} ${worst.toFixed(3)}x (informational)`);
