// Browser host for tinyc.wasm. Drives the full pipeline:
//   1. fetch tinyc.wasm + runtime_wasm.wasm.o + start_wasm.wasm.o
//   2. compile tinyC source via `tinyc.wasm --emit=wasm` (in an in-memory
//      FS over the WASI shim we already use for Node.js / wasmtime)
//   3. link the produced object + the runtime/start objects via
//      `tinyc.wasm --link --export=_start`
//   4. instantiate the resulting wasm + the same WASI shim, route its
//      stdout/stderr into the output panel.

import { makeWasi, makeMemoryFS, ProcExit } from
    "../../../corec/platform/js/wasi.js";

const $ = (id) => document.getElementById(id);
const status = (msg, isErr) => {
    $("status").textContent = msg;
    $("status").className = isErr ? "err" : "";
};
const appendOutput = (txt) => { $("output").textContent += txt; };
const clearOutput = () => { $("output").textContent = ""; };

// Pre-fetched assets, populated once at page load.
let tinycWasmBytes = null;
let runtimeObj    = null;
let startObj      = null;

async function loadAssets() {
    status("loading tinyc.wasm…");
    const [tc, rt, st] = await Promise.all([
        fetch("./tinyc.wasm").then(r => r.arrayBuffer()),
        fetch("./runtime_wasm.wasm.o").then(r => r.arrayBuffer()),
        fetch("./start_wasm.wasm.o").then(r => r.arrayBuffer()),
    ]);
    tinycWasmBytes = new Uint8Array(tc);
    runtimeObj    = new Uint8Array(rt);
    startObj      = new Uint8Array(st);
    status(`ready (${(tinycWasmBytes.length / 1024).toFixed(0)} KB)`);
    $("runBtn").disabled = false;
}

// Run tinyc.wasm with a given argv + virtual FS. Returns the exit status
// and any captured stderr text. stdout is left in the FS as a side
// effect (since we always pass `-o <path>` on the argv).
async function runTinyc(argv, fs) {
    let stdout = "";
    let stderr = "";
    const io = {
        argv,
        stdin:  { read(_n) { return new Uint8Array(); } },
        stdout: { write(b) { stdout += new TextDecoder().decode(b); } },
        stderr: { write(b) { stderr += new TextDecoder().decode(b); } },
        fs,
    };
    const wasi = makeWasi(io);
    const mod = await WebAssembly.instantiate(tinycWasmBytes, wasi.imports);
    wasi.setMemory(mod.instance.exports.memory);
    let st = 0;
    try { mod.instance.exports._start(); }
    catch (e) {
        if (e instanceof ProcExit) st = e.status; else throw e;
    }
    return { status: st, stdout, stderr };
}

async function compileAndRun() {
    clearOutput();
    $("runBtn").disabled = true;
    try {
        const source = $("source").value;

        // -------- Stage 1: tinyc.wasm --emit=wasm  ------------------------
        status("compiling…");
        const fs1 = makeMemoryFS({
            "hello.tc":          source,
        });
        const r1 = await runTinyc(
            ["tinyc", "--emit=wasm", "--lowering=native",
             "-o", "hello.wasm.o", "hello.tc"], fs1);
        if (r1.stderr) appendOutput(r1.stderr);
        if (r1.status !== 0) {
            status(`compile failed (exit ${r1.status})`, true);
            return;
        }
        const obj = fs1.files.get("hello.wasm.o").data;

        // -------- Stage 2: tinyc.wasm --link  -----------------------------
        status("linking…");
        const fs2 = makeMemoryFS({
            "hello.wasm.o":      obj,
            "runtime_wasm.o":    runtimeObj,
            "start_wasm.o":      startObj,
        });
        const r2 = await runTinyc(
            ["tinyc", "--link", "--export=_start",
             "-o", "hello.wasm",
             "hello.wasm.o", "runtime_wasm.o", "start_wasm.o"], fs2);
        if (r2.stderr) appendOutput(r2.stderr);
        if (r2.status !== 0) {
            status(`link failed (exit ${r2.status})`, true);
            return;
        }
        const linked = fs2.files.get("hello.wasm").data;

        // -------- Stage 3: run the linked program ------------------------
        status(`running (${linked.length} bytes)…`);
        const fs3 = makeMemoryFS();
        const io = {
            argv:   ["program"],
            stdin:  { read(_n) { return new Uint8Array(); } },
            stdout: { write(b) { appendOutput(new TextDecoder().decode(b)); } },
            stderr: { write(b) { appendOutput(new TextDecoder().decode(b)); } },
            fs: fs3,
        };
        const wasi = makeWasi(io);
        const mod = await WebAssembly.instantiate(linked, wasi.imports);
        wasi.setMemory(mod.instance.exports.memory);
        let st = 0;
        try { mod.instance.exports._start(); }
        catch (e) {
            if (e instanceof ProcExit) st = e.status; else throw e;
        }
        status(`exit ${st}`);
    } catch (e) {
        status(`error: ${e.message}`, true);
        appendOutput(`\n[host] ${e.stack || e}\n`);
    } finally {
        $("runBtn").disabled = false;
    }
}

$("runBtn").addEventListener("click", compileAndRun);
loadAssets().catch(e => status(`load failed: ${e.message}`, true));
