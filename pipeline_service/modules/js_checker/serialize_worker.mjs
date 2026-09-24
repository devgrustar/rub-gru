import { parentPort, workerData } from 'node:worker_threads';
import * as THREE from 'three';
import { postValidate } from './validator/postValidation.js';

const SafeFunction = Function;
const NowMs = Date.now.bind(Date);

const TRAPPED_GLOBALS = [
  'setTimeout', 'setInterval', 'setImmediate', 'queueMicrotask',
  'fetch', 'XMLHttpRequest', 'WebSocket',
  'crypto',
  'Date', 'performance',
  'eval', 'Function',
  'process', 'global',
  'WeakRef', 'FinalizationRegistry',
  'SharedArrayBuffer', 'Atomics',
];

const CODEGEN_PROTOTYPES = [
  SafeFunction.prototype,
  Object.getPrototypeOf(function*(){}),
  Object.getPrototypeOf(async function(){}),
  Object.getPrototypeOf(async function*(){}),
];

function trapViolation(name) {
  throw new Error(`Runtime violation: ${name} is forbidden`);
}

function send(msg) {
  parentPort.postMessage(msg);
}

function geometryIsDirty(geo) {
  const attrs = geo.attributes || {};
  for (const name in attrs) {
    if (attrs[name] && attrs[name].version > 0) return true;
  }
  if (geo.index && geo.index.version > 0) return true;
  const morph = geo.morphAttributes || {};
  for (const name in morph) {
    const list = morph[name];
    if (Array.isArray(list)) {
      for (const a of list) if (a && a.version > 0) return true;
    }
  }
  return false;
}

// Curve classes ObjectLoader can rebuild (three/src/extras/curves). A user-defined `class X extends THREE.Curve`
// (or a CurvePath) used as a TubeGeometry path / ExtrudeGeometry extrudePath makes ObjectLoader throw
// "Curves[data.path.type] is not a constructor" on the render side, so such geometries must be baked.
const BUILTIN_CURVES = new Set([
  'ArcCurve', 'CatmullRomCurve3', 'CubicBezierCurve', 'CubicBezierCurve3', 'EllipseCurve',
  'LineCurve', 'LineCurve3', 'QuadraticBezierCurve', 'QuadraticBezierCurve3', 'SplineCurve',
]);

// Parametric geometries whose JSON form ObjectLoader cannot rehydrate:
//  - dirty attributes (the module edited vertices after construction) -> parameters no longer describe the mesh
//  - no `static fromJSON` on the class (EdgesGeometry, WireframeGeometry, user subclasses) -> "Geometries[data.type].fromJSON is not a function"
//  - TubeGeometry / ExtrudeGeometry driven by a non-built-in curve -> "Curves[data.path.type] is not a constructor"
function geometryNeedsBake(geo) {
  if (geometryIsDirty(geo)) return true;
  const ctor = THREE[geo.type];
  if (!ctor || typeof ctor.fromJSON !== 'function') return true;
  if (geo.type === 'TubeGeometry') {
    const path = geo.parameters && geo.parameters.path;
    if (!path || !BUILTIN_CURVES.has(path.type)) return true;
  }
  if (geo.type === 'ExtrudeGeometry') {
    const ep = geo.parameters && geo.parameters.options && geo.parameters.options.extrudePath;
    if (ep && !BUILTIN_CURVES.has(ep.type)) return true;
  }
  return false;
}

function bakeModifiedParametricGeometries(root) {
  const baked = new Map();
  root.traverse((o) => {
    const geo = o.geometry;
    if (!geo || geo.parameters === undefined || !geometryNeedsBake(geo)) return;
    if (!baked.has(geo)) {
      const bg = new THREE.BufferGeometry().copy(geo);
      bg.userData = {
        ...(geo.userData || {}),
        originalGeometry: { type: geo.type, parameters: { ...geo.parameters } },
      };
      baked.set(geo, bg);
    }
    o.geometry = baked.get(geo);
  });
}

function run() {
  const { source } = workerData;

  const origRandom = Math.random;
  const savedGlobals = new Map();
  const savedCtors = [];

  let moduleLoadMs = 0;
  let executionMs = 0;
  const moduleStart = NowMs();

  try {
    let seed = 0x12345678;
    Math.random = () => {
      seed |= 0; seed = seed + 0x6D2B79F5 | 0;
      let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };

    for (const name of TRAPPED_GLOBALS) {
      if (!(name in globalThis)) continue;
      savedGlobals.set(name, Object.getOwnPropertyDescriptor(globalThis, name));
      try {
        Object.defineProperty(globalThis, name, {
          get() { trapViolation(name); },
          configurable: true,
        });
      } catch {
        try { globalThis[name] = undefined; } catch {}
      }
    }

    for (const proto of CODEGEN_PROTOTYPES) {
      const desc = Object.getOwnPropertyDescriptor(proto, 'constructor');
      savedCtors.push({ proto, desc });
      Object.defineProperty(proto, 'constructor', {
        get() { trapViolation('Function constructor'); },
        configurable: true,
      });
    }

    const body = `'use strict';\n${source}`;
    let factory;
    try {
      factory = new SafeFunction(body);
    } catch (err) {
      moduleLoadMs = NowMs() - moduleStart;
      send({
        failures: [{
          stage: 'module_load',
          rule: 'EXECUTION_THREW',
          detail: `compilation failed: ${err && err.message ? err.message : String(err)}`,
        }],
        metrics: null,
        moduleLoadMs,
        executionMs: 0,
      });
      return;
    }
    moduleLoadMs = NowMs() - moduleStart;

    const execStart = NowMs();
    let generate;
    try {
      generate = factory();
    } catch (err) {
      executionMs = NowMs() - execStart;
      const rule = err && err.message && err.message.includes('Runtime violation')
        ? 'RUNTIME_VIOLATION'
        : 'EXECUTION_THREW';
      send({
        failures: [{ stage: 'module_load', rule, detail: err && err.message ? err.message : String(err) }],
        metrics: null,
        moduleLoadMs,
        executionMs,
      });
      return;
    }

    if (typeof generate !== 'function') {
      executionMs = NowMs() - execStart;
      send({
        failures: [{
          stage: 'module_load',
          rule: 'INVALID_RETURN_TYPE',
          detail: `default export is not a function (got ${typeof generate})`,
        }],
        metrics: null,
        moduleLoadMs,
        executionMs,
      });
      return;
    }

    let rawResult;
    try {
      rawResult = generate(THREE);
    } catch (err) {
      executionMs = NowMs() - execStart;
      const rule = err && err.message && err.message.includes('Runtime violation')
        ? 'RUNTIME_VIOLATION'
        : 'EXECUTION_THREW';
      send({
        failures: [{ stage: 'execution', rule, detail: err && err.message ? err.message : String(err) }],
        metrics: null,
        moduleLoadMs,
        executionMs,
      });
      return;
    }
    executionMs = NowMs() - execStart;

    if (
      rawResult !== null &&
      (typeof rawResult === 'object' || typeof rawResult === 'function') &&
      typeof rawResult.then === 'function'
    ) {
      send({
        failures: [{
          stage: 'execution',
          rule: 'ASYNC_NOT_ALLOWED',
          detail: 'generate() returned a Promise or thenable',
        }],
        metrics: null,
        moduleLoadMs,
        executionMs,
      });
      return;
    }

    let postResult;
    try {
      postResult = postValidate(rawResult);
    } catch (err) {
      send({
        failures: [{
          stage: 'post_validation',
          rule: 'EXECUTION_THREW',
          detail: err && err.message ? err.message : String(err),
        }],
        metrics: null,
        moduleLoadMs,
        executionMs,
      });
      return;
    }

    let object = null;
    if (postResult.failures.length === 0) {
      try {
        bakeModifiedParametricGeometries(rawResult);
        object = rawResult.toJSON();
      } catch (err) {
        send({
          failures: [{
            stage: 'post_validation',
            rule: 'SERIALIZE_THREW',
            detail: `toJSON() failed: ${err && err.message ? err.message : String(err)}`,
          }],
          metrics: postResult.metrics,
          moduleLoadMs,
          executionMs,
        });
        return;
      }
    }

    send({
      failures: postResult.failures,
      metrics: postResult.metrics,
      moduleLoadMs,
      executionMs,
      object,
    });
  } finally {
    Math.random = origRandom;
    for (const [name, desc] of savedGlobals) {
      try { Object.defineProperty(globalThis, name, desc); }
      catch { try { globalThis[name] = desc?.value; } catch {} }
    }
    for (const { proto, desc } of savedCtors) {
      try { Object.defineProperty(proto, 'constructor', desc); }
      catch {}
    }
  }
}

try {
  run();
} catch (err) {
  send({
    failures: [{
      stage: 'execution',
      rule: 'EXECUTION_THREW',
      detail: err && err.stack ? err.stack : String(err),
    }],
    metrics: null,
    moduleLoadMs: 0,
    executionMs: 0,
  });
}
