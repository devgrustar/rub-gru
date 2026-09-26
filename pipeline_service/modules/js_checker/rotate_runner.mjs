/**
 * Writes a rotation of the returned root into a candidate module's own `generate()`.
 *
 * Usage: node rotate_runner.mjs <module.mjs> <axis> <angleExpr> [<axis> <angleExpr> ...]
 * Each pair adds one `rotateOnWorldAxis` call in front of every return that belongs to generate()
 * itself, in the order given. World-axis rotation composes with whatever rotation the root already
 * carries, so a yaw followed by a tip lands where the probe measured it.
 * Prints the rewritten source on stdout; exits 1 with a reason on stderr when nothing was rewritten.
 */

import fs from 'node:fs';
import _traverse from '@babel/traverse';
import { parseSource } from './validator/parse.js';

const traverse = _traverse.default || _traverse;

const AXIS_VECTORS = { x: '1, 0, 0', y: '0, 1, 0', z: '0, 0, 1' };

function die(reason) {
  process.stderr.write(`${reason}\n`);
  process.exit(1);
}

/** The generate() function of `export default function ...`, or null. */
function defaultExportFunction(ast) {
  const stmt = ast.program.body.find(s => s.type === 'ExportDefaultDeclaration');
  if (!stmt) return null;
  const decl = stmt.declaration;
  const isFn =
    decl.type === 'FunctionDeclaration' ||
    decl.type === 'FunctionExpression' ||
    decl.type === 'ArrowFunctionExpression';
  return isFn ? decl : null;
}

/** Returns owned by `fn` itself — returns inside nested helper functions are skipped. */
function ownReturns(ast, fn) {
  const found = [];
  traverse(ast, {
    ReturnStatement(path) {
      if (path.getFunctionParent()?.node === fn) found.push(path.node);
    },
  });
  return found;
}

function indentOf(source, node) {
  const lineStart = source.lastIndexOf('\n', node.start) + 1;
  const prefix = source.slice(lineStart, node.start);
  return /^[ \t]*$/.test(prefix) ? prefix : '  ';
}

function rotationBlock(source, node, three, steps, indent) {
  const inner = `${indent}  `;
  const expr = source.slice(node.argument.start, node.argument.end);
  const lines = [`const __oriented = ${expr};`];
  for (const { axis, angle } of steps) {
    lines.push(`__oriented.rotateOnWorldAxis(new ${three}.Vector3(${AXIS_VECTORS[axis]}), ${angle});`);
  }
  lines.push(`__oriented.position.sub(new ${three}.Box3().setFromObject(__oriented).getCenter(new ${three}.Vector3()));`);
  lines.push('return __oriented;');
  return `{\n${lines.map(l => `${inner}${l}\n`).join('')}${indent}}`;
}

function rewrite(source, steps) {
  const { ast, failures } = parseSource(source);
  if (failures.length > 0) die(`parse failed: ${failures[0].rule} ${failures[0].detail || ''}`);

  const fn = defaultExportFunction(ast);
  if (!fn) die('no default-exported function found');
  const three = fn.params[0]?.type === 'Identifier' ? fn.params[0].name : null;
  if (!three) die('generate() has no named THREE parameter');

  const returns = ownReturns(ast, fn).filter(r => r.argument != null);
  if (returns.length === 0) die('generate() has no value-returning statement');

  let out = source;
  for (const node of [...returns].sort((a, b) => b.start - a.start)) {
    const block = rotationBlock(source, node, three, steps, indentOf(source, node));
    out = out.slice(0, node.start) + block + out.slice(node.end);
  }
  return out;
}

const [codePath, ...rest] = process.argv.slice(2);
if (!codePath) die('no source file path provided');
if (rest.length === 0 || rest.length % 2 !== 0) die('expected one or more <axis> <angleExpr> pairs');

const steps = [];
for (let i = 0; i < rest.length; i += 2) {
  const axis = rest[i];
  if (!(axis in AXIS_VECTORS)) die(`unknown axis '${axis}'`);
  steps.push({ axis, angle: rest[i + 1] });
}

let source;
try {
  source = fs.readFileSync(codePath, 'utf8');
} catch (err) {
  die(`cannot read ${codePath}: ${err.message || err}`);
}

process.stdout.write(rewrite(source, steps));
