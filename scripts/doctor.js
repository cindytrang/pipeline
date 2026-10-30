const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const root = path.resolve(__dirname, '..');
let missing = false;
if (Number(process.versions.node.split('.')[0]) === 24) {
  console.log(`OK: Node ${process.versions.node} matches the project runtime.`);
} else {
  console.error(`Use Node 24 for the tested project setup; current version is ${process.versions.node}.`);
  missing = true;
}
for (const dependency of ['express', 'hbs']) {
  try {
    require.resolve(dependency, { paths: [path.join(root, 'frontend')] });
    console.log(`OK: ${dependency} is installed.`);
  } catch {
    console.error(`Missing ${dependency}. Run npm run setup.`);
    missing = true;
  }
}
const python = spawnSync(process.env.PYTHON || 'python3', ['--version'], { encoding: 'utf8' });
console.log(python.status === 0 ? `OK for importer tests: ${(python.stdout || python.stderr).trim()}` : 'Python is unavailable: the demo can run, but importer tests need Python 3.');
console.log(fs.existsSync(path.join(root, 'pipeline', 'resources')) ? 'Model resources folder exists; verify your run artifacts separately.' : 'Model resources are not installed. They are optional for the synthetic frontend demo.');
process.exitCode = missing ? 1 : 0;
