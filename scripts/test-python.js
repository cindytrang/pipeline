const { spawnSync } = require('node:child_process');
const path = require('node:path');
const executable = process.env.PYTHON || 'python3';
const result = spawnSync(executable, ['-m', 'unittest', 'discover', '-s', 'tests', '-v'], {
  cwd: path.resolve(__dirname, '..'), stdio: 'inherit',
});
if (result.error) {
  console.error(`Could not run ${executable}. Install Python 3 or set PYTHON to its executable.`);
  process.exit(1);
}
process.exit(result.status === null ? 1 : result.status);
