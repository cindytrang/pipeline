# Local development and testing

## First run

```sh
npm run setup
npm run doctor
npm run test:all
npm start
```

Open http://127.0.0.1:3000 and select the synthetic datapack, then its example chat. Check that the three messages, labels, heatmap and prediction chart are visible. The synthetic scores are illustrative, not model results.

## Development loop

1. Choose a GitHub issue with clear acceptance criteria.
2. Create a branch, for example `git switch -c update/improve-chart-labels`.
3. Make a small change and run the relevant tests.
4. Use `npm run dev` to try it in the browser. Stop it with Ctrl+C.
5. Run `npm run test:all`, review `git diff`, commit, and open a pull request.
6. Review the GitHub Actions results before merging.

`npm test` remains a frontend-only shortcut. `npm run test:pipeline` runs the JSON handoff tests; it does not run the legacy ML pipeline. Set `PYTHON=python3.12` if you want to select a specific Python executable.

## Test a real annotated output

```sh
python3 scripts/import_datapack.py /absolute/path/to/output.json --name my-local-run.json
npm start
```

The importer refuses overwrites by default. The output stays local and is ignored by Git. Restart the app after adding a datapack. Only import trusted input; the validator is not an HTML sanitizer.

## Common setup problems

- **Missing Express or hbs:** run `npm run setup` from the root repository folder.
- **Wrong Node version:** use `nvm install` and `nvm use`, or install Node 24 directly.
- **Python not found:** install Python 3 for the importer tests; the synthetic frontend needs only Node.
- **Port 3000 is occupied:** start with `PORT=3001 npm start`.
- **Model artifacts are missing:** use the synthetic demo while retrieving the datasets and trained runs needed by the pipeline.
- **Templates or watch behaviour changed:** run `npm run dev` again; nodemon is a development dependency and is not installed with `--omit=dev`.

## CI jobs

- `demo`: locked frontend installation and HTTP/rendering checks.
- `datapack`: standard-library Python importer tests.
- `dependency-audit`: npm audit, failing for high/critical advisories. Moderate findings remain visible and must be triaged separately.

Use the Actions tab's Run workflow button to trigger the workflow manually. Workflow definitions can be read in `.github/workflows/checks.yml`; local tests use the same commands as CI.
