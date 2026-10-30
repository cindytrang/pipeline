# Dependency health

Audit date: **2026-10-30**. Findings describe the scanned versions on that date and will change as new advisories are published.

## Frontend

Before the development-tool update, npm audit reported three high-severity affected package entries in the nodemon → simple-update-notifier → semver chain, plus one moderate hbs finding. These are affected dependency entries, not four independent exploit reports.

This change moves nodemon into development dependencies and upgrades it from 2.0.22 to 3.1.14. The updated lockfile audit reports **zero high/critical findings and one moderate finding**.

The remaining finding is hbs 4.4.0, GHSA-7f5c-rpf4-86p8 / CVE-2021-32822. The advisory describes possible file disclosure depending on how template data and render options are used, and lists no patched hbs release. [Issue #3](https://github.com/cindytrang/pipeline/issues/3) tracks replacing or isolating this integration. This audit is not evidence that the local app has been exploited.

- Advisory: https://github.com/advisories/GHSA-7f5c-rpf4-86p8
- Recheck: `npm run audit:frontend`
- CI fails on high/critical frontend findings; moderate findings remain visible and need separate triage.
- Express 4.22.3 is installed. Express 5 is a planned major-version migration ([issue #9](https://github.com/cindytrang/pipeline/issues/9)); the current npm audit did not identify an Express vulnerability.

Dependabot configuration proposes weekly npm and GitHub Actions updates. It takes effect when `.github/dependabot.yml` is on the default branch. Updates do not merge automatically.

## Python ML dependency snapshot

A pip-audit scan checked the 169 entries in `pipeline/requirements.txt`; advisories were reported for **33 packages**. The scan used `--no-deps --disable-pip`, so it checked the saved pinned versions without resolving or executing a complete environment. It does not establish which components are reachable during a particular model run, whether the environment currently installs, or whether an exploit is possible in this project.

| Package | Pinned version | Reported advisory entries |
| --- | --- | --- |
| accelerate | 0.20.3 | 1 |
| aiohttp | 3.8.6 | 88 |
| bleach | 6.0.0 | 2 |
| certifi | 2023.11.17 | 4 |
| datasets | 2.13.2 | 1 |
| filelock | 3.12.2 | 6 |
| flair | 0.12.2 | 2 |
| fsspec | 2023.1.0 | 1 |
| gdown | 4.4.0 | 2 |
| httplib2 | 0.22.0 | 2 |
| idna | 3.6 | 4 |
| keras | 2.9.0 | 25 |
| lxml | 4.9.3 | 2 |
| markdown | 3.4.4 | 2 |
| numpy | 1.21.6 | 1 |
| oauthlib | 3.2.2 | 2 |
| pillow | 9.5.0 | 68 |
| protobuf | 3.19.6 | 4 |
| pyarrow | 12.0.1 | 3 |
| pyasn1 | 0.5.1 | 8 |
| requests | 2.31.0 | 6 |
| scikit-learn | 1.0.2 | 2 |
| scipy | 1.7.3 | 2 |
| sentencepiece | 0.1.99 | 2 |
| soupsieve | 2.4.1 | 6 |
| sqlitedict | 2.1.0 | 2 |
| tensorflow | 2.9.3 | 44 |
| torch | 1.13.1 | 28 |
| tqdm | 4.66.1 | 2 |
| transformers | 4.30.2 | 49 |
| urllib3 | 1.25.11 | 19 |
| werkzeug | 2.2.3 | 15 |
| zipp | 3.15.0 | 2 |

This scan covers `pipeline/requirements.txt` only; it is not a complete audit of every historical environment or bundled browser asset.

Reproduce the saved-pin audit in a separate environment with a Python version supported by pip-audit:

```sh
python3 -m venv .audit-venv
.audit-venv/bin/python -m pip install pip-audit
.audit-venv/bin/python -m pip_audit -r pipeline/requirements.txt --no-deps --disable-pip
```

Do not interpret this as the model installation procedure. The Python files currently disagree on runtime versions. [Issue #4](https://github.com/cindytrang/pipeline/issues/4) tracks a supported and tested environment baseline. Upgrading Torch, TensorFlow, Flair and Transformers can affect artifact compatibility and prediction behaviour; validate changes against a restored inference baseline.
