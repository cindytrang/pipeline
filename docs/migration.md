# Migration and provenance

## Original source revisions

| Component | Repository | Original commit |
| --- | --- | --- |
| Pipeline | https://github.com/cindytrang/pipeline | `b55cd3033f9b47ae881db0bb0a43ec1470d82eda` |
| Demo | https://github.com/cindytrang/chat-grooming-demo | `8184e42d9bf23ed21678e2c02ab97501e2053b4c` |

The combined commit has both source revisions as parents. No existing commit is rewritten. The original repositories are not deleted or archived by this local preparation.

## File moves and exclusions

- Pipeline-owned code, environment snapshots, plots, job scripts and license files are under `pipeline/`.
- Demo server, templates, CSS, JavaScript, images and required browser vendor assets are under `frontend/`.
- Demo-side Python scripts and their dependency snapshot are retained under `frontend/legacy_pipeline/` so experimental variants are not silently overwritten by similarly named pipeline files. They are not used to start the JavaScript demo.
- The checked-in `new_env/` environment from the demo and the `Python-3.7.5/` interpreter source tree from the pipeline are excluded from the combined working tree.
- Original frontend datapacks are not copied to the current source tree. A newly written, harmless synthetic fixture makes the interface testable without research data.
- Frontend static paths are resolved relative to the frontend directory. The server has a direct Node start command, a dependency lockfile and automated smoke tests.

## History size and data

Keeping both histories means the removed environment, interpreter sources and historical datapacks remain reachable through old commits. This cleanup reduces the current source tree, **not** the historical object store. Do not describe this as removal of historical secrets or data. A separate history rewrite would change commit IDs and requires a different migration strategy.

## Publication

Choose whether to publish as a branch in the pipeline repository or as a separate repository. Do not force-push or delete the originals as part of the merge. Verify the demo and retrieve the original model artifacts before deciding whether to archive the source repositories.
