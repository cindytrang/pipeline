# Project improvement backlog

Start with the local demo, then restore the model pipeline in dependency order.

| Order | Task | GitHub issue |
| --- | --- | --- |
| 1 | Resolve the hbs dependency advisory | [#3](https://github.com/cindytrang/pipeline/issues/3) |
| 2 | Define a supported Python ML environment | [#4](https://github.com/cindytrang/pipeline/issues/4) |
| 3 | Locate trained models, run settings and datasets | [#5](https://github.com/cindytrang/pipeline/issues/5) |
| 4 | Establish an inference regression baseline | [#6](https://github.com/cindytrang/pipeline/issues/6) |
| 5 | Test heatmaps, labels and warning thresholds in a browser | [#7](https://github.com/cindytrang/pipeline/issues/7) |
| 6 | Refactor model configuration and execution | [#8](https://github.com/cindytrang/pipeline/issues/8) |
| 7 | Assess Express 5 compatibility | [#9](https://github.com/cindytrang/pipeline/issues/9) |

Model setup and artifact restoration can proceed together. Inference regression checks depend on both; use them to protect behaviour before refactoring model code. The Express major upgrade is a planned compatibility task, not a reported Express security fix.
