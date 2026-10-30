# Historical pipeline setup and limitations

The scripts are retained for research continuity; they have not been upgraded or retrained as part of the repository merge.

## Environment evidence

- `pipeline/.python-version` specifies Python 3.7.17.
- `pipeline/environment.yml` specifies Python 3.9.15 and TensorFlow 2.6.0.
- The other requirement/environment snapshots contain additional versions, including older TensorFlow, Torch and Flair packages.

These files are historical alternatives, not a single verified dependency lock. Do not assume installing them into the current system Python will work. Reconstruct a compatible isolated environment (for example from the original training machine) before running model code. Keep the frontend environment independent.

## Required local files

- The dataset expected by the chosen script. `add_predictions.py` currently selects `PJZ.json` inside the `data_dir` stored in run settings.
- `pipeline/resources/<run-id>/run_settings.json` and the corresponding trained classifier artifacts.
- spaCy’s `en_core_web_sm` model and the downloaded Transformers/TorchVision resources used by the historical scripts.

The model runs and datasets are not present in this repository. Several university job scripts contain historical machine paths and need adapting. No original research chat data is bundled in the current frontend tree.

## Historical commands

Run from the pipeline directory, inside the reconstructed ML environment:

```sh
cd pipeline
python train_flair.py --dataset Corpus --project flair --seq_len 512 --model bert_classifier
```

`Corpus` is an example dataset indicator, not a bundled dataset. Annotation expects an existing run:

```sh
python add_predictions.py --run YOUR_EXISTING_RUN_ID --model_version non_quantized --eval_mode segments --window_size 124
```

The script looks up `resources/YOUR_EXISTING_RUN_ID/run_settings.json` and writes a JSON datapack under that run’s `non_quantized/message_based_eval/` directory. `add_predictions.py` uses a `normalised-datapack-...json` filename. Inspect the actual run directory to locate the output, then import it using the root `scripts/import_datapack.py` script.

## Deliberately unresolved research-code issues

The historical scripts perform heavyweight model loading at import time, use local data paths, and contain duplicated experimental logic. Runtime compatibility and model correctness need a separate, focused validation pass with the original artifacts. The merge does not claim to fix or validate these behaviours. Use `add_predictions.py` as the documented original entry point; `annotate_datapack_with_predictions.py` remains a historical variant.
