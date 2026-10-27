# Pipeline-to-frontend data contract

The frontend loads JSON objects with a `datapackID` string and a `chats` object. Each chat contains a `content` array. Messages have `type: "message"`, an author, body and optional Unix timestamp in milliseconds. The frontend treats `prediction` as a message score between 0 and 1 and displays string `labels` when present. Separators have `type: "separator"`.

```json
{
  "datapackID": "my-local-result",
  "description": "An annotated local run",
  "chats": {
    "example": {
      "authorOnRightSide": "Alex",
      "content": [
        {
          "type": "message",
          "author": "Alex",
          "body": "A harmless example message.",
          "time": 1704067200000,
          "prediction": 0.2,
          "labels": ["Example label"]
        }
      ]
    }
  }
}
```

The importer rejects malformed chat/content structures, non-string message bodies and labels, and non-numeric, infinite, NaN or out-of-range predictions. Scores can be omitted for unscored messages. Additional historical metadata is retained. The importer checks the input contract but does not perform model inference or transform label names.

Chat-level `className` and `predictedClassName` support the existing classification summaries. `HTMLclassNames: ["warning"]` supports the historical warning display. These fields are optional.

Do not interpret arbitrary scores or sample labels as validated predictions. The bundled synthetic fixture intentionally uses made-up scores to exercise the visualisation.

Imported JSON is ignored by the root `.gitignore`; only `frontend/datapacks/synthetic-demo.json` is tracked. Imported data is trusted local input: validation is not a general-purpose content sanitizer. Imported datapack IDs should be distinct, since the legacy loader indexes objects by their ID.
