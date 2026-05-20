# Artifact Handling Notes

Keep repository history focused on source code, configuration, and curated documentation.

## Generated Artifacts

Keep these outputs outside normal Git history:

- Raw datasets and local split exports.
- Generated embeddings.
- Model checkpoints.
- Prediction arrays and temporary CSV exports.
- Long training logs.

## Result Bundles

For a reported experiment, preserve the command line, config snapshot, checkpoint identifier, metric output, and prediction export in the same external folder.

## Documentation

When committing result notes, include the script name, data split, metric implementation, and model configuration so the number can be traced later.
