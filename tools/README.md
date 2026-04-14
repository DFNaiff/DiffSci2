# tools/

Small, focused helpers that sit outside the numbered pipelines.

| Script                         | Purpose                                                                       |
|--------------------------------|-------------------------------------------------------------------------------|
| `download_checkpoints.py`      | Fetch the published VAE + base + FC-257 + FC-129 bundle into `savedmodels/`.  |
| `download_data.py`             | Fetch the Imperial College four-rock micro-CT volumes into `saveddata/raw/`.  |
| `clean_corrupted_notebook.py`  | Recover source from a corrupted Jupyter notebook.                             |
| `diagnose_conditioning.py`     | Inspect an existing checkpoint's conditioning wiring.                          |
| `test_spatial_parallel.py`     | Integration test for `diffsci2.distributed`.                                  |

Each downloader is idempotent: re-running it only fetches what is missing
or corrupt.
