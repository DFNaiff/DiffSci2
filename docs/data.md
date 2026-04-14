# Data

The four rocks used in the paper are from Imperial College London's
pore-scale modelling group:

<https://www.imperial.ac.uk/earth-science/research/research-groups/pore-scale-modelling/micro-ct-images-and-networks/>

## Files expected

Under `$DIFFSCI2_DATA_ROOT` (default: `<repo>/saveddata/raw/imperial_college`):

| Rock         | Voxel size (µm) | `.raw` filename                            |
|--------------|-----------------|--------------------------------------------|
| Bentheimer   | 3.0035          | `Bentheimer_1000c_3p0035um.raw`            |
| Doddington   | 2.6929          | `Doddington_1000c_2p6929um.raw`            |
| Estaillades  | 3.31136         | `Estaillades_1000c_3p31136um.raw`          |
| Ketton       | 3.00006         | `Ketton_1000c_3p00006um.raw`               |

Each `.raw` is a $1000^3$ binary volume (uint8). Each has matching
`.mhd` metadata and a pre-extracted `.network.npz` (SNOW2 pore-network
dump).

## Fetch

Run:

```bash
python tools/download_data.py
```

On the first run, without a manifest, this prints the list of missing
files and the canonical source URL. Provide a local manifest at
`tools/data_manifest.json` with per-file URLs + sha256s to make the
downloader idempotent.

## License

The four-rock dataset is distributed by Imperial College London with
citation requested. Cite the original papers listed on their page when
reusing.
