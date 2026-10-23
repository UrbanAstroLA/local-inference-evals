# Writer census scripts (amendment 22ar)

Zero-GPU static analysis behind `../P2-WRITER-CENSUS.md`. Run on CPU, with the source trees extracted read-only from
the stock 0.9.1 base image `a9e985c0` (commands in `../P2-WRITER-CENSUS.md` §2).

- `census4.py`: taint pass from the KV-cache carriers to every call site, in-place op and store. Writes
  `out/census4.json` and `out/sinks4.tsv`.
- `classify.py`: assigns each sink to groups G1-G21. Writes `out/groups.json`.
- `b12x_stores.py`: lists the global-store targets in the b12x sparse-MLA kernel bodies.

`ROOT`/`OUT` at the top of each script point at the working directory used for the run (`<census-work>`). Set them
to the location of the extracted trees (`src/py/vllm`, `src/py/b12x`) to re-run: `python3 census4.py && python3
classify.py`, about 5 minutes. The extracted trees (329 MB) are not included; they are reproducible from the image
by digest. File hashes: see `../P2-WRITER-CENSUS.md` §7.
