# Commands actually run in this consolidation

Baseline `40b9b0708d888a015abe5043bb273c3c6ee601ae`. `<repo>` is `/home/ubuntu/repos/NUDT5-Inhibitor-Breast-Cancer-Discovery`; `<scratch>` is `/home/ubuntu/path_b_consolidation`. Manuscript SHA256 `5ee7257be5a09c9862c2ead2f98416e1e8d1b2fa875624ca009df5cb95934bbb`. All source inspections are read-only; every output is written under `<scratch>`.

```sh
git -C <repo> rev-parse HEAD
git -C <repo> status --porcelain
git -C <repo> rev-parse --is-shallow-repository
git -C <repo> ls-tree -r --name-only 8c2a1b6990df1e140e15ab5a0eabea69b70eee14
git -C <repo> show 8c2a1b6990df1e140e15ab5a0eabea69b70eee14:<each of the five released files>
sha256sum <repo>/research/manuscript.md
python <scratch>/inspect_inputs.py          # claim IDs, exact slices, line coverage, partitions, enums
python <scratch>/validate_sources.py        # archives, manifests, 171 file hashes, Methods quotes, DOCX re-extraction, supplements, primary XML, frozen geometry
python <scratch>/build_claims.py            # unified claim table, annotated Methods files, reconciliation log
python <scratch>/add_caption_audit.py       # 10 generated captions/site notes from AST/frozen SVG
python <scratch>/build_gaps.py              # material gap register
python <scratch>/build_brief.py             # Path B brief, lab specification, UNSENT inquiry
python <scratch>/build_methods_summary.py   # corrected human-readable Priority 0 audit
python <scratch>/build_package.py           # manifest, integrity report, this file, ZIP
python <scratch>/validate_final.py          # final independent output/constraint/ZIP checks
python -m compileall -q <scratch>/final/audit_scripts
```

Direct inspection used `sed`, `grep`, `find`, `ls` and inline Python with `csv`, `json`, `hashlib`, `ast`, `zipfile` and `xml.etree.ElementTree` to read repository code, results, the geometry contract, the lab handoff, the archived primary XML and the upstream bundles. `read_environment_config` was used to read the enterprise blueprint; no configuration change was proposed.

Corrections made during the run, recorded rather than hidden: the first validator rejected the legitimate `projection` claim type and was fixed; a basename collision between unrelated bundles was initially asserted as a mismatch and was reclassified as a namespace difference; and the DOCX table-cell join initially stripped trailing whitespace, which produced eight cosmetic quote differences until the join rule matched python-docx. All three were re-run to completion.

Not run here: model fitting, permutation, coordinate recomputation, repository test suite/lint/type checking, figure regeneration, fresh publisher-source HTTP retrieval, experiments or Git history changes. A read-only clone/fetch and detached checkout obtained the baseline earlier in this session; no branch, commit or push was made. Scratch audit scripts were syntax-checked and final data/ZIP checks were run.
