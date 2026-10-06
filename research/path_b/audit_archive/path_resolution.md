# Path resolution for evidence locators

Baseline `40b9b0708d888a015abe5043bb273c3c6ee601ae`. Manuscript SHA256 `5ee7257be5a09c9862c2ead2f98416e1e8d1b2fa875624ca009df5cb95934bbb`. Exact inspected scope and commands: input_manifest.json; commands_run.md.

Evidence strings inside the preserved upstream audit rows are the locators those audits used at the time. They are not assertions that those paths exist on any particular machine.

- `research/...`, `scripts/...`, `tests/...` and repository-root filenames refer to the immutable checkout at `40b9b0708d888a015abe5043bb273c3c6ee601ae`. Verify with the hashes in `input_manifest.json`.
- `/home/ubuntu/p1_audit/...`, `/home/ubuntu/p2_audit/...`, `/home/ubuntu/p3_audit/...` and bare scratch filenames such as `sources/marques.xml` refer to the upstream audits' own scratch directories, which are not part of this package. The corresponding bytes are hashed in each bundle's manifest, and all 171 upstream manifest entries were re-verified here (51+72+48, with overlapping files).
- `sources/` inside an upstream bundle means that bundle's evidence copy, not a repository path.
- Two basenames collide across namespaces: `input_hashes.json` and `artifact_manifest.json` each exist in more than one bundle with different contents. They are distinct artifacts, not mismatches.
- Raw copyrighted primary and historical sources are excluded from this ZIP and identified by path, size and SHA256 in `input_manifest.json` (including every upstream file).
- `rows['9']` in a selectivity locator selects the row whose `source_compound` is `9`; it is not list index 9.
- P3 audit flags use D4 for identity/reference boundaries and D5 for rediscovery; P1/P2 use D4 for candidate identity and D5 for reference mismatch. These source-local flags are preserved, not silently swapped. The unified CSV Flag namespace column makes this explicit; the material gap register uses the project-wide P1/P2 convention.
- Source-local discrepancy IDs in the selectivity artifacts are a different namespace from the project D1-D7 flags.
