# Source provenance

This repository publishes the actual P3 Labs competition source and selected
results. The full original Git history, teammate branches and model archives
are retained separately in a private repository. The public history starts
with the cleaned source release to keep private data out of Git history.

`src/` and the historical `scripts/` files were copied byte for byte from research
commit `2b044b59cb95fa289fb992ab3b79336d68577277`. Their Git blob hashes were
verified when retrieved. The source includes 116 model/inference modules and
152 historical scripts. Private archive publication and old repository-specific
administration checks were left out.

The later experiment sources were recovered from the saved 7 November review
package, whose SHA-256 is
`d650bc105bd26babf381ffe98cb95cf73b92411a0d7b312108965a64df845a0a`.
Those Python files and experiment plans are unchanged. Their original research
archive bundlers are excluded because they assemble private inputs for transfer.

`source_manifest.json` records SHA-256 hashes for the retained original files.
The new source-check script and regression tests are clearly separate additions.
The private `artifacts/manifest.json` index is omitted. Release specifications
remain as records of the required source/model dependencies; the model files
themselves are not published here.

The score records are dated participant-supplied portal observations. The
9 November organizer announcement is the source for ninth position on the
Phase 2 advancement list. Team correspondence and account details are omitted.

This was a team project maintained by Faith Olopade, with AI assistance during
implementation and documentation. Competition datasets, organizer attachments,
trained weights and row-level prediction arrays are not distributed. The code
license does not grant rights to those external materials or imply endorsement
by the organizers.
