# cairn_new snapshot

This repository contains the readable project files from cairn_new. The complete
capture-time snapshot, including prepared data, model checkpoints, binary reports
and readable files, is distributed as split .tar.zst assets in the matching GitHub
Release. cairn_dev is deferred and is not uploaded.

Authentication directories, credentials, Git internal metadata and transient
runtime/upload files are excluded; SNAPSHOT_MANIFEST.json lists every exclusion.
No PAT is stored in this repository or its Release.

The capture does not stop training. Results written after the capture are not
part of this snapshot. Existing scripts may refer to ../cairn_dev for the Python
environment or raw DRKG input; that directory is not included in this upload.

## Restore the complete snapshot

Download all cairn_new_snapshot.tar.zst.partNNN assets from one Release, then:

    cat cairn_new_snapshot.tar.zst.part* > cairn_new_snapshot.tar.zst
    zstd -d -c cairn_new_snapshot.tar.zst | tar -xf -

The resulting directory is cairn_new/. Individual Git-tracked files are a
browsable subset; the Release archive is the complete included file set.
