# Data

The repository does not contain images, videos, frames or weights. Git ignores everything in `/data/` except this README.
The source data sets show identifiable faces. Do not commit them, and do not commit outputs that show faces.

## Sources

| Source | Where | Terms | Content |
|---|---|---|---|
| DAiSEE (Dataset for Affective States in E-Environments) | Request access on the official DAiSEE page | Research use only, see its license | Videos of learners in online sessions. Each clip has levels 0-3 for boredom, engagement, confusion and frustration |
| YawDD (Yawning Detection Dataset) | On request from the YawDD authors (IEEE DataPort) | Research use only, see its terms | Driver videos with normal, talking and yawning segments |

Use the data only for research, under the terms of each source, and with the consent rules of your organisation.

## State labels

| Source label | engagecam state | Rule |
|---|---|---|
| DAiSEE clip levels | `boredom`, `confusion`, `frustration`, `engagement` or `active` | `states.daisee_state`: the strongest of boredom, confusion, frustration at level 2 or more. Else `engagement` at level 2 or more. Else `active` |
| YawDD `normal`, `talking` | `active` | `states.yawdd_state` |
| YawDD `yawning` | `yawn` | `states.yawdd_state` |
| `sleep` | `sleep` | No public source in this list. Add a source with eyes-closed clips and keep its subject ids |

## Expected layout

Extract frames (for example 1 frame per second), crop the face, and save each frame as:

```
data/frames/<source>/<state>/<subject>__<clip>__<frame>.jpg
```

Then build the manifest:

```bash
engagecam manifest data/frames --out data/manifest.csv
engagecam validate data/manifest.csv
```

The manifest columns are `path, subject_id, clip_id, source, state, frame`. The subject and clip ids are necessary for the subject-disjoint split.

## Synthetic data (no download)

```bash
engagecam synth --out data/synthetic      # drawn face-like frames, no real person
engagecam demo                            # in memory, nothing is written
```

The tests and the offline demo use only synthetic data.
