# Multi-Object Tracker Notes

## What is a Tentative Track?

When a tracker sees a new detection it doesn't know yet whether it's a real object or a noise/false-positive from the detector. To protect against creating junk tracks, it holds the detection in a **tentative** state for a few frames before committing to it.

```
Frame 1:  detection appears  →  track created (tentative, TID assigned internally)
Frame 2:  detection seen again →  still tentative
Frame 3:  detection seen again →  min_hits reached → track CONFIRMED, TID emitted in output
Frame 4+: normal confirmed track
```

If the detection disappears before `min_hits` is reached, the tentative track is silently dropped — no TID is ever emitted.

### Key parameters

| Parameter | What it does |
|---|---|
| `min_hits` | Consecutive frames a detection must appear before TID is output. Default is usually 3. Set to **1** to emit TIDs immediately. |
| `track_buffer` / `max_age` | How many frames a confirmed track survives without a matching detection before being deleted. |

---

## Trackers Available (BoxMot)

### ByteTrack ⭐ recommended for speed
- Uses **both high- and low-confidence** detections. High-conf detections create/confirm tracks; low-conf detections are used only to re-associate existing tracks.
- No appearance (re-ID) model — purely motion-based (Kalman filter + IoU).
- Fast, good for real-time.
- `min_hits=1` makes it instant.

### OcSort (Observation-Centric SORT)
- Builds on SORT but re-links tracks using the **observation history** rather than predicted positions, which handles occlusions better.
- Still motion-only (no re-ID).
- Better ID consistency than ByteTrack when objects are temporarily occluded.

### SFSORT (Simple Fast SORT)
- Lightweight variant tuned for high-frame-rate scenarios.
- Good balance between speed and accuracy.
- Motion-only.

### BoostTrack
- State-of-the-art accuracy.
- Combines **appearance embeddings** (re-ID) + confidence boosting + detection refinement.
- Significantly reduces ID switches compared to the others.
- Slowest of the four — best suited for offline processing.

---

## Comparison

| Tracker | Appearance (re-ID) | Occlusion handling | Speed | Accuracy |
|---|---|---|---|---|
| ByteTrack | No | OK | Fastest | Good |
| OcSort | No | Better | Fast | Better |
| SFSORT | No | OK | Fast | Good |
| BoostTrack | Yes | Best | Slowest | Best |

---

## This Project's Setup

All trackers are initialised with `min_hits=1` so every detection gets a TID on its first frame — no tentative waiting period. The `track_buffer` (ByteTrack only) controls how long a confirmed track persists through missed frames before being dropped.

```python
boxmot.ByteTrack(track_buffer=track_buffer, min_hits=1)
boxmot.OcSort(min_hits=1)
boxmot.SFSORT(min_hits=1)
boxmot.BoostTrack(min_hits=1)
```
