# E3 routing-lens demo asset

This directory contains the two-minute E3 acceptance demonstration referenced by the routing-interpretability guide.

| File | Purpose |
|---|---|
| `e3-p2-demo.zh-CN.mp4` | 120-second H.264/AAC demonstration with Chinese narration and burned captions |
| `e3-p2-demo.zh-CN.srt` | Auditable caption wording and timing |
| `e3-p2-demo.json` | Duration, codec, size, SHA-256, caption, font, and voice metadata |
| `e3-p2-caption-proof.png` | Start, middle, and end caption spot-check |

The video is generated from archived routing evidence and a browser UI. It does not modify model forward behavior. Dominant-expert colors are categorical argmax assignments rather than clusters or semantic labels.
