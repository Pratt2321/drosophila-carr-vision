# 🪰 Drosophila Embodied 3D Compound-Eye Vision & Looming Escape Simulation

A biologically grounded 3D embodied simulation pipeline that models *Drosophila melanogaster* in FlyGym's MuJoCo physics arena reacting to basketball broadcast footage (Coen Carr fast break highlights).

![Drosophila watching Coen Carr](output/carr_escape_jump.gif)

---

## 🔬 Scientific & Computational Architecture

```
Video Stream (YouTube URL / Local MP4)
       │
       ├──────────────────────────────────────────────┐
       ▼                                              ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────────────┐
│  FlyGym Retina (~721 Ommatidia)      │  │  Virtual Screen Inset (Top-Right)   │
│  - Letterbox/pillarbox aspect fit    │  │  - Dynamic source header indicator   │
│  - Pale / Yellow spectral types      │  │  - Broadcast video playback          │
│  - Green & UV photoreceptor channels │  └──────────────────────────────────────┘
└──────────────────┬───────────────────┘
                   │ Ommatidia Readout (721, 2)
                   ▼
┌──────────────────────────────────────┐
│  Lobula Plate & Giant Fiber Circuit  │
│  - Temporal derivative ΔI(t)         │
│  - Radial angular symmetry filter    │
│  - Simultaneous ring contrast test   │
│  - Central field expansion rate      │
└──────────────────┬───────────────────┘
                   │ Giant Fiber Spike
                   ▼
┌────────────────────────────────────────────────────────┐
│  Embodied NeuroMechFly 3D Physics (MuJoCo)            │
│  - 3rd-person over-the-shoulder arena camera           │
│  - 42 active leg joint position servos                 │
│  - Stereotyped jump reaction:                          │
│    FREEZE (crouch) -> JUMP (kick) -> FLIGHT -> RECOVER │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  Composited Scientific Viewport (960×540)              │
│  - Main: 3D fly body in MuJoCo arena                   │
│  - Top-Right: Virtual broadcast screen inset           │
│  - Bottom-Right: 721-facet ommatidia readout inset     │
│  - Bottom-Left: Telemetry HUD, looming bar & GF badge  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
                   H.264 (yuv420p) MP4
```

---

## 🏆 Tested Showcase Highlights

While **Coen Carr's explosive fast breaks** serve as the primary showcase, the pipeline has also been benchmarked and verified across classic NBA transition sequences:

- **Blake Griffin vs. Kendrick Perkins (2012 Poster Dunk):**
  - **Source:** YouTube (`OZ9o7owSUcQ` at 02:08)
  - **Benchmark Result:** The automated transition filter successfully detects and suppresses the metallic circular broadcast countdown wipe ("#3") at 00:03. The Giant Fiber escape reflex triggers with high biological fidelity only when Griffin elevates directly toward the camera to finish the dunk at 00:10.

---

## 🚀 Quickstart: Test Any Video

### 1. Environment Setup
```bash
python3.11 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

### 2. Run the Default Coen Carr Showcase
```bash
./run.sh
```

### 3. Run Custom Videos (YouTube or Local File)
Run directly from an arbitrary YouTube video (automatically downloaded and cached in `inputs/`):
```bash
# Run directly from a YouTube link
./run.sh --url "https://youtu.be/<ID>" --start-sec 45 --count 500

# Or run a local file
./run.sh --video "path/to/video.mp4" --start-sec 10 --count 400
```

---

## ⚙️ Command-Line Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--url` | | `None` | YouTube video URL to download and process |
| `--video` | `-v` | `coen_carr.mp4` | Local video file path fallback (used when `--url` is not set) |
| `--output` | `-o` | `output/fly_reaction.mp4` | Destination MP4 output path |
| `--start-sec` | | `140.0` | Start timestamp in seconds (default: 02:20 = Carr fast break) |
| `--start` | `-s` | `None` | Start frame index (overrides `--start-sec` if provided) |
| `--count` | `-n` | `600` | Number of sequential frames to process |
| `--duration-sec` | | `None` | Duration in seconds (overrides `--count`, using stream FPS) |

---

## 📊 Pipeline Components & Features

### 1. Dynamic Video Ingestion & Caching (`resolve_video`)
- **`--url`**: Downloads the optimal MP4+M4A stream via `yt-dlp` into `inputs/`, indexed by a deterministic SHA-256 hash. Cached streams are reused automatically on subsequent runs.
- **`--video`**: Seamlessly falls back to local media files.
- **Dynamic Header**: Insets display `VIRTUAL SCREEN // YOUTUBE: <VIDEO_ID>` for streams or `VIRTUAL SCREEN // <FILENAME>` for local clips.

### 2. Aspect-Ratio Preserving Letterboxing (`letterbox_resize`)
- Any incoming video aspect ratio (16:9, 4:3, square 1:1, or 9:16 vertical mobile) is safely letterboxed/pillarboxed with black padding before projecting onto the compound-eye retina or HUD display, preventing artificial optical stretching.

### 3. Stream FPS Adaptation
- Queries stream metadata via `cv2.CAP_PROP_FPS` to drive simulation timing, timecode badges, and frame rate encoding (falling back to standard 24 fps if unspecified).

### 4. Graphic Wipe & Radial Transition Filter (`LoomingDetector`)
- Television broadcasts often feature synthetic wipes, spinning logos, or metallic countdown graphics that cause whole-screen brightness shifts. The detector filters out non-biological transitions via:
  - **Simultaneous Ring Contrast:** Detects uniform contrast surges across both peripheral ($r > 0.60$) and central ($r < 0.40$) rings.
  - **Circular Symmetry:** Evaluates coefficient of variation across 8 radial angular sectors ($CV_\theta < 0.52$).
  - **Peripheral Dominance:** Identifies outward edge-sweeps where peripheral expansion outpaces central expansion.

### 5. Biological Retinal Mapping (`HexRetina`)
- Maps video frames to 721 ommatidia per eye using FlyGym's hexagonal photoreceptor model, simulating green (pale) and UV/violet (yellow) spectral sensitivity.

### 6. Embodied NeuroMechFly 3D Physics (`Embodied3DFlySim`)
- FlyGym's `NeuroMechFly` model in `FlatGroundWorld` with active leg joints and articulated wing hinges (`l_wing_yaw`, `l_wing_pitch`, `r_wing_yaw`, `r_wing_pitch`).
- Giant Fiber activation initiates a stereotyped kinematic reflex:
  - **FREEZE:** Crouch lowering center of mass.
  - **JUMP:** Ballistic leg extension impulse launching into positive vertical clearance.
  - **FLIGHT:** High-frequency wing flutter and leg retraction.
  - **RECOVERY:** Smooth cosine-interpolated return to neutral standing posture.

### 7. Web & VS Code H.264 Transcoding (`transcode_to_h264`)
- Automatically encodes outputs using `libx264` (YUV420p) with `+faststart` metadata for instant streaming in VS Code, QuickTime, and modern web browsers.

---

## 🔬 Known Biological & Computational Limitations

When running arbitrary videos through the pipeline, visual responses reflect the specific neuroethology of fly looming detection:

1. **Looming Velocity & Scale ($\frac{d\theta}{dt}$):**
   - The Giant Fiber escape circuit specifically responds to accelerating radial optical expansion. Objects filmed from a distance, moving tangentially across the visual field (lateral motion), or shown in slow motion do not produce sufficient central angular velocity to trip the reflex.
2. **Framing & Central Receptive Field:**
   - In nature, impending predatory threats (e.g. birds, swatters) loom directly toward the head. In videos, the approaching subject must be reasonably centered and occupy a significant fraction of the field of view. Far-away subjects will not trip the threshold unless zoomed or cropped.
3. **Broadcast & Compression Artifacts:**
   - Sudden camera cuts, flash photography, strobe lighting, and rapid digital transitions can occasionally trigger brief transient spikes. While the global shift filter and radial circular symmetry filter suppress most television wipes, extreme full-screen strobe sequences can occasionally challenge threshold adaptation.
