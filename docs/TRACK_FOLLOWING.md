# Track following

## Pipeline

```text
camera frame
  └─ PathRibbonEncoder          track/ribbon.py
       ├─ PathObservation       lateral, heading, confidence (row-by-row stripe search)
       └─ ribbon_frames         160x320 white ribbon on black, `track_substeps` per frame
  └─ track brain                track/brains.py
       1. place cursor:  y = (1 - anchor) * y + anchor * observation.lateral
       2. perceive:      y += brain motion over the ribbon frames
       └─ BrainEstimate         lateral = y, velocity = motion added this frame
  └─ TrackController            track/steer.py
       turn = clip(kp * lateral + kd * velocity, -1, 1)
       forward = base_speed * (1 - 0.5 |turn|)
       left, right = forward ± turn_gain * turn     → MotorDecoder → motor_command
       or forward/yaw_rate/lateral/confidence       → track_command
       lost for `track_lost_frames` → zero output + emergency_stop
```

### The chart analogy

| price chart (research) | camera path (this project) |
|---|---|
| candle body centres, time left → right | path centre per image row, near → far |
| price up = ribbon up | path right of vehicle = ribbon up |
| blank prediction column at the right | blank column beyond the look-ahead |
| chart sweep in 16 sub-steps | interpolated centreline in `track_substeps` sub-steps |
| cursor placed on last price (`start_y`) | cursor placed on measured path (`track_anchor`) |
| T4c/T5c − T4d/T5d cursor | same readout (connectome) or a vertical Reichardt array |

Sign conventions: lateral > 0 and yaw_rate > 0 both mean "path to the right,
turn right". Image columns grow to the right; image rows grow downward.

## The research behind it

In `bee/reports/FINDINGS.md` the connectome cursor followed price at rho
0.84-0.91, peaking at lag +1, and never led. Randomly wired brains followed
better (0.97). Two checks were run while building this fork.

**1. What produced the following.** The runner placed the cursor at the last
body centre before each observation (`cursor.start_reference: last_body_center`),
and the reported rho is `final_y` against price position. On
`exp_2026_09_visual_a_real_litusdt_white_001/simulation_LITUSDT.parquet`:

| quantity | Spearman rho |
|---|---|
| start_y vs final_y | 0.93 |
| brain delta vs change in price position, same observation | 0.005 |
| brain delta vs next change | 0.006 |
| brain delta vs previous change | 0.011 |

The follower was the placement. Random wiring followed "better" because its delta
disturbed the placement less.

**2. Direction selectivity.** `scripts/grating_test.py` drives the male MaleCNS
graph with full-field gratings. The per-cycle firing fraction is nearly the same
for every direction (e.g. T4c 0.0098 up / 0.0093 down, T4d 0.0123 / 0.0106). The
V0 engine has a single leak and threshold per neuron with no cell-type-specific
delays, so the correlator timing that makes real T4/T5 direction-selective is
missing. The up-minus-down cursor therefore carries no direction information.

**What transfers.** A follower that sits on the line and lags by a step is exactly
what a line-following vehicle needs, as long as "sitting on the line" comes from
the camera. That is the default here (`track_anchor: 1.0`). The motion signal
comes from a Hassenstein-Reichardt array, which is direction-selective by
construction (`tests/test_track.py::test_reichardt_detector_is_direction_selective`).

## Results

### Open loop (`scripts/open_loop_test.py`, 400 frames, seed 3, anchor 0)

The vehicle moves along a straight path with a smooth random offset (±0.1 m)
and heading (±0.3 rad), independent of any steering. The table correlates the
brain's per-frame motion with the measured change in path offset.

| brain | rho | best lag (frames) | rate |
|---|---|---|---|
| Reichardt | 0.779 | −2 (0.800) | 146 Hz |
| male connectome, real | 0.080 | −4 (0.120) | 12 Hz |
| male connectome, random | 0.029 | −3 (0.036) | 6 Hz |

The Reichardt signal's small negative lag is look-ahead, not prediction: the
ribbon includes far rows that shift before the near rows used for the measured
offset.

### Closed loop (CLI defaults, upstream decoder smoothing 0.25, dead zone 5)

Every run below completed its course (99%) with no lost frames.

| track | kd 0 | kd 0.3 (default) | kd 1.0 |
|---|---|---|---|
| curvy seed 1 | 12.1 mm | 14.5 mm | 18.3 mm |
| curvy seed 2 | 13.0 mm | 13.8 mm | 16.9 mm |
| curvy seed 3 | 7.8 mm | 7.8 mm | 11.9 mm |
| sine | 21.3 mm | 24.2 mm | 29.8 mm |

In this simulator the motion term does not reduce error; it is on by default
at a light weight. Set `track_kd: 0` for pure placement.

Connectome arms, curvy seed 1, 450 frames, decoder smoothing 0.5 / dead zone 2:

| setting | RMS | max | rate |
|---|---|---|---|
| placement only | 4 mm | 7 mm | 93 Hz |
| placement + real connectome | 35 mm | 108 mm | 12 Hz |
| placement + random connectome | 5 mm | 11 mm | 5 Hz |
| real connectome, anchor 0 | lost after ~1% | | 15 Hz |

### Reproducing the numbers

```bash
python scripts/open_loop_test.py
python scripts/open_loop_test.py --backend connectome-track \
    --connectome ../bee/data/processed/male_connectome.npz --wiring random
python scripts/grating_test.py ../bee/data/processed/male_connectome.npz
printf 'track_kd: 0.0\n' > kd0.yaml
flybrain-track sim --kind curvy --seed 2 --config kd0.yaml
```

## Connectome backend

```bash
pip install -e ".[connectome]"
cat > config.yaml <<'EOF'
backend: connectome-track
track_connectome_path: ../bee/data/processed/male_connectome.npz   # or female_connectome.npz
track_wiring: real        # real | shuffled | random
EOF
python -m flybrain_robot.main --synthetic-track --steps 300
```

The `.npz` files come from the beedictor preparation pipeline (FlyWire FAFB v783
female, 134k neurons; MaleCNS v1.0 male, 164k neurons; ≥5 synapses, log1p,
per-target normalised). They are not distributed with this repository. Engine
parameters are the research's calibrated values
(`track/connectome/defaults.py`). A useful next experiment is to add
cell-type-specific delays (fast Mi1/Tm3 versus slow Mi4/Mi9 inputs to T4), rerun
`grating_test.py`, and only then retest following against the shuffled and
random arms.

## Drones

`--output track` sends one `track_command` per frame:

| field | range | meaning |
|---|---|---|
| `forward` | −100..100 | forward speed, same scale as `max_speed` |
| `yaw_rate` | −100..100 | turn rate; positive = right |
| `lateral` | −1..1 | path offset in the image, positive = right |
| `confidence` | 0..1 | fraction of look-ahead rows where the path was found |
| `emergency_stop` | bool | true when telemetry is stale or the path is lost |

On a companion computer, scale `forward` and `yaw_rate` to body-frame velocity
and yaw-rate setpoints (for example PX4 offboard or ArduPilot guided mode). You
can optionally feed `lateral` into a sideways-velocity correction. On
`emergency_stop` or no packet for 500 ms, hold position or land. The bridge
expects `telemetry` datagrams back from the vehicle, as in the upstream
protocol, and stops when they go stale. Keep a downward or forward-down camera
at a fixed height so the path's width and look-ahead stay consistent.

## Tuning a real camera

Work from recordings: `flybrain-track check --video clip.mp4` for detection, then
`flybrain-track run --video clip.mp4 --record out.mp4` to see the detected path
(green dots), the search region, the brain stimulus and the steering for every
frame.

- **`track_roi`:** use the image fraction that shows the ground just ahead, where the path is sharp.
- **`track_polarity`:** `dark` for black tape on a light floor.
- **`min_contrast` / `max_width`** (`PathRibbonEncoder` arguments, not yet config settings): raise `min_contrast` on noisy floors, and lower `max_width` if large bright areas get picked up as the path.
- **`track_kp`, `track_turn_gain`, `track_base_speed`:** start slow, and raise `kp` until the vehicle oscillates, then back off.
- **`dead_zone`:** the upstream default of 5 hides small corrections at low speed; 2 works better for line following.
