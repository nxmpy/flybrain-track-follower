"""Track-following brain backends.

Both backends read the ribbon frames produced by PathRibbonEncoder, turn vertical
ribbon motion (lateral path motion) into a signed velocity, and integrate it into
a lateral cursor as the beedictor readout did.

Every frame the cursor is first placed toward the measured path, like the
research's `cursor.begin(start_y)` placed the bee on the last price. That
placement is what makes the system a follower: in the research the brain's own
delta correlated with price moves at rho ~0.005, and here an anchor-only run
follows as well as any brain (see docs/TRACK_FOLLOWING.md). `anchor=0` removes
the placement and leaves the brain on its own.
"""

import numpy as np

from ..brain.base import BrainBackend, MotorActivity
from .steer import BrainEstimate, TrackController


class TrackBrain(BrainBackend):
    names = ("lateral", "velocity", "turn", "forward", "confidence")

    def __init__(self, encoder, controller=None, anchor=1.0):
        if not 0 <= anchor <= 1:
            raise ValueError("anchor must be between 0 and 1")
        self.encoder = encoder
        self.controller = controller or TrackController()
        self.anchor = anchor
        self.y = 0.0
        self.estimate = BrainEstimate()
        self.activity = MotorActivity()

    @property
    def state(self):
        c = self.controller.command
        return np.array(
            [self.estimate.lateral, self.estimate.velocity, c.turn, c.forward, c.confidence]
        )

    def reset(self):
        self.y = 0.0
        self.estimate = BrainEstimate()
        self.activity = MotorActivity()
        self.controller.reset()

    def perceive(self, frames, y):
        """Run the frames through the brain; return the new cursor position."""
        raise NotImplementedError

    def step(self, sensory_input, telemetry, dt):
        if not (dt > 0 and np.isfinite(dt)):
            raise ValueError("dt must be finite and positive")
        observation = self.encoder.observation
        start = self.y
        if observation.confidence >= self.controller.min_confidence:
            start = (1 - self.anchor) * self.y + self.anchor * observation.lateral
        self.y = float(self.perceive(self.encoder.ribbon_frames, start))
        self.estimate = BrainEstimate(lateral=self.y, velocity=self.y - start)
        self.activity = self.controller.update(self.estimate, observation)
        return self.activity

    def get_motor_activity(self):
        return self.activity


class MotionNormaliser:
    """Causal running baseline and scale, as in the research Cursor."""

    def __init__(self, min_samples=50):
        self.mean, self.mad, self.count, self.min_samples = 0.0, 0.0, 0, min_samples

    def __call__(self, value):
        ready = self.count >= self.min_samples and self.mad > 1e-9
        z = float(np.clip((value - self.mean) / self.mad, -3, 3)) if ready else 0.0
        self.count += 1
        rate = 1 / self.count if self.count <= 2000 else 5e-4
        self.mean += rate * (value - self.mean)
        self.mad += rate * (abs(value - self.mean) - self.mad)
        return z


class OptomotorTrackBrain(TrackBrain):
    """Hassenstein-Reichardt correlator array: the textbook insect motion detector.

    NumPy only and fast, so it runs on small boards and in CI. It is a model of
    the T4/T5 computation, not a connectome.
    """

    def __init__(self, encoder, controller=None, anchor=1.0, gain=0.02, adapt_tau=0.35,
                 delay=0.5):
        super().__init__(encoder, controller, anchor)
        self.gain, self.adapt_tau, self.delay = gain, adapt_tau, delay
        self.reset()

    def reset(self):
        super().reset()
        self.baseline = None
        self.delayed = None
        self.normaliser = MotionNormaliser()

    def perceive(self, frames, y):
        for frame in frames:
            lum = frame[::2, ::2]
            if self.baseline is None:
                self.baseline = lum.copy()
                self.delayed = np.zeros_like(lum)
            contrast = lum - self.baseline  # L1/L2-like temporal contrast
            self.baseline += self.adapt_tau * (lum - self.baseline)
            # Image rows grow downward, so upward motion reaches row r+1 first.
            up = self.delayed[1:] * contrast[:-1] - contrast[1:] * self.delayed[:-1]
            self.delayed += self.delay * (contrast - self.delayed)
            velocity = float(up.mean())
            y = float(np.clip(y + self.gain * self.normaliser(velocity), -2, 2))
        return y


class ConnectomeTrackBrain(TrackBrain):
    """EXPERIMENTAL: the research connectome simulator with the T4/T5 cursor readout.

    Loads a prepared FlyWire (female) or MaleCNS (male) `.npz` from the beedictor
    pipeline. `wiring` selects the research control arms: real, shuffled
    (degree-preserving) or random. With the V0 threshold engine the T4/T5
    populations are not direction-selective (scripts/grating_test.py), so this
    backend adds error rather than tracking; it is kept to test future engines.
    """

    def __init__(self, encoder, path, controller=None, anchor=1.0, wiring="real",
                 research_config=None):
        super().__init__(encoder, controller, anchor)
        if not path:
            raise ValueError("track_connectome_path is required for connectome-track")
        from .connectome import csr, retinotopy
        from .connectome.cursor import Cursor
        from .connectome.defaults import research_config as defaults
        from .connectome.engine import NUMBA, Brain
        from .connectome.retina import Encoder, build_map

        if not NUMBA:
            raise ValueError('connectome-track needs numba: pip install -e ".[connectome]"')

        cfg = research_config or defaults()
        visual = cfg["visual"]
        if (encoder.height, encoder.width) != (visual["viewport_h"], visual["viewport_w"]):
            raise ValueError("Ribbon viewport must match the research viewport")
        connectome = csr.load(path)
        seed = int(cfg["project"]["seed"])
        if wiring == "shuffled":
            connectome = csr.shuffle_preserving_degree(connectome, seed)
        elif wiring == "random":
            connectome = csr.random_graph(connectome, seed)
        elif wiring != "real":
            raise ValueError("wiring must be real, shuffled or random")
        retinotopy.assign(connectome, cfg)
        self.wiring = wiring
        self.connectome = connectome
        self.cycles = int(cfg["simulator"]["cycles_per_step"])
        self.brain = Brain(connectome, cfg)
        self.retina = Encoder(build_map(connectome, cfg), cfg, connectome.n_neurons)
        self.cursor = Cursor(connectome, cfg)
        self.traversals = 0
        self.frames_seen = 0

    def reset(self):
        super().reset()
        if hasattr(self, "brain"):
            self.brain.reset()
            self.retina.reset()

    def perceive(self, frames, y):
        self.cursor.begin(y)
        for frame in frames:
            drive = self.retina.encode(frame)
            for _ in range(self.cycles):
                fired, traversals = self.brain.step(drive)
                self.traversals += traversals
                self.cursor.update(fired)
                # Drive applies on the first cycle only; later cycles propagate.
                drive = np.zeros_like(drive)
        self.frames_seen += 1
        if self.frames_seen % 30 == 0:
            self.brain.health_check()
        return self.cursor.y

    @property
    def participation(self):
        return self.brain.participation
