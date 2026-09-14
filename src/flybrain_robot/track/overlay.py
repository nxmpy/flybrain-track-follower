"""Annotated preview: what the camera saw, what was detected and how it steers."""

import cv2
import numpy as np

# BGR
GREEN, AMBER, RED = (120, 230, 100), (80, 190, 245), (70, 70, 235)
BLUE, GREY = (230, 200, 110), (170, 170, 170)


def _text(image, text, origin, color=(235, 235, 235), scale=0.45, thickness=1):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2,
                cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def camera_panel(frame, encoder, size=(320, 240)):
    """Camera frame with the search region and detected path centres drawn on it."""
    w, h = size
    image = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    top = int(round((1 - encoder.roi) * (h - 1)))
    shade = image.copy()
    shade[:top] = (shade[:top] * 0.45).astype(np.uint8)
    image = shade
    cv2.line(image, (0, top), (w - 1, top), BLUE, 1, cv2.LINE_AA)
    _text(image, "search region", (6, top + 16), BLUE, 0.4)
    centres = encoder.observation.centres
    for fraction, centre in zip(encoder.row_fractions(), centres):
        if np.isnan(centre):
            continue
        x = int(round((centre + 1) / 2 * (w - 1)))
        cv2.circle(image, (x, int(round(fraction * (h - 1)))), 3, GREEN, -1, cv2.LINE_AA)
    cv2.line(image, (w // 2, h - 12), (w // 2, h - 1), GREY, 1)
    return image


def status_panel(encoder, brain, left, right, lost, size=(320, 240)):
    w, h = size
    image = np.full((h, w, 3), 24, np.uint8)
    _text(image, "brain stimulus (path right = up)", (10, 17), GREY, 0.4)
    if encoder.ribbon_frames:
        ribbon = (encoder.ribbon_frames[-1] * 255).astype(np.uint8)
        ribbon = cv2.resize(ribbon, (200, 100), interpolation=cv2.INTER_AREA)
        image[26:126, 10:210] = ribbon[..., None]
    obs, steer = encoder.observation, brain.controller.command
    side = "right" if obs.lateral > 0.05 else ("left" if obs.lateral < -0.05 else "ahead")
    turn = "right" if steer.turn > 0.05 else ("left" if steer.turn < -0.05 else "straight")
    _text(image, f"path  {side} {obs.lateral:+.2f}", (10, 150))
    _text(image, f"seen  {obs.confidence:.0%}", (10, 175))
    ok = obs.confidence >= brain.controller.min_confidence
    cv2.rectangle(image, (100, 165), (210, 177), (60, 60, 60), -1)
    cv2.rectangle(image, (100, 165), (100 + int(110 * obs.confidence), 177),
                  GREEN if ok else AMBER, -1)
    _text(image, f"steer {turn} {steer.turn:+.2f}", (10, 200))
    _text(image, f"wheels  L {left:+d}   R {right:+d}", (10, 225), GREEN)
    # steering dial on the right
    cx, cy = 265, 150
    cv2.circle(image, (cx, cy), 42, (60, 60, 60), 1, cv2.LINE_AA)
    angle = steer.turn * 1.2
    tip = (int(cx + 38 * np.sin(angle)), int(cy - 38 * np.cos(angle)))
    cv2.arrowedLine(image, (cx, cy), tip, AMBER, 3, cv2.LINE_AA, tipLength=0.3)
    _text(image, "steering", (cx - 32, cy + 60), GREY, 0.4)
    if lost:
        cv2.rectangle(image, (0, 0), (w - 1, h - 1), RED, 4)
        _text(image, "PATH LOST - STOPPED", (w // 2 - 95, 80), RED, 0.6, 2)
    return image


def render(frame, encoder, brain, left, right, lost):
    """Side-by-side 640x240 BGR preview."""
    return np.hstack(
        [camera_panel(frame, encoder), status_panel(encoder, brain, left, right, lost)]
    )


class Preview:
    """Writes annotated frames to a video file and/or a live window."""

    def __init__(self, record=None, show=False, fps=30):
        self.record, self.show, self.fps = record, show, fps
        self.writer = None
        if show:
            try:
                cv2.namedWindow("flybrain-track", cv2.WINDOW_AUTOSIZE)
            except cv2.error as exc:
                raise ValueError(
                    "Live preview needs an OpenCV build with GUI support "
                    "(pip install opencv-python). Use --record preview.mp4 instead."
                ) from exc

    def __call__(self, image):
        """Returns False when the user closes the live window."""
        if self.record:
            if self.writer is None:
                h, w = image.shape[:2]
                self.writer = cv2.VideoWriter(
                    str(self.record), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h)
                )
                if not self.writer.isOpened():
                    raise ValueError(f"Cannot write preview video: {self.record}")
            self.writer.write(image)
        if self.show:
            cv2.imshow("flybrain-track", image)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                return False
        return True

    def close(self):
        if self.writer is not None:
            self.writer.release()
        if self.show:
            cv2.destroyAllWindows()
