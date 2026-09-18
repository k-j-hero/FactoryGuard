"""Deterministic per-pair event state; no inference or video dependencies."""
from dataclasses import asdict, dataclass
from math import ceil, hypot


@dataclass(frozen=True)
class Track:
    track_id: int
    label: str
    bbox: tuple[float, float, float, float]
    confidence: float

    @property
    def anchor(self):
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)

    def anchor_for(self, mode):
        x1, y1, x2, y2 = self.bbox
        if mode == "center":
            return ((x1 + x2) / 2, (y1 + y2) / 2)
        if mode == "bottom_center":
            return ((x1 + x2) / 2, y2)
        raise ValueError(f"Unknown anchor mode: {mode}")


def normalized_proximity(person, forklift, width, height, anchor_mode="bottom_center"):
    """Smaller means closer in image space; this is not a physical distance."""
    if width <= 0 or height <= 0:
        raise ValueError("Frame dimensions must be positive")
    a = person.anchor_for(anchor_mode)
    b = forklift.anchor_for(anchor_mode)
    return hypot(a[0] - b[0], a[1] - b[1]) / hypot(width, height)


@dataclass
class Event:
    event_id: str
    person_id: int
    forklift_id: int
    start_frame: int
    end_frame: int  # inclusive; last observed close frame
    confirmed_frame: int
    start_sec: float
    end_sec: float  # exclusive; (end_frame + 1) / fps
    min_normalized_proximity: float
    observed_close_sec: float
    close_reason: str
    label: str = "potential near-miss candidate"

    def to_dict(self):
        return asdict(self)


@dataclass
class PairState:
    start: int
    last: int
    minimum: float
    samples: int = 1
    confirmed: int | None = None


class ProximityEngine:
    def __init__(self, fps, enter_threshold=0.08, exit_threshold=0.11,
                 min_duration_sec=0.5, lost_tolerance_sec=0.3,
                 anchor_mode="bottom_center"):
        if fps <= 0 or not 0 < enter_threshold < exit_threshold <= 1:
            raise ValueError("Invalid FPS or proximity thresholds")
        if min_duration_sec <= 0 or lost_tolerance_sec < 0:
            raise ValueError("Invalid temporal thresholds")
        if anchor_mode not in {"bottom_center", "center"}:
            raise ValueError("Invalid anchor mode")
        self.fps = fps
        self.enter = enter_threshold
        self.exit = exit_threshold
        self.required = max(1, ceil(min_duration_sec * fps))
        self.allowed_missing = int(lost_tolerance_sec * fps)
        self.anchor_mode = anchor_mode
        self.states = {}
        self.sequence = 0
        self.previous_frame = -1

    def _close(self, pair, reason):
        state = self.states.pop(pair)
        if state.confirmed is None:
            return None
        self.sequence += 1
        return Event(f"event_{self.sequence:05d}", *pair, state.start, state.last,
                     state.confirmed, state.start / self.fps,
                     (state.last + 1) / self.fps, state.minimum,
                     state.samples / self.fps, reason)

    def update(self, frame_index, tracks, width, height):
        if frame_index != self.previous_frame + 1:
            raise ValueError("Pass every decoded frame in order, including empty frames")
        self.previous_frame = frame_index
        people = [t for t in tracks if t.label == "person"]
        forklifts = [t for t in tracks if t.label == "forklift"]
        distances = {(p.track_id, f.track_id): normalized_proximity(
                     p, f, width, height, self.anchor_mode)
                     for p in people for f in forklifts}
        closed = []
        for pair in list(self.states):
            state = self.states[pair]
            missing_frames = frame_index - state.last - (1 if pair in distances else 0)
            if missing_frames > self.allowed_missing:
                event = self._close(pair, "track_lost")
                if event:
                    closed.append(event)
            elif pair in distances and distances[pair] > self.exit:
                event = self._close(pair, "separated")
                if event:
                    closed.append(event)
        for pair, distance in distances.items():
            if pair not in self.states:
                if distance > self.enter:
                    continue
                self.states[pair] = PairState(frame_index, frame_index, distance)
            else:
                state = self.states[pair]
                state.last = frame_index
                state.samples += 1
                state.minimum = min(state.minimum, distance)
            state = self.states[pair]
            # Missing frames do not count towards the observed close duration.
            if state.confirmed is None and state.samples >= self.required:
                state.confirmed = frame_index
        active = {pair for pair, state in self.states.items()
                  if state.confirmed is not None and pair in distances}
        return distances, active, closed

    def finish(self, reason="end_of_video"):
        events = [self._close(pair, reason) for pair in list(self.states)]
        return [event for event in events if event is not None]
