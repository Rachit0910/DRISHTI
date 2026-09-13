"""Salience and narration policy — the layer between raw detections and speech.

This module is the reason Drishti is usable rather than merely functional. A naive
assistive app pipes every detection into a text-to-speech engine and produces an
unlistenable stream: "chair chair chair person chair". Speech is a strictly serial,
low-bandwidth channel, so the hard problem is not detecting things, it is deciding
which one thing is worth a second and a half of the user's attention right now.

The policy has four parts:

  1. Salience scoring   - hazard class x proximity x centrality x confidence
  2. Critical override  - imminent hazards interrupt, bypassing the budget
  3. Novelty suppression- never repeat the same fact until it changes or decays
  4. Utterance budgeting- a hard ceiling on words per unit time

Everything here is pure and deterministic given a clock value, which is what makes
it unit-testable without a camera, an NPU, or a speaker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable

from .phrases import ENGLISH, PhraseBook


class Hazard(IntEnum):
    """How much a class of object matters to someone walking."""

    IGNORE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


# Keyed by COCO class names, which is what YOLOv8-Detection emits.
# Anything unlisted falls back to Hazard.LOW.
HAZARD_BY_LABEL: dict[str, Hazard] = {
    # Things that will hurt you
    "car": Hazard.CRITICAL,
    "bus": Hazard.CRITICAL,
    "truck": Hazard.CRITICAL,
    "motorcycle": Hazard.CRITICAL,
    "train": Hazard.CRITICAL,
    "bicycle": Hazard.HIGH,
    # Things that move unpredictably
    "person": Hazard.HIGH,
    "dog": Hazard.HIGH,
    "cat": Hazard.MEDIUM,
    # Things you walk into
    "chair": Hazard.MEDIUM,
    "bench": Hazard.MEDIUM,
    "couch": Hazard.MEDIUM,
    "dining table": Hazard.MEDIUM,
    "potted plant": Hazard.MEDIUM,
    "fire hydrant": Hazard.MEDIUM,
    "stop sign": Hazard.LOW,
    "parking meter": Hazard.MEDIUM,
    "traffic light": Hazard.LOW,
    # Things that are useful context but not obstacles
    "tv": Hazard.IGNORE,
    "laptop": Hazard.IGNORE,
    "cell phone": Hazard.IGNORE,
    "book": Hazard.IGNORE,
}

# Below this distance a HIGH or CRITICAL hazard interrupts whatever is being said.
CRITICAL_DISTANCE_M = 1.2

# Proximity falls off exponentially with this length scale, so "close" dominates.
PROXIMITY_TAU_M = 2.5

# Objects outside the walking path still matter, just less. This is the floor.
PERIPHERAL_WEIGHT = 0.25

# An object must close at least this fraction of its last-announced distance before
# it counts as "approaching". Monocular depth is noisy; without this hysteresis an
# estimate wobbling either side of a band boundary re-announces the same obstacle
# several times a second, which is exactly the failure mode this policy exists to stop.
APPROACH_HYSTERESIS = 0.25


@dataclass(frozen=True)
class Observation:
    """One detected object, fused from the detector and the depth model."""

    label: str
    confidence: float
    # Normalised box in [0, 1]: (x1, y1, x2, y2)
    bbox: tuple[float, float, float, float]
    distance_m: float

    @property
    def center_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def hazard(self) -> Hazard:
        return HAZARD_BY_LABEL.get(self.label, Hazard.LOW)


@dataclass(frozen=True)
class Utterance:
    """A sentence the policy has decided is worth saying."""

    text: str
    salience: float
    interrupt: bool = False


def bearing_of(center_x: float, phrases: PhraseBook = ENGLISH) -> str:
    """Turn a horizontal position into a spoken direction."""
    return phrases.bearing(center_x)


def distance_phrase(distance_m: float, phrases: PhraseBook = ENGLISH) -> str:
    """Turn metres into something a person can act on."""
    return phrases.distance(distance_m)


def _distance_bucket(distance_m: float) -> int:
    """Coarse distance band, used as part of the novelty key.

    Buckets are deliberately chunky: a person drifting from 3.1 m to 2.9 m has not
    told you anything new, but crossing from 3 m to 1 m has.
    """
    if distance_m < 1.0:
        return 0
    if distance_m < 2.0:
        return 1
    if distance_m < 3.5:
        return 2
    if distance_m < 6.0:
        return 3
    return 4


def salience_of(obs: Observation) -> float:
    """Score how much this observation deserves the speech channel.

    hazard x proximity x centrality x confidence, each in a sane range, so the
    product stays interpretable when debugging why something did or did not get said.
    """
    if obs.hazard is Hazard.IGNORE:
        return 0.0

    proximity = math.exp(-max(obs.distance_m, 0.0) / PROXIMITY_TAU_M)

    # 1.0 dead ahead, falling to PERIPHERAL_WEIGHT at the frame edges.
    offset = min(abs(obs.center_x - 0.5) * 2.0, 1.0)
    centrality = PERIPHERAL_WEIGHT + (1.0 - PERIPHERAL_WEIGHT) * (1.0 - offset)

    return float(obs.hazard) * proximity * centrality * obs.confidence


@dataclass
class NarrationPolicy:
    """Decides what gets spoken, and refuses to say anything else.

    Args:
        max_utterances: ceiling on non-critical utterances per `window_s`.
        window_s: the budget window.
        repeat_cooldown_s: how long before the same fact may be repeated.
        min_salience: floor below which nothing is worth saying at all.
    """

    max_utterances: int = 3
    window_s: float = 10.0
    repeat_cooldown_s: float = 12.0
    min_salience: float = 0.35
    phrases: PhraseBook = ENGLISH

    # (label, bearing) -> (when last spoken, which distance band, distance in metres)
    _spoken_at: dict[tuple[str, str], tuple[float, int, float]] = field(default_factory=dict)
    _recent: list[float] = field(default_factory=list)

    def tick(self, observations: Iterable[Observation], now: float) -> list[Utterance]:
        """Return the utterances to speak for this frame — usually none.

        Args:
            observations: fused detections for the current frame.
            now: monotonic seconds. Passed in rather than read so this is testable.
        """
        scored = sorted(
            ((salience_of(o), o) for o in observations),
            key=lambda pair: pair[0],
            reverse=True,
        )
        scored = [(s, o) for s, o in scored if s >= self.min_salience]
        if not scored:
            return []

        self._expire(now)
        out: list[Utterance] = []

        for score, obs in scored:
            critical = (
                obs.distance_m <= CRITICAL_DISTANCE_M
                and obs.hazard >= Hazard.HIGH
            )
            key = (obs.label, bearing_of(obs.center_x, self.phrases))
            bucket = _distance_bucket(obs.distance_m)

            if not self._is_novel(key, bucket, obs.distance_m, now) and not critical:
                continue

            if not critical and len(self._recent) >= self.max_utterances:
                # Budget exhausted this window. Stay quiet rather than babble.
                break

            out.append(
                Utterance(
                    text=self._phrase(obs, critical=critical),
                    salience=score,
                    interrupt=critical,
                )
            )
            self._spoken_at[key] = (now, bucket, obs.distance_m)
            if not critical:
                self._recent.append(now)

            if critical:
                # A critical hazard is the only thing worth hearing right now.
                return out
            # At most one routine utterance per frame; speech is serial.
            break

        return out

    def _is_novel(
        self, key: tuple[str, str], bucket: int, distance_m: float, now: float
    ) -> bool:
        """Has anything changed that the user actually needs to hear?

        Three things count as novel: a thing never mentioned, a thing whose mention
        has decayed past the cooldown, and a thing that is genuinely approaching —
        meaning it has both crossed into a nearer distance band *and* closed at least
        APPROACH_HYSTERESIS of its last-announced distance. Both conditions are
        required: the band alone is too twitchy for noisy monocular depth, and the
        margin alone would re-fire within a band.

        Moving further away is deliberately never novel. It is not actionable, and
        announcing it would double the chatter for no benefit.
        """
        last = self._spoken_at.get(key)
        if last is None:
            return True
        last_time, last_bucket, last_distance = last
        if (now - last_time) >= self.repeat_cooldown_s:
            return True
        approaching = distance_m <= last_distance * (1.0 - APPROACH_HYSTERESIS)
        return bucket < last_bucket and approaching

    def _expire(self, now: float) -> None:
        cutoff = now - self.window_s
        self._recent = [t for t in self._recent if t > cutoff]
        horizon = self.repeat_cooldown_s * 3
        stale = [k for k, (t, _, _) in self._spoken_at.items() if now - t > horizon]
        for k in stale:
            del self._spoken_at[k]

    def _phrase(self, obs: Observation, *, critical: bool) -> str:
        """Render the sentence. Every word comes from the phrasebook, never from here."""
        book = self.phrases
        label = obs.label.capitalize()
        where = book.distance(obs.distance_m)
        bearing = book.bearing(obs.center_x)

        if critical:
            # Front-load the actionable word; the user may only hear the first syllable.
            return book.critical.format(label=label, distance=where, bearing=bearing)
        if where == book.immediate:
            return book.routine_immediate.format(label=label, distance=where, bearing=bearing)
        return book.routine.format(label=label, distance=where, bearing=bearing)
