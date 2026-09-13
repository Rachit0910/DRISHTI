"""Every word the user hears, in one place.

India is multilingual, and an assistive device that only speaks English excludes a
large share of the people it is meant to serve. Localising Drishti must therefore be
a data change, not a code change — so no user-facing string appears anywhere in the
logic. The policy engine decides *what* to say; a PhraseBook decides *how it sounds*.

Adding a language means adding one PhraseBook and installing a matching system voice.
It does not mean touching the salience engine, the depth calibrator or the scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PhraseBook:
    """The complete user-facing vocabulary for one language."""

    locale: str

    # Spoken directions, ordered left to right.
    bearings: tuple[str, str, str, str, str]

    # Distance vocabulary.
    immediate: str          # closer than one metre
    unit_singular: str      # "one metre"
    unit_plural: str        # "{n} metres"
    far: str                # beyond the useful range

    # Sentence templates. {label}, {distance}, {bearing} are substituted.
    routine: str
    routine_immediate: str
    critical: str

    # Camera state messages, keyed by the quality gate's reason codes.
    camera: dict[str, str] = field(default_factory=dict)

    def bearing(self, center_x: float) -> str:
        if center_x < 0.30:
            return self.bearings[0]
        if center_x < 0.42:
            return self.bearings[1]
        if center_x <= 0.58:
            return self.bearings[2]
        if center_x <= 0.70:
            return self.bearings[3]
        return self.bearings[4]

    def distance(self, distance_m: float) -> str:
        if distance_m < 1.0:
            return self.immediate
        if distance_m < 5.0:
            metres = int(round(distance_m))
            return self.unit_singular if metres == 1 else self.unit_plural.format(n=metres)
        return self.far


ENGLISH = PhraseBook(
    locale="en",
    bearings=("to your left", "slightly left", "ahead", "slightly right", "to your right"),
    immediate="right in front of you",
    unit_singular="one metre",
    unit_plural="{n} metres",
    far="further ahead",
    routine="{label}, {distance}, {bearing}.",
    routine_immediate="{label} {distance}.",
    critical="Stop. {label} {distance}.",
    camera={
        "occluded": "Camera is covered.",
        "dark": "Too dark to see.",
        "blurred": "Camera image is blurred.",
        "clear": "Camera clear.",
    },
)


# A second language, included to prove the abstraction holds rather than merely
# claiming it would. Verify the wording with a native speaker before shipping, and
# note that speaking it requires a matching Hindi voice installed on the system.
HINDI = PhraseBook(
    locale="hi",
    bearings=("बाईं ओर", "थोड़ा बाएं", "सामने", "थोड़ा दाएं", "दाईं ओर"),
    immediate="बिल्कुल सामने",
    unit_singular="एक मीटर",
    unit_plural="{n} मीटर",
    far="और आगे",
    routine="{label}, {distance}, {bearing}।",
    routine_immediate="{label} {distance}।",
    critical="रुकिए। {label} {distance}।",
    camera={
        "occluded": "कैमरा ढका हुआ है।",
        "dark": "देखने के लिए बहुत अंधेरा है।",
        "blurred": "कैमरे की तस्वीर धुंधली है।",
        "clear": "कैमरा साफ़ है।",
    },
)


PHRASEBOOKS: dict[str, PhraseBook] = {book.locale: book for book in (ENGLISH, HINDI)}


def get(locale: str) -> PhraseBook:
    """Look up a phrasebook, falling back to English for anything unsupported."""
    return PHRASEBOOKS.get(locale.split("-")[0].lower(), ENGLISH)
