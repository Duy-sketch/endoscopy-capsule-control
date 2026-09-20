"""Runtime uncertainty switches for plant experiments."""

from __future__ import annotations

from dataclasses import dataclass


_VALID_NAMES = {"current", "localization", "robot"}


@dataclass(frozen=True)
class NoiseSwitches:
    """Select which plant uncertainties are active for one run."""

    current: bool = False
    localization: bool = False
    robot: bool = False

    @classmethod
    def none(cls) -> "NoiseSwitches":
        return cls()

    @classmethod
    def all(cls) -> "NoiseSwitches":
        return cls(current=True, localization=True, robot=True)

    @classmethod
    def from_names(cls, names: list[str] | tuple[str, ...]) -> "NoiseSwitches":
        normalized = [str(name).strip().lower() for name in names]

        if not normalized or normalized == ["none"]:
            return cls.none()

        if "all" in normalized:
            return cls.all()

        unknown = set(normalized) - _VALID_NAMES
        if unknown:
            raise ValueError(
                "Unknown noise source(s): " + ", ".join(sorted(unknown))
            )

        return cls(
            current="current" in normalized,
            localization="localization" in normalized,
            robot="robot" in normalized,
        )

    def enabled_names(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.current:
            names.append("current")
        if self.localization:
            names.append("localization")
        if self.robot:
            names.append("robot")
        return tuple(names)
