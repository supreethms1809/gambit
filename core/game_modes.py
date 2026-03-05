from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class ContrastiveGameConfig:
    mode: str
    use_shared: bool
    lambda_margin: float
    lambda_overlap: float
    lambda_disjoint: float
    lambda_partition: float


@dataclass(frozen=True)
class ShiftGameConfig:
    mode: str
    lambda_mean: float
    lambda_var: float
    lambda_gap: float
    lambda_shortcut: float
    lambda_disjoint: float
    lambda_sparse: float


_CONTRASTIVE_PRESETS: Dict[str, ContrastiveGameConfig] = {
    # Prior default behavior in examples.
    "mixed": ContrastiveGameConfig(
        mode="mixed",
        use_shared=True,
        lambda_margin=1.0,
        lambda_overlap=0.2,
        lambda_disjoint=0.1,
        lambda_partition=0.1,
    ),
    # Cooperative: no explicit competition terms; shared evidence is allowed.
    "cooperative": ContrastiveGameConfig(
        mode="cooperative",
        use_shared=True,
        lambda_margin=0.0,
        lambda_overlap=0.0,
        lambda_disjoint=0.0,
        lambda_partition=0.2,
    ),
    # Competitive: no shared mask, stronger unique separation terms.
    "competitive": ContrastiveGameConfig(
        mode="competitive",
        use_shared=False,
        lambda_margin=1.5,
        lambda_overlap=0.35,
        lambda_disjoint=0.35,
        lambda_partition=0.0,
    ),
}


_SHIFT_PRESETS: Dict[str, ShiftGameConfig] = {
    # Prior default behavior in shift script.
    "mixed": ShiftGameConfig(
        mode="mixed",
        lambda_mean=1.0,
        lambda_var=0.5,
        lambda_gap=1.0,
        lambda_shortcut=0.0,
        lambda_disjoint=0.2,
        lambda_sparse=0.05,
    ),
    # Cooperative: do not force ID-vs-OOD shortcut rivalry or disjointness.
    "cooperative": ShiftGameConfig(
        mode="cooperative",
        lambda_mean=1.0,
        lambda_var=0.5,
        lambda_gap=0.0,
        lambda_shortcut=1.0,
        lambda_disjoint=0.0,
        lambda_sparse=0.05,
    ),
    # Competitive: emphasize shortcut ID-vs-OOD gap and robust/shortcut separation.
    "competitive": ShiftGameConfig(
        mode="competitive",
        lambda_mean=1.0,
        lambda_var=0.5,
        lambda_gap=1.5,
        lambda_shortcut=0.0,
        lambda_disjoint=0.4,
        lambda_sparse=0.05,
    ),
}


def contrastive_game_modes() -> Tuple[str, ...]:
    return ("mixed", "cooperative", "competitive", "manual")


def shift_game_modes() -> Tuple[str, ...]:
    return ("mixed", "cooperative", "competitive", "manual")


def _validate_nonnegative(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def resolve_contrastive_game(
    mode: str,
    *,
    use_shared: Optional[bool] = None,
    lambda_margin: Optional[float] = None,
    lambda_overlap: Optional[float] = None,
    lambda_disjoint: Optional[float] = None,
    lambda_partition: Optional[float] = None,
) -> ContrastiveGameConfig:
    mode = mode.lower()
    if mode == "manual":
        if (
            use_shared is None
            or lambda_margin is None
            or lambda_overlap is None
            or lambda_disjoint is None
            or lambda_partition is None
        ):
            raise ValueError(
                "manual contrastive mode requires use_shared, lambda_margin, "
                "lambda_overlap, lambda_disjoint, and lambda_partition"
            )
        _validate_nonnegative("lambda_margin", lambda_margin)
        _validate_nonnegative("lambda_overlap", lambda_overlap)
        _validate_nonnegative("lambda_disjoint", lambda_disjoint)
        _validate_nonnegative("lambda_partition", lambda_partition)
        return ContrastiveGameConfig(
            mode=mode,
            use_shared=bool(use_shared),
            lambda_margin=float(lambda_margin),
            lambda_overlap=float(lambda_overlap),
            lambda_disjoint=float(lambda_disjoint),
            lambda_partition=float(lambda_partition),
        )

    if mode not in _CONTRASTIVE_PRESETS:
        raise ValueError(f"Unknown contrastive game mode: {mode}")
    if any(v is not None for v in (use_shared, lambda_margin, lambda_overlap, lambda_disjoint, lambda_partition)):
        raise ValueError("Preset contrastive modes do not accept manual lambda/use_shared overrides")
    return _CONTRASTIVE_PRESETS[mode]


def resolve_shift_game(
    mode: str,
    *,
    lambda_mean: Optional[float] = None,
    lambda_var: Optional[float] = None,
    lambda_gap: Optional[float] = None,
    lambda_shortcut: Optional[float] = None,
    lambda_disjoint: Optional[float] = None,
    lambda_sparse: Optional[float] = None,
) -> ShiftGameConfig:
    mode = mode.lower()
    if mode == "manual":
        vals = (lambda_mean, lambda_var, lambda_gap, lambda_shortcut, lambda_disjoint, lambda_sparse)
        if any(v is None for v in vals):
            raise ValueError(
                "manual shift mode requires lambda_mean, lambda_var, lambda_gap, lambda_shortcut, "
                "lambda_disjoint, and lambda_sparse"
            )
        assert lambda_mean is not None and lambda_var is not None and lambda_gap is not None
        assert lambda_shortcut is not None
        assert lambda_disjoint is not None and lambda_sparse is not None
        _validate_nonnegative("lambda_mean", lambda_mean)
        _validate_nonnegative("lambda_var", lambda_var)
        _validate_nonnegative("lambda_gap", lambda_gap)
        _validate_nonnegative("lambda_shortcut", lambda_shortcut)
        _validate_nonnegative("lambda_disjoint", lambda_disjoint)
        _validate_nonnegative("lambda_sparse", lambda_sparse)
        return ShiftGameConfig(
            mode=mode,
            lambda_mean=float(lambda_mean),
            lambda_var=float(lambda_var),
            lambda_gap=float(lambda_gap),
            lambda_shortcut=float(lambda_shortcut),
            lambda_disjoint=float(lambda_disjoint),
            lambda_sparse=float(lambda_sparse),
        )

    if mode not in _SHIFT_PRESETS:
        raise ValueError(f"Unknown shift game mode: {mode}")
    if any(v is not None for v in (lambda_mean, lambda_var, lambda_gap, lambda_shortcut, lambda_disjoint, lambda_sparse)):
        raise ValueError("Preset shift modes do not accept manual lambda overrides")
    return _SHIFT_PRESETS[mode]
