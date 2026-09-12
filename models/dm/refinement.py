"""Terminal energy refinement in normalized grasp coordinates."""

from time import perf_counter
from typing import Callable, Dict, Tuple

import torch


EnergyFunction = Callable[[torch.Tensor], torch.Tensor]


def _synchronize(tensor: torch.Tensor) -> None:
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def refine_terminal_state(
    terminal_state: torch.Tensor,
    energy_fn: EnergyFunction,
    learning_rate: float,
    iterations: int,
    lower_bound: float = -1.0,
    upper_bound: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Minimize terminal physical energy in normalized translation+joint space.

    ``terminal_state`` has shape ``(B, 27)``. The caller supplies an energy
    function returning one scalar per candidate. Adam updates only this state;
    every update is projected to the legal normalized interval.
    """
    if terminal_state.ndim != 2 or terminal_state.shape[1] != 27:
        raise ValueError(f"expected terminal state (B, 27), got {tuple(terminal_state.shape)}")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not torch.isfinite(terminal_state).all():
        raise FloatingPointError("terminal state contains NaN or Inf")

    initial_outside = ((terminal_state < lower_bound) | (terminal_state > upper_bound)).any(dim=1)
    state = terminal_state.detach().clone().clamp_(lower_bound, upper_bound)
    state.requires_grad_(True)
    optimizer = torch.optim.Adam([state], lr=learning_rate)

    initial_energy = None
    _synchronize(state)
    started = perf_counter()
    for step in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            energy = energy_fn(state)
        if energy.ndim != 1 or energy.shape[0] != state.shape[0]:
            raise ValueError(f"energy function must return (B,), got {tuple(energy.shape)}")
        if not torch.isfinite(energy).all():
            raise FloatingPointError(f"non-finite energy at refinement step {step}")
        if not energy.requires_grad:
            raise RuntimeError("physical energy is not differentiable with respect to terminal state")
        if initial_energy is None:
            initial_energy = energy.detach()
        energy.mean().backward()
        if state.grad is None or not torch.isfinite(state.grad).all():
            raise FloatingPointError(f"non-finite or missing state gradient at refinement step {step}")
        optimizer.step()
        with torch.no_grad():
            state.clamp_(lower_bound, upper_bound)

    with torch.no_grad():
        final_energy = energy_fn(state)
    if not torch.isfinite(final_energy).all():
        raise FloatingPointError("non-finite final energy")
    _synchronize(state)
    elapsed = perf_counter() - started

    refined = state.detach()
    invalid = (~torch.isfinite(refined)).any(dim=1)
    stats = {
        "learning_rate": float(learning_rate),
        "iterations": int(iterations),
        "energy_evaluations": int(iterations + 1),
        "refinement_time_sec": float(elapsed),
        "initial_energy_mean": float(initial_energy.mean().item()),
        "final_energy_mean": float(final_energy.mean().item()),
        "invalid_candidate_count": int(invalid.sum().item()),
        "clamped_initial_candidate_count": int(initial_outside.sum().item()),
    }
    return refined, stats
