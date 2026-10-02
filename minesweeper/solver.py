from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .recognition import Cell, CellKind


Position = tuple[int, int]


@dataclass(frozen=True)
class Constraint:
    cells: frozenset[Position]
    mines: int


@dataclass(frozen=True)
class SolveResult:
    safe: frozenset[Position]
    mines: frozenset[Position]
    constraints: tuple[Constraint, ...]
    contradictions: tuple[str, ...]


# The exhaustive stage only ever reports a cell after proving it has the same
# value in every consistent assignment, so truncation must discard results.
ENUMERATION_NODE_BUDGET = 400
_ENUMERATION_CACHE: dict[tuple, tuple[frozenset[Position], frozenset[Position], bool]] = {}
_ENUMERATION_CACHE_LIMIT = 128


def _neighbors(row: int, column: int) -> list[Position]:
    return [
        (other_row, other_column)
        for other_row in range(row - 1, row + 2)
        for other_column in range(column - 1, column + 2)
        if (other_row, other_column) != (row, column)
    ]


def _reduce(
    constraints: Iterable[Constraint],
    safe: set[Position],
    mines: set[Position],
) -> tuple[set[Position], set[Position], set[Constraint], bool]:
    """Fixpoint of normalization, direct deduction and subset differences.

    Used by the exhaustive stage to prune branches. `safe` and `mine` are
    mutated in place; a `True` contradiction invalidates the branch."""
    current = set(constraints)
    while True:
        by_cells: dict[frozenset[Position], int] = {}
        contradiction = False
        for constraint in current:
            cells = constraint.cells - safe - mines
            count = constraint.mines - len(constraint.cells & mines)
            if count < 0 or count > len(cells):
                contradiction = True
                break
            if not cells:
                if count != 0:
                    contradiction = True
                    break
                continue
            key = frozenset(cells)
            previous = by_cells.get(key)
            if previous is not None and previous != count:
                contradiction = True
                break
            by_cells[key] = count
        if contradiction:
            return safe, mines, set(), True
        normalized = {Constraint(key, count) for key, count in by_cells.items()}
        direct_safe = {cell for con in normalized if con.mines == 0 for cell in con.cells}
        direct_mines = {cell for con in normalized if con.mines == len(con.cells) for cell in con.cells}
        unresolved = {con for con in normalized if 0 < con.mines < len(con.cells)}
        direct_safe -= safe
        direct_mines -= mines
        if direct_safe or direct_mines:
            safe |= direct_safe
            mines |= direct_mines
            continue
        derived: set[Constraint] = set()
        # Normalization merged same-cell constraints, so a useful difference
        # can only come from two constraints that actually share a cell.
        touching: dict[Position, list[Constraint]] = {}
        for constraint in unresolved:
            for cell in constraint.cells:
                touching.setdefault(cell, []).append(constraint)
        seen_pairs: set[tuple[Constraint, Constraint]] = set()
        for candidates in touching.values():
            for index, left in enumerate(candidates):
                for right in candidates[index + 1 :]:
                    pair = (left, right) if left.cells < right.cells else (right, left)
                    if not pair[0].cells < pair[1].cells or pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    difference = pair[1].cells - pair[0].cells
                    mine_difference = pair[1].mines - pair[0].mines
                    if mine_difference < 0 or mine_difference > len(difference):
                        return safe, mines, set(), True
                    if difference:
                        derived.add(Constraint(frozenset(difference), mine_difference))
        if derived <= unresolved:
            return safe, mines, unresolved, False
        current = unresolved | derived


def _constraint_components(constraints: Iterable[Constraint]) -> list[list[Constraint]]:
    ordered = sorted(constraints, key=lambda item: (sorted(item.cells), item.mines))
    parent = list(range(len(ordered)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owners: dict[Position, int] = {}
    for index, constraint in enumerate(ordered):
        for cell in constraint.cells:
            if cell in owners:
                left, right = find(index), find(owners[cell])
                if left != right:
                    parent[right] = left
            else:
                owners[cell] = index
    groups: dict[int, list[Constraint]] = {}
    for index, constraint in enumerate(ordered):
        groups.setdefault(find(index), []).append(constraint)
    return list(groups.values())


def _enumerate_component(
    constraints: frozenset[Constraint],
) -> tuple[frozenset[Position], frozenset[Position], bool]:
    """Branch over every consistent assignment of one normalized component.

    A cell is reported only when it holds the same value in every completed
    branch, which is exactly the class of deductions a human spots on patterns
    such as 1-2-1. Returns `(mines, safe, truncated)`; truncated results are
    discarded by the caller because they would not be sound."""
    always_mine: set[Position] | None = None
    never_mine: set[Position] | None = None
    state = {"nodes": 0, "truncated": False}

    def record(safe: set[Position], mines: set[Position]) -> None:
        nonlocal always_mine, never_mine
        assigned_safe = (safe | mines) - mines
        if always_mine is None:
            always_mine = set(mines)
            never_mine = assigned_safe
        else:
            always_mine &= mines
            never_mine &= assigned_safe

    def visit(safe: set[Position], mines: set[Position], cons: frozenset[Constraint]) -> None:
        if state["truncated"]:
            return
        if always_mine is not None and not always_mine and not never_mine:
            return
        state["nodes"] += 1
        if state["nodes"] > ENUMERATION_NODE_BUDGET:
            state["truncated"] = True
            return
        safe, mines, remaining, contradiction = _reduce(cons, safe, mines)
        if contradiction:
            return
        if not remaining:
            record(safe, mines)
            return
        target = min(remaining, key=lambda item: (len(item.cells), item.mines, sorted(item.cells)))
        branch = min(target.cells)
        visit(set(safe), mines | {branch}, remaining)
        visit(safe | {branch}, set(mines), remaining)

    visit(set(), set(), constraints)
    if state["truncated"]:
        return frozenset(), frozenset(), True
    return (
        frozenset(always_mine or set()),
        frozenset(never_mine or set()),
        False,
    )


def _enumerate_component_cached(
    constraints: frozenset[Constraint],
) -> tuple[frozenset[Position], frozenset[Position], bool]:
    key = tuple(sorted((tuple(sorted(constraint.cells)), constraint.mines) for constraint in constraints))
    cached = _ENUMERATION_CACHE.get(key)
    if cached is not None:
        return cached
    result = _enumerate_component(constraints)
    if len(_ENUMERATION_CACHE) >= _ENUMERATION_CACHE_LIMIT:
        _ENUMERATION_CACHE.clear()
    _ENUMERATION_CACHE[key] = result
    return result


def solve_local(grid: list[list[Cell]]) -> SolveResult:
    if not grid or not grid[0]:
        return SolveResult(frozenset(), frozenset(), tuple(), ("空棋盘",))
    rows = len(grid)
    columns = len(grid[0])
    known_mines = {
        (cell.row, cell.column)
        for row in grid
        for cell in row
        if cell.kind in {CellKind.MINE_MARK, CellKind.OPEN_MINE}
    }
    safe: set[Position] = set()
    mines: set[Position] = set()
    contradictions: list[str] = []
    constraints: set[Constraint] = set()

    # Ignore the outer ring: this is a viewport into a much larger board, so
    # numbers on the rim may have unseen neighbors outside the screenshot.
    for row in range(1, rows - 1):
        for column in range(1, columns - 1):
            cell = grid[row][column]
            if cell.kind != CellKind.NUMBER or cell.number is None:
                continue
            neighbors = _neighbors(row, column)
            marked = sum(position in known_mines for position in neighbors)
            unknown = frozenset(
                position
                for position in neighbors
                if grid[position[0]][position[1]].kind == CellKind.CLOSED
            )
            remaining = cell.number - marked
            if remaining < 0 or remaining > len(unknown):
                contradictions.append(
                    f"数字 ({row},{column})={cell.number} 与周围状态矛盾：已标雷 {marked}，未知 {len(unknown)}"
                )
                continue
            if unknown:
                constraints.add(Constraint(unknown, remaining))

    while True:
        normalized_by_cells: dict[frozenset[Position], int] = {}
        for constraint in constraints:
            remaining_cells = constraint.cells - safe - mines
            remaining_mines = constraint.mines - len(constraint.cells & mines)
            if remaining_mines < 0 or remaining_mines > len(remaining_cells):
                contradictions.append("局部约束在推导过程中发生矛盾")
                continue
            if not remaining_cells:
                if remaining_mines != 0:
                    contradictions.append("空约束仍要求地雷，推导结果矛盾")
                continue
            previous = normalized_by_cells.get(frozenset(remaining_cells))
            if previous is not None and previous != remaining_mines:
                contradictions.append("同一组格子出现了不同的地雷数量约束")
                continue
            normalized_by_cells[frozenset(remaining_cells)] = remaining_mines

        normalized = {
            Constraint(cells, mine_count) for cells, mine_count in normalized_by_cells.items()
        }
        direct_safe: set[Position] = set()
        direct_mines: set[Position] = set()
        unresolved: set[Constraint] = set()
        for constraint in normalized:
            if constraint.mines == 0:
                direct_safe.update(constraint.cells)
            elif constraint.mines == len(constraint.cells):
                direct_mines.update(constraint.cells)
            else:
                unresolved.add(constraint)
        if direct_safe & direct_mines:
            contradictions.append("同一个格子被同时推导为安全格和地雷")
            break
        direct_safe.difference_update(safe)
        direct_mines.difference_update(mines)
        if direct_safe or direct_mines:
            safe.update(direct_safe)
            mines.update(direct_mines)
            constraints = unresolved
            continue

        derived: set[Constraint] = set()
        constraint_list = sorted(
            unresolved,
            key=lambda item: (len(item.cells), item.mines, sorted(item.cells)),
        )
        for left in constraint_list:
            for right in constraint_list:
                if left == right or not left.cells < right.cells:
                    continue
                difference = right.cells - left.cells
                mine_difference = right.mines - left.mines
                if mine_difference < 0 or mine_difference > len(difference):
                    contradictions.append("集合差分产生了不可能的地雷数量")
                elif difference:
                    derived.add(Constraint(frozenset(difference), mine_difference))
        new_constraints = derived - unresolved
        constraints = unresolved | derived
        if new_constraints:
            continue

        # Subset differences are exhausted. Enumerate the remaining components
        # to surface pattern deductions (1-2-1 and friends) that set algebra
        # alone cannot express. Still deterministic: only unanimous cells.
        if contradictions:
            break
        component_safe: set[Position] = set()
        component_mines: set[Position] = set()
        truncated = False
        for component in _constraint_components(constraints):
            mines_here, safe_here, component_truncated = _enumerate_component_cached(frozenset(component))
            truncated = truncated or component_truncated
            component_mines |= mines_here
            component_safe |= safe_here
            if truncated:
                break
        component_safe -= safe
        component_mines -= mines
        if component_safe & component_mines:
            contradictions.append("穷举推导同时得到安全格和地雷")
            break
        if truncated or not (component_safe or component_mines):
            break
        safe |= component_safe
        mines |= component_mines

    safe.difference_update(mines)
    return SolveResult(
        frozenset(safe),
        frozenset(mines),
        tuple(sorted(constraints, key=lambda item: (len(item.cells), item.mines, sorted(item.cells)))),
        tuple(dict.fromkeys(contradictions)),
    )
