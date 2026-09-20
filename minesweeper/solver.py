from __future__ import annotations

from dataclasses import dataclass

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


def _neighbors(row: int, column: int) -> list[Position]:
    return [
        (other_row, other_column)
        for other_row in range(row - 1, row + 2)
        for other_column in range(column - 1, column + 2)
        if (other_row, other_column) != (row, column)
    ]


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
        if not new_constraints:
            break

    safe.difference_update(mines)
    return SolveResult(
        frozenset(safe),
        frozenset(mines),
        tuple(sorted(constraints, key=lambda item: (len(item.cells), item.mines, sorted(item.cells)))),
        tuple(dict.fromkeys(contradictions)),
    )
