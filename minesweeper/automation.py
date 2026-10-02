from __future__ import annotations

import hashlib
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from .capture import CaptureFrame
from .dialogs import (
    detect_cyan_dialog,
    detect_death_dialog,
    detect_revive_button,
    detect_welfare_confirm,
)
from .input_control import BatchExecution, BatchPoints, WindowInputController
from .observer import Observation, WindowObserver
from .recognition import Cell, CellKind
from .statistics import StatisticsStore


class ActionKind(str, Enum):
    BATCH = "batch"
    DRAG = "drag"
    COMPLETE = "complete"


class RecognitionAnomaly(RuntimeError):
    pass


@dataclass(frozen=True)
class PlannedAction:
    kind: ActionKind
    marks: tuple[tuple[int, int], ...] = ()
    safe: tuple[tuple[int, int], ...] = ()
    direction: str | None = None
    drag_start: tuple[int, int] | None = None
    drag_end: tuple[int, int] | None = None
    navigation_mode: str | None = None


@dataclass(frozen=True)
class StepOutcome:
    observation: Observation
    action: PlannedAction
    executed: int
    message: str
    advice: str | None = None
    opened: int = 0
    flagged: int = 0


def view_signature(observation: Observation) -> str:
    values = []
    for row in observation.grid:
        values.append(
            ",".join(
                f"{cell.kind.value}:{cell.number if cell.number is not None else ''}:{cell.variant or ''}"
                for cell in row
            )
        )
    return hashlib.sha1("|".join(values).encode("utf-8")).hexdigest()


def recognition_issue(observation: Observation) -> str | None:
    if observation.result.contradictions:
        return "识别结果存在约束矛盾"
    unknown_count = sum(
        cell.kind == CellKind.UNRECOGNIZED
        for row in observation.grid
        for cell in row
    )
    if unknown_count:
        return f"当前视口存在 {unknown_count} 个未识别格子"
    return None


def frontier_analysis(
    observation: Observation,
) -> tuple[list[tuple[int, int]], dict[str, float]]:
    grid = observation.grid
    rows, columns = len(grid), len(grid[0])
    frontier: list[tuple[int, int]] = []
    closed_normals = {direction: 0.0 for direction in LocalNavigator.DIRECTIONS}
    open_kinds = {
        CellKind.OPEN,
        CellKind.NUMBER,
        CellKind.TREASURE,
        CellKind.OPEN_MINE,
        CellKind.MINE_MARK,
    }
    cardinal = ((-1, 0, "up"), (1, 0, "down"), (0, -1, "left"), (0, 1, "right"))
    for row in range(rows):
        for column in range(columns):
            if grid[row][column].kind not in open_kinds:
                continue
            touches_closed = False
            for row_delta, column_delta, direction in cardinal:
                other_row = row + row_delta
                other_column = column + column_delta
                if not (0 <= other_row < rows and 0 <= other_column < columns):
                    continue
                if grid[other_row][other_column].kind == CellKind.CLOSED:
                    closed_normals[direction] += 1.0
                    touches_closed = True
            if touches_closed:
                frontier.append((row, column))
    return frontier, closed_normals


def closed_ratio(observation: Observation) -> float:
    cells = [cell for row in observation.grid for cell in row]
    if not cells:
        return 0.0
    return sum(cell.kind == CellKind.CLOSED for cell in cells) / len(cells)


def region_direction(
    observation: Observation,
    target_kinds: set[CellKind],
) -> str | None:
    positions = [
        (cell.row, cell.column)
        for row in observation.grid
        for cell in row
        if cell.kind in target_kinds
    ]
    if not positions:
        return None
    center_row = (len(observation.grid) - 1) / 2.0
    center_column = (len(observation.grid[0]) - 1) / 2.0
    mean_row = sum(row for row, _ in positions) / len(positions)
    mean_column = sum(column for _, column in positions) / len(positions)
    row_offset = (mean_row - center_row) / max(1.0, center_row)
    column_offset = (mean_column - center_column) / max(1.0, center_column)
    if abs(row_offset) >= abs(column_offset):
        return "up" if row_offset < 0 else "down"
    return "left" if column_offset < 0 else "right"


def interior_closed_island(
    observation: Observation,
    *,
    maximum_ratio: float = 0.25,
    margin: int = 2,
) -> tuple[float, float] | None:
    positions = [
        (cell.row, cell.column)
        for row in observation.grid
        for cell in row
        if cell.kind == CellKind.CLOSED
    ]
    if not positions or closed_ratio(observation) > maximum_ratio:
        return None
    rows = len(observation.grid)
    columns = len(observation.grid[0])
    minimum_row = min(row for row, _ in positions)
    maximum_row = max(row for row, _ in positions)
    minimum_column = min(column for _, column in positions)
    maximum_column = max(column for _, column in positions)
    if (
        minimum_row < margin
        or minimum_column < margin
        or maximum_row >= rows - margin
        or maximum_column >= columns - margin
    ):
        return None
    return (
        sum(row for row, _ in positions) / len(positions),
        sum(column for _, column in positions) / len(positions),
    )


class LocalNavigator:
    DIRECTIONS = ("right", "down", "left", "up")
    OPPOSITE = {"right": "left", "left": "right", "up": "down", "down": "up"}
    # Nominal viewport travel per drag in board cells, mirroring plan_drag.
    STEP_CELLS = {"right": (0.0, 12.0), "left": (0.0, -12.0), "down": (8.0, 0.0), "up": (-8.0, 0.0)}
    # Two viewport centres closer than this share most of their content, so
    # arriving there counts as revisiting the same spot.
    REVISIT_RADIUS = 6.0

    def __init__(self) -> None:
        self.rotation = 0
        self.last_direction: str | None = None
        self.recent_views: deque[str] = deque(maxlen=24)
        self.repeated_views = 0
        self.last_open_direction: str | None = None
        self.last_closed_direction: str | None = None
        self.island_escape_direction: str | None = None
        self.last_mode = "frontier"
        # Single-run trail of viewport centres in virtual board coordinates.
        # It is deliberately not a world map: only "where have we been" is
        # kept, so exploration can avoid returning to familiar ground.
        self.position: tuple[float, float] = (0.0, 0.0)
        self.trail: list[tuple[float, float]] = [(0.0, 0.0)]

    def move_viewport(self, row_delta: float, column_delta: float) -> None:
        self.position = (self.position[0] + row_delta, self.position[1] + column_delta)
        self.trail.append(self.position)

    def revisit_count(self, direction: str) -> int:
        row_step, column_step = self.STEP_CELLS[direction]
        target_row = self.position[0] + row_step
        target_column = self.position[1] + column_step
        return sum(
            1
            for row, column in self.trail
            if abs(row - target_row) <= self.REVISIT_RADIUS
            and abs(column - target_column) <= self.REVISIT_RADIUS
        )

    def _avoid_revisit(self, direction: str, scores: dict[str, float]) -> tuple[str, bool]:
        visits = self.revisit_count(direction)
        if visits == 0:
            return direction, False
        any_positive = any(score > 0 for score in scores.values())
        origin = self.DIRECTIONS.index(direction)
        alternatives: list[str] = []
        for candidate in self.DIRECTIONS:
            if candidate == direction or self.revisit_count(candidate) >= visits:
                continue
            if any_positive and scores.get(candidate, 0.0) <= 0:
                continue
            alternatives.append(candidate)
        if not alternatives:
            return direction, False
        alternatives.sort(
            key=lambda item: (
                -scores.get(item, 0.0),
                (self.DIRECTIONS.index(item) - origin) % len(self.DIRECTIONS),
            )
        )
        return alternatives[0], True

    def remember(self, signature: str) -> None:
        if signature in self.recent_views:
            self.repeated_views += 1
        else:
            self.repeated_views = max(0, self.repeated_views - 1)
        self.recent_views.append(signature)

    def choose_direction(self, observation: Observation) -> str:
        grid = observation.grid
        rows, columns = len(grid), len(grid[0])
        band = max(2, min(3, min(rows, columns) // 4))
        frontier, closed_normals = frontier_analysis(observation)

        open_kinds = {
            CellKind.OPEN,
            CellKind.NUMBER,
            CellKind.TREASURE,
            CellKind.OPEN_MINE,
            CellKind.MINE_MARK,
        }
        visible_open_direction = region_direction(observation, open_kinds)
        visible_closed_direction = region_direction(observation, {CellKind.CLOSED})
        if visible_open_direction is not None:
            self.last_open_direction = visible_open_direction
        if visible_closed_direction is not None:
            self.last_closed_direction = visible_closed_direction
        ratio = closed_ratio(observation)
        island = interior_closed_island(observation)
        if ratio == 0.0:
            self.island_escape_direction = None
        if island is not None and self.island_escape_direction is None:
            island_row, island_column = island
            distances = {
                "up": island_row,
                "down": rows - 1 - island_row,
                "left": island_column,
                "right": columns - 1 - island_column,
            }
            nearest_distance = min(distances.values())
            tied = [
                direction
                for direction in self.DIRECTIONS
                if abs(distances[direction] - nearest_distance) < 0.5
            ]
            self.island_escape_direction = (
                self.last_direction if self.last_direction in tied else tied[0]
            )
        if self.island_escape_direction is not None and 0.0 < ratio <= 0.30:
            direction = self.island_escape_direction or self.last_direction or self.DIRECTIONS[self.rotation]
            self.last_mode = "island_escape"
            self.rotation = (self.DIRECTIONS.index(direction) + 1) % len(self.DIRECTIONS)
            self.last_direction = direction
            return direction
        if ratio > 0.30:
            self.island_escape_direction = None
        if ratio >= 0.70:
            direction = visible_open_direction or self.last_open_direction
            if direction is not None:
                self.rotation = (self.DIRECTIONS.index(direction) + 1) % len(self.DIRECTIONS)
                self.last_direction = direction
                self.last_mode = "ratio_rescue"
                return direction
        if ratio <= 0.30:
            direction = visible_closed_direction or self.last_closed_direction
            if direction is not None:
                self.rotation = (self.DIRECTIONS.index(direction) + 1) % len(self.DIRECTIONS)
                self.last_direction = direction
                self.last_mode = "ratio_rescue"
                return direction

        def closed_count(positions: list[tuple[int, int]]) -> int:
            return sum(grid[row][column].kind == CellKind.CLOSED for row, column in positions)

        if frontier:
            # Clockwise around the unopened region keeps CLOSED cells on the
            # right-hand side of travel.
            scores = {direction: 0.0 for direction in self.DIRECTIONS}
            clockwise_tangent = {
                "right": "up",
                "down": "right",
                "left": "down",
                "up": "left",
            }
            for closed_side, strength in closed_normals.items():
                scores[clockwise_tangent[closed_side]] += strength * 3.0
            for row, column in frontier:
                if row < band:
                    scores["up"] += float((band - row) * 2)
                if row >= rows - band:
                    scores["down"] += float((row - (rows - band) + 1) * 2)
                if column < band:
                    scores["left"] += float((band - column) * 2)
                if column >= columns - band:
                    scores["right"] += float((column - (columns - band) + 1) * 2)
        else:
            scores = {
                "left": float(closed_count([(row, column) for row in range(rows) for column in range(band)])),
                "right": float(closed_count([(row, column) for row in range(rows) for column in range(columns - band, columns)])),
                "up": float(closed_count([(row, column) for row in range(band) for column in range(columns)])),
                "down": float(closed_count([(row, column) for row in range(rows - band, rows) for column in range(columns)])),
            }

        if self.last_direction is None:
            order = list(self.DIRECTIONS[self.rotation :] + self.DIRECTIONS[: self.rotation])
            direction = max(order, key=lambda candidate: scores[candidate])
        else:
            index = self.DIRECTIONS.index(self.last_direction)
            clockwise = self.DIRECTIONS[(index + 1) % len(self.DIRECTIONS)]
            counterclockwise = self.DIRECTIONS[(index - 1) % len(self.DIRECTIONS)]
            reverse = self.OPPOSITE[self.last_direction]
            same_score = scores[self.last_direction]
            clockwise_score = scores[clockwise]
            if clockwise_score >= max(2.0, same_score * 0.70):
                direction = clockwise
            elif same_score > 0:
                direction = self.last_direction
            elif clockwise_score > 0:
                direction = clockwise
            elif scores[counterclockwise] > 0:
                direction = counterclockwise
            else:
                direction = reverse
        direction, avoided = self._avoid_revisit(direction, scores)
        self.rotation = (self.DIRECTIONS.index(direction) + 1) % len(self.DIRECTIONS)
        self.last_direction = direction
        self.last_mode = "trail" if avoided else ("frontier" if frontier else "density")
        return direction

    def turn_clockwise(self) -> str:
        if self.last_direction is None:
            self.last_direction = self.DIRECTIONS[self.rotation]
        else:
            index = self.DIRECTIONS.index(self.last_direction)
            self.last_direction = self.DIRECTIONS[(index + 1) % len(self.DIRECTIONS)]
        self.rotation = (self.DIRECTIONS.index(self.last_direction) + 1) % len(self.DIRECTIONS)
        if self.last_mode == "island_escape":
            self.island_escape_direction = self.last_direction
        return self.last_direction


def plan_drag(
    observation: Observation,
    navigator: LocalNavigator,
    forced_direction: str | None = None,
) -> PlannedAction:
    rows = len(observation.grid)
    columns = len(observation.grid[0])
    center = (rows / 2.0, columns / 2.0)
    direction = forced_direction or navigator.choose_direction(observation)
    if forced_direction is not None:
        navigator.last_direction = forced_direction
        navigator.rotation = (navigator.DIRECTIONS.index(forced_direction) + 1) % len(navigator.DIRECTIONS)
        navigator.last_mode = "forced"
    geometry = observation.geometry
    frontier, _ = frontier_analysis(observation)
    # Dragging is a viewport gesture in this game, not a click. Every complete
    # interior tile is therefore a valid anchor, regardless of its state.
    candidates = [
        cell
        for row in observation.grid
        for cell in row
        if 1 <= cell.row < rows - 1
        and 1 <= cell.column < columns - 1
    ]
    if not candidates:
        raise RuntimeError("可见棋盘过小，找不到内部拖动坐标")
    horizontal = min(geometry.pitch * 12.0, geometry.full_columns * geometry.pitch * 0.52)
    vertical = min(geometry.pitch * 8.0, geometry.full_rows * geometry.pitch * 0.75)
    ratio = closed_ratio(observation)
    rescue_mode = ratio >= 0.70 or ratio <= 0.30
    if navigator.last_mode == "island_escape":
        horizontal = geometry.pitch * 8.0
        vertical = geometry.pitch * 6.0
    elif rescue_mode:
        imbalance = min(1.0, abs(ratio - 0.5) / 0.5)
        horizontal = geometry.pitch * (3.0 + imbalance * 3.0)
        vertical = geometry.pitch * (2.5 + imbalance * 2.5)
    delta_x, delta_y = {
        "right": (-horizontal, 0.0),
        "left": (horizontal, 0.0),
        "down": (0.0, -vertical),
        "up": (0.0, vertical),
    }[direction]
    if frontier and not rescue_mode:
        frontier_row = sum(row for row, _ in frontier) / len(frontier)
        frontier_column = sum(column for _, column in frontier) / len(frontier)
        viewport_row = (rows - 1) / 2.0
        viewport_column = (columns - 1) / 2.0
        if direction in {"up", "down"}:
            delta_x += -(frontier_column - viewport_column) * geometry.pitch
        else:
            delta_y += -(frontier_row - viewport_row) * geometry.pitch

    closed_preference = lambda cell: 0 if cell.kind == CellKind.CLOSED else 1

    def anchor_key(cell: Cell) -> tuple[float, int, float]:
        column = cell.column
        row = cell.row
        horizontal_room = column if delta_x < 0 else (columns - 1 - column)
        vertical_room = row if delta_y < 0 else (rows - 1 - row)
        travel_room = horizontal_room * abs(delta_x) + vertical_room * abs(delta_y)
        return (-travel_room, closed_preference(cell), abs(row - center[0]) + abs(column - center[1]))

    anchor = min(candidates, key=anchor_key)
    start_x, start_y = geometry.center(anchor.row, anchor.column)
    min_x = geometry.origin_x + geometry.pitch
    max_x = geometry.origin_x + (geometry.full_columns - 1) * geometry.pitch
    min_y = geometry.origin_y + geometry.pitch
    max_y = geometry.origin_y + (geometry.full_rows - 1) * geometry.pitch
    end = (
        round(max(min_x, min(max_x, start_x + delta_x))),
        round(max(min_y, min(max_y, start_y + delta_y))),
    )
    if abs(end[0] - start_x) + abs(end[1] - start_y) < geometry.pitch * 2:
        raise RuntimeError("当前已打开区域不足以完成安全拖动")
    return PlannedAction(
        ActionKind.DRAG,
        direction=direction,
        drag_start=(start_x, start_y),
        drag_end=end,
        navigation_mode=navigator.last_mode,
    )


def plan_action(observation: Observation, navigator: LocalNavigator) -> PlannedAction:
    issue = recognition_issue(observation)
    if issue is not None:
        raise RecognitionAnomaly(issue)

    rows = len(observation.grid)
    columns = len(observation.grid[0])
    center = (rows / 2.0, columns / 2.0)
    sort_key = lambda position: (position[0] - center[0]) ** 2 + (position[1] - center[1]) ** 2
    marks = tuple(sorted(observation.result.mines, key=sort_key))
    safe = tuple(sorted(observation.result.safe, key=sort_key))
    if marks or safe:
        return PlannedAction(ActionKind.BATCH, marks=marks, safe=safe)

    closed = [cell for row in observation.grid for cell in row if cell.kind == CellKind.CLOSED]
    if not closed:
        return PlannedAction(ActionKind.COMPLETE)
    return plan_drag(observation, navigator)


class LocalAutomation:
    MAX_RECOVERY_DRAGS = 8
    # An emptied viewport is explored by dragging until either closed cells
    # reappear or this budget runs out; drags alone are not a stop condition.
    EMPTY_VIEW_TIMEOUT_SECONDS = 300.0
    LAST_ACTION_DIRECTION_DRAGS = 1
    # Stepping on a mine opens a death dialog whose revive button only
    # appears after a six-second countdown, so the wait must exceed it.
    REVIVE_WAIT_SECONDS = 15.0
    REVIVE_POLL_INTERVAL = 0.3
    # Welfare/reward dialogs after clicking a closed chest.
    CHEST_DIALOG_TIMEOUT = 8.0
    CHEST_POLL_INTERVAL = 0.25
    MAX_CHEST_FAILURES = 3

    def __init__(
        self,
        observer: Callable[[], Observation] | None = None,
        controller: WindowInputController | None = None,
        clock: Callable[[], float] = time.monotonic,
        statistics: StatisticsStore | None = None,
    ) -> None:
        self.observer = observer or WindowObserver()
        self.controller = controller or WindowInputController()
        self.navigator = LocalNavigator()
        self.last_observation: Observation | None = None
        self.empty_view_drags = 0
        self.empty_view_started_at: float | None = None
        self.last_action_direction: str | None = None
        self.clock = clock
        self.statistics = statistics
        self.search_started_at: float | None = None
        self.recovery_note: str | None = None
        self.chest_failures = 0

    @staticmethod
    def _wait(stop_event: threading.Event, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while not stop_event.is_set() and time.monotonic() < deadline:
            time.sleep(min(0.025, deadline - time.monotonic()))

    def perform_cycle(self, stop_event: threading.Event) -> StepOutcome:
        self.recovery_note = None
        activate_target = getattr(self.controller, "activate_target", None)
        active_rect: tuple[int, int, int, int] | None = None
        if callable(activate_target):
            _, active_rect = activate_target()
        try:
            before = self.observer()
        except (ValueError, RuntimeError) as error:
            before = self._recover_recognition(stop_event, self.last_observation, active_rect, str(error))
        issue = recognition_issue(before)
        if issue is not None:
            before = self._recover_recognition(stop_event, before, active_rect, issue)
        self.last_observation = before
        chest_center = self._closed_chest_center(before)
        if chest_center is not None and self.chest_failures < self.MAX_CHEST_FAILURES:
            if stop_event.is_set():
                return StepOutcome(before, PlannedAction(ActionKind.BATCH), 0, "已暂停")
            handled = self._open_closed_chest(stop_event, before.frame.window_rect, chest_center)
            if handled:
                self.chest_failures = 0
                self.recovery_note = "已自动开启宝箱并领取奖励"
            else:
                self.chest_failures += 1
            if stop_event.is_set():
                return StepOutcome(before, PlannedAction(ActionKind.BATCH), 0, "已暂停")
            before = self._observe_after_chest(stop_event, before)
            self.last_observation = before
        before_signature = view_signature(before)
        self.navigator.remember(before_signature)
        action = plan_action(before, self.navigator)
        exploring_empty_view = action.kind == ActionKind.COMPLETE
        forced_last_action_probe = False
        if exploring_empty_view:
            if self.empty_view_started_at is None:
                self.empty_view_started_at = self.clock()
                self.search_started_at = None
            elif self.clock() - self.empty_view_started_at >= self.EMPTY_VIEW_TIMEOUT_SECONDS:
                minutes = int(self.EMPTY_VIEW_TIMEOUT_SECONDS // 60)
                return StepOutcome(
                    before,
                    action,
                    0,
                    f"连续拖动 {minutes} 分钟仍未发现未开启格，自动运行结束",
                )
            forced_direction = (
                self.last_action_direction
                if self.empty_view_drags < self.LAST_ACTION_DIRECTION_DRAGS
                else None
            )
            forced_last_action_probe = forced_direction is not None
            action = plan_drag(before, self.navigator, forced_direction=forced_direction)
            self.empty_view_drags += 1
        else:
            self.empty_view_drags = 0
            self.empty_view_started_at = None
        if action.kind == ActionKind.BATCH:
            self.search_started_at = None
            last_position = action.safe[-1] if action.safe else action.marks[-1]
            self.last_action_direction = self._position_direction(before, last_position)
        elif action.kind == ActionKind.DRAG and self.search_started_at is None:
            self.search_started_at = self.clock()
        if stop_event.is_set():
            return StepOutcome(before, action, 0, "已暂停")

        if action.kind == ActionKind.BATCH:
            marks = tuple(before.geometry.center(row, column) for row, column in action.marks)
            safe = tuple(before.geometry.center(row, column) for row, column in action.safe)
            execution = self.controller.click_batch(
                before.frame.window_rect,
                BatchPoints(marks=marks, safe=safe),
                stop_event,
            )
            if isinstance(execution, int):
                execution = BatchExecution(
                    flagged=min(execution, len(marks)),
                    opened=max(0, execution - len(marks)),
                )
            executed = execution.total
            if self.statistics is not None and executed:
                self.statistics.add(opened=execution.opened, flagged=execution.flagged)
            if stop_event.is_set():
                return StepOutcome(
                    before,
                    action,
                    executed,
                    "已暂停",
                    opened=execution.opened,
                    flagged=execution.flagged,
                )
            wait_for_stable = getattr(self.observer, "wait_for_stable", None)
            try:
                if callable(wait_for_stable):
                    after = wait_for_stable(
                        stop_event,
                        force_redetect=False,
                        initial_delay=0.75,
                    )
                else:
                    self._wait(stop_event, 1.05)
                    after = self.observer()
            except (ValueError, RuntimeError) as error:
                after = self._recover_recognition(
                    stop_event,
                    before,
                    before.frame.window_rect,
                    str(error),
                )
            issue = recognition_issue(after)
            if issue is not None:
                after = self._recover_recognition(stop_event, after, after.frame.window_rect, issue)
            self.last_observation = after
            if not stop_event.is_set() and view_signature(after) == before_signature:
                raise RuntimeError("批量点击后棋盘没有变化，自动模式已停止")
            message = self._with_note(
                f"本批执行 {len(action.marks)} 个标雷、{len(action.safe)} 个开格"
            )
            return StepOutcome(
                after,
                action,
                executed,
                message,
                self._search_advice(),
                opened=execution.opened,
                flagged=execution.flagged,
            )

        assert action.drag_start is not None and action.drag_end is not None
        self.controller.drag(
            before.frame.window_rect,
            action.drag_start,
            action.drag_end,
            stop_event,
        )
        if stop_event.is_set():
            return StepOutcome(before, action, 0, "已暂停")
        wait_for_stable = getattr(self.observer, "wait_for_stable", None)
        try:
            if callable(wait_for_stable):
                after = wait_for_stable(
                    stop_event,
                    force_redetect=True,
                    initial_delay=0.08,
                )
            else:
                self._wait(stop_event, 0.65)
                after = self.observer()
        except (ValueError, RuntimeError) as error:
            after = self._recover_recognition(
                stop_event,
                before,
                before.frame.window_rect,
                str(error),
            )
        issue = recognition_issue(after)
        if issue is not None:
            after = self._recover_recognition(stop_event, after, after.frame.window_rect, issue)
        self.last_observation = after
        after_signature = view_signature(after)
        # Empty-view exploration is a sanctioned long search: the one-minute
        # frontier advice would only mislead while it runs.
        advice = None if exploring_empty_view else self._search_advice()
        if not stop_event.is_set() and after_signature == before_signature:
            if forced_last_action_probe:
                return StepOutcome(
                    after,
                    action,
                    1,
                    f"{self._empty_exploration_message(action)}，本次拖动未改变视口",
                    advice,
                )
            next_direction = self.navigator.turn_clockwise()
            message = f"当前方向已到边界，下一步顺时针转向 {next_direction}"
            return StepOutcome(after, action, 1, message, advice)
        self.navigator.remember(after_signature)
        pitch = before.geometry.pitch
        self.navigator.move_viewport(
            -(action.drag_end[1] - action.drag_start[1]) / pitch,
            -(action.drag_end[0] - action.drag_start[0]) / pitch,
        )
        if exploring_empty_view:
            has_closed = any(
                cell.kind == CellKind.CLOSED
                for row in after.grid
                for cell in row
            )
            if has_closed:
                explored = self.empty_view_drags
                self.empty_view_drags = 0
                self.empty_view_started_at = None
                self.search_started_at = None
                return StepOutcome(
                    after,
                    action,
                    1,
                    f"拖动 {explored} 次后发现新的未开启格",
                    advice,
                )
            return StepOutcome(
                after,
                action,
                1,
                self._empty_exploration_message(action),
                advice,
            )
        if action.navigation_mode == "island_escape":
            navigation_message = f"跳过中央孤立未开区，向{action.direction}寻找新的外围边界"
        elif action.navigation_mode == "trail":
            navigation_message = f"该方向区域已探索过，转向{action.direction}避免重复"
        else:
            navigation_message = f"顺时针沿未开区域边缘向{action.direction}探索，并校正开/未开比例"
        return StepOutcome(
            after,
            action,
            1,
            self._with_note(navigation_message),
            self._search_advice(),
        )

    def _search_advice(self) -> str | None:
        if self.search_started_at is None:
            return None
        if self.clock() - self.search_started_at < 60.0:
            return None
        return "已沿前沿搜索超过 1 分钟仍无可开区域；建议按 F8 停止，并手动更换区域。"

    @staticmethod
    def _position_direction(observation: Observation, position: tuple[int, int]) -> str:
        row, column = position
        center_row = (len(observation.grid) - 1) / 2.0
        center_column = (len(observation.grid[0]) - 1) / 2.0
        normalized_row = (row - center_row) / max(1.0, center_row)
        normalized_column = (column - center_column) / max(1.0, center_column)
        if abs(normalized_row) >= abs(normalized_column):
            return "up" if normalized_row < 0 else "down"
        return "left" if normalized_column < 0 else "right"

    def _empty_exploration_message(self, action: PlannedAction) -> str:
        if (
            self.last_action_direction is not None
            and self.empty_view_drags <= self.LAST_ACTION_DIRECTION_DRAGS
        ):
            return (
                f"当前视口已清空，沿最后操作方向 {action.direction} 查找"
                f"（{self.empty_view_drags}/{self.LAST_ACTION_DIRECTION_DRAGS}）"
            )
        elapsed = 0.0
        if self.empty_view_started_at is not None:
            elapsed = max(0.0, self.clock() - self.empty_view_started_at)
        budget = int(self.EMPTY_VIEW_TIMEOUT_SECONDS // 60)
        return (
            "当前视口已清空，继续探索"
            f"（已 {int(elapsed // 60)} 分 {int(elapsed % 60)} 秒 / {budget} 分钟）"
        )

    def _generic_recovery_action(
        self,
        rect: tuple[int, int, int, int],
        attempt: int,
    ) -> PlannedAction:
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        direction = LocalNavigator.DIRECTIONS[attempt % len(LocalNavigator.DIRECTIONS)]
        start = (round(width * 0.52), round(height * 0.58))
        delta = {
            "right": (-round(width * 0.40), 0),
            "left": (round(width * 0.40), 0),
            "down": (0, -round(height * 0.34)),
            "up": (0, round(height * 0.34)),
        }[direction]
        return PlannedAction(
            ActionKind.DRAG,
            direction=direction,
            drag_start=start,
            drag_end=(start[0] + delta[0], start[1] + delta[1]),
        )

    def _with_note(self, message: str) -> str:
        if self.recovery_note:
            return f"{message}；{self.recovery_note}"
        return message

    def _fresh_image(self) -> tuple[CaptureFrame, np.ndarray] | None:
        raw_frame = getattr(self.observer, "raw_frame", None)
        if not callable(raw_frame):
            return None
        try:
            frame = raw_frame()
        except (RuntimeError, ValueError):
            return None
        return frame, np.array(frame.image.convert("RGB"))

    def _click_point(self, frame: CaptureFrame, point: tuple[int, int], stop_event: threading.Event) -> None:
        self.controller.click_batch(
            frame.window_rect,
            BatchPoints(marks=(), safe=(point,)),
            stop_event,
        )

    def _dismiss_modals(self, stop_event: threading.Event) -> bool:
        """Detect and dismiss one step of any known modal dialog.

        Returns False immediately when the frame shows no known dialog so
        ordinary recognition problems go straight to recovery drags. Dialog
        clicks are never counted as opened cells."""
        captured = self._fresh_image()
        if captured is None:
            return False
        frame, image = captured
        if detect_cyan_dialog(image):
            amber = detect_welfare_confirm(image)
            if amber is not None:
                self._click_point(frame, amber, stop_event)
                self.recovery_note = "已确认福利宝箱"
                return True
        green = detect_revive_button(image)
        if green is not None and not detect_death_dialog(image):
            self._click_point(frame, green, stop_event)
            self.recovery_note = "已领取宝箱奖励"
            return True
        if detect_death_dialog(image):
            return self._wait_and_revive(stop_event)
        return False

    def _wait_and_revive(self, stop_event: threading.Event) -> bool:
        """Wait out the death countdown, then click the revive button."""
        deadline = time.monotonic() + self.REVIVE_WAIT_SECONDS
        captured = self._fresh_image()
        button = detect_revive_button(captured[1]) if captured else None
        while button is None:
            if stop_event.is_set() or stop_event.wait(self.REVIVE_POLL_INTERVAL):
                return False
            if time.monotonic() >= deadline:
                return False
            captured = self._fresh_image()
            if captured is None:
                return False
            button = detect_revive_button(captured[1])
        frame, _ = captured
        self._click_point(frame, button, stop_event)
        self.recovery_note = "检测到踩雷，已自动点击复活"
        return True

    def _modal_present(self, image: np.ndarray) -> bool:
        if not detect_cyan_dialog(image):
            return False
        return (
            detect_welfare_confirm(image) is not None
            or detect_revive_button(image) is not None
            or detect_death_dialog(image)
        )

    def _wait_for_modal(self, stop_event: threading.Event, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            captured = self._fresh_image()
            if captured is not None and self._modal_present(captured[1]):
                return True
            if stop_event.is_set() or stop_event.wait(self.CHEST_POLL_INTERVAL):
                return False
            if time.monotonic() >= deadline:
                return False

    def _open_closed_chest(
        self,
        stop_event: threading.Event,
        rect: tuple[int, int, int, int],
        center: tuple[int, int],
    ) -> bool:
        """Click a closed chest cell, then dismiss the welfare and reward
        dialogs. Returns False when the expected dialogs never appear."""
        self.controller.click_batch(
            rect,
            BatchPoints(marks=(), safe=(center,)),
            stop_event,
        )
        if not self._wait_for_modal(stop_event, self.CHEST_DIALOG_TIMEOUT):
            return False
        self._dismiss_modals(stop_event)
        if self._wait_for_modal(stop_event, self.CHEST_DIALOG_TIMEOUT):
            self._dismiss_modals(stop_event)
        return True

    def _closed_chest_center(self, observation: Observation) -> tuple[int, int] | None:
        candidates = [
            cell
            for row in observation.grid
            for cell in row
            if cell.kind == CellKind.TREASURE and cell.variant == "closed"
        ]
        if not candidates:
            return None
        rows = len(observation.grid)
        columns = len(observation.grid[0])
        center_row = (rows - 1) / 2.0
        center_column = (columns - 1) / 2.0
        target = min(
            candidates,
            key=lambda cell: (cell.row - center_row) ** 2 + (cell.column - center_column) ** 2,
        )
        return observation.geometry.center(target.row, target.column)

    def _observe_after_chest(self, stop_event: threading.Event, previous: Observation) -> Observation:
        wait_for_stable = getattr(self.observer, "wait_for_stable", None)
        try:
            if callable(wait_for_stable):
                candidate = wait_for_stable(
                    stop_event,
                    force_redetect=True,
                    initial_delay=0.3,
                )
            else:
                self._wait(stop_event, 0.8)
                candidate = self.observer()
        except (ValueError, RuntimeError) as error:
            return self._recover_recognition(stop_event, previous, previous.frame.window_rect, str(error))
        issue = recognition_issue(candidate)
        if issue is not None:
            return self._recover_recognition(stop_event, candidate, candidate.frame.window_rect, issue)
        return candidate

    def _recover_recognition(
        self,
        stop_event: threading.Event,
        observation: Observation | None,
        active_rect: tuple[int, int, int, int] | None,
        reason: str,
    ) -> Observation:
        current = observation
        last_error = reason
        wait_for_stable = getattr(self.observer, "wait_for_stable", None)
        for attempt in range(self.MAX_RECOVERY_DRAGS):
            if stop_event.is_set():
                raise RuntimeError("自动运行已停止")
            if self._dismiss_modals(stop_event):
                try:
                    if callable(wait_for_stable):
                        candidate = wait_for_stable(
                            stop_event,
                            force_redetect=True,
                            initial_delay=0.4,
                        )
                    else:
                        self._wait(stop_event, 0.8)
                        candidate = self.observer()
                except (ValueError, RuntimeError):
                    continue
                if recognition_issue(candidate) is None:
                    self.last_observation = candidate
                    return candidate
                continue
            if current is not None:
                action = plan_drag(current, self.navigator)
                rect = current.frame.window_rect
            else:
                if active_rect is None:
                    raise RuntimeError(f"{reason}；且无法获得窗口位置进行恢复拖动")
                action = self._generic_recovery_action(active_rect, attempt)
                rect = active_rect
            assert action.drag_start is not None and action.drag_end is not None
            self.controller.drag(rect, action.drag_start, action.drag_end, stop_event)
            if stop_event.is_set():
                raise RuntimeError("自动运行已停止")
            try:
                if callable(wait_for_stable):
                    candidate = wait_for_stable(
                        stop_event,
                        force_redetect=True,
                        initial_delay=0.08,
                    )
                else:
                    self._wait(stop_event, 0.65)
                    candidate = self.observer()
            except (ValueError, RuntimeError) as error:
                last_error = str(error)
                current = None
                continue
            issue = recognition_issue(candidate)
            if issue is None:
                if current is not None:
                    pitch = current.geometry.pitch
                    self.navigator.move_viewport(
                        -(action.drag_end[1] - action.drag_start[1]) / pitch,
                        -(action.drag_end[0] - action.drag_start[0]) / pitch,
                    )
                self.last_observation = candidate
                return candidate
            last_error = issue
            current = candidate
        raise RuntimeError(
            f"识别异常后已完成 {self.MAX_RECOVERY_DRAGS} 次恢复拖动，仍未恢复：{last_error}"
        )
