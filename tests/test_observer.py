from __future__ import annotations

import json
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image

from minesweeper.automation import (
    ActionKind,
    LocalAutomation,
    LocalNavigator,
    PlannedAction,
    closed_ratio,
    interior_closed_island,
    plan_action,
    plan_drag,
)
from minesweeper.capture import CaptureFrame
from minesweeper.input_control import BatchExecution, WindowInputController
from minesweeper.observer import WindowObserver, board_motion_ratio, observe_image
from minesweeper.recognition import Cell, CellKind, classify_cell
from minesweeper.dialogs import (
    detect_cyan_dialog,
    detect_death_dialog,
    detect_revive_button,
    detect_welfare_confirm,
)
from minesweeper.solver import Constraint, SolveResult, _enumerate_component, solve_local
from minesweeper.statistics import StatisticsStore


ROOT = Path(__file__).resolve().parents[1]


def grid_from_cells(
    rows: int,
    columns: int,
    overrides: list[tuple[int, int, CellKind, int | None]],
) -> list[list[Cell]]:
    grid = [
        [Cell(row, column, CellKind.OPEN, number=0) for column in range(columns)]
        for row in range(rows)
    ]
    for row, column, kind, number in overrides:
        grid[row][column] = Cell(row, column, kind, number=number)
    return grid


class RecognitionTests(unittest.TestCase):
    def test_plain_gray_tile_is_closed(self) -> None:
        tile = np.full((33, 33, 3), 169, dtype=np.uint8)
        self.assertEqual(classify_cell(tile, 0, 0).kind, CellKind.CLOSED)

    def test_any_artwork_on_gray_tile_is_a_mine(self) -> None:
        for color in ((0, 0, 0), (255, 210, 20), (70, 130, 180)):
            with self.subTest(color=color):
                tile = np.full((33, 33, 3), 169, dtype=np.uint8)
                tile[11:22, 11:22] = color
                self.assertEqual(classify_cell(tile, 0, 0).kind, CellKind.MINE_MARK)

    def test_open_tile_is_not_a_mine(self) -> None:
        tile = np.full((33, 33, 3), 240, dtype=np.uint8)
        self.assertEqual(classify_cell(tile, 0, 0).kind, CellKind.OPEN)

    def test_click_timing_defaults_favor_reliable_input(self) -> None:
        controller = WindowInputController()
        self.assertGreaterEqual(controller.move_settle, 0.010)
        self.assertGreaterEqual(controller.press_duration, 0.020)
        self.assertGreaterEqual(controller.click_interval, 0.075)

    def test_statistics_persist_across_store_restarts_until_cleared(self) -> None:
        with TemporaryDirectory(dir=ROOT / "tests") as directory:
            path = Path(directory) / "statistics.json"
            first = StatisticsStore(path)
            first.add(opened=12, flagged=5)
            restarted = StatisticsStore(path)
            self.assertEqual(restarted.snapshot().opened, 12)
            self.assertEqual(restarted.snapshot().flagged, 5)
            restarted.add(opened=3, flagged=2)
            continued = StatisticsStore(path)
            self.assertEqual(continued.snapshot().opened, 15)
            self.assertEqual(continued.snapshot().flagged, 7)
            continued.clear()
            cleared = StatisticsStore(path)
            self.assertEqual(cleared.snapshot().opened, 0)
            self.assertEqual(cleared.snapshot().flagged, 0)

    def test_invalid_statistics_file_recovers_without_negative_counts(self) -> None:
        with TemporaryDirectory(dir=ROOT / "tests") as directory:
            path = Path(directory) / "statistics.json"
            path.write_text('{"opened": -4, "flagged": "3"}', encoding="utf-8")
            snapshot = StatisticsStore(path).snapshot()
            self.assertEqual(snapshot.opened, 0)
            self.assertEqual(snapshot.flagged, 3)


class ReplayTests(unittest.TestCase):
    def test_saved_sample_matches_all_calibrated_cells(self) -> None:
        observation = observe_image(Image.open(ROOT / "samples" / "initial.png"))
        expected = json.loads((ROOT / "samples" / "initial-analysis.json").read_text(encoding="utf-8"))["grid"]
        symbols = {
            CellKind.CLOSED: "?",
            CellKind.OPEN: "0",
            CellKind.MINE_MARK: "M",
            CellKind.UNRECOGNIZED: "U",
        }
        actual: list[list[str]] = []
        for row in observation.grid:
            actual.append(
                [str(cell.number) if cell.kind == CellKind.NUMBER else symbols[cell.kind] for cell in row]
            )
        self.assertEqual(actual, [row[:26] for row in expected[:11]])
        self.assertFalse(observation.result.contradictions)
        # Direct rules and subset differences find 7/2; the exhaustive
        # component stage surfaces four more pattern deductions.
        self.assertEqual(len(observation.result.safe), 11)
        self.assertEqual(len(observation.result.mines), 3)

    def test_current_view_is_planned_as_one_fast_batch(self) -> None:
        observation = observe_image(Image.open(ROOT / "samples" / "initial.png"))
        action = plan_action(observation, LocalNavigator())
        self.assertEqual(action.kind, ActionKind.BATCH)
        self.assertEqual(len(action.safe), 11)
        self.assertEqual(len(action.marks), 3)

    def test_no_moves_plans_a_local_drag_without_world_map(self) -> None:
        observation = observe_image(Image.open(ROOT / "samples" / "initial.png"))
        no_moves = replace(
            observation,
            result=SolveResult(frozenset(), frozenset(), tuple(), tuple()),
        )
        navigator = LocalNavigator()
        action = plan_action(no_moves, navigator)
        self.assertEqual(action.kind, ActionKind.DRAG)
        self.assertIsNotNone(action.drag_start)
        self.assertIsNotNone(action.drag_end)
        self.assertLessEqual(navigator.recent_views.maxlen or 0, 24)
        assert action.drag_start is not None
        assert action.drag_end is not None
        start_column = round((action.drag_start[0] - observation.geometry.origin_x) / observation.geometry.pitch - 0.5)
        start_row = round((action.drag_start[1] - observation.geometry.origin_y) / observation.geometry.pitch - 0.5)
        self.assertIn(
            observation.grid[start_row][start_column].kind,
            {CellKind.CLOSED, CellKind.OPEN, CellKind.NUMBER, CellKind.TREASURE},
        )
        drag_distance = abs(action.drag_end[0] - action.drag_start[0]) + abs(action.drag_end[1] - action.drag_start[1])
        self.assertGreaterEqual(drag_distance, observation.geometry.pitch * 2)

    def test_drag_can_start_on_a_closed_cell(self) -> None:
        source = observe_image(Image.open(ROOT / "samples" / "initial.png"))
        closed_grid = [
            [replace(cell, kind=CellKind.CLOSED, number=None, confidence=1.0) for cell in row]
            for row in source.grid
        ]
        closed_view = replace(
            source,
            grid=closed_grid,
            result=SolveResult(frozenset(), frozenset(), tuple(), tuple()),
        )
        action = plan_drag(closed_view, LocalNavigator())
        assert action.drag_start is not None
        column = round((action.drag_start[0] - source.geometry.origin_x) / source.geometry.pitch - 0.5)
        row = round((action.drag_start[1] - source.geometry.origin_y) / source.geometry.pitch - 0.5)
        self.assertEqual(closed_view.grid[row][column].kind, CellKind.CLOSED)

    def test_every_cell_kind_can_be_used_as_a_drag_anchor(self) -> None:
        source = observe_image(Image.open(ROOT / "samples" / "initial.png"))
        for kind in CellKind:
            with self.subTest(kind=kind.value):
                uniform_grid = [
                    [
                        replace(
                            cell,
                            kind=kind,
                            number=1 if kind == CellKind.NUMBER else None,
                            confidence=1.0,
                        )
                        for cell in row
                    ]
                    for row in source.grid
                ]
                view = replace(
                    source,
                    grid=uniform_grid,
                    result=SolveResult(frozenset(), frozenset(), tuple(), tuple()),
                )
                action = plan_drag(view, LocalNavigator())
                assert action.drag_start is not None
                column = round((action.drag_start[0] - source.geometry.origin_x) / source.geometry.pitch - 0.5)
                row = round((action.drag_start[1] - source.geometry.origin_y) / source.geometry.pitch - 0.5)
                self.assertEqual(view.grid[row][column].kind, kind)

    def test_resized_window_and_mixed_mine_art_are_stable(self) -> None:
        observation = observe_image(Image.open(ROOT / "samples" / "resized-raw.png"))
        self.assertEqual((observation.geometry.full_columns, observation.geometry.full_rows), (30, 11))
        self.assertFalse(observation.result.contradictions)
        self.assertFalse(
            [cell for row in observation.grid for cell in row if cell.kind == CellKind.UNRECOGNIZED]
        )
        self.assertEqual(len(observation.result.safe), 24)
        self.assertEqual(len(observation.result.mines), 11)

    def test_yellow_chest_is_an_open_safe_cell(self) -> None:
        observation = observe_image(Image.open(ROOT / "samples" / "chest-raw.png"))
        treasures = [
            cell for row in observation.grid for cell in row if cell.kind == CellKind.TREASURE
        ]
        self.assertEqual((observation.geometry.full_columns, observation.geometry.full_rows), (30, 11))
        self.assertEqual([(cell.row, cell.column) for cell in treasures], [(8, 12)])
        self.assertFalse(
            [cell for row in observation.grid for cell in row if cell.kind == CellKind.UNRECOGNIZED]
        )
        self.assertFalse(observation.result.contradictions)
        self.assertNotIn((8, 12), observation.result.mines)

    def test_closed_chest_art_is_also_an_open_safe_cell(self) -> None:
        observation = observe_image(Image.open(ROOT / "samples" / "closed-chest-raw.png"))
        treasures = [
            cell for row in observation.grid for cell in row if cell.kind == CellKind.TREASURE
        ]
        self.assertEqual([(cell.row, cell.column) for cell in treasures], [(3, 2)])
        self.assertFalse(
            [cell for row in observation.grid for cell in row if cell.kind == CellKind.UNRECOGNIZED]
        )
        self.assertFalse(observation.result.contradictions)
        self.assertNotIn((3, 2), observation.result.mines)

    def test_number_six_sample_is_recognized(self) -> None:
        observation = observe_image(Image.open(ROOT / "samples" / "number-6-raw.png"))
        six = observation.grid[4][3]
        self.assertEqual(six.kind, CellKind.NUMBER)
        self.assertEqual(six.number, 6)
        self.assertFalse(
            [cell for row in observation.grid for cell in row if cell.kind == CellKind.UNRECOGNIZED]
        )
        self.assertFalse(observation.result.contradictions)

    def test_open_mine_is_counted_but_never_scheduled_for_input(self) -> None:
        observation = observe_image(Image.open(ROOT / "samples" / "open-mine-raw.png"))
        open_mines = [
            cell for row in observation.grid for cell in row if cell.kind == CellKind.OPEN_MINE
        ]
        self.assertEqual([(cell.row, cell.column) for cell in open_mines], [(6, 5)])
        self.assertFalse(
            [cell for row in observation.grid for cell in row if cell.kind == CellKind.UNRECOGNIZED]
        )
        self.assertFalse(observation.result.contradictions)
        self.assertNotIn((6, 5), observation.result.safe)
        self.assertNotIn((6, 5), observation.result.mines)

    def test_drag_frames_are_ignored_until_three_stable_pairs(self) -> None:
        base_image = Image.open(ROOT / "samples" / "chest-raw.png").convert("RGB")
        base = np.array(base_image)
        moving_one = base.copy()
        moving_two = base.copy()
        moving_one[98:] = np.roll(moving_one[98:], 5, axis=1)
        moving_two[98:] = np.roll(moving_two[98:], 11, axis=1)
        self.assertGreater(board_motion_ratio(moving_one, moving_two, 98), 0.025)
        self.assertEqual(board_motion_ratio(base, base.copy(), 98), 0.0)

        initial = observe_image(base_image)
        observer = WindowObserver(initial=initial)
        frames = [
            CaptureFrame(Image.fromarray(moving_one), (0, 0, 995, 470), False),
            CaptureFrame(Image.fromarray(moving_two), (0, 0, 995, 470), False),
            CaptureFrame(base_image, (0, 0, 995, 470), False),
            CaptureFrame(base_image, (0, 0, 995, 470), False),
            CaptureFrame(base_image, (0, 0, 995, 470), False),
            CaptureFrame(base_image, (0, 0, 995, 470), False),
        ]
        with patch("minesweeper.observer.capture_window", side_effect=frames) as capture:
            result = observer.wait_for_stable(
                threading.Event(),
                force_redetect=True,
                timeout=1.0,
                interval=0.0,
            )
        self.assertEqual(capture.call_count, 6)
        self.assertEqual(result.geometry.full_rows, 11)
        self.assertFalse(result.result.contradictions)

    def test_recognition_recovery_stops_only_after_eight_drags(self) -> None:
        valid = observe_image(Image.open(ROOT / "samples" / "initial.png"))
        abnormal = replace(
            valid,
            result=SolveResult(frozenset(), frozenset(), tuple(), ("模拟识别异常",)),
        )

        class FakeObserver:
            def wait_for_stable(self, *args: object, **kwargs: object):
                return abnormal

        class FakeController:
            def __init__(self) -> None:
                self.drags = 0

            def drag(self, *args: object, **kwargs: object) -> None:
                self.drags += 1

        controller = FakeController()
        automation = LocalAutomation(observer=FakeObserver(), controller=controller)
        with self.assertRaisesRegex(RuntimeError, "8 次恢复拖动"):
            automation._recover_recognition(
                threading.Event(),
                abnormal,
                abnormal.frame.window_rect,
                "模拟识别异常",
            )
        self.assertEqual(controller.drags, 8)

    def test_recognition_recovery_resumes_as_soon_as_view_is_valid(self) -> None:
        valid = observe_image(Image.open(ROOT / "samples" / "initial.png"))
        abnormal = replace(
            valid,
            result=SolveResult(frozenset(), frozenset(), tuple(), ("模拟识别异常",)),
        )

        class FakeObserver:
            def __init__(self) -> None:
                self.results = iter((abnormal, abnormal, valid))

            def wait_for_stable(self, *args: object, **kwargs: object):
                return next(self.results)

        class FakeController:
            def __init__(self) -> None:
                self.drags = 0

            def drag(self, *args: object, **kwargs: object) -> None:
                self.drags += 1

        controller = FakeController()
        automation = LocalAutomation(observer=FakeObserver(), controller=controller)
        recovered = automation._recover_recognition(
            threading.Event(),
            abnormal,
            abnormal.frame.window_rect,
            "模拟识别异常",
        )
        self.assertIs(recovered, valid)
        self.assertEqual(controller.drags, 3)

    def test_empty_view_keeps_exploring_until_five_minutes(self) -> None:
        source = observe_image(Image.open(ROOT / "samples" / "initial.png"))
        empty_grid = [
            [replace(cell, kind=CellKind.OPEN, number=0, confidence=1.0) for cell in row]
            for row in source.grid
        ]
        empty = replace(
            source,
            grid=empty_grid,
            result=SolveResult(frozenset(), frozenset(), tuple(), tuple()),
        )

        class FakeObserver:
            def __call__(self):
                return empty

            def wait_for_stable(self, *args: object, **kwargs: object):
                return empty

        class FakeController:
            def __init__(self) -> None:
                self.drags = 0

            def activate_target(self):
                return 1, empty.frame.window_rect

            def drag(self, *args: object, **kwargs: object) -> None:
                self.drags += 1

        now = [1000.0]
        controller = FakeController()
        automation = LocalAutomation(
            observer=FakeObserver(),
            controller=controller,
            clock=lambda: now[0],
        )
        outcomes = [automation.perform_cycle(threading.Event()) for _ in range(9)]
        self.assertTrue(all(outcome.action.kind == ActionKind.DRAG for outcome in outcomes))
        self.assertEqual(controller.drags, 9)
        now[0] += automation.EMPTY_VIEW_TIMEOUT_SECONDS + 1.0
        final = automation.perform_cycle(threading.Event())
        self.assertEqual(final.action.kind, ActionKind.COMPLETE)
        self.assertIn("5 分钟", final.message)
        self.assertIn("自动运行结束", final.message)

    def test_empty_view_timer_resets_when_closed_cells_reappear(self) -> None:
        source = observe_image(Image.open(ROOT / "samples" / "initial.png"))
        empty_grid = [
            [replace(cell, kind=CellKind.OPEN, number=0, confidence=1.0) for cell in row]
            for row in source.grid
        ]
        empty = replace(
            source,
            grid=empty_grid,
            result=SolveResult(frozenset(), frozenset(), tuple(), tuple()),
        )

        class FakeObserver:
            def __call__(self):
                return empty

            def wait_for_stable(self, *args: object, **kwargs: object):
                return source

        class FakeController:
            def activate_target(self):
                return 1, empty.frame.window_rect

            def drag(self, *args: object, **kwargs: object) -> None:
                return None

        automation = LocalAutomation(observer=FakeObserver(), controller=FakeController())
        outcome = automation.perform_cycle(threading.Event())
        self.assertIn("发现新的未开启格", outcome.message)
        self.assertEqual(automation.empty_view_drags, 0)
        self.assertIsNone(automation.empty_view_started_at)

    def test_cleared_view_first_searches_toward_last_action(self) -> None:
        source = observe_image(Image.open(ROOT / "samples" / "initial.png"))
        empty_grid = [
            [replace(cell, kind=CellKind.OPEN, number=0, confidence=1.0) for cell in row]
            for row in source.grid
        ]
        empty = replace(
            source,
            grid=empty_grid,
            result=SolveResult(frozenset(), frozenset(), tuple(), tuple()),
        )

        class FakeObserver:
            def __call__(self):
                return empty

            def wait_for_stable(self, *args: object, **kwargs: object):
                return empty

        class FakeController:
            def activate_target(self):
                return 1, empty.frame.window_rect

            def drag(self, *args: object, **kwargs: object) -> None:
                return None

        automation = LocalAutomation(observer=FakeObserver(), controller=FakeController())
        automation.last_action_direction = "up"
        first = automation.perform_cycle(threading.Event())
        second = automation.perform_cycle(threading.Event())
        self.assertEqual(first.action.direction, "up")
        self.assertIn("沿最后操作方向", first.message)
        self.assertNotIn("沿最后操作方向", second.message)

    def test_last_action_position_maps_to_view_direction(self) -> None:
        source = observe_image(Image.open(ROOT / "samples" / "initial.png"))
        rows = len(source.grid)
        columns = len(source.grid[0])
        cases = {
            (0, columns // 2): "up",
            (rows - 1, columns // 2): "down",
            (rows // 2, 0): "left",
            (rows // 2, columns - 1): "right",
        }
        for position, expected in cases.items():
            with self.subTest(position=position):
                self.assertEqual(LocalAutomation._position_direction(source, position), expected)

    def test_navigation_follows_a_vertical_open_frontier(self) -> None:
        source = observe_image(Image.open(ROOT / "samples" / "initial.png"))
        split_column = len(source.grid[0]) // 2
        frontier_grid = []
        for row in source.grid:
            frontier_grid.append(
                [
                    replace(
                        cell,
                        kind=CellKind.OPEN if cell.column < split_column else CellKind.CLOSED,
                        number=0 if cell.column < split_column else None,
                        confidence=1.0,
                    )
                    for cell in row
                ]
            )
        frontier_view = replace(
            source,
            grid=frontier_grid,
            result=SolveResult(frozenset(), frozenset(), tuple(), tuple()),
        )
        navigator = LocalNavigator()
        directions = [navigator.choose_direction(frontier_view) for _ in range(3)]
        self.assertTrue(all(direction == "up" for direction in directions))
        self.assertEqual(len(set(directions)), 1)

    def test_navigation_clockwise_mapping_around_closed_region(self) -> None:
        source = observe_image(Image.open(ROOT / "samples" / "initial.png"))
        rows = len(source.grid)
        columns = len(source.grid[0])
        cases = {
            "closed_right": (lambda row, column: column >= columns // 2, "up"),
            "closed_below": (lambda row, column: row >= rows // 2, "right"),
            "closed_left": (lambda row, column: column < columns // 2, "down"),
            "closed_above": (lambda row, column: row < rows // 2, "left"),
        }
        for label, (is_closed, expected) in cases.items():
            with self.subTest(case=label):
                grid = [
                    [
                        replace(
                            cell,
                            kind=CellKind.CLOSED if is_closed(cell.row, cell.column) else CellKind.OPEN,
                            number=None if is_closed(cell.row, cell.column) else 0,
                            confidence=1.0,
                        )
                        for cell in row
                    ]
                    for row in source.grid
                ]
                view = replace(
                    source,
                    grid=grid,
                    result=SolveResult(frozenset(), frozenset(), tuple(), tuple()),
                )
                self.assertEqual(LocalNavigator().choose_direction(view), expected)

    def test_drag_recenters_open_closed_frontier_toward_half_view(self) -> None:
        source = observe_image(Image.open(ROOT / "samples" / "initial.png"))
        split_column = int(len(source.grid[0]) * 0.65)
        grid = [
            [
                replace(
                    cell,
                    kind=CellKind.OPEN if cell.column < split_column else CellKind.CLOSED,
                    number=0 if cell.column < split_column else None,
                    confidence=1.0,
                )
                for cell in row
            ]
            for row in source.grid
        ]
        view = replace(
            source,
            grid=grid,
            result=SolveResult(frozenset(), frozenset(), tuple(), tuple()),
        )
        action = plan_drag(view, LocalNavigator())
        self.assertEqual(action.direction, "up")
        assert action.drag_start is not None and action.drag_end is not None
        self.assertLess(action.drag_end[0], action.drag_start[0])
        self.assertGreater(action.drag_end[1], action.drag_start[1])

    def test_closed_heavy_view_moves_back_toward_visible_open_region(self) -> None:
        source = observe_image(Image.open(ROOT / "samples" / "initial.png"))
        columns = len(source.grid[0])
        split = max(2, columns // 5)
        grid = [
            [
                replace(
                    cell,
                    kind=CellKind.OPEN if cell.column < split else CellKind.CLOSED,
                    number=0 if cell.column < split else None,
                    confidence=1.0,
                )
                for cell in row
            ]
            for row in source.grid
        ]
        view = replace(source, grid=grid, result=SolveResult(frozenset(), frozenset(), tuple(), tuple()))
        self.assertGreaterEqual(closed_ratio(view), 0.70)
        navigator = LocalNavigator()
        action = plan_drag(view, navigator)
        self.assertEqual(action.direction, "left")
        assert action.drag_start is not None and action.drag_end is not None
        self.assertGreater(action.drag_end[0], action.drag_start[0])

    def test_all_closed_view_uses_last_known_open_direction(self) -> None:
        source = observe_image(Image.open(ROOT / "samples" / "initial.png"))
        columns = len(source.grid[0])
        split = max(2, columns // 5)
        mixed_grid = [
            [
                replace(
                    cell,
                    kind=CellKind.OPEN if cell.column < split else CellKind.CLOSED,
                    number=0 if cell.column < split else None,
                    confidence=1.0,
                )
                for cell in row
            ]
            for row in source.grid
        ]
        all_closed_grid = [
            [replace(cell, kind=CellKind.CLOSED, number=None, confidence=1.0) for cell in row]
            for row in source.grid
        ]
        navigator = LocalNavigator()
        mixed = replace(source, grid=mixed_grid, result=SolveResult(frozenset(), frozenset(), tuple(), tuple()))
        all_closed = replace(source, grid=all_closed_grid, result=SolveResult(frozenset(), frozenset(), tuple(), tuple()))
        self.assertEqual(navigator.choose_direction(mixed), "left")
        self.assertEqual(navigator.choose_direction(all_closed), "left")

    def test_open_heavy_view_moves_toward_closed_region(self) -> None:
        source = observe_image(Image.open(ROOT / "samples" / "initial.png"))
        columns = len(source.grid[0])
        split = int(columns * 0.8)
        grid = [
            [
                replace(
                    cell,
                    kind=CellKind.OPEN if cell.column < split else CellKind.CLOSED,
                    number=0 if cell.column < split else None,
                    confidence=1.0,
                )
                for cell in row
            ]
            for row in source.grid
        ]
        view = replace(source, grid=grid, result=SolveResult(frozenset(), frozenset(), tuple(), tuple()))
        self.assertLessEqual(closed_ratio(view), 0.30)
        self.assertEqual(LocalNavigator().choose_direction(view), "right")

    def test_center_closed_island_triggers_escape_directly(self) -> None:
        source = observe_image(Image.open(ROOT / "samples" / "initial.png"))
        rows = len(source.grid)
        columns = len(source.grid[0])
        center_row = rows // 2
        center_column = columns // 2
        grid = [
            [
                replace(
                    cell,
                    kind=(
                        CellKind.CLOSED
                        if abs(cell.row - center_row) <= 1 and abs(cell.column - center_column) <= 1
                        else CellKind.OPEN
                    ),
                    number=(
                        None
                        if abs(cell.row - center_row) <= 1 and abs(cell.column - center_column) <= 1
                        else 0
                    ),
                    confidence=1.0,
                )
                for cell in row
            ]
            for row in source.grid
        ]
        view = replace(source, grid=grid, result=SolveResult(frozenset(), frozenset(), tuple(), tuple()))
        self.assertIsNotNone(interior_closed_island(view))
        automation = LocalAutomation(observer=None, controller=None)
        automation.rng = _FixedRandom("up")
        action = plan_drag(view, automation.navigator)
        action, note = automation._apply_escape_or_record(view, action)
        self.assertEqual(action.direction, "up")
        self.assertEqual(automation.escape_direction, "up")
        self.assertEqual(automation.escape_moves_done, 1)
        self.assertIn("中央孤立未开区", note)
        self.assertIn("脱困移动", note)

    def test_small_closed_region_touching_edge_is_not_a_center_island(self) -> None:
        source = observe_image(Image.open(ROOT / "samples" / "initial.png"))
        grid = [
            [
                replace(
                    cell,
                    kind=CellKind.CLOSED if cell.column <= 2 else CellKind.OPEN,
                    number=None if cell.column <= 2 else 0,
                    confidence=1.0,
                )
                for cell in row
            ]
            for row in source.grid
        ]
        view = replace(source, grid=grid, result=SolveResult(frozenset(), frozenset(), tuple(), tuple()))
        self.assertIsNone(interior_closed_island(view))

    def test_blocked_frontier_turns_clockwise_instead_of_reversing(self) -> None:
        navigator = LocalNavigator()
        navigator.last_direction = "right"
        self.assertEqual(navigator.turn_clockwise(), "down")
        self.assertEqual(navigator.turn_clockwise(), "left")
        self.assertEqual(navigator.turn_clockwise(), "up")

    def test_one_minute_frontier_search_shows_f8_advice(self) -> None:
        now = [99.9]
        automation = LocalAutomation(clock=lambda: now[0])
        automation.search_started_at = 40.0
        self.assertIsNone(automation._search_advice())
        now[0] = 100.1
        advice = automation._search_advice()
        self.assertIsNotNone(advice)
        self.assertIn("F8", advice or "")
        self.assertIn("更换区域", advice or "")


class DeductionTests(unittest.TestCase):
    def test_1_2_1_pattern_is_fully_deduced(self) -> None:
        # Numbers 1-2-1 over a row of five closed cells have exactly one
        # consistent assignment; subset differences alone find none of it.
        grid = grid_from_cells(
            5,
            7,
            [
                (1, 0, CellKind.CLOSED, None),
                (1, 1, CellKind.CLOSED, None),
                (1, 2, CellKind.CLOSED, None),
                (1, 3, CellKind.CLOSED, None),
                (1, 4, CellKind.CLOSED, None),
                (2, 1, CellKind.NUMBER, 1),
                (2, 2, CellKind.NUMBER, 2),
                (2, 3, CellKind.NUMBER, 1),
            ],
        )
        result = solve_local(grid)
        self.assertFalse(result.contradictions)
        self.assertEqual(result.mines, {(1, 1), (1, 3)})
        self.assertTrue({(1, 0), (1, 2), (1, 4)} <= result.safe)

    def test_overlapping_pair_with_different_counts_is_deduced(self) -> None:
        # {a,b}=1 next to {b,c}=2 has the unique solution b=c=1, a=0; the two
        # constraints are the same size, so subset differences cannot see it.
        grid = grid_from_cells(
            5,
            7,
            [
                (1, 1, CellKind.CLOSED, None),
                (1, 2, CellKind.CLOSED, None),
                (1, 3, CellKind.CLOSED, None),
                (2, 1, CellKind.NUMBER, 1),
                (2, 3, CellKind.NUMBER, 2),
            ],
        )
        result = solve_local(grid)
        self.assertFalse(result.contradictions)
        self.assertEqual(result.mines, {(1, 2), (1, 3)})
        self.assertIn((1, 1), result.safe)

    def test_loose_constraint_never_reports_guesses(self) -> None:
        # One constraint over 30 cells with 10 mines has an astronomical
        # solution space and no definite cell; the enumeration must exhaust
        # its budget (or abort early) without ever reporting a guess.
        cells = frozenset((100, column) for column in range(30))
        mines, safe, _truncated = _enumerate_component(frozenset({Constraint(cells, 10)}))
        self.assertEqual(mines, frozenset())
        self.assertEqual(safe, frozenset())


class ViewportTrailTests(unittest.TestCase):
    def test_trail_records_viewport_positions(self) -> None:
        navigator = LocalNavigator()
        navigator.move_viewport(0.0, 12.0)
        self.assertEqual(navigator.position, (0.0, 12.0))
        self.assertEqual(len(navigator.trail), 2)
        self.assertGreaterEqual(navigator.revisit_count("left"), 1)
        self.assertEqual(navigator.revisit_count("right"), 0)

    def test_straight_line_progress_is_never_penalized(self) -> None:
        for direction, step in LocalNavigator.STEP_CELLS.items():
            with self.subTest(direction=direction):
                navigator = LocalNavigator()
                navigator.move_viewport(step[0], step[1])
                self.assertEqual(navigator.revisit_count(direction), 0)

    def test_avoid_revisit_prefers_fresh_frontier_direction(self) -> None:
        navigator = LocalNavigator()
        navigator.move_viewport(0.0, 12.0)
        scores = {"right": 10.0, "down": 5.0, "left": 0.0, "up": 0.0}
        # Going left would return to the origin viewport; right is fresh and
        # scores highest, so it wins over down.
        direction, avoided = navigator._avoid_revisit("left", scores)
        self.assertTrue(avoided)
        self.assertEqual(direction, "right")

    def test_avoid_revisit_keeps_direction_without_alternatives(self) -> None:
        navigator = LocalNavigator()
        navigator.position = (0.0, 12.0)
        # Every direction's nominal target has been visited once already, so
        # there is no fresher alternative and the original choice must stand.
        navigator.trail = [
            (0.0, 12.0),
            (0.0, 24.0),
            (8.0, 12.0),
            (0.0, 0.0),
            (-8.0, 12.0),
        ]
        scores = {"right": 10.0, "down": 0.0, "left": 0.0, "up": 0.0}
        direction, avoided = navigator._avoid_revisit("right", scores)
        self.assertFalse(avoided)
        self.assertEqual(direction, "right")

    def test_performed_drag_is_recorded_in_trail(self) -> None:
        source = observe_image(Image.open(ROOT / "samples" / "initial.png"))
        columns = len(source.grid[0])
        split = columns // 2
        view = [
            [
                replace(
                    cell,
                    kind=CellKind.OPEN if cell.column < split else CellKind.CLOSED,
                    number=0 if cell.column < split else None,
                    confidence=1.0,
                )
                for cell in row
            ]
            for row in source.grid
        ]
        before = replace(source, grid=view, result=SolveResult(frozenset(), frozenset(), tuple(), tuple()))
        opened_cell = replace(before.grid[5][split], kind=CellKind.OPEN, number=0, confidence=1.0)
        after_grid = [row[:] for row in before.grid]
        after_grid[5][split] = opened_cell
        after = replace(before, grid=after_grid)

        class FakeObserver:
            def __call__(self):
                return before

            def wait_for_stable(self, *args: object, **kwargs: object):
                return after

        class FakeController:
            def activate_target(self):
                return 1, before.frame.window_rect

            def drag(self, *args: object, **kwargs: object) -> None:
                return None

        automation = LocalAutomation(observer=FakeObserver(), controller=FakeController())
        outcome = automation.perform_cycle(threading.Event())
        self.assertEqual(outcome.action.kind, ActionKind.DRAG)
        self.assertEqual(len(automation.navigator.trail), 2)
        self.assertNotEqual(automation.navigator.position, (0.0, 0.0))

    def test_failed_drag_leaves_trail_unchanged(self) -> None:
        source = observe_image(Image.open(ROOT / "samples" / "initial.png"))
        empty_grid = [
            [replace(cell, kind=CellKind.OPEN, number=0, confidence=1.0) for cell in row]
            for row in source.grid
        ]
        empty = replace(
            source,
            grid=empty_grid,
            result=SolveResult(frozenset(), frozenset(), tuple(), tuple()),
        )

        class FakeObserver:
            def __call__(self):
                return empty

            def wait_for_stable(self, *args: object, **kwargs: object):
                return empty

        class FakeController:
            def activate_target(self):
                return 1, empty.frame.window_rect

            def drag(self, *args: object, **kwargs: object) -> None:
                return None

        automation = LocalAutomation(observer=FakeObserver(), controller=FakeController())
        automation.perform_cycle(threading.Event())
        self.assertEqual(automation.navigator.trail, [(0.0, 0.0)])
        self.assertEqual(automation.navigator.position, (0.0, 0.0))


class DialogTests(unittest.TestCase):
    @staticmethod
    def _death_array() -> np.ndarray:
        return np.array(Image.open(ROOT / "samples" / "death-state.png").convert("RGB"))

    @classmethod
    def _dialog_only_array(cls) -> np.ndarray:
        # The countdown phase: death dialog present, revive button not yet.
        array = cls._death_array().copy()
        array[265:315, 305:415] = (228, 247, 247)
        return array

    @staticmethod
    def _sample_array(name: str) -> np.ndarray:
        return np.array(Image.open(ROOT / "samples" / name).convert("RGB"))

    @staticmethod
    def _frame(image: Image.Image) -> CaptureFrame:
        return CaptureFrame(image, (0, 0, image.width, image.height), False)

    def test_death_dialog_and_button_are_detected(self) -> None:
        death = self._death_array()
        self.assertTrue(detect_death_dialog(death))
        button = detect_revive_button(death)
        self.assertIsNotNone(button)
        assert button is not None
        self.assertAlmostEqual(button[0], 360, delta=6)
        self.assertAlmostEqual(button[1], 288, delta=6)

    def test_normal_frames_do_not_trigger_any_dialog(self) -> None:
        for name in ("initial.png", "chest-raw.png", "resized-raw.png"):
            with self.subTest(sample=name):
                array = self._sample_array(name)
                self.assertFalse(detect_cyan_dialog(array))
                self.assertFalse(detect_death_dialog(array))
                self.assertIsNone(detect_revive_button(array))
                self.assertIsNone(detect_welfare_confirm(array))

    def test_welfare_dialog_is_detected_without_death(self) -> None:
        welfare = self._sample_array("welfare-chest-dialog.png")
        self.assertTrue(detect_cyan_dialog(welfare))
        self.assertFalse(detect_death_dialog(welfare))
        self.assertIsNone(detect_revive_button(welfare))
        confirm = detect_welfare_confirm(welfare)
        self.assertIsNotNone(confirm)
        assert confirm is not None
        self.assertAlmostEqual(confirm[0], 352, delta=6)
        self.assertAlmostEqual(confirm[1], 348, delta=6)

    def test_reward_dialog_keeps_green_button_but_is_not_death(self) -> None:
        reward = self._sample_array("chest-reward-dialog.png")
        self.assertTrue(detect_cyan_dialog(reward))
        self.assertFalse(detect_death_dialog(reward))
        button = detect_revive_button(reward)
        self.assertIsNotNone(button)
        assert button is not None
        self.assertAlmostEqual(button[0], 353, delta=6)
        self.assertAlmostEqual(button[1], 282, delta=6)

    def test_chest_variant_classification(self) -> None:
        closed = observe_image(Image.open(ROOT / "samples" / "closed-chest-raw.png"))
        opened = observe_image(Image.open(ROOT / "samples" / "chest-raw.png"))
        closed_cells = [
            cell for row in closed.grid for cell in row if cell.kind == CellKind.TREASURE
        ]
        opened_cells = [
            cell for row in opened.grid for cell in row if cell.kind == CellKind.TREASURE
        ]
        self.assertEqual([(cell.variant) for cell in closed_cells], ["closed"])
        self.assertEqual([(cell.variant) for cell in opened_cells], ["open"])

    def test_dismiss_modals_clicks_welfare_confirm(self) -> None:
        welfare_image = Image.fromarray(self._sample_array("welfare-chest-dialog.png"))

        class FakeObserver:
            def raw_frame(self) -> CaptureFrame:
                return DialogTests._frame(welfare_image)

        class FakeController:
            def __init__(self) -> None:
                self.clicks: list[tuple[int, int]] = []

            def click_batch(self, _rect, points, _stop_event) -> BatchExecution:
                self.clicks.extend(points.safe)
                return BatchExecution(flagged=0, opened=len(points.safe))

        controller = FakeController()
        automation = LocalAutomation(observer=FakeObserver(), controller=controller)
        self.assertTrue(automation._dismiss_modals(threading.Event()))
        self.assertEqual(len(controller.clicks), 1)
        self.assertAlmostEqual(controller.clicks[0][0], 352, delta=6)
        self.assertAlmostEqual(controller.clicks[0][1], 348, delta=6)
        self.assertIn("福利宝箱", automation.recovery_note or "")

    def test_dismiss_modals_clicks_reward_confirm_not_revive(self) -> None:
        reward_image = Image.fromarray(self._sample_array("chest-reward-dialog.png"))

        class FakeObserver:
            def raw_frame(self) -> CaptureFrame:
                return DialogTests._frame(reward_image)

        class FakeController:
            def __init__(self) -> None:
                self.clicks: list[tuple[int, int]] = []

            def click_batch(self, _rect, points, _stop_event) -> BatchExecution:
                self.clicks.extend(points.safe)
                return BatchExecution(flagged=0, opened=len(points.safe))

        controller = FakeController()
        automation = LocalAutomation(observer=FakeObserver(), controller=controller)
        self.assertTrue(automation._dismiss_modals(threading.Event()))
        self.assertAlmostEqual(controller.clicks[0][0], 353, delta=6)
        self.assertAlmostEqual(controller.clicks[0][1], 282, delta=6)
        self.assertIn("宝箱奖励", automation.recovery_note or "")
        self.assertNotIn("复活", automation.recovery_note or "")

    def test_dismiss_modals_clicks_revive_on_death_frame(self) -> None:
        death_image = Image.fromarray(self._death_array())

        class FakeObserver:
            def raw_frame(self) -> CaptureFrame:
                return DialogTests._frame(death_image)

            def wait_for_stable(self, *args: object, **kwargs: object):
                return valid

        class FakeController:
            def __init__(self) -> None:
                self.clicks: list[tuple[int, int]] = []

            def click_batch(self, _rect, points, _stop_event) -> BatchExecution:
                self.clicks.extend(points.safe)
                return BatchExecution(flagged=0, opened=len(points.safe))

        valid = observe_image(Image.open(ROOT / "samples" / "initial.png"))
        controller = FakeController()
        automation = LocalAutomation(observer=FakeObserver(), controller=controller)
        self.assertTrue(automation._dismiss_modals(threading.Event()))
        self.assertAlmostEqual(controller.clicks[0][0], 360, delta=6)
        self.assertAlmostEqual(controller.clicks[0][1], 288, delta=6)
        self.assertIn("复活", automation.recovery_note or "")

    def test_dismiss_modals_waits_for_revive_during_countdown(self) -> None:
        dialog_image = Image.fromarray(self._dialog_only_array())
        button_image = Image.fromarray(self._death_array())

        class FakeObserver:
            def __init__(self) -> None:
                self.frames = [dialog_image, dialog_image, button_image]
                self.captures = 0

            def raw_frame(self) -> CaptureFrame:
                self.captures += 1
                return DialogTests._frame(
                    self.frames.pop(0) if len(self.frames) > 1 else self.frames[0]
                )

            def wait_for_stable(self, *args: object, **kwargs: object):
                return valid

        class FakeController:
            def __init__(self) -> None:
                self.clicks: list[tuple[int, int]] = []

            def click_batch(self, _rect, points, _stop_event) -> BatchExecution:
                self.clicks.extend(points.safe)
                return BatchExecution(flagged=0, opened=len(points.safe))

        valid = observe_image(Image.open(ROOT / "samples" / "initial.png"))
        controller = FakeController()
        automation = LocalAutomation(observer=FakeObserver(), controller=controller)
        self.assertTrue(automation._dismiss_modals(threading.Event()))
        self.assertEqual(len(controller.clicks), 1)
        self.assertEqual(automation.observer.captures, 3)
        self.assertIn("复活", automation.recovery_note or "")

    def test_dismiss_modals_times_out_without_button(self) -> None:
        dialog_image = Image.fromarray(self._dialog_only_array())

        class FakeObserver:
            def raw_frame(self) -> CaptureFrame:
                return DialogTests._frame(dialog_image)

        class FakeController:
            def __init__(self) -> None:
                self.clicks: list[tuple[int, int]] = []

            def click_batch(self, _rect, points, _stop_event) -> BatchExecution:
                self.clicks.extend(points.safe)
                return BatchExecution(flagged=0, opened=len(points.safe))

        automation = LocalAutomation(observer=FakeObserver(), controller=FakeController())
        automation.REVIVE_WAIT_SECONDS = 0.2
        automation.REVIVE_POLL_INTERVAL = 0.05
        self.assertFalse(automation._dismiss_modals(threading.Event()))
        self.assertIsNone(automation.recovery_note)

    def test_dismiss_modals_skips_normal_frames(self) -> None:
        normal_image = Image.open(ROOT / "samples" / "initial.png").convert("RGB")

        class FakeObserver:
            def raw_frame(self) -> CaptureFrame:
                return DialogTests._frame(normal_image)

        class FakeController:
            def __init__(self) -> None:
                self.clicks: list[tuple[int, int]] = []

            def click_batch(self, _rect, points, _stop_event) -> BatchExecution:
                self.clicks.extend(points.safe)
                return BatchExecution(flagged=0, opened=len(points.safe))

        controller = FakeController()
        automation = LocalAutomation(observer=FakeObserver(), controller=controller)
        self.assertFalse(automation._dismiss_modals(threading.Event()))
        self.assertEqual(controller.clicks, [])
        self.assertIsNone(automation.recovery_note)

    def test_recovery_prefers_dialog_handling_over_recovery_drags(self) -> None:
        valid = observe_image(Image.open(ROOT / "samples" / "initial.png"))
        abnormal = replace(
            valid,
            result=SolveResult(frozenset(), frozenset(), tuple(), ("模拟识别异常",)),
        )
        death_image = Image.fromarray(self._death_array())

        class FakeObserver:
            def raw_frame(self) -> CaptureFrame:
                return DialogTests._frame(death_image)

            def wait_for_stable(self, *args: object, **kwargs: object):
                return valid

        class FakeController:
            def __init__(self) -> None:
                self.clicks = 0
                self.drags = 0

            def click_batch(self, _rect, points, _stop_event) -> BatchExecution:
                self.clicks += len(points.safe)
                return BatchExecution(flagged=0, opened=len(points.safe))

            def drag(self, *args: object, **kwargs: object) -> None:
                self.drags += 1

        controller = FakeController()
        automation = LocalAutomation(observer=FakeObserver(), controller=controller)
        recovered = automation._recover_recognition(
            threading.Event(),
            abnormal,
            abnormal.frame.window_rect,
            "模拟识别异常",
        )
        self.assertIs(recovered, valid)
        self.assertEqual(controller.drags, 0)
        self.assertEqual(controller.clicks, 1)
        self.assertIn("复活", automation.recovery_note or "")

    def test_recovery_note_is_appended_to_messages(self) -> None:
        automation = LocalAutomation(observer=None, controller=None)
        self.assertEqual(automation._with_note("本批执行 1 个标雷"), "本批执行 1 个标雷")
        automation.recovery_note = "已自动开启宝箱并领取奖励"
        self.assertEqual(
            automation._with_note("本批执行 1 个标雷"),
            "本批执行 1 个标雷；已自动开启宝箱并领取奖励",
        )


class ChestFlowTests(unittest.TestCase):
    @staticmethod
    def _frame(image: Image.Image) -> CaptureFrame:
        return CaptureFrame(image, (0, 0, image.width, image.height), False)

    @staticmethod
    def _image(name: str) -> Image.Image:
        return Image.open(ROOT / "samples" / name).convert("RGB")

    def test_closed_chest_center_targets_treasure_cell(self) -> None:
        closed = observe_image(self._image("closed-chest-raw.png"))
        opened = observe_image(self._image("chest-raw.png"))
        automation = LocalAutomation(observer=None, controller=None)
        self.assertEqual(
            automation._closed_chest_center(closed),
            closed.geometry.center(3, 2),
        )
        self.assertIsNone(automation._closed_chest_center(opened))

    def test_open_closed_chest_clicks_cell_then_both_confirms(self) -> None:
        welfare_image = self._image("welfare-chest-dialog.png")
        reward_image = self._image("chest-reward-dialog.png")

        class FakeObserver:
            def __init__(self) -> None:
                self.frames = [welfare_image, welfare_image, reward_image, reward_image]

            def raw_frame(self) -> CaptureFrame:
                return ChestFlowTests._frame(
                    self.frames.pop(0) if len(self.frames) > 1 else self.frames[0]
                )

        class FakeController:
            def __init__(self) -> None:
                self.clicks: list[tuple[int, int]] = []

            def click_batch(self, _rect, points, _stop_event) -> BatchExecution:
                self.clicks.extend(points.safe)
                return BatchExecution(flagged=0, opened=len(points.safe))

        controller = FakeController()
        automation = LocalAutomation(observer=FakeObserver(), controller=controller)
        handled = automation._open_closed_chest(
            threading.Event(),
            (0, 0, 708, 454),
            (150, 200),
        )
        self.assertTrue(handled)
        self.assertEqual(len(controller.clicks), 3)
        self.assertEqual(controller.clicks[0], (150, 200))
        self.assertAlmostEqual(controller.clicks[1][0], 352, delta=6)
        self.assertAlmostEqual(controller.clicks[1][1], 348, delta=6)
        self.assertAlmostEqual(controller.clicks[2][0], 353, delta=6)
        self.assertAlmostEqual(controller.clicks[2][1], 282, delta=6)
        self.assertIn("宝箱奖励", automation.recovery_note or "")

    def test_chest_flow_fails_without_dialog(self) -> None:
        normal_image = self._image("initial.png")

        class FakeObserver:
            def raw_frame(self) -> CaptureFrame:
                return ChestFlowTests._frame(normal_image)

        class FakeController:
            def __init__(self) -> None:
                self.clicks: list[tuple[int, int]] = []

            def click_batch(self, _rect, points, _stop_event) -> BatchExecution:
                self.clicks.extend(points.safe)
                return BatchExecution(flagged=0, opened=len(points.safe))

        controller = FakeController()
        automation = LocalAutomation(observer=FakeObserver(), controller=controller)
        automation.CHEST_DIALOG_TIMEOUT = 0.2
        automation.CHEST_POLL_INTERVAL = 0.05
        self.assertFalse(
            automation._open_closed_chest(threading.Event(), (0, 0, 721, 460), (100, 200))
        )
        self.assertEqual(automation.chest_failures, 0)
        self.assertEqual(len(controller.clicks), 1)

    def test_perform_cycle_opens_chest_before_batch(self) -> None:
        closed = observe_image(self._image("closed-chest-raw.png"))
        other = observe_image(self._image("initial.png"))
        welfare_image = self._image("welfare-chest-dialog.png")
        reward_image = self._image("chest-reward-dialog.png")

        class FakeObserver:
            def __init__(self) -> None:
                self.stable = iter((closed, other))
                self.raw_frames = iter(
                    [
                        ChestFlowTests._frame(welfare_image),
                        ChestFlowTests._frame(welfare_image),
                        ChestFlowTests._frame(reward_image),
                        ChestFlowTests._frame(reward_image),
                    ]
                )

            def __call__(self):
                return closed

            def raw_frame(self) -> CaptureFrame:
                return next(self.raw_frames)

            def wait_for_stable(self, *args: object, **kwargs: object):
                return next(self.stable)

        class FakeController:
            def __init__(self) -> None:
                self.clicks: list[tuple[int, int]] = []
                self.drags = 0

            def activate_target(self):
                return 1, closed.frame.window_rect

            def click_batch(self, _rect, points, _stop_event) -> BatchExecution:
                self.clicks.extend(points.safe)
                return BatchExecution(flagged=0, opened=len(points.safe))

            def drag(self, *args: object, **kwargs: object) -> None:
                self.drags += 1

        controller = FakeController()
        automation = LocalAutomation(observer=FakeObserver(), controller=controller)
        outcome = automation.perform_cycle(threading.Event())
        self.assertEqual(outcome.action.kind, ActionKind.BATCH)
        self.assertGreaterEqual(len(controller.clicks), 3)
        self.assertEqual(
            controller.clicks[0],
            closed.geometry.center(3, 2),
        )
        self.assertAlmostEqual(controller.clicks[1][0], 352, delta=6)
        self.assertAlmostEqual(controller.clicks[1][1], 348, delta=6)
        self.assertAlmostEqual(controller.clicks[2][0], 353, delta=6)
        self.assertAlmostEqual(controller.clicks[2][1], 282, delta=6)
        self.assertIn("已自动开启宝箱", outcome.message)
        self.assertEqual(automation.chest_failures, 0)

    def test_chest_opening_stops_after_repeated_failures(self) -> None:
        closed = observe_image(self._image("closed-chest-raw.png"))
        other = observe_image(self._image("initial.png"))

        class FakeObserver:
            def __call__(self):
                return closed

            def raw_frame(self) -> CaptureFrame:
                raise AssertionError("宝箱开启已被禁用，不应再截取对话框")

            def wait_for_stable(self, *args: object, **kwargs: object):
                return other

        class FakeController:
            def __init__(self) -> None:
                self.clicks: list[tuple[int, int]] = []
                self.drags = 0

            def activate_target(self):
                return 1, closed.frame.window_rect

            def click_batch(self, _rect, points, _stop_event) -> BatchExecution:
                self.clicks.extend(points.safe)
                return BatchExecution(flagged=0, opened=len(points.safe))

            def drag(self, *args: object, **kwargs: object) -> None:
                self.drags += 1

        controller = FakeController()
        automation = LocalAutomation(observer=FakeObserver(), controller=FakeController())
        automation.chest_failures = automation.MAX_CHEST_FAILURES
        outcome = automation.perform_cycle(threading.Event())
        self.assertNotIn("宝箱", outcome.message)
        self.assertEqual(automation.chest_failures, automation.MAX_CHEST_FAILURES)
        # The cycle still works normally on the same observation.
        self.assertEqual(outcome.action.kind, ActionKind.BATCH)


class _FixedRandom:
    """Deterministic stand-in for random.Random in escape tests."""

    def __init__(self, value: str) -> None:
        self.value = value

    def choice(self, seq):
        return self.value


class EscapeTests(unittest.TestCase):
    @staticmethod
    def _drag_view():
        source = observe_image(Image.open(ROOT / "samples" / "initial.png"))
        return replace(source, result=SolveResult(frozenset(), frozenset(), tuple(), tuple()))

    @staticmethod
    def _drag_action(direction: str) -> PlannedAction:
        return PlannedAction(
            ActionKind.DRAG,
            direction=direction,
            drag_start=(0, 0),
            drag_end=(10, 10),
        )

    def test_stuck_pattern_detection(self) -> None:
        automation = LocalAutomation(observer=None, controller=None)
        self.assertIsNone(automation._stuck_pattern())
        # Fewer than eight alternating moves never triggers.
        automation.recent_drag_directions.extend(["up", "down", "up", "down"])
        self.assertIsNone(automation._stuck_pattern())
        automation.recent_drag_directions.extend(["up", "down", "up", "down"])
        self.assertEqual(automation._stuck_pattern(), "oscillation")
        automation.recent_drag_directions.clear()
        automation.recent_drag_directions.extend(["up", "left", "down", "right"])
        self.assertEqual(automation._stuck_pattern(), "rotation")
        automation.recent_drag_directions.clear()
        automation.recent_drag_directions.extend(["down", "left", "up", "right"])
        self.assertEqual(automation._stuck_pattern(), "rotation")
        automation.recent_drag_directions.clear()
        automation.recent_drag_directions.extend(["up", "right", "down", "left"])
        self.assertEqual(automation._stuck_pattern(), "rotation")
        automation.recent_drag_directions.clear()
        automation.recent_drag_directions.extend(["up", "up", "down", "down"])
        self.assertIsNone(automation._stuck_pattern())
        automation.recent_drag_directions.clear()
        automation.recent_drag_directions.extend(["up", "down", "up", "left"])
        self.assertIsNone(automation._stuck_pattern())

    def test_oscillation_triggers_escape_burst_of_eight(self) -> None:
        view = self._drag_view()
        automation = LocalAutomation(observer=None, controller=None)
        automation.rng = _FixedRandom("left")
        automation.recent_drag_directions.extend(["up", "down", "up", "down", "up", "down", "up"])
        action, note = automation._apply_escape_or_record(view, self._drag_action("down"))
        self.assertEqual(action.direction, "left")
        self.assertEqual(automation.escape_direction, "left")
        self.assertEqual(automation.escape_moves_done, 1)
        self.assertIn("方向往复振荡", note)
        self.assertIn("1/8", note)
        for _ in range(7):
            action, note = automation._apply_escape_or_record(view, self._drag_action("left"))
            self.assertEqual(action.direction, "left")
        self.assertIn("脱困移动完成", note)
        self.assertIsNone(automation.escape_direction)
        self.assertEqual(automation.escape_moves_done, 0)
        self.assertEqual(len(automation.recent_drag_directions), 0)

    def test_rotation_triggers_escape_burst(self) -> None:
        view = self._drag_view()
        automation = LocalAutomation(observer=None, controller=None)
        automation.rng = _FixedRandom("right")
        automation.recent_drag_directions.extend(["down", "left", "up"])
        action, note = automation._apply_escape_or_record(view, self._drag_action("right"))
        self.assertEqual(automation._stuck_pattern(), "rotation")
        self.assertEqual(action.direction, "right")
        self.assertEqual(automation.escape_direction, "right")
        self.assertIn("方向绕圈旋转", note)
        self.assertIn("1/8", note)

    def test_perform_cycle_runs_eight_forced_escape_drags(self) -> None:
        source = observe_image(Image.open(ROOT / "samples" / "initial.png"))
        empty_grid = [
            [replace(cell, kind=CellKind.OPEN, number=0, confidence=1.0) for cell in row]
            for row in source.grid
        ]
        empty = replace(
            source,
            grid=empty_grid,
            result=SolveResult(frozenset(), frozenset(), tuple(), tuple()),
        )

        class FakeObserver:
            def __call__(self):
                return empty

            def wait_for_stable(self, *args: object, **kwargs: object):
                return empty

        class FakeController:
            def activate_target(self):
                return 1, empty.frame.window_rect

            def drag(self, *args: object, **kwargs: object) -> None:
                return None

        automation = LocalAutomation(observer=FakeObserver(), controller=FakeController())
        automation.escape_direction = "left"
        outcomes = [automation.perform_cycle(threading.Event()) for _ in range(8)]
        self.assertTrue(all(outcome.action.kind == ActionKind.DRAG for outcome in outcomes))
        self.assertTrue(all(outcome.action.direction == "left" for outcome in outcomes))
        self.assertIn("脱困移动完成", outcomes[7].message)
        self.assertIsNone(automation.escape_direction)
        ninth = automation.perform_cycle(threading.Event())
        self.assertNotIn("脱困", ninth.message)

    def test_batch_breaks_escape_and_direction_history(self) -> None:
        batch_view = observe_image(Image.open(ROOT / "samples" / "initial.png"))
        other = observe_image(Image.open(ROOT / "samples" / "chest-raw.png"))

        class FakeObserver:
            def __call__(self):
                return batch_view

            def wait_for_stable(self, *args: object, **kwargs: object):
                return other

        class FakeController:
            def activate_target(self):
                return 1, batch_view.frame.window_rect

            def click_batch(self, _rect, points, _stop_event) -> BatchExecution:
                return BatchExecution(flagged=len(points.marks), opened=len(points.safe))

            def drag(self, *args: object, **kwargs: object) -> None:
                return None

        automation = LocalAutomation(observer=FakeObserver(), controller=FakeController())
        automation.escape_direction = "up"
        automation.escape_moves_done = 2
        automation.recent_drag_directions.extend(["up", "down", "up", "down"])
        outcome = automation.perform_cycle(threading.Event())
        self.assertEqual(outcome.action.kind, ActionKind.BATCH)
        self.assertIsNone(automation.escape_direction)
        self.assertEqual(automation.escape_moves_done, 0)
        self.assertEqual(len(automation.recent_drag_directions), 0)


if __name__ == "__main__":
    unittest.main()
