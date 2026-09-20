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
    closed_ratio,
    interior_closed_island,
    plan_action,
    plan_drag,
)
from minesweeper.capture import CaptureFrame
from minesweeper.input_control import WindowInputController
from minesweeper.observer import WindowObserver, board_motion_ratio, observe_image
from minesweeper.recognition import CellKind, classify_cell
from minesweeper.solver import SolveResult
from minesweeper.statistics import StatisticsStore


ROOT = Path(__file__).resolve().parents[1]


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
        self.assertEqual(len(observation.result.safe), 7)
        self.assertEqual(len(observation.result.mines), 2)

    def test_current_view_is_planned_as_one_fast_batch(self) -> None:
        observation = observe_image(Image.open(ROOT / "samples" / "initial.png"))
        action = plan_action(observation, LocalNavigator())
        self.assertEqual(action.kind, ActionKind.BATCH)
        self.assertEqual(len(action.safe), 7)
        self.assertEqual(len(action.marks), 2)

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
        self.assertEqual(len(observation.result.safe), 23)
        self.assertEqual(len(observation.result.mines), 10)

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

    def test_empty_view_drags_eight_times_before_completing(self) -> None:
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

        controller = FakeController()
        automation = LocalAutomation(observer=FakeObserver(), controller=controller)
        outcomes = [automation.perform_cycle(threading.Event()) for _ in range(9)]
        self.assertEqual(controller.drags, 8)
        self.assertTrue(all(outcome.action.kind == ActionKind.DRAG for outcome in outcomes[:8]))
        self.assertEqual(outcomes[8].action.kind, ActionKind.COMPLETE)
        self.assertIn("连续 8 次", outcomes[8].message)

    def test_empty_view_counter_resets_when_closed_cells_reappear(self) -> None:
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

    def test_small_center_closed_island_is_not_orbited(self) -> None:
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
        navigator = LocalNavigator()
        actions = [plan_drag(view, navigator) for _ in range(4)]
        self.assertTrue(all(action.navigation_mode == "island_escape" for action in actions))
        self.assertEqual(len({action.direction for action in actions}), 1)

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


if __name__ == "__main__":
    unittest.main()
