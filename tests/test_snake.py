"""tests/snake.py 的游戏逻辑测试。

只测 SnakeGame 的纯逻辑，不启动 curses 界面。
"""

import random
import sys
from pathlib import Path

import pytest

# tests/ 不是包，把它的父目录加进 sys.path 才能 import snake
sys.path.insert(0, str(Path(__file__).resolve().parent))

from snake import DOWN, LEFT, RIGHT, UP, SnakeGame  # noqa: E402


def make_game(height=10, width=10, seed=0):
    """造一个固定随机种子的游戏，方便断言食物位置。"""
    return SnakeGame(height, width, rng=random.Random(seed))


# ================================================================ 初始化


def test_initial_state():
    game = make_game()
    assert len(game.snake) == 3
    assert game.score == 0
    assert game.game_over is False
    assert game.direction == RIGHT
    # 蛇头在最右边，蛇身向左延伸
    head_r, head_c = game.snake[0]
    assert list(game.snake) == [(head_r, head_c), (head_r, head_c - 1), (head_r, head_c - 2)]


def test_food_never_on_snake():
    for seed in range(20):
        game = make_game(seed=seed)
        assert game.food not in set(game.snake)


def test_area_too_small():
    with pytest.raises(ValueError):
        SnakeGame(4, 10)
    with pytest.raises(ValueError):
        SnakeGame(10, 4)


# ================================================================ 移动


def test_step_moves_forward():
    game = make_game()
    head = game.snake[0]
    game.step()
    assert game.snake[0] == (head[0], head[1] + 1)
    # 没吃到食物时长度不变
    assert len(game.snake) == 3


def test_turn_changes_direction():
    game = make_game()
    game.turn(UP)
    assert game.direction == UP
    head = game.snake[0]
    game.step()
    assert game.snake[0] == (head[0] - 1, head[1])


def test_cannot_reverse_into_itself():
    """朝右走时按左键应被忽略，否则会立刻撞到自己。"""
    game = make_game()
    game.turn(LEFT)
    assert game.direction == RIGHT


def test_reverse_allowed_after_turn():
    """先转向上，再按向下才算掉头，同样要被忽略。"""
    game = make_game()
    game.turn(UP)
    game.turn(DOWN)
    assert game.direction == UP


# ================================================================ 吃食物与生长


def test_eating_grows_and_scores():
    game = make_game()
    head_r, head_c = game.snake[0]
    # 把食物摆在蛇头正前方
    game.food = (head_r, head_c + 1)
    assert game.step() is True
    assert len(game.snake) == 4
    assert game.score == 10
    # 吃完后立刻补一个新食物，且不在蛇身上
    assert game.food is not None
    assert game.food not in set(game.snake)


def test_not_eating_keeps_length():
    game = make_game()
    head_r, head_c = game.snake[0]
    game.food = (0, 0)  # 放到远处，确保这一步吃不到
    assert game.step() is False
    assert len(game.snake) == 3


def test_tail_cell_is_free_to_enter():
    """尾巴这一步会移走，所以蛇头可以走进尾巴原来的格子。"""
    game = make_game()
    head_r, head_c = game.snake[0]
    # 造一个 2x2 的方块蛇：头在 (5,5)，身体绕一圈回到 (5,4)
    game.snake.clear()
    game.snake.extend([(5, 5), (6, 5), (6, 4), (5, 4)])
    game.direction = LEFT
    game.food = (0, 0)
    assert game.step() is False
    assert game.game_over is False
    assert game.snake[0] == (5, 4)


# ================================================================ 结束条件


def test_hit_wall():
    game = make_game(height=5, width=5)
    # 一路向右撞墙
    for _ in range(10):
        game.step()
        if game.game_over:
            break
    assert game.game_over is True


def test_hit_self():
    game = make_game()
    # 摆一个 U 形，向右走一步就会撞到自己的身体
    game.snake.clear()
    game.snake.extend([(5, 5), (5, 6), (4, 6), (4, 5), (4, 4)])
    game.direction = RIGHT
    game.food = (0, 0)
    game.step()
    assert game.game_over is True


def test_step_after_game_over_is_noop():
    game = make_game(height=5, width=5)
    game.game_over = True
    snapshot = list(game.snake)
    assert game.step() is False
    assert list(game.snake) == snapshot


def test_turn_ignored_after_game_over():
    game = make_game()
    game.game_over = True
    game.turn(UP)
    assert game.direction == RIGHT


# ================================================================ 重开


def test_reset_restores_initial_state():
    game = make_game()
    game.food = (game.snake[0][0], game.snake[0][1] + 1)
    game.step()
    game.game_over = True
    game.reset()
    assert len(game.snake) == 3
    assert game.score == 0
    assert game.game_over is False
    assert game.direction == RIGHT
    assert game.food not in set(game.snake)


def test_win_when_board_is_full():
    """棋盘被填满时没有空格放食物，food 为 None 且不报错。"""
    game = make_game(height=5, width=5)
    game.snake.clear()
    # 除 (0, 0) 外全部占满
    for r in range(5):
        for c in range(5):
            if (r, c) != (0, 0):
                game.snake.append((r, c))
    game.food = (0, 0)
    # 把 (0, 0) 也占上，棋盘全满，补食物时应该找不到位置
    game.snake.append((0, 0))
    game._place_food()
    assert game.food is None
