"""贪吃蛇小游戏（终端版）。

运行方式：
    python tests/snake.py

操作：
    方向键 或 WASD 控制方向，Q 退出，R 重开。
    撞墙、撞到自己都会结束游戏。

依赖：标准库 curses（Windows 上需要 pip install windows-curses）。
"""

import curses
import random
from collections import deque

# 初始蛇身长度
INITIAL_LENGTH = 3
# 每吃到一个食物加多少分
SCORE_PER_FOOD = 10

# 四个方向：(行增量, 列增量)
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)

# 按键到方向的映射，方向键和 WASD 都支持
KEY_TO_DIRECTION = {
    curses.KEY_UP: UP,
    curses.KEY_DOWN: DOWN,
    curses.KEY_LEFT: LEFT,
    curses.KEY_RIGHT: RIGHT,
    ord("w"): UP,
    ord("s"): DOWN,
    ord("a"): LEFT,
    ord("d"): RIGHT,
}


class SnakeGame:
    """贪吃蛇的游戏逻辑，不涉及任何界面绘制。

    坐标统一用 (行, 列)，原点在左上角。
    """

    def __init__(self, height, width, rng=None):
        """height/width 是可用游戏区域的行数和列数，至少为 5。"""
        if height < 5 or width < 5:
            raise ValueError("游戏区域至少需要 5x5")

        self.height = height
        self.width = width
        self.rng = rng or random.Random()

        # 蛇身用 deque 存，索引 0 是蛇头
        self.snake = deque()
        self.direction = RIGHT
        self.score = 0
        self.game_over = False
        self.food = None
        self.reset()

    # ------------------------------------------------------------ 初始化

    def reset(self):
        """把游戏恢复到初始状态。"""
        self.snake.clear()
        # 蛇从中间偏左的位置开始，朝右，这样有足够的空间生长
        row = self.height // 2
        col = max(INITIAL_LENGTH - 1, self.width // 4)
        for i in range(INITIAL_LENGTH):
            self.snake.append((row, col - i))

        self.direction = RIGHT
        self.score = 0
        self.game_over = False
        self.food = None
        self._place_food()

    def _place_food(self):
        """在空格子里随机放一个食物；没有空格子时返回 None（通关）。"""
        occupied = set(self.snake)
        empty = [
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if (r, c) not in occupied
        ]
        self.food = self.rng.choice(empty) if empty else None

    # ------------------------------------------------------------ 游戏逻辑

    def turn(self, direction):
        """改变前进方向。不允许 180 度掉头（会立刻撞到自己）。"""
        if direction is None or self.game_over:
            return
        dr, dc = direction
        cur_dr, cur_dc = self.direction
        # 新方向与当前方向相反时忽略
        if (dr, dc) == (-cur_dr, -cur_dc):
            return
        self.direction = direction

    def step(self):
        """推进一帧。返回 True 表示这一步吃到了食物。"""
        if self.game_over:
            return False

        head_r, head_c = self.snake[0]
        dr, dc = self.direction
        new_head = (head_r + dr, head_c + dc)

        # 撞墙
        if not (0 <= new_head[0] < self.height and 0 <= new_head[1] < self.width):
            self.game_over = True
            return False

        ate = new_head == self.food
        # 撞到自己：不吃食物时尾巴会移走，所以尾巴那一格不算撞
        body = set(self.snake)
        if not ate:
            body.discard(self.snake[-1])
        if new_head in body:
            self.game_over = True
            return False

        self.snake.appendleft(new_head)
        if ate:
            self.score += SCORE_PER_FOOD
            self._place_food()
        else:
            self.snake.pop()
        return ate


# ================================================================ 界面绘制


def _draw(stdscr, game):
    """把当前游戏状态画到屏幕上。"""
    stdscr.erase()
    height, width = stdscr.getmaxyx()

    # 顶部状态栏
    status = f" 得分: {game.score}   长度: {len(game.snake)}   Q 退出 / R 重开"
    stdscr.addnstr(0, 0, status, width - 1, curses.A_BOLD)

    # 游戏区域从第 2 行开始，四周留 1 格边框
    top, left = 2, 1
    bottom = min(top + game.height, height - 1)
    right = min(left + game.width, width - 1)

    # 画边框
    for c in range(left - 1, right + 1):
        if c < width:
            stdscr.addch(top - 1, c, "-")
            if bottom < height - 1:
                stdscr.addch(bottom, c, "-")
    for r in range(top - 1, bottom + 1):
        if r < height - 1:
            stdscr.addch(r, left - 1, "|")
            if right < width:
                stdscr.addch(r, right, "|")

    # 画食物
    if game.food is not None:
        fr, fc = game.food
        if top + fr < height - 1 and left + fc < width:
            stdscr.addch(top + fr, left + fc, "*", curses.A_BOLD)

    # 画蛇，蛇头用 @ 区分
    for i, (r, c) in enumerate(game.snake):
        if top + r >= height - 1 or left + c >= width:
            continue
        stdscr.addch(top + r, left + c, "@" if i == 0 else "o")

    if game.game_over:
        msg = f" 游戏结束！得分 {game.score}，按 R 重开，Q 退出 "
        stdscr.addnstr(height // 2, max(0, (width - len(msg)) // 2), msg, width - 1,
                       curses.A_REVERSE | curses.A_BOLD)

    stdscr.refresh()


def main(stdscr):
    """curses 主循环。"""
    curses.curs_set(0)  # 隐藏光标
    stdscr.nodelay(True)  # getch 不阻塞
    stdscr.keypad(True)  # 让方向键返回 KEY_UP 这类常量

    height, width = stdscr.getmaxyx()
    # 留出状态栏、边框和底部一行
    game = SnakeGame(height=max(5, height - 4), width=max(5, width - 3))

    tick = 0
    speed = 8  # 每 8 帧走一步，数字越大越慢
    while True:
        key = stdscr.getch()
        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("r"), ord("R")):
            game.reset()
        elif key in KEY_TO_DIRECTION:
            game.turn(KEY_TO_DIRECTION[key])

        tick += 1
        if tick >= speed:
            tick = 0
            game.step()

        _draw(stdscr, game)
        curses.napms(30)  # 约 33 FPS


if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
