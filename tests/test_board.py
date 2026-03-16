from tetris_rl.env.board import Board

def test_board_initializes_empty():
    board = Board(height=20, width=10)

    assert board.grid.shape == (20, 10)
    assert board.grid.sum() == 0

def test_clone_created_independent_copy():
    board = Board(height=4, width=4)
    board.grid[0, 0] = 1

    cloned = board.clone()
    cloned.grid[0, 1] = 1

    assert board.grid[0, 1] == 0
    assert cloned.grid[0, 1] == 1

def test_check_collision_with_left_wall():
    board = Board(height=6, width=6)
    piece = [[1, 1]]

    assert board.check_collision(piece, top=0, left=-1) is True

def test_check_collision_with_right_wall():
    board = Board(height=6, width=6)
    piece = [[1, 1, 1]]

    assert board.check_collision(piece, top=0, left=4) is True

def test_check_collision_with_bottom():
    board = Board(height=6, width=6)
    piece = [[1], [1]]

    assert board.check_collision(piece, top=5, left=0) is True

def test_no_collision_for_valid_placement():
    board = Board(height=6, width=6)
    piece = [[1, 1], [1, 1]]

    assert board.check_collision(piece, top=2, left=2) is False

def test_collision_with_existing_blocks():
    board = Board(height=6, width=6)
    board.grid[3, 3] = 1

    piece = [[1]]
    assert board.check_collision(piece, top=3, left=3) is True

def test_place_piece_updates_grid():
    board = Board(height=6, width=6)
    piece = [[1, 1], [1, 1]]

    board.place_piece(piece, top=2, left=3)

    assert board.grid[2, 3] == 1
    assert board.grid[2, 4] == 1
    assert board.grid[3, 3] == 1
    assert board.grid[3, 4] == 1

def test_clear_lines_removes_full_row():
    board = Board(height=4, width=4)
    board.grid[3] = [1, 1, 1, 1]

    cleared = board.clear_lines()

    assert cleared == 1
    assert board.grid[3].sum() == 0
    assert board.grid.sum() == 0

def test_clear_lines_shifts_rows_down():
    board = Board(height=4, width=4)
    board.grid[2, 0] = 1
    board.grid[3] = [1, 1, 1, 1]

    cleared = board.clear_lines()

    assert cleared == 1
    assert board.grid[3, 0].sum() == 1
    assert board.grid.sum() == 1

def test_is_game_over_false_for_empty_top_row():
    board = Board(height=4, width=4)

    assert board.is_game_over() is False

def test_is_game_over_true_when_top_row_is_filled():
    board = Board(height=4, width=4)
    board.grid[0, 2] = 1

    assert board.is_game_over() is True
