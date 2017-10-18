"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    return eval_function7(game, player)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return eval_function8(game, player)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return 1.


def find_avail_neighbors(game, player):

    loc = game.get_player_location(player)

    # The first move always results in 8 unoccupied neighbors
    if loc == game.NOT_MOVED:
        return 8.
    r, c = loc
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                      (0, 1), (1, -1), (1, 0), (1, 1)]
    
    avail_neighbors = [(r + dr, c + dc) for dr, dc in neighbors
                       if game.move_is_legal((r + dr, c + dc))]
    
    return avail_neighbors

# The number of neighbors of a player
def eval_function1(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    avail_neighbors = find_avail_neighbors(game, player)
    
    return float(len(avail_neighbors))

# The number of neighbors of current player minus the number of neighbors of the opponent
# Better than eval_function1 when using chasing after opponents
def eval_function2(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    own_neighbors = len(find_avail_neighbors(game, player))
    opp_neighbors = len(find_avail_neighbors(game, game.get_opponent(player)))
    
    return float(own_neighbors - opp_neighbors)

# The number of player's legal moves minus the number of opponent's moves
def eval_function3(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    return float(own_moves - opp_moves)

# Distance from the center
def eval_function4(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    center_y, center_x = int(game.height/2), int(game.width/2)
    player_y, player_x = game.get_player_location(player)

    return -float((center_y - player_y)**2 + (center_x - player_x)**2)    


# The number of neighbors of current player minus 3*the number of neighbors of the opponent
def eval_function5(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    own_neighbors = len(find_avail_neighbors(game, player))
    opp_neighbors = len(find_avail_neighbors(game, game.get_opponent(player)))
    
    return float(own_neighbors - 3*opp_neighbors)


# The number of player's legal moves minus 3*the number of opponent's moves
def eval_function6(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    return float(own_moves - 3*opp_moves)


# Use different strategies for beginning and middle game
def eval_function7(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    percent_occupied = float(game.move_count / (game.width * game.height))

    # Beginning: If less than a quarter of the board is occupied, use own_moves - opp_moves
    if percent_occupied < 0.25:
        own_moves = len(game.get_legal_moves(player))
        opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
        return float(own_moves - opp_moves)
    # Middle: If a quarter to three-quarters of the board are occupied, use own_neighbors - opp_neighbors
    else:
        own_neighbors = len(find_avail_neighbors(game, player))
        opp_neighbors = len(find_avail_neighbors(game, game.get_opponent(player)))
        return float(own_neighbors - opp_neighbors)        


# Use different strategies for beginning and middle game
def eval_function8(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    percent_occupied = float(game.move_count / (game.width * game.height))

    # Beginning: If less than a quarter of the board is occupied, use own_moves - opp_moves
    if percent_occupied < 0.25:
        own_neighbors = len(find_avail_neighbors(game, player))
        opp_neighbors = len(find_avail_neighbors(game, game.get_opponent(player)))
        return float(own_neighbors - opp_neighbors)  
    # Middle: If a quarter to three-quarters of the board are occupied, use own_neighbors - opp_neighbors
    else:
        own_moves = len(game.get_legal_moves(player))
        opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
        return float(own_moves - opp_moves)
 

# Evaluation functions to be used for end game
# Find all paths for the current player, assuming their opponent doesn't move
def all_paths(game, player, path=None):
    if path is None: 
        path = [game.get_player_location(player)]
    paths = []
    #print ("current player: ", player)
    #print ("own location: ", game.get_player_location(player), "opp location: ", game.get_player_location(game.get_opponent(player)))
    for m in game.get_legal_moves(player):
        current_path = path + [m]
        paths.append(tuple(current_path))
        paths.extend(all_paths(game.forecast_move(m), game.forecast_move(m)._inactive_player, current_path))
    return paths


def longest_path(all_paths):
    max_path = ()
    for p in all_paths:
        if len(p) > len(max_path):
            max_path = p
    return max_path


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()
        best_score = float("-inf")
        best_move = (-1, -1)
        if not legal_moves:
        	return best_move
        for m in legal_moves:
        	best_score, best_move = max((best_score, best_move), (self.min_value(game.forecast_move(m), depth), m))
        return best_move

    def max_value(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_score = float("-inf")
        legal_moves = game.get_legal_moves()
        if not legal_moves or depth <= 1:
            return self.score(game, self)

        for m in legal_moves:
        	best_score = max(best_score, self.min_value(game.forecast_move(m), depth-1))
        return best_score

    def min_value(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_score = float("inf")
        legal_moves = game.get_legal_moves()
        if not legal_moves or depth <= 1:
            return self.score(game, self)

        for m in legal_moves:
        	best_score = min(best_score, self.max_value(game.forecast_move(m), depth-1))
        return best_score


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            legal_moves = game.get_legal_moves()

            if not legal_moves:
                return best_move

            if len(legal_moves) == 1:
                return legal_moves[0]
            center_y, center_x = int(game.height/2), int(game.width/2)
            # For its first move, player 1 should occupy the center position
            if game.move_count == 0:
                return (center_y, center_x)
            # If player 1 didn't occupy the center position it its first move, player 2 should do so
            elif game.move_count == 1 and (center_y, center_x) in game.get_legal_moves():
                print ("move count = 1", (center_y, center_x))
                return (center_y, center_x)
            
            percent_occupied = float(game.move_count / (game.width * game.height))
            # End game: If more than three-quarters of the board are occupied, use longest_path
            if percent_occupied > 0.6:
                path = longest_path(all_paths(game, game._active_player))
                best_move = path[1]
                return best_move

            depth = 1
            while True:
                best_move = self.alphabeta(game, depth, float("-inf"), float("inf"))
                depth += 1

        except SearchTimeout:
            pass

        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()
        best_score = float("-inf")
        best_move = (-1, -1)
        if not legal_moves:
            return best_move
        for m in legal_moves:
            best_score, best_move = max((best_score, best_move), (self.min_value(game.forecast_move(m), depth, alpha, beta), m))
            if best_score >= beta:
                return m                
            alpha = max(alpha, best_score)

        return best_move

    def max_value(self, game, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_score = float("-inf")
        legal_moves = game.get_legal_moves()
        if not legal_moves or depth <= 1:
            return self.score(game, self)

        for m in legal_moves:
            best_score = max(best_score, self.min_value(game.forecast_move(m), depth-1, alpha, beta))
            if best_score >= beta:
                return best_score
            alpha = max(alpha, best_score)
        return best_score

    def min_value(self, game, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_score = float("inf")
        legal_moves = game.get_legal_moves()
        if not legal_moves or depth <= 1:
            return self.score(game, self)

        for m in legal_moves:
            best_score = min(best_score, self.max_value(game.forecast_move(m), depth-1, alpha, beta))
            if best_score <= alpha:
                return best_score
            beta = min(beta, best_score)            
        return best_score
