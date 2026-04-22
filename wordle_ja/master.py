import random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import logging
import numpy as np
import re

from clemcore.backends import Model
from clemcore.clemgame import GameSpec, GameMaster, GameBenchmark, Player, ResponseError, ParseError, RuleViolationError
from clemcore.clemgame.legacy.scorer import GameScorer
from clemcore.clemgame.legacy.master import DialogueGameMaster
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, METRIC_REQUEST_COUNT, \
    METRIC_REQUEST_COUNT_VIOLATED, METRIC_REQUEST_COUNT_PARSED, BENCH_SCORE

from utils.guessvalidator import GuessValidator
from utils.compute_metrics import turns_closeness, turns_strategy

logger = logging.getLogger(__name__)


class UnknownFiveLetterWordError(RuleViolationError):
    """Raised when the word is 5-letters but not part of the game's vocabulary"""
    pass


class WordLengthError(RuleViolationError):
    """Raised when the word is 5-letters but not part of the game's vocabulary"""
    pass


class WordFormatError(RuleViolationError):
    """Raised when the word is 5-letters but not part of the game's vocabulary"""
    pass


class ProtocolError(ResponseError):
    """Raised when a message does not follow the communication protocol expected by the game master."""
    pass


class ParseError(ProtocolError):
    """
    This error is supposed to be raised when player messages cannot be parsed or understood by the game master e.g.
    because the response does not start with a specified prefix.
    For example:
        - taboo: clue giver messages should start with 'CLUE:'
        - wordle: guesser messages should start with 'GUESS:'
    """
    pass

class ResponseFormatter:

    def __init__(self, words):
        self.words = words

    # noinspection PyMethodMayBeStatic
    def to_gm_turn_stats(self, stats: Dict):
        return '\n'.join(f'{key} = {value}' for key, value in stats.items())

    def to_gm_reprompt_for_guesser(self, error: ResponseError):
        return (f"{self.words['error_prompt_text'][error.key]} "  # only white space separated
                f"{self.words['error_prompt_text']['RETRY']}\n\n"  # Please try again.
                f"{self.words['error_prompt_text']['INVALID_FORMAT']}\n"  # Provide your response only in this format.
                f"{self.words['explanation_lang']} {self.words['explanataion_details_lang']}\n"
                f"{self.words['guess_lang']} {self.words['guess_word_lang']}"
                )

    def to_gm_response_for_guesser(self, feedback: str):
        return (f"{self.words['guess_feedback_lang']} {feedback}\n\n"
                f"{self.words['error_prompt_text']['INVALID_FORMAT']}\n"  # Provide your response only in this format.
                f"{self.words['explanation_lang']} {self.words['explanataion_details_lang']}\n"
                f"{self.words['guess_lang']} {self.words['guess_word_lang']}"
                )


class WordGuesser(Player):
    def __init__(self, model: Model, words: Dict, target_word: str):
        super().__init__(model)
        self.target_word = target_word
        self.words = words
        self._custom_responses = ["apple", "beach", "crane",
                                  "pathy",  # throw in an invalid word
                                  "after", "those", "horse"]

    def to_guesser_response(self, explanation: str, guess: str):
        """ Only for custom response behavior (mock); documents the expected response format """
        return (f"{self.words['explanation_lang']} {explanation}\n"
                f"{self.words['guess_lang']} {guess}")

    def _terminal_response(self, context: Dict) -> str:
        guess = input("Enter your guess: ")
        return self.to_guesser_response("human guesser", guess)

    def _custom_response(self, messages):  # for playing with_critic we need doulbe the amoutn of responses
        guess = self._custom_responses.pop(0)
        if random.randint(0, 100) < 10:  # let the player occasionally win
            guess = self.target_word
        if random.randint(0, 100) > 90:  # let the player occasionally abort (not 5-letter word)
            guess = "scrumbled eggs"
        return self.to_guesser_response("custom guesser", guess)


def parse_response(player: Player, response: str, words: Dict) -> Tuple[str, str]:
    """Parse guesser response and extract guess and explanation"""

    response = response.replace("<|im_end|>", "").replace("*", "").replace("推測:  \n", "推測: ").replace("推測: \n", "推測: ").replace("推測:  ", "推測: ")

    if not response or not response.startswith(words["explanation_lang"]):
        # raise ParseError(f"The response should always start with the keyword '{words['explanation_lang']}'",

        # Let's try to see if there is really no way to do this right...

        if "assistantfinal" in response:
            resp_parts = response.split("assistantfinal")
            response = resp_parts[1]
        if words["explanation_lang"] in response:
            resp_parts = response.split(words["explanation_lang"])
            response = words["explanation_lang"] + resp_parts[1]
        else:
            raise ParseError(f"答えは常にキーワードで始まる必要があります '{words['explanation_lang']}'",
                         key="INVALID_START_WORD")

    response = response.strip()
    # lines = response.split("\n")
    # if len(lines) > 2:
    #     raise ParseError(f"The response should contain only the '{words['guess_lang']}' and "
    #                      f"'{words['explanation_lang']}' keywords and associated information.",
    #                      key="UNKNOWN_TAGS")

    # Extract explanation and guess
    explanation_pattern = re.compile(rf"{words['explanation_lang']}([^\n]*)", re.IGNORECASE)

    content_prefix = words['guess_lang']
    content_pattern = re.compile(rf"{content_prefix}([^\n]*)", re.IGNORECASE)

    explanation_match = explanation_pattern.search(response)
    content_match = content_pattern.findall(response)

    if len(content_match) != 1:
        raise ParseError(f"応答には '{content_prefix}' キーワードが 1 回だけ含まれている必要があります。",
                         key="MORE_THAN_ONE_GUESS")

    content = content_match[0].strip().lower()
    explanation = explanation_match.group(1).strip() if explanation_match else ""

    return content, explanation


def validate_guess(guess: str, words: Dict):
    """Validate guess format and content"""
    import jaconv
    guess = jaconv.kata2hira(guess)

    if not guess.isalpha() or " " in guess:
        raise WordFormatError("推測する単語は 1 つで、カタカナのみで構成されている必要があります。",
                                 key="INVALID_FORMAT")

    if len(guess) != words["max_word_length"]:
        raise WordLengthError(f"推測の長さは{words['max_word_length']}ではありません.",
                                 key="INVALID_WORD_LENGTH")

    if guess not in words["official_words_list"]:
        raise UnknownFiveLetterWordError(f"このゲームでは「推測」は有効な単語ではありません。",
                                         key="NOT_VALID_WORD_FOR_GAME")


@dataclass
class WordleGameState:
    # Wordle
    target_word: str
    words: Dict[str, str]
    max_rounds: int
    max_retry_per_error: int
    guesser_initial_prompt: str
    success: bool = False
    failure: bool = False
    aborted: bool = False
    valid_response: bool = False
    reprompt_attempts: int = 0
    error: Optional[ResponseError] = None
    current_guess: str = None
    current_explanation: str = None
    guess_feedback: str = None


# interaction keys to log structured data for scoring or logging
GUESSER_GUESSES = "Guesser Guesses"
GUESSER_EXPLANATIONS = "Guesser Explanations"
GUESSER_FEEDBACKS = "Guesser Feedbacks"


class Wordle(DialogueGameMaster):
    """Basic Wordle game without clue or critic"""

    def __init__(self, game_spec: GameSpec, experiment: Dict, player_models: List[Model]):
        super().__init__(game_spec, experiment, player_models)
        # game specific logging
        self.request_counts: int = 0
        self.parsed_request_counts: int = 0
        self.violated_request_counts: int = 0
        self.guesser_guesses: List[str] = []
        self.guesser_explanations: List[str] = []
        self.guesser_feedbacks: List[str] = []

    def _on_setup(self, **game_instance):
        self.state = WordleGameState(
            target_word=game_instance["target_word"].strip().lower(),
            words=self.experiment["lang_keywords"],
            max_rounds=self.experiment["common_config"]["n_turns"],
            # NOT_VALID_WORD_FOR_GAME is the only entry in the dict; we only handle this case in the game for now
            max_retry_per_error=self.experiment["common_config"]["max_retry_per_error"],
            guesser_initial_prompt=self.experiment["guesser_prompt"]
        )
        self.guess_validator = GuessValidator(self.state.target_word)
        self.formatter = ResponseFormatter(self.state.words)
        self._add_players()

    def _add_players(self):
        self.guesser = WordGuesser(self.player_models[0], self.state.words, self.state.target_word)
        self.add_player(self.guesser, initial_context=self.state.guesser_initial_prompt)

    def _does_game_proceed(self):
        return not (self.state.success or self.state.failure or self.state.aborted)

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        self.request_counts += 1
        try:
            # Parse response of the only player: the guesser
            guess, explanation = parse_response(player, utterance, self.state.words)
            self.state.current_guess = guess
            self.state.current_explanation = explanation
            # Validate guess
            validate_guess(guess, self.state.words)
            self.parsed_request_counts += 1
            # Reset re-prompting states
            self.state.valid_response = True
            self.state.reprompt_attempts = 0
            self.state.error = None
            return True
        except (ParseError, RuleViolationError) as e:
            if isinstance(e, UnknownFiveLetterWordError) or isinstance(e, WordLengthError) or isinstance(e, WordFormatError):
                self.parsed_request_counts += 1  # in this case still count toward parsed requests, but re-prompt
            else:
                self.violated_request_counts += 1
            self.state.valid_response = False
            self.state.error = e
            self.log_to_self("metadata", f"エラー: {e.reason}")
            return False

    def _should_pass_turn(self):
        if not self.state.valid_response:
            if isinstance(self.state.error, UnknownFiveLetterWordError):
                # perform re-prompting up to N times
                self.state.reprompt_attempts += 1
                if self.state.reprompt_attempts > self.state.max_retry_per_error["NOT_VALID_WORD_FOR_GAME"]:
                    self.log_to_self("invalid format", "ゲーム_結果 = 放棄")
                    self.state.aborted = True
                else:  # adjust re-prompt text
                    self.set_context_for(self.guesser, self.formatter.to_gm_reprompt_for_guesser(self.state.error))

            elif isinstance(self.state.error, WordLengthError):
                # perform re-prompting up to N times
                self.state.reprompt_attempts += 1
                if self.state.reprompt_attempts > self.state.max_retry_per_error["INVALID_WORD_LENGTH"]:
                    self.log_to_self("invalid format", "ゲーム_結果 = 放棄")
                    self.state.aborted = True
                else:  # adjust re-prompt text
                    self.set_context_for(self.guesser, self.formatter.to_gm_reprompt_for_guesser(self.state.error))

            elif isinstance(self.state.error, WordFormatError):
                # perform re-prompting up to N times
                self.state.reprompt_attempts += 1
                if self.state.reprompt_attempts > self.state.max_retry_per_error["INVALID_FORMAT"]:
                    self.log_to_self("invalid format", "ゲーム_結果 = 放棄")
                    self.state.aborted = True
                else:  # adjust re-prompt text
                    self.set_context_for(self.guesser, self.formatter.to_gm_reprompt_for_guesser(self.state.error))
            else:
                self.log_to_self("invalid format", "ゲーム_結果 = 放棄")
                self.state.aborted = True
            return False
        return True

    def _start_next_round(self) -> bool:
        return self.state.valid_response

    def _on_valid_player_response(self, player: Player, parsed_response: str):
        self.state.guess_feedback = self.guess_validator.validate(self.state.current_guess)
        self.log_to_self("metadata", self.formatter.to_gm_turn_stats(self.get_turn_stats()))
        self.guesser_feedbacks.append(self.state.guess_feedback)
        self.guesser_guesses.append(self.state.current_guess)
        self.guesser_explanations.append(self.state.current_explanation)
        # Check terminal conditions
        if self.state.target_word == self.state.current_guess:
            self.log_to_self("correct guess", "ゲーム_結果 = 勝利")
            self.state.success = True
        elif self.current_round + 1 >= self.state.max_rounds:  # zero-based rounds
            self.log_to_self("max rounds played", "ゲーム_結果 = 損失")
            self.state.failure = True
        else:  # Provide word validation feedback to guesser for next round
            content = self.formatter.to_gm_response_for_guesser(self.state.guess_feedback)
            self.set_context_for(self.guesser, content)

    def get_turn_stats(self):
        return {
            "attempts": self.current_round + 1,
            "target_word": self.state.target_word,
            "guess": self.state.current_guess,
            "guess_feedback": self.state.guess_feedback
        }

    def compute_response_score(self, response, context):
        return 1 if self.state.success else 0

    def compute_episode_score(self):
        if self.state.success:
            return 100 / self.current_round
        return 0

    def _on_after_game(self):
        self.log_key(METRIC_ABORTED, int(self.state.aborted))
        self.log_key(METRIC_LOSE, int(self.state.failure))
        self.log_key(METRIC_SUCCESS, int(self.state.success))

        self.log_key(METRIC_REQUEST_COUNT, self.request_counts)
        self.log_key(METRIC_REQUEST_COUNT_PARSED, self.parsed_request_counts)
        self.log_key(METRIC_REQUEST_COUNT_VIOLATED, self.violated_request_counts)

        self.log_key(GUESSER_GUESSES, self.guesser_guesses)
        self.log_key(GUESSER_FEEDBACKS, self.guesser_feedbacks)
        self.log_key(GUESSER_EXPLANATIONS, self.guesser_explanations)


SPEED_SCORES = {
    1: 100,
    2: 100,
    3: 100,
    4: 50,
    5: 30,
    6: 20
}
GUESS_REPETITIONS = "Guess Repetitions"
CLOSENESS_SCORE = "Closeness Score"  # turn metric
STRATEGY_SCORE = "Strategy Score"  # turn metric


class WordleScorer(GameScorer):
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def score_turns(self, episode_interactions: Dict) -> None:
        guesser_feedbacks = episode_interactions[GUESSER_FEEDBACKS]

        if not guesser_feedbacks:
            self.log_turn_score(0, CLOSENESS_SCORE, np.nan)
            self.log_turn_score(0, STRATEGY_SCORE, np.nan)
            return

        closeness_scores = turns_closeness(guesser_feedbacks)
        for idx, score in enumerate(closeness_scores):
            self.log_turn_score(idx + 1, CLOSENESS_SCORE, score)

        strategy_scores = turns_strategy(guesser_feedbacks, is_aborted=episode_interactions[METRIC_ABORTED])
        for idx, score in enumerate(strategy_scores):
            self.log_turn_score(idx + 1, STRATEGY_SCORE, score)

    def compute_speed(self, episode_interactions):
        """
        Rank is computed based on the number of turns taken to guess the word.
        The lesser the number of turns, the higher the speed
        """
        num_rounds: int = len(episode_interactions["turns"])
        if self.game_name == "wordle":
            return SPEED_SCORES[num_rounds]
        return round(100 / num_rounds, 2)

    def compute_guess_repetition(self, episode_interactions):
        guesses = episode_interactions[GUESSER_GUESSES]
        return len(guesses) - len(set(guesses))

    def log_main_score(self, episode_interactions: Dict):
        if episode_interactions[METRIC_ABORTED]:
            self.log_episode_score(BENCH_SCORE, np.nan)
            self.log_episode_score(GUESS_REPETITIONS, np.nan)
        elif episode_interactions[METRIC_LOSE]:
            self.log_episode_score(BENCH_SCORE, 0)
            self.log_episode_score(GUESS_REPETITIONS, self.compute_guess_repetition(episode_interactions))
        elif episode_interactions[METRIC_SUCCESS]:
            self.log_episode_score(BENCH_SCORE, self.compute_speed(episode_interactions))
            self.log_episode_score(GUESS_REPETITIONS, self.compute_guess_repetition(episode_interactions))
        else:
            raise RuntimeError("Cannot compute BENCH_SCORE because neither aborted, lose nor success is set.")


class WordleGameBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return Wordle(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return WordleScorer(self.game_name, experiment, game_instance)
