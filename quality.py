# RPS.py
BEATS = {"R": "P", "P": "S", "S": "R"}  # BEATS[x] is the move that beats x


class _State:
    def __init__(self):
        self.reset()

    def reset(self):
        self.my_hist = []
        self.opp_hist = []
        self.scores = {"quincy": 0.0, "kris": 0.0, "mrugesh": 0.0, "abbey": 0.0}
        self.last_preds = {}
        # abbey model state
        self.abbey_hist = []
        self.abbey_counts = {a + b: 0 for a in "RPS" for b in "RPS"}


S = _State()


def predict_quincy(my_hist):
    # quincy cycles R, R, P, P, S (its counter starts at 1)
    return ["R", "R", "P", "P", "S"][(len(my_hist) + 1) % 5]


def predict_kris(my_hist):
    # kris beats my previous move (assumes "R" on the first game)
    return BEATS[my_hist[-1]] if my_hist else "P"


def predict_mrugesh(my_hist):
    # mrugesh beats my most frequent move in the last 10 games
    last_ten = ([""] + my_hist)[-10:]
    most_frequent = max(set(last_ten), key=last_ten.count)
    if most_frequent == "":
        most_frequent = "S"
    return BEATS[most_frequent]


def predict_abbey(my_hist):
    # abbey keeps pair counts of my moves and beats my predicted next move
    prev = my_hist[-1] if my_hist else "R"
    S.abbey_hist.append(prev)
    last_two = "".join(S.abbey_hist[-2:])
    if len(last_two) == 2:
        S.abbey_counts[last_two] += 1
    options = {prev + x: S.abbey_counts[prev + x] for x in "RPS"}
    prediction = max(options, key=options.get)[-1]
    return BEATS[prediction]


def player(prev_play, opponent_history=[]):
    # A new match starts with an empty string: reset all state
    if prev_play == "":
        S.reset()
    else:
        S.opp_hist.append(prev_play)
        # Score each model on whether it predicted the opponent's last move
        for name, pred in S.last_preds.items():
            S.scores[name] = S.scores[name] * 0.9 + (1.0 if pred == prev_play else 0.0)

    # Each model's prediction of the opponent's next move
    S.last_preds = {
        "quincy": predict_quincy(S.my_hist),
        "kris": predict_kris(S.my_hist),
        "mrugesh": predict_mrugesh(S.my_hist),
        "abbey": predict_abbey(S.my_hist),
    }

    best = max(S.scores, key=S.scores.get)
    my_move = BEATS[S.last_preds[best]]
    S.my_hist.append(my_move)
    return my_move
