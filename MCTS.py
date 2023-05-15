import logging
import math

import numpy as np

from parse import parse_mcts_actions

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS:
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.board_to_id = {"__ROOT__": 0}  # associates to each board state a unique id
        self.local_id = {
            "__ROOT__": 0
        }  # ids in the current reachable tree, preorder sorted
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s
        self.last_a = self.game.getActionSize()  # the first invalid action token
        self.last_s = "__ROOT__"
        self.children = (
            {}
        )  # for each board s, stores its children in the MCTS tree (with their boards s) as a list
        self.vs = (
            {}
        )  # save evaluations to simulate first encounters in the case of diamond merging
        self.edge_targets = {}  # for each (s, a), its end node s'
        self.traces = []  # collect outputs of simulations
        self._trace_state = ""  # current simulation trace: search state
        self._trace_actions = ""  # current simulation trace: search actions
        self.reset_trace()
        self.search_start = "__ROOT__"  # starting state of the search, for restarting
        self._visited = set()  # for preorder()

    def reset_trace(self):
        self._trace_state = "STATE:\n"
        self._trace_actions = ""

    def submit_trace(self):
        # to make `_trace_state` a prompt for `_trace_actions`, append "ACTIONS:\n" to `_trace_state`
        self.trace("ACTIONS:", to="state")
        self.traces.append((self._trace_state, self._trace_actions))
        self.reset_trace()

    @property
    def trace_empty(self):
        return self._trace_state == "" and self._trace_actions == ""

    def trace(self, s, to="state", end="\n"):
        assert to in ["state", "actions"]
        if to == "state":
            self._trace_state += s + end
        else:
            self._trace_actions += s + end
        if self.args.verbose:
            print(s, end=end)

    def board_str(self, board):
        return self.game.stringRepresentationReadable(board, human=self.args.human)

    def getActionProb(self, canonicalBoard, temp=1, lm=None):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        If `lm` is set, call `lm` on MCTS state prompt to compute an MCTS action string
        which is parsed t0n retrieve the node expansion and updates.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            # print("=" * 20 + f"Simulation {i} " + "=" * 20)
            self.last_s = "__ROOT__"
            if lm is None:
                self.search(canonicalBoard)
            else:
                self.lm_search(canonicalBoard, lm=lm)

        s = self.board_str(canonicalBoard)
        counts = [
            self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
            for a in range(self.game.getActionSize())
        ]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1.0 / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def print_node(self, s, local: bool = True):
        sid = self.local_id[s] if local else self.board_to_id[s]
        cs = [self.local_id[x] for x in self.children.get(s, [])]
        if self.args.human:
            # terminal nodes don't record an Ns[s] entry, take 0
            self.trace(f"{sid:3} {s} N={self.Ns.get(s, 0):3} {cs}")
        else:
            cs_str = " ".join(f"{c:03}" for c in cs)
            self.trace(f"{sid:03} {s} {self.Ns.get(s, 0):03} {cs_str}")

    def print_edge(self, s, a, local: bool = True):
        sid = self.local_id[s] if local else self.board_to_id[s]
        s2 = self.edge_targets[(s, a)]
        sid2 = self.local_id[s2] if local else self.board_to_id[s2]
        if self.args.human:
            self.trace(
                f"s={sid:3} a={a:2} s'={sid2:3} Nsa={self.Nsa[(s, a)]:3} Qsa={self.Qsa[(s, a)]:+5.3f}"
            )
        else:
            self.trace(
                f"{sid:03} {a:02} {sid2:03} {self.Nsa[(s, a)]:03} {self.Qsa[(s, a)]:+5.3f}"
            )

    def preorder(self, f, s):
        """Preorder traversal for graphs - maintains a set of visited nodes to prevent cycles."""
        self._visited = set()
        self._preorder(f, s)

    def _preorder(self, f, s):
        if s in self._visited:
            return
        f(s)
        self._visited.add(s)
        for c in self.children.get(s, []):
            self._preorder(f, c)

    def compute_reachable_tree(self, s):
        self.local_id = {"__ROOT__": 0}
        if not s in self.Ns:
            return

        def assign_local_id(s):
            self.local_id[s] = len(self.local_id)  # assign next local id

        self.preorder(assign_local_id, s)

    def print_reachable_tree(self, s):
        """Prints the reachable part of the MCTS tree from the current board position `s`."""
        if not s in self.local_id:
            return
        self.preorder(self.print_node, s)

    def print_reachable_edges(self, s):
        for s, a in self.Qsa:
            if s in self.local_id:
                self.print_edge(s, a)

    def search(self, canonicalBoard):
        while True:
            try:
                s = self.board_str(canonicalBoard)
                self.search_start = canonicalBoard
                self.compute_reachable_tree(s)
                self.print_reachable_tree(s)
                self.print_reachable_edges(s)
                result = self._search(canonicalBoard)
                self.submit_trace()
                return result
            except RuntimeError:
                self.reset_trace()
                # self.children has been updated, so the next attempt is more likely to succeed

    def lm_search(self, canonicalBoard, lm):
        while True:
            try:
                s = self.board_str(canonicalBoard)
                self.compute_reachable_tree(s)
                self.print_reachable_tree(s)
                self.print_reachable_edges(s)
                actions_str = lm(self._trace_state + "ACTIONS:\n")
                _, expand, updates = parse_mcts_actions()  # vists not needed
                self.trace(actions_str, to="actions")
                self.lm_step(expand, updates)
                result = self._search(canonicalBoard)
                self.submit_trace()
                return result
            except RuntimeError:
                self.reset_trace()
                # self.children has been updated, so the next attempt is more likely to succeed

    def _register_node(self, s):
        assert s not in self.board_to_id
        self.board_to_id[s] = len(self.board_to_id)
        self.children[self.last_s] = self.children.get(self.last_s, []) + [s]
        self.edge_targets[(self.last_s, self.last_a)] = s

    def _search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        s = self.board_str(canonicalBoard)
        # Diamond Merging:
        # `self.children` and `self.local_id` maintain a local search tree that needs to be
        # extended if `s` is known in the global search tree, but with another path from the root.
        # In this case, print an `expand ...` line, add the node to the local search tree, print it out
        # and pretend to be in a new iteration. The same applies if there is a diamond in the local search tree,
        # hence we check in `self.children` not in `self.local_id`.
        if s in self.board_to_id and not s in self.children.get(self.last_s, []):
            # add the node to the current local tree
            self.children[self.last_s] = self.children.get(self.last_s, []) + [s]
            # print(self.last_s, self.children[self.last_s])
            self.edge_targets[(self.last_s, self.last_a)] = s

            # abort and restart the current search in the try/catch loop of search()
            raise RuntimeError()

            # self.trace(
            #     f"RESTARTED at: {s} last: {self.last_s} start: {self.board_str(self.search_start)}"
            # )
            # if self.args.human:
            #     self.trace += f"simulate expand {self.local_id[self.last_s]:3} {self.last_a:2} {s} v={self.vs[s]:+5.3f}\n"
            # else:
            #     self.trace += f"expand {self.local_id[self.last_s]:03} {self.last_a:02} {s} {self.vs[s]:+5.3f}\n"

        # print(f"visiting {s2} N={self.Ns.get(s, 0)}")
        if s in self.local_id:
            if self.args.human:
                self.trace(f"visit {self.local_id[s]:3}", to="actions")
            else:
                self.trace(f"visit {self.local_id[s]:03}", to="actions")

        # self.print_node(s)
        # print(self.children[s])

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            # terminal node
            if not s in self.board_to_id:
                self._register_node(s)

            if self.args.human:
                self.trace(f"terminal {s} v={-self.Es[s]:+2}", to="actions")
            else:
                self.trace(f"terminal {s} {-self.Es[s]:+2}", to="actions")

            return -self.Es[s]

        if s not in self.Ps:
            # leaf node: expand
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            v = v[0]
            self.vs[s] = v
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            self._register_node(s)
            if self.args.human:
                self.trace(
                    f"expand {self.local_id[self.last_s]:3} {self.last_a:2} {s} v={v:+5.3f}",
                    to="actions",
                )
            else:
                self.trace(
                    f"expand {self.local_id[self.last_s]:03} {self.last_a:02} {s} {v:+5.3f}",
                    to="actions",
                )
            return -v

        valids = self.Vs[s]
        cur_best = -float("inf")
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(
                        self.Ns[s]
                    ) / (1 + self.Nsa[(s, a)])
                else:
                    u = (
                        self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)
                    )  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)
        self.last_s = s
        self.last_a = best_act
        v = self._search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (
                self.Nsa[(s, a)] + 1
            )
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
            assert (s, a) in self.edge_targets, (self.local_id.get(s, None), a)

        self.Ns[s] += 1
        # print(f"update {self.board_to_id[s]:3} N={self.Ns[s]:3}")
        if self.args.human:
            self.trace(
                f"update {self.local_id[s]:3} {a:2} Ns={self.Ns[s]:3} Nsa={self.Nsa[(s, a)]:3} Qsa={self.Qsa[(s, a)]:+5.3f}",
                to="actions",
            )
        else:
            self.trace(
                f"update {self.local_id[s]:03} {a:02} {self.Ns[s]:03} {self.Nsa[(s, a)]:03} {self.Qsa[(s, a)]:+5.3f}",
                to="actions",
            )

        return -v

    def lm_step(self, expand, updates):
        # TODO: terminal nodes
        if expand:
            node, action, s, _ = expand
            last_s = self.local_id[node]
            if s in self.board_to_id:
                # add known node to the current local tree
                self.children[last_s] = self.children.get(last_s, []) + [s]
                # print(self.last_s, self.children[self.last_s])
                self.edge_targets[(last_s, action)] = s
                # abort and restart the current search in the try/catch loop of lm_search()
                raise RuntimeError()
            else:
                # register new node
                # set parent for _register_node
                self.last_s = last_s
                self.last_a = action
                # note: v is not needed, only implicit in updates
                self.Ns[s] = 0
                self._register_node(s)

        for update in updates:
            assert update is not None
            node, action, Ns, Nsa, Qsa = update
            s = self.local_id[node]
            self.Ns[s] = Ns
            self.Nsa[(s, action)] = Nsa
            self.Qsa[(s, action)] = Qsa
