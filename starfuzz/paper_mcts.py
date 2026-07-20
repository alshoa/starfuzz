from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .transforms import Action, semantic_constraint


@dataclass
class PaperEvaluation:
    stability_coverage: float
    pred: int


EvaluateFn = Callable[[list[np.ndarray]], PaperEvaluation]


@dataclass
class PaperNode:
    image: np.ndarray
    action: Action | None = None
    parent: "PaperNode | None" = None
    depth: int = 0
    visits: int = 0
    reward_sum: float = 0.0
    avg_reward: float = 0.0
    stability_coverage_cache: float = 0.0
    children: list["PaperNode"] = field(default_factory=list)

    def path(self) -> list["PaperNode"]:
        nodes = []
        node: PaperNode | None = self
        while node is not None:
            nodes.append(node)
            node = node.parent
        return list(reversed(nodes))


class PaperMCTSTree:
    """Algorithm-1 style stability-coverage MCTS.

    This keeps the paper's important mechanics:
    global UCB selection over the current tree, root-based mutation,
    path-set rollout stability coverage, parent SCov cache reward, and
    path back-propagation.
    """

    def __init__(
        self,
        seed_image: np.ndarray,
        label: int,
        actions: list[Action],
        evaluate: EvaluateFn,
        rollout_depth: int = 8,
        max_attempts: int = 20,
        cp: float = 1.4,
        alpha: float = 0.08,
        rng: random.Random | None = None,
    ):
        self.seed_image = seed_image
        self.label = int(label)
        self.actions = actions
        self.evaluate = evaluate
        self.rollout_depth = int(rollout_depth)
        self.max_attempts = int(max_attempts)
        self.cp = float(cp)
        self.alpha = float(alpha)
        self.rng = rng or random.Random()
        self.root = PaperNode(seed_image, depth=0)
        self.nodes: list[PaperNode] = [self.root]
        self.root.stability_coverage_cache = self.evaluate([seed_image]).stability_coverage
        self.rollout_failures = 0

    def iterate(self) -> PaperNode:
        selected = self._select()
        child = self._expand(selected)
        eval_result = self._simulate(child)
        parent_scov = child.parent.stability_coverage_cache if child.parent is not None else 0.0
        reward = eval_result.stability_coverage - parent_scov
        if eval_result.pred != self.label:
            self.rollout_failures += 1
        child.stability_coverage_cache = eval_result.stability_coverage
        self._backpropagate(child, reward)
        return child

    def run(self, budget: int) -> None:
        for _ in range(int(budget)):
            self.iterate()

    def _select(self) -> PaperNode:
        expandable = [node for node in self.nodes if node.depth < self.rollout_depth]
        if not expandable:
            return self.root
        return max(expandable, key=self._ucb)

    def _ucb(self, node: PaperNode) -> float:
        if node is self.root and node.visits == 0:
            return float("inf")
        if node.visits == 0:
            return float("inf")
        parent_visits = max(1, node.parent.visits if node.parent is not None else node.visits)
        explore = self.cp * math.sqrt(math.log(parent_visits + 1) / (node.visits + 1e-9))
        depth_bonus = self.alpha * math.log(node.depth + 1)
        return node.avg_reward + explore + depth_bonus

    def _expand(self, node: PaperNode) -> PaperNode:
        if node.depth >= self.rollout_depth:
            return node
        for _ in range(self.max_attempts):
            action = self.rng.choice(self.actions)
            image = action.apply(self.seed_image)
            if semantic_constraint(self.seed_image, image):
                child = PaperNode(image=image, action=action, parent=node, depth=node.depth + 1)
                node.children.append(child)
                self.nodes.append(child)
                return child
        return node

    def _simulate(self, node: PaperNode) -> PaperEvaluation:
        images = [n.image for n in node.path()]
        while len(images) < self.rollout_depth + 1:
            accepted = False
            for _ in range(self.max_attempts):
                action = self.rng.choice(self.actions)
                image = action.apply(self.seed_image)
                if semantic_constraint(self.seed_image, image):
                    images.append(image)
                    accepted = True
                    break
            if not accepted:
                break
        return self.evaluate(images)

    def _backpropagate(self, node: PaperNode, reward: float) -> None:
        while node is not None:
            node.visits += 1
            node.reward_sum += reward
            node.avg_reward = node.reward_sum / node.visits
            node = node.parent

    def best_nodes(self, top_k: int) -> list[PaperNode]:
        candidates = [n for n in self.nodes if n is not self.root and n.visits > 0]
        return sorted(candidates, key=lambda n: n.avg_reward, reverse=True)[:top_k]
