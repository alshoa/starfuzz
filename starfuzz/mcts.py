from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .transforms import Action, semantic_constraint


@dataclass
class Evaluation:
    stability_coverage: float
    pred: int


EvaluateFn = Callable[[list[np.ndarray]], Evaluation]


@dataclass
class Node:
    image: np.ndarray
    action: Action | None = None
    parent: "Node | None" = None
    depth: int = 0
    visits: int = 0
    total_reward: float = 0.0
    stability_coverage: float = 0.0
    children: list["Node"] = field(default_factory=list)
    untried: list[Action] = field(default_factory=list)

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.visits if self.visits else 0.0

    def path(self) -> list["Node"]:
        out = []
        node: Node | None = self
        while node is not None:
            out.append(node)
            node = node.parent
        return list(reversed(out))


@dataclass
class FuzzResult:
    best_nodes: list[Node]
    total_iterations: int
    total_failures: int


class MCTSFuzzer:
    def __init__(
        self,
        seed_image: np.ndarray,
        label: int,
        actions: list[Action],
        evaluate: EvaluateFn,
        rollout_depth: int = 20,
        max_attempts: int = 20,
        exploration: float = 1.4,
        depth_bias: float = 0.08,
        rng: random.Random | None = None,
    ):
        self.seed_image = seed_image
        self.label = int(label)
        self.actions = actions
        self.evaluate = evaluate
        self.rollout_depth = int(rollout_depth)
        self.max_attempts = int(max_attempts)
        self.exploration = float(exploration)
        self.depth_bias = float(depth_bias)
        self.rng = rng or random.Random()
        self.root = Node(image=seed_image, depth=0, untried=list(actions))
        self.root.stability_coverage = self.evaluate([seed_image]).stability_coverage

    def run(self, budget: int, top_k: int = 20) -> FuzzResult:
        failures = 0
        for _ in range(int(budget)):
            node = self._select(self.root)
            child = self._expand(node)
            eval_result = self._simulate(child)
            parent_scov = child.parent.stability_coverage if child.parent else 0.0
            reward = eval_result.stability_coverage - parent_scov
            if eval_result.pred != self.label:
                failures += 1
            child.stability_coverage = eval_result.stability_coverage
            self._backpropagate(child, reward)
        return FuzzResult(best_nodes=self.best_nodes(top_k), total_iterations=int(budget), total_failures=failures)

    def _select(self, node: Node) -> Node:
        while not node.untried and node.children:
            node = max(node.children, key=self._ucb)
        return node

    def _ucb(self, node: Node) -> float:
        if node.visits == 0:
            return float("inf")
        parent_visits = max(1, node.parent.visits if node.parent else 1)
        explore = self.exploration * math.sqrt(math.log(parent_visits + 1) / (node.visits + 1e-9))
        return node.avg_reward + explore + self.depth_bias * math.log(node.depth + 1)

    def _expand(self, node: Node) -> Node:
        if not node.untried:
            return node
        self.rng.shuffle(node.untried)
        while node.untried:
            action = node.untried.pop()
            for _ in range(self.max_attempts):
                image = action.apply(self.seed_image)
                if semantic_constraint(self.seed_image, image):
                    child = Node(
                        image=image,
                        action=action,
                        parent=node,
                        depth=node.depth + 1,
                        untried=list(self.actions),
                    )
                    node.children.append(child)
                    return child
        return node

    def _simulate(self, node: Node) -> Evaluation:
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

    def _backpropagate(self, node: Node, reward: float) -> None:
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def best_nodes(self, top_k: int) -> list[Node]:
        nodes: list[Node] = []

        def visit(node: Node) -> None:
            if node is not self.root and node.visits > 0:
                nodes.append(node)
            for child in node.children:
                visit(child)

        visit(self.root)
        return sorted(nodes, key=lambda n: n.avg_reward, reverse=True)[:top_k]
