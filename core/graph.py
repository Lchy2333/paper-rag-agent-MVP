from typing import Callable, Dict


START = "__START__"
END = "__END__"


class StateGraph:
    def __init__(self):
        self.nodes: Dict = {}
        self.edges: Dict = {}
        self.conditional_edges: Dict = {}

    def add_node(self, name: str, fn: Callable):
        self.nodes[name] = fn

    def add_edge(self, start: str, target: str):
        self.edges[start] = target

    def add_conditional_edge(self, start: str, router):
        self.conditional_edges[start] = router

    def compile(self) -> "CompileGraph":
        if START not in self.edges and START not in self.conditional_edges:
            raise ValueError("START 节点没有出边")

        for start, target in self.edges.items():
            if target not in self.nodes and target != END:
                raise ValueError(f"边 {start} -> {target} 指向不存在的节点: {target}")

        for start in list(self.edges) + list(self.conditional_edges):
            if start not in self.nodes and start != START:
                raise ValueError(f"节点 {start} 未注册，却有出边")

        return CompileGraph(self)


class CompileGraph:
    def __init__(self, state_graph: StateGraph):
        self.nodes = dict(state_graph.nodes)
        self.edges = dict(state_graph.edges)
        self.conditional_edges = dict(state_graph.conditional_edges)

    def invoke(self, state: Dict, max_steps:int = 10) -> Dict:
        current = START
        steps = 0

        while current != END:
            if steps >= max_steps:
                raise RuntimeError(f"执行超过 {max_steps} 步，疑似死循环")

            if current == START:
                if current in self.conditional_edges:
                    current = self.conditional_edges[current](state)
                else:
                    current = self.edges[current]

                continue

            fn = self.nodes[current]
            update = fn(state)
            state = {**state, **update}

            if current in self.conditional_edges:
                current = self.conditional_edges[current](state)
            else:
                current = self.edges[current]
            steps += 1

        return state
