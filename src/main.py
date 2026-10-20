import networkx as nx
import pyomo.environ as pyo
from pyomo.core.util import sum_product


class MinimumSpanningTree(pyo.ConcreteModel):
    def __init__(self, G: nx.Graph) -> None:
        self._graph = G
        self.__set_composite_sets()
        self.__set_parameters()
        self.__set_decision_variables()
        self.__set_objective_function()

    def __set_composite_sets(self) -> None:
        self.V = pyo.Set(initialize=list(self._graph.nodes))
        self.r = pyo.Set(initialize=list(self._graph.nodes)[0], domain=self.V)
        self.E = pyo.Set(initialize=list(self._graph.to_directed().edges), dimen=2)
        self.P = pyo.Set(initialize=[((v, w), (w, v)) for (v, w) in self._graph.edges], dimen=4)

    def __set_parameters(self) -> None:
        self.n = pyo.Param(self._graph.number_of_nodes, domain=pyo.PositiveIntegers)
        self.c = pyo.Param(initialize=self._graph.edges(data="distance"))

    def __set_decision_variables(self) -> None:
        self.x = pyo.Var(self.E, domain=pyo.Binary, initialize=0)
        self.u = pyo.Var(self.V, domain=pyo.PositiveIntegers, initialize=1, bounds=(1, self.n))

    def __set_objective_function(self) -> None:
        self.obj = pyo.Objective(sense=pyo.minimize, expr=sum_product(self.c, self.x, index=self.E))

    def __set_constraints(self) -> None:
        self.edge_selection_constraints = pyo.Constraint(
            self.P, rule=lambda model, v1, w1, v2, w2: model.x[v1,w1] + model.x[v2,w2] <= 1
        )
        self.out_degree_from_root_constraint = pyo.Constraint()
        self.in_degree_for_non_root_nodes = pyo.Constraint()
        self.mtz_subtour_elimination_constraints = pyo.Constraint(
            self.E, rule=lambda model, v, w:
                model.u[w] >= model.u[v] + 1 - model.n * (1 - model.x[v,w]) + (model.n - 2) * model.x[w,v]
        )

if __name__ == "__main__":
    G = nx.Graph()
    G.add_nodes_from([
        ( 1, {'cidade': 'Torina'   }),
        ( 2, {'cidade': 'Milano'   }),
        ( 3, {'cidade': 'Bolzano'  }),
        ( 4, {'cidade': 'Verona'   }),
        ( 5, {'cidade': 'Venezia'  }),
        ( 6, {'cidade': 'Trieste'  }),
        ( 7, {'cidade': 'Genova'   }),
        ( 8, {'cidade': 'Bolgano'  }),
        ( 9, {'cidade': 'Ancona'   }),
        (10, {'cidade': 'Pisa'     }),
        (11, {'cidade': 'Firenze'  }),
        (12, {'cidade': 'Roma'     }),
        (13, {'cidade': 'Peragia'  }),
        (14, {'cidade': 'Caligiari'}),
        (15, {'cidade': 'Napoli'   }),
        (16, {'cidade': 'Pesacara' }),
        (17, {'cidade': 'Palermo'  }),
        (18, {'cidade': 'Catamin'  }),
        (19, {'cidade': 'Potenza'  }),
        (20, {'cidade': 'Catanzaro'}),
        (21, {'cidade': 'Bari'     })
    ])
    G.add_edges_from([
        ( 1,  2, {'distância':  95}),
        ( 1,  7, {'distância':  90}),
        ( 2,  3, {'distância': 140}),
        ( 2,  4, {'distância': 110}),
        ( 2,  7, {'distância':  90}),
        ( 3,  4, {'distância': 110}),
        ( 3,  6, {'distância': 210}),
        ( 4,  5, {'distância':  90}),
        ( 4,  8, {'distância':  95}),
        ( 5,  6, {'distância':  85}),
        ( 5,  8, {'distância':  95}),
        ( 7,  8, {'distância': 130}),
        ( 7, 10, {'distância': 120}),
        ( 7, 11, {'distância': 150}),
        ( 8,  9, {'distância': 200}),
        ( 8, 11, {'distância':  55}),
        ( 9, 16, {'distância': 130}),
        (10, 11, {'distância':  60}),
        (10, 12, {'distância': 190}),
        (11, 12, {'distância': 180}),
        (11, 13, {'distância': 110}),
        (12, 13, {'distância': 120}),
        (12, 14, {'distância': 480}),
        (12, 15, {'distância': 180}),
        (13, 16, {'distância': 170}),
        (14, 17, {'distância': 420}),
        (15, 16, {'distância': 200}),
        (15, 17, {'distância': 310}),
        (15, 18, {'distância': 350}),
        (15, 19, {'distância':  90}),
        (15, 21, {'distância': 210}),
        (16, 21, {'distância': 270}),
        (17, 18, {'distância': 150}),
        (18, 20, {'distância': 210}),
        (19, 20, {'distância': 200}),
        (19, 21, {'distância': 100})
    ])
    MST = MinimumSpanningTree(G)