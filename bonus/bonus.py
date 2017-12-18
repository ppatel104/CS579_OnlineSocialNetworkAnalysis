import networkx as nx


def jaccard_wt(graph, node):
  """
  The weighted jaccard score, defined above.
  Args:
    graph....a networkx graph
    node.....a node to score potential new edges for.
  Returns:
    A list of ((node, ni), score) tuples, representing the 
              score assigned to edge (node, ni)
              (note the edge order)
  """
  neighbours = set(graph.neighbors(node))
  scores = []
  for n in graph.nodes():
    neighbours2 = set(graph.neighbors(n))
    if not (graph.has_edge(node,n)) and node != n:
        if(len(list(neighbours & neighbours2))!=0):
            score1 = 0
            score2 = 0
            score3 = 0
            for j in list(neighbours & neighbours2):
                score1 += 1/(graph.degree(j))
            for k in neighbours:
                score2 += (graph.degree(k))
            for p in neighbours2:
                score3 += (graph.degree(p))
            scores.append(tuple(((node,n),(score1/((1/score2)+(1/score3))))))
        else:
            scores.append(tuple(((node,n),0.0)))
  scores = sorted(scores, key=lambda x: (-x[1], x[0]))
  result = []
  for i in range(0, len(scores)):
    result.append(scores[i])
  return result
