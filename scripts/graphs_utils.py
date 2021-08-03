import numpy as np


def prima(m, title=None):
    """ Prints a matrix to the terminal """
    if title:
        print(title)
    for row in m:
        print(', '.join([str(x) for x in row]))
    print('')


def path(m):
    """ Returns a path matrix """
    p = [list(row) for row in m]
    n = len(p)
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                continue
            if p[j][i]:
                for k in range(0, n):
                    if p[j][k] == 0:
                        p[j][k] = p[i][k]
    return p


def hsu(m):
    """ Transforms a given directed acyclic graph into its minimal equivalent """
    n = len(m)
    for j in range(n):
        for i in range(n):
            if m[i][j]:
                for k in range(n):
                    if m[j][k]:
                        m[i][k] = 0


def get_minimal_graph(g):
    p = path(g)
    hsu(p)
    return p


def triplets_to_adj_matrix(triplets):
    triplets = np.array(triplets).copy()
    triplets_concatenated = np.array(np.concatenate([triplets[:, :1], triplets[:, 2:]], axis=1))
    N = int(np.max(triplets_concatenated) + 1)
    grid = np.zeros((N, N), dtype='uint8')
    for k in range(triplets_concatenated.shape[0]):
        i, j = triplets_concatenated[k]
        grid[i, j] = 1
    return grid.tolist()


def matrix_to_triplets(m, rel_idx):
    rows, cols = np.where(np.array(m, dtype='uint8') == 1)
    rels = np.ones(len(rows)) * rel_idx
    return np.stack([rows, rels, cols], axis=1)


def triplets_to_minimal(triplets):
    if len(triplets) < 3:
        return triplets
    m = triplets_to_adj_matrix(triplets)
    m = get_minimal_graph(m)
    new_triplets = matrix_to_triplets(m, triplets[0][1])
    # print("Old vs new: %s, %s"%(len(triplets), len(new_triplets)))
    return new_triplets


def reduce_transitive_edges(triplets, p_keep=0.5):
    if len(triplets) < 3:
        return triplets
    mat = triplets_to_adj_matrix(triplets)
    min_graph = get_minimal_graph(mat)
    prob_mat = np.random.uniform(0, 1, (len(mat), len(mat)))
    new_mat = (prob_mat * (np.array(mat) - np.array(min_graph)) > (1 - p_keep)).astype('uint8') + np.array(min_graph)
    new_triplets = matrix_to_triplets(new_mat, triplets[0][1])
    return new_triplets


def get_maximal_transitive_triplets(triplets):
    if len(triplets) < 2:
        return triplets
    mat = triplets_to_adj_matrix(triplets)
    maximal_graph = path(mat)
    return matrix_to_triplets(np.array(maximal_graph), triplets[0][1])


def get_minimal_and_transitive_triplets(triplets):
    mat = triplets_to_adj_matrix(triplets)
    min_graph = get_minimal_graph(mat)
    maximal_graph = path(mat)
    return matrix_to_triplets(min_graph, triplets[0][1]), matrix_to_triplets(
        np.array(maximal_graph) - np.array(min_graph), triplets[0][1])


def get_current_and_transitive_triplets(triplets):
    min_graph = triplets_to_adj_matrix(triplets)
    maximal_graph = path(min_graph)
    return matrix_to_triplets(min_graph, triplets[0][1]), matrix_to_triplets(
        np.array(maximal_graph) - np.array(min_graph), triplets[0][1])


def get_symmetric_triplets(triplets):
    triplets = np.array(triplets)
    return triplets[:, ::-1]


def get_edge_antisymmetric_triplets(triplets, vocab):
    pred_id = int(triplets[0, 1])
    antisymmetric_candidates = list(set(vocab['pred_name_to_idx'].values()) - set([pred_id]))
    antisymmetric_edges = []
    for p in antisymmetric_candidates:
        symmetric_triplets = triplets.copy()[:, ::-1]
        symmetric_triplets[:, 1] = p
        antisymmetric_edges.extend(symmetric_triplets)
    return antisymmetric_edges


def test_reduce_transitive_edges():
    triplets = [[0, 1, 1],  # redundant
                [0, 1, 2],  # redundant
                [0, 1, 3],
                [1, 1, 2],
                [3, 1, 1],
                [3, 1, 2]]  # redundant

    reduced_triplets = [[0, 1, 3],
                        [1, 1, 2],
                        [3, 1, 1]]

    gt_mat = \
        [[0, 1, 1, 1],
         [0, 0, 1, 0],
         [0, 0, 0, 0],
         [0, 1, 1, 0]]

    output_mat = triplets_to_adj_matrix(triplets)
    result_triplets = matrix_to_triplets(output_mat, 1)

    assert np.all(np.array(gt_mat) == np.array(gt_mat))
    assert np.all(np.array(result_triplets) == np.array(triplets))

    reduced_trip = reduce_transitive_edges(triplets, 1)
    assert np.all(np.array(reduced_triplets) == np.array(reduced_trip))


if __name__ == "__main__":
    test_reduce_transitive_edges()
