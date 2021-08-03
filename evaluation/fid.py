import numpy as np
from scipy import stats
from scipy.linalg import sqrtm, norm
import pdb
import tqdm




def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    new_q = q[q != 0]
    new_p = p[q != 0]
    return np.sum(np.where(new_p != 0, new_p * np.log(new_p / new_q), 0))


def quant(pred_dist, action_set):
    """
    :param num_classes: int
    :param pred_dist: ndarray [num, num_classes] for predicted categories
    :return:
    """
    # get the histogram of gt and pred_hist
    overall_dist = np.mean(pred_dist, axis=0)

    # get the predicted_class
    predicted_class = np.argmax(pred_dist, axis=1)

    klds = []
    Intra_Es = []
    class_Intra_Es = {}
    for idx in range(len(pred_dist)):
        intra_E = stats.entropy(pred_dist[idx])
        klds.append(kl(pred_dist[idx], overall_dist))
        Intra_Es.append(intra_E)
        # get action_class
        action_class = action_set[predicted_class[idx] % len(action_set)]
        class_Intra_E = class_Intra_Es.get(action_class, [])
        class_Intra_E.append(intra_E)
        class_Intra_Es[action_class] = class_Intra_E

    I_score = np.exp(np.mean(klds))
    Intra_E = np.mean(Intra_Es)
    for k, v in class_Intra_Es.items():
        class_Intra_Es[k] = float(np.mean(v))
    Inter_E = stats.entropy(overall_dist)
    return float(I_score), float(Intra_E), Inter_E, class_Intra_Es


def confusion_matrix(num_classes, pred_cat, gt_cat):
    matrix = np.zeros((num_classes, num_classes))
    num_case = len(pred_cat)
    for idx in range(num_case):
        matrix[gt_cat[idx], pred_cat[idx]] += 1
    total = np.sum(matrix, axis=1)
    for idx in range(num_classes):
        matrix[idx] = matrix[idx] / total[idx] * 100
    return matrix


def get_mean_covar(samples):
    mean = samples.mean(axis=0)
    num_sample = samples.shape[0]
    mean_tile = np.tile(mean, [num_sample, 1])
    white_samples = samples - mean_tile
    covar = np.matmul(white_samples.transpose((1, 0)), white_samples) / num_sample
    return mean, covar


def get_fid(train_feature, test_feature, train_cat, test_cat):
    classes = np.unique(test_cat).tolist()
    distances = []
    for cls in tqdm.tqdm(classes):
        test_index = np.argwhere(test_cat == cls)[:, 0].tolist()
        train_index = np.argwhere(train_cat == cls)[:, 0].tolist()
        subcls_train_feature = train_feature[train_index]
        subcls_test_feature = test_feature[test_index]
        train_mean, train_covar = get_mean_covar(subcls_train_feature)
        test_mean, test_covar = get_mean_covar(subcls_test_feature)
        distance = norm(test_mean - train_mean) ** 2
        #            + np.trace(train_covar + test_covar -
        #                       2 * sqrtm(np.matmul(train_covar, test_covar)))
        print('distance: ', distance)
        distances.append(distance)
    FID = np.array(distances).mean()
    return FID
