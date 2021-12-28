import copy
from Data import DataProccess
import numpy as np
import Model


def main():

    file_path = "./data/sudoku.csv"
    dp = DataProccess(file_path)

    x_train, x_test, y_train, y_test = dp.get_data()

    model = Model.get_compiled_model()

    model.load_weights("training/cp.ckpt")

    # summary
    model.summary()

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=32)
    print("test loss, test acc:", results)

    print("test accuracy: for 1000 games using all-in-one-go technique:\n")
    res_all_in_one = test_accuracy_all_in_one(x_test[:100], y_test[:100], model)

    print("test accuracy: for 1000 games using one by one technique:\n")
    res_one_by_one = test_accuracy_one_by_one(x_test[:100], y_test[:100], model)


def solve_one_by_one(sample, model):
    '''
        This function solve the sudoku by filling blank positions one by one.
    '''

    feat = copy.copy(sample)

    while (1):

        out = model.predict(feat.reshape((1, 9, 9, 1)))
        out = out.squeeze()

        pred = np.argmax(out, axis=1).reshape((9, 9)) + 1

        prob = np.around(np.max(out, axis=1).reshape((9, 9)), 2)

        feat = DataProccess.denorm(feat).reshape((9, 9))

        mask = (feat == 0)

        if mask.sum() == 0:
            break

        prob_new = prob * mask

        ind = np.argmax(prob_new)
        x, y = (ind // 9), (ind % 9)

        val = pred[x][y]
        feat[x][y] = val
        feat = DataProccess.norm(feat)

    return pred


def test_accuracy_all_in_one(feats, labels, model):
    correct = 0
    results = []

    for i, feat in enumerate(feats):

        out = model.predict(feat.reshape((1, 9, 9, 1)))
        out = out.squeeze()

        pred = np.argmax(out, axis=1).reshape((9, 9)) + 1

        true = labels[i].reshape((9, 9)) + 1

        if abs(true - pred).sum() == 0:
            correct += 1
            results.append(True)
        else:
            results.append(False)

    print("all-in-one accuracy: ", correct / feats.shape[0])
    return results


def test_accuracy_one_by_one(feats, labels, model):
    correct = 0
    results = []

    for i, feat in enumerate(feats):

        pred = solve_one_by_one(feat, model)

        true = labels[i].reshape((9, 9)) + 1

        if abs(true - pred).sum() == 0:
            correct += 1
            results.append(True)
        else:
            results.append(False)

    print(f"one-by-one accuracy: {correct / feats.shape[0]}")
    return results


if __name__ == '__main__':
    main()
