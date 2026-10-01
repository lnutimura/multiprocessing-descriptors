import numpy as np
import pytest

from mpdescriptors.evaluation import evaluate


def test_perfect_separation():
    labels = np.repeat(np.arange(4), 3)
    features = labels[:, None] * 100.0 + np.random.default_rng(0).random((12, 2))
    result = evaluate(features, labels)
    assert result.mean_average_precision == 1.0
    assert result.auc_pr == pytest.approx(1.0)
    np.testing.assert_allclose(result.recall, [0.5, 1.0])


def test_hand_computed_ranking():
    labels = np.array([0, 1, 0, 1])
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    # Query 1 ties tiles 0 and 2; the stable sort keeps collection order.
    result = evaluate(features, labels)
    np.testing.assert_allclose(result.precision[:, 0], [1 / 2, 1 / 3, 1 / 3, 1 / 2])
    assert result.mean_average_precision == pytest.approx(5 / 12)
    assert result.auc_pr == pytest.approx(5 / 12)


def test_precision_at_each_relevant_hit():
    labels = np.array([0, 0, 0, 1, 1, 1])
    features = np.array([[0.0], [1.0], [5.0], [2.0], [3.0], [6.0]])
    # Query 0 retrieves 1, 3, 4, 2, 5: relevant hits at ranks 1 and 4.
    np.testing.assert_allclose(evaluate(features, labels).precision[0], [1.0, 0.5])


def test_chunking_does_not_change_results():
    rng = np.random.default_rng(1)
    labels = np.repeat(np.arange(5), 4)
    features = rng.random((20, 3))
    np.testing.assert_array_equal(
        evaluate(features, labels, chunk_size=3).precision, evaluate(features, labels).precision
    )


def test_rejects_unequal_classes():
    with pytest.raises(ValueError):
        evaluate(np.zeros((3, 1)), np.array([0, 0, 1]))
