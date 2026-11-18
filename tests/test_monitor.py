import json

import pytest

from embedding_drift.cli import main
from embedding_drift.monitor import compare

BASE = [[1, 0], [0.99, 0.1], [0.99, -0.1], [0.9, 0.2]]


def test_identical_no_drift():
    r = compare(BASE, BASE)
    assert r.js_divergence == pytest.approx(0)
    assert not r.alert and r.reference_count == r.current_count == 4


def test_shift_is_detected():
    r = compare(BASE, [[0, 1], [0.1, 1], [-0.1, 1]])
    assert r.alert and r.js_divergence == pytest.approx(1)
    assert r.current_mean_similarity < r.reference_mean_similarity


def test_threshold_is_strict():
    r = compare(BASE, [[0, 1], [0, 2]], threshold=1)
    assert r.js_divergence == pytest.approx(1) and not r.alert


def test_scale_invariant():
    assert compare(BASE, [[4, 0], [3.96, 0.4], [3.96, -0.4], [3.6, 0.8]]).js_divergence == 0


@pytest.mark.parametrize("ref,cur", [([], BASE), ([[1, 0]], BASE), ([[0, 0], [1, 0]], BASE),
                                    ([[float('nan'), 0], [1, 0]], BASE),
                                    ([[1, 0], [0, 1]], [[1, 2, 3], [2, 3, 4]]),
                                    ([[1, 0], [-1, 0]], BASE),
                                    ([[1, 2], [1]], BASE)])
def test_invalid_inputs(ref, cur):
    with pytest.raises(ValueError):
        compare(ref, cur)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float('nan')])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        compare(BASE, BASE, threshold=threshold)


@pytest.mark.parametrize("bins", [1, 101, True, 2.5])
def test_invalid_bins(bins):
    with pytest.raises(ValueError):
        compare(BASE, BASE, bins=bins)


def test_large_finite_vectors():
    r = compare([[1e300, 0], [1e300, 1e299]], [[1e300, 0], [1e300, 1e299]])
    assert r.js_divergence == 0


def test_cli_alert_and_json(capsys):
    assert main(["--reference", "examples/reference.json", "--current", "examples/current.json"]) == 1
    assert json.loads(capsys.readouterr().out)["alert"] is True


def test_cli_missing(capsys):
    assert main(["--reference", "missing", "--current", "examples/current.json"]) == 2
    assert "error" in json.loads(capsys.readouterr().out)
