import pandas as pd

from data_pipeline import _as_true, _split_for_id, load_public_domain_rows


def test_public_domain_boolean_parser():
    values = pd.Series([True, "TRUE", "yes", 1, False, "no", ""])
    assert _as_true(values).tolist() == [True, True, True, True, False, False, False]


def test_split_is_deterministic_and_valid():
    first = _split_for_id("123")
    assert first == _split_for_id("123")
    assert first in {"train", "valid", "test"}


def test_loader_excludes_non_public_and_missing_images(tmp_path):
    path = tmp_path / "objects.csv"
    pd.DataFrame(
        [
            {
                "Object ID": 1,
                "Is Public Domain": True,
                "Primary Image": "https://example.com/1.jpg",
                "Primary Image Small": "",
                "Title": "Allowed",
            },
            {
                "Object ID": 2,
                "Is Public Domain": False,
                "Primary Image": "https://example.com/2.jpg",
                "Primary Image Small": "",
                "Title": "Copyrighted",
            },
            {
                "Object ID": 3,
                "Is Public Domain": True,
                "Primary Image": "",
                "Primary Image Small": "",
                "Title": "No image",
            },
        ]
    ).to_csv(path, index=False)

    result = load_public_domain_rows(path)
    assert result["Object ID"].tolist() == ["1"]
