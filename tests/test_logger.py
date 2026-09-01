import csv

from utils.logger import append_csv


def test_append_csv_expands_header_when_row_schema_changes(tmp_path):
    path = tmp_path / "log.csv"

    append_csv(path, {"epoch": 1, "loss": 0.5})
    append_csv(path, {"epoch": 2, "loss": 0.4, "dynamic_refresh_epoch": 2})

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert reader.fieldnames == ["epoch", "loss", "dynamic_refresh_epoch"]
    assert rows[0] == {"epoch": "1", "loss": "0.5", "dynamic_refresh_epoch": ""}
    assert rows[1] == {"epoch": "2", "loss": "0.4", "dynamic_refresh_epoch": "2"}
