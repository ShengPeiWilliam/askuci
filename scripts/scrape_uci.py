import requests
import pandas as pd
import time

LIST_URL = "https://archive.ics.uci.edu/api/datasets/list"
DETAIL_URL = "https://archive.ics.uci.edu/api/dataset"
OUTPUT_PATH = "data/uci_datasets.csv"


def fetch_all_ids() -> list[int]:
    resp = requests.get(LIST_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    datasets = data.get("data", [])
    return [d["id"] for d in datasets]


def fetch_dataset(dataset_id: int) -> dict | None:
    resp = requests.get(DETAIL_URL, params={"id": dataset_id}, timeout=30)
    if resp.status_code != 200:
        print(f"  Skipping id={dataset_id} (status {resp.status_code})")
        return None
    raw = resp.json().get("data", {})
    return {
        "id": raw.get("uci_id"),
        "name": raw.get("name", ""),
        "abstract": raw.get("abstract", ""),
        "area": raw.get("area", ""),
        "tasks": ", ".join(raw.get("tasks", []) or []),
        "num_instances": raw.get("num_instances"),
        "num_features": raw.get("num_features"),
        "has_missing_values": raw.get("has_missing_values"),
        "year_of_dataset_creation": raw.get("year_of_dataset_creation"),
    }


def main():
    print("Fetching dataset list...")
    ids = fetch_all_ids()
    print(f"Found {len(ids)} datasets. Fetching details...")

    records = []
    for i, dataset_id in enumerate(ids, 1):
        print(f"[{i}/{len(ids)}] id={dataset_id}", end="\r")
        record = fetch_dataset(dataset_id)
        if record:
            records.append(record)
        time.sleep(0.2)

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} datasets to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
