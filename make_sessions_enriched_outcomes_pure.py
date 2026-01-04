import csv, json, os

IN_ENRICHED = os.path.join("data", "processed", "sessions_enriched.json")
IN_CSV = os.path.join("data", "raw", "ecommerce_clickstream_transactions.csv") # put next to index.html, or change path
OUT_PATH    = os.path.join("data", "processed", "sessions_enriched_outcomes.json")

CORE = {"page_view", "product_view", "add_to_cart", "purchase"}

def main():
    # 1) Load your enriched sessions (already has x/y/z + entropy/purity)
    with open(IN_ENRICHED, "r", encoding="utf-8") as f:
        sessions = json.load(f)

    # 2) Aggregate outcomes from raw clickstream CSV (pure python)
    agg = {}  # id -> dict
    with open(IN_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            sid = f"{row['UserID']}::{row['SessionID']}"
            ev  = row.get("EventType", "")

            if sid not in agg:
                agg[sid] = {
                    "converted": 0,
                    "revenue": 0.0,
                    "n_purchase": 0,
                    "n_core": 0
                }

            if ev == "purchase":
                agg[sid]["converted"] = 1
                agg[sid]["n_purchase"] += 1
                amt = row.get("Amount")
                if amt:
                    try:
                        agg[sid]["revenue"] += float(amt)
                    except ValueError:
                        pass

            if ev in CORE:
                agg[sid]["n_core"] += 1

    # 3) Attach outcomes onto each session record
    for s in sessions:
        m = agg.get(s["id"])
        if m:
            s.update(m)
        else:
            # should not happen, but keep safe defaults
            s.update({"converted": 0, "revenue": 0.0, "n_purchase": 0, "n_core": 0})

    # 4) Save
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(sessions, f)

    print("✅ wrote", OUT_PATH, "records:", len(sessions))

if __name__ == "__main__":
    main()
