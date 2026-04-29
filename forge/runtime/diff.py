def compute_diff(before: dict, after: dict) -> dict:
    added: dict = {}
    changed: dict = {}
    removed: dict = {}

    all_collections = set(before.keys()) | set(after.keys())

    for collection in all_collections:
        before_col = before.get(collection, {})
        after_col = after.get(collection, {})

        # Skip non-dict collection values (e.g. scalar fields like actor_id)
        if not isinstance(before_col, dict) or not isinstance(after_col, dict):
            if before_col != after_col:
                changed[collection] = {"before": before_col, "after": after_col}
            continue

        before_ids = set(before_col.keys())
        after_ids = set(after_col.keys())

        for entity_id in after_ids - before_ids:
            added[f"{collection}.{entity_id}"] = after_col[entity_id]

        for entity_id in before_ids - after_ids:
            removed[f"{collection}.{entity_id}"] = before_col[entity_id]

        for entity_id in before_ids & after_ids:
            b_entity = before_col[entity_id]
            a_entity = after_col[entity_id]
            all_fields = set(b_entity.keys()) | set(a_entity.keys())
            for field in all_fields:
                b_val = b_entity.get(field)
                a_val = a_entity.get(field)
                if b_val != a_val:
                    changed[f"{collection}.{entity_id}.{field}"] = {
                        "before": b_val,
                        "after": a_val,
                    }

    return {"added": added, "changed": changed, "removed": removed}
