from abc import ABC, abstractmethod

from pymongo import MongoClient


class Repository(ABC):
    @abstractmethod
    def insert(self, collection: str, document: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get(self, collection: str, document_id: str) -> dict | None:
        raise NotImplementedError

    @abstractmethod
    def find_one(self, collection: str, query: dict) -> dict | None:
        raise NotImplementedError

    @abstractmethod
    def find(self, collection: str, query: dict | None = None) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def set_fields(self, collection: str, document_id: str, fields: dict) -> bool:
        raise NotImplementedError

    @abstractmethod
    def set_many(self, collection: str, query: dict, fields: dict) -> int:
        raise NotImplementedError

    @abstractmethod
    def complete_plan_job(self, job_id: str, plan_id: str, job_fields: dict, result: dict | None, history: dict | None) -> bool:
        raise NotImplementedError


def _matches(document: dict, query: dict) -> bool:
    for key, expected in query.items():
        if key == "$or":
            if not any(_matches(document, clause) for clause in expected):
                return False
            continue
        if isinstance(expected, dict) and "$in" in expected:
            if document.get(key) not in expected["$in"]:
                return False
        elif document.get(key) != expected:
            return False
    return True


def _set_nested(document: dict, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    target = document
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


class MemoryRepository(Repository):
    def __init__(self):
        self._collections: dict[str, dict[str, dict]] = {}

    def reset(self) -> None:
        self._collections.clear()

    def insert(self, collection: str, document: dict) -> dict:
        store = self._collections.setdefault(collection, {})
        if document["id"] in store:
            raise KeyError(f"duplicate id in {collection}")
        store[document["id"]] = dict(document)
        return dict(document)

    def get(self, collection: str, document_id: str) -> dict | None:
        document = self._collections.get(collection, {}).get(document_id)
        return dict(document) if document is not None else None

    def find_one(self, collection: str, query: dict) -> dict | None:
        for document in self._collections.get(collection, {}).values():
            if _matches(document, query):
                return dict(document)
        return None

    def find(self, collection: str, query: dict | None = None) -> list[dict]:
        query = query or {}
        return [dict(document) for document in self._collections.get(collection, {}).values() if _matches(document, query)]

    def set_fields(self, collection: str, document_id: str, fields: dict) -> bool:
        document = self._collections.get(collection, {}).get(document_id)
        if document is None:
            return False
        for key, value in fields.items():
            _set_nested(document, key, value)
        return True

    def set_many(self, collection: str, query: dict, fields: dict) -> int:
        changed = 0
        for document in self._collections.get(collection, {}).values():
            if _matches(document, query):
                for key, value in fields.items():
                    _set_nested(document, key, value)
                changed += 1
        return changed

    def complete_plan_job(self, job_id: str, plan_id: str, job_fields: dict, result: dict | None, history: dict | None) -> bool:
        job = self._collections.get("jobs", {}).get(job_id)
        plan = self._collections.get("plans", {}).get(plan_id)
        if job is None or plan is None:
            return False
        if plan.get("active_job_id") != job_id or job.get("status") in {"cancelled", "superseded"}:
            return False
        if job.get("status") in {"completed", "failed"}:
            return False
        job.update(job_fields)
        if result is not None:
            old_result = plan.get("latest_result")
            if history is None and old_result is not None:
                plan["history_result"] = old_result
            elif history is not None:
                plan["history_result"] = history
            plan["latest_result"] = result
        plan["active_job_id"] = None
        return True


class MongoRepository(Repository):
    def __init__(self, uri: str, database: str):
        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        self.db = self.client[database]
        self._indexes()

    def _indexes(self) -> None:
        self.db.plans.create_index("active_job_id")
        self.db.jobs.create_index([("plan_id", 1), ("status", 1)])
        self.db.point_versions.create_index("point_id")
        self.db.almanacs.create_index("created_at")

    def insert(self, collection: str, document: dict) -> dict:
        self.db[collection].insert_one(dict(document))
        return dict(document)

    def get(self, collection: str, document_id: str) -> dict | None:
        value = self.db[collection].find_one({"id": document_id}, {"_id": False})
        return dict(value) if value is not None else None

    def find_one(self, collection: str, query: dict) -> dict | None:
        value = self.db[collection].find_one(query, {"_id": False})
        return dict(value) if value is not None else None

    def find(self, collection: str, query: dict | None = None) -> list[dict]:
        return [dict(item) for item in self.db[collection].find(query or {}, {"_id": False})]

    def set_fields(self, collection: str, document_id: str, fields: dict) -> bool:
        result = self.db[collection].update_one({"id": document_id}, {"$set": fields})
        return result.matched_count == 1

    def set_many(self, collection: str, query: dict, fields: dict) -> int:
        result = self.db[collection].update_many(query, {"$set": fields})
        return result.matched_count

    def complete_plan_job(self, job_id: str, plan_id: str, job_fields: dict, result: dict | None, history: dict | None) -> bool:
        plan_collection = self.db.plans
        plan = plan_collection.find_one({"id": plan_id}, {"_id": False})
        if plan is None or plan.get("active_job_id") != job_id:
            return False
        set_fields = dict(job_fields)
        update = {"$set": set_fields}
        if result is not None:
            update["$set"]["latest_result"] = result
            update["$set"]["history_result"] = history if history is not None else plan.get("latest_result")
            update["$set"]["active_job_id"] = None
        updated_plan = plan_collection.find_one_and_update(
            {"id": plan_id, "active_job_id": job_id}, update, return_document=False
        )
        if updated_plan is None:
            return False
        self.db.jobs.update_one(
            {"id": job_id, "status": {"$nin": ["cancelled", "superseded"]}},
            {"$set": job_fields},
        )
        return True
