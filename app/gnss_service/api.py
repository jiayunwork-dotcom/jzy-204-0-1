import os

from flask import Flask, jsonify, request

from .computation import ObservationEngine
from .models import ValidationError
from .repositories import MemoryRepository, MongoRepository
from .service import PlanningService
from .timeutils import parse_time


def create_app(service: PlanningService | None = None) -> Flask:
    app = Flask(__name__)
    if service is None:
        mongo_uri = os.getenv("MONGO_URI", "mongodb://mongo:27017/gnss_planning")
        if os.getenv("USE_MEMORY_REPOSITORY") == "1":
            repository = MemoryRepository()
        else:
            repository = MongoRepository(mongo_uri, mongo_uri.rsplit("/", 1)[-1].split("?", 1)[0])
        service = PlanningService(repository, ObservationEngine())

    def error_response(code: int, message: str):
        return jsonify({"error": message}), code

    @app.errorhandler(ValidationError)
    def handle_validation(exc):
        return error_response(400, str(exc))

    @app.errorhandler(KeyError)
    def handle_missing(exc):
        return error_response(404, str(exc).strip("'"))

    @app.errorhandler(ValueError)
    def handle_value(exc):
        return error_response(400, str(exc))

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/almanacs")
    def import_almanac():
        payload = request.get_json(force=True)
        return jsonify(service.import_almanac(payload)), 201

    @app.get("/api/almanacs")
    def list_almanacs():
        return jsonify(service.list_almanacs())

    @app.get("/api/almanacs/<almanac_id>")
    def get_almanac(almanac_id):
        return jsonify(service.get_almanac(almanac_id))

    @app.put("/api/points/<point_id>")
    def upsert_point(point_id):
        created = service.repo.get("points", point_id) is None
        result = service.upsert_point(point_id, request.get_json(force=True))
        return jsonify(result), 201 if created else 200

    @app.get("/api/points")
    def list_points():
        return jsonify(service.list_points())

    @app.get("/api/points/<point_id>")
    def get_point(point_id):
        return jsonify(service.get_point(point_id))

    @app.get("/api/query")
    def query_instant():
        at = request.args.get("time")
        if not at:
            return error_response(400, "time query parameter is required")
        parse_time(at)  # validate before entering the service
        return jsonify(
            service.query_instant(
                request.args["almanac_version_id"], request.args["point_id"], at
            )
        )

    @app.post("/api/plans")
    def create_plan():
        return jsonify(service.create_plan(request.get_json(force=True))), 202

    @app.get("/api/plans")
    def list_plans():
        return jsonify(service.list_plans())

    @app.get("/api/plans/<plan_id>")
    def get_plan(plan_id):
        return jsonify(service.get_plan(plan_id))

    @app.get("/api/plans/<plan_id>/results")
    def get_result(plan_id):
        historical = request.args.get("history", "0") in {"1", "true", "yes"}
        result = service.get_result(plan_id, historical=historical)
        if result is None:
            return error_response(404, "result not found")
        return jsonify(result)

    @app.get("/api/plans/<plan_id>/diff")
    def get_diff(plan_id):
        current = service.get_result(plan_id)
        historical = service.get_result(plan_id, historical=True)
        if current is None:
            return error_response(404, "current result not found")
        return jsonify(service.compare_results(historical, current))

    @app.get("/api/jobs/<job_id>")
    def get_job(job_id):
        return jsonify(service.get_job(job_id))

    @app.post("/api/jobs/<job_id>/cancel")
    def cancel_job(job_id):
        return jsonify(service.cancel_job(job_id))

    app.config["PLANNING_SERVICE"] = service
    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
