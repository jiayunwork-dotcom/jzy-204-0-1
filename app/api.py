"""Flask HTTP 接口层。

路由
----
历书
  POST /api/almanacs                 导入新历书 -> 新版本号，自动触发重规划
  GET  /api/almanacs                 历书版本列表
  GET  /api/almanacs/<ver>           某版本历书详情
点位
  PUT  /api/points                   新增/更新点位（版本自增），触发相关重规划
  GET  /api/points                   点位列表
  GET  /api/points/<pid>             点位当前版本
  GET  /api/points/<pid>/versions/<v> 点位历史版本
可见性
  POST /api/sky/observe              单时刻可见卫星与 DOP
规划
  POST /api/plans                    提交规划（异步作业，返回作业号）
  GET  /api/jobs/<job_id>            查询进度
  POST /api/jobs/<job_id>/wait       等待完成（测试/同步使用）
  POST /api/jobs/<job_id>/cancel     取消作业
  GET  /api/plans                    规划列表
  GET  /api/plans/<plan_id>          规划定义
  GET  /api/plans/<plan_id>/result   当前结果（含绑定的版本信息）
  GET  /api/plans/<plan_id>/history  历史结果列表
差异
  POST /api/plans/<plan_id>/diff     对比历史/任意历书版本或另一份规划
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
from flask import Flask, jsonify, request

from .cache import SharedSkyCache
from .config import Config
from .diff import diff_results
from .errors import JobError, ValidationError
from .mask import build_point
from .orbit import build_almanac
from .planning import validate_params
from .scheduler import Scheduler
from .timeutils import parse_time
from .visibility import VisibilityEngine


def _json_default(obj):
    if isinstance(obj, datetime):
        return obj.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    raise TypeError(f"不可序列化对象 {type(obj)!r}")


def create_app(repo=None, scheduler: Scheduler | None = None,
               config: Config | None = None, db=None,
               per_point_hook=None):
    """应用工厂。测试可直接传入 repo（如 mongomock 库）。"""
    config = config or Config
    app = Flask(__name__)
    app.json.default = _json_default  # Flask 3 的 JSON provider

    if repo is None:
        from pymongo import MongoClient
        client = MongoClient(config.MONGO_URL, serverSelectionTimeoutMS=5000)
        db = client[config.MONGO_DB]
    from .repository import Repository
    if repo is None:
        repo = Repository(db)
    repo.setup_indexes()

    if scheduler is None:
        scheduler = Scheduler(
            repo, cache=SharedSkyCache(maxsize=config.CACHE_SIZE),
            workers=config.WORKERS, per_point_hook=per_point_hook)
        scheduler.startup()
        app.config["_OWNS_SCHEDULER"] = True
    else:
        app.config["_OWNS_SCHEDULER"] = False
    app.config["REPO"] = repo
    app.config["SCHED"] = scheduler

    register_routes(app, repo, scheduler)

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.errorhandler(ValidationError)
    def _bad_request(err):
        return jsonify({"error": "validation_error",
                        "message": err.message,
                        "field": err.field}), 400

    @app.errorhandler(JobError)
    def _job_error(err):
        return jsonify({"error": "job_error", "message": str(err)}), 409

    @app.errorhandler(404)
    def _not_found(err):
        return jsonify({"error": "not_found", "message": str(err)}), 404

    return app


def register_routes(app: Flask, repo, scheduler: Scheduler):

    # ------------------------------------------------------------- 历书
    @app.post("/api/almanacs")
    def import_almanac():
        body = request.get_json(force=True, silent=False)
        version = repo.next_almanac_version()
        almanac = build_almanac(body, version=version)
        repo.insert_almanac(almanac)
        job_ids = scheduler.trigger_almanac_replan(version)
        return jsonify({
            "almanac_version": version,
            "n_sat": len(almanac.sats),
            "replan_jobs": job_ids,
        }), 201

    @app.get("/api/almanacs")
    def list_almanacs():
        return jsonify({"almanacs": repo.list_almanacs()})

    @app.get("/api/almanacs/<int:version>")
    def get_almanac(version):
        return jsonify(repo.get_almanac(version).to_dict())

    # ------------------------------------------------------------- 点位
    @app.put("/api/points")
    def upsert_point():
        body = request.get_json(force=True)
        point_id = str(body.get("point_id", "")).strip()
        if not point_id:
            raise ValidationError("点位缺少 point_id", field="point_id")
        version = repo.next_point_version(point_id)
        point = build_point(body, version=version)
        repo.upsert_point(point)
        job_ids = scheduler.trigger_point_replan(point_id, version)
        return jsonify({
            "point_id": point_id,
            "point_version": version,
            "replan_jobs": job_ids,
        }), 200

    @app.get("/api/points")
    def list_points():
        return jsonify({"points": repo.list_points()})

    @app.get("/api/points/<point_id>")
    def get_point(point_id):
        return jsonify(repo.get_point(point_id).to_dict())

    @app.get("/api/points/<point_id>/versions/<int:version>")
    def get_point_version(point_id, version):
        return jsonify(repo.get_point(point_id, version).to_dict())

    # ----------------------------------------------------------- 单时刻
    @app.post("/api/sky/observe")
    def observe():
        body = request.get_json(force=True)
        t = parse_time(body.get("time"))
        almanac = _resolve_almanac(body.get("almanac_version"), repo)
        point = _resolve_point(body, repo)
        # 可选阈值：给了则 available 同时要求 GDOP≤阈值；
        # 不给时 available 仅表示"≥4 星且 DOP 可解算（有限）"。
        threshold = body.get("gdop_threshold")
        if threshold is not None:
            threshold = float(threshold)
            if not threshold > 0:
                raise ValidationError("GDOP 阈值必须为正",
                                      field="gdop_threshold")
        engine = VisibilityEngine(almanac, point)
        ok_flag, gdop, n_vis, dop, details = engine.state_at(t)
        from .timeutils import to_iso
        solvable = bool(ok_flag and n_vis >= 4 and np.isfinite(gdop))
        available = solvable and (threshold is None or gdop <= threshold)
        return jsonify({
            "time": to_iso(t),
            "almanac_version": almanac.version,
            "point_id": point.point_id,
            "point_version": point.version,
            "n_visible": n_vis,
            "available": available,
            "within_threshold": (None if threshold is None else
                                 bool(solvable and gdop <= threshold)),
            "gdop_threshold": threshold,
            "dop": dop.to_dict(),
            "satellites": details,
        })

    # ------------------------------------------------------------- 规划
    @app.post("/api/plans")
    def create_plan():
        body = request.get_json(force=True)
        params = validate_params(body)
        point_ids = body.get("point_ids")
        if not isinstance(point_ids, list) or not point_ids:
            raise ValidationError("point_ids 必须为非空列表",
                                  field="point_ids")
        point_ids = [str(x) for x in point_ids]
        if len(set(point_ids)) != len(point_ids):
            raise ValidationError("point_ids 有重复", field="point_ids")
        for pid in point_ids:
            repo.get_point(pid)  # 不存在即 400
        alm = _resolve_almanac(body.get("almanac_version"), repo)
        plan_id = f"plan-{repo.next_id('plan'):06d}"
        plan_doc = {
            "plan_id": plan_id,
            "point_ids": point_ids,
            "almanac_version": alm.version,
            "params": params,
            "status": "active",
        }
        repo.insert_plan(plan_doc)
        job_id = scheduler.submit_plan_job(plan_doc, triggered_by="manual")
        return jsonify({"plan_id": plan_id, "job_id": job_id,
                        "almanac_version": alm.version}), 202

    @app.get("/api/plans")
    def list_plans():
        plans = [{k: v for k, v in p.items() if k != "_id"}
                 for p in repo.plans.find(sort=[("plan_id", 1)])] \
            if hasattr(repo, "plans") else []
        return jsonify({"plans": plans})

    @app.get("/api/plans/<plan_id>")
    def get_plan(plan_id):
        return jsonify(repo.get_plan(plan_id))

    # ------------------------------------------------------------- 作业
    @app.get("/api/jobs/<job_id>")
    def get_job(job_id):
        return jsonify(scheduler.get_job(job_id))

    @app.post("/api/jobs/<job_id>/wait")
    def wait_job(job_id):
        timeout = float(request.args.get("timeout", "30"))
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = scheduler.get_job(job_id)
            if job["status"] in ("done", "failed", "canceled",
                                 "interrupted"):
                return jsonify(job)
            time.sleep(0.02)
        return jsonify({"error": "timeout", **scheduler.get_job(job_id)}), 202

    @app.post("/api/jobs/<job_id>/cancel")
    def cancel_job(job_id):
        doc = scheduler.cancel(job_id)
        return jsonify({"job_id": job_id, "status": doc["status"],
                        "cancel_reason": doc.get("cancel_reason")})

    # ------------------------------------------------------------- 结果
    @app.get("/api/plans/<plan_id>/result")
    def get_result(plan_id):
        return jsonify(_public_result(repo.get_result(plan_id)))

    @app.get("/api/plans/<plan_id>/history")
    def get_history(plan_id):
        history = repo.list_history(plan_id)
        return jsonify({
            "plan_id": plan_id,
            "history": [_brief(h) for h in history],
        })

    @app.post("/api/plans/<plan_id>/diff")
    def result_diff(plan_id):
        body = request.get_json(force=True, silent=True) or {}
        current = repo.get_result(plan_id)
        # 基线：默认取该规划保存的上一版历史；也可指定历史序号或另一规划
        if body.get("other_plan_id"):
            other = repo.get_result(body["other_plan_id"])
            old_results = other["results"]
            base_desc = f"plan:{body['other_plan_id']}@" \
                        f"v{other['almanac_version']}"
        elif "history_index" in body:
            history = current.get("history", [])
            idx = int(body["history_index"])
            if idx < 0 or idx >= len(history):
                raise ValidationError("历史序号越界", field="history_index")
            old_results = history[idx]["results"]
            base_desc = f"history[{idx}]:v" \
                        f"{history[idx]['almanac_version']}"
        else:
            history = current.get("history", [])
            if not history:
                return jsonify({"error": "no_baseline",
                                "message": "该规划没有历史结果可对比"}), 409
            old_results = history[-1]["results"]
            base_desc = f"history[-1]:v{history[-1]['almanac_version']}"
        result = diff_results(old_results, current["results"],
                              shift_tol_s=int(body.get("shift_tol_s", 1)))
        result["plan_id"] = plan_id
        result["baseline"] = base_desc
        return jsonify(result)


# ----------------------------------------------------------------- 辅助
def _resolve_almanac(version, repo):
    if version is None:
        alm = repo.get_latest_almanac()
        if alm is None:
            raise ValidationError("尚未导入任何历书", field="almanac_version")
        return alm
    return repo.get_almanac(int(version))


def _resolve_point(body, repo):
    """点位可引用已维护的 point_id(+版本)，也可内联给经纬度/遮挡。"""
    if body.get("point_id"):
        return repo.get_point(str(body["point_id"]),
                              body.get("point_version"))
    inline = body.get("point")
    if inline is None:
        raise ValidationError("需要 point_id 或内联 point", field="point_id")
    return build_point(inline)


def _brief(history_entry: dict) -> dict:
    return {
        "almanac_version": history_entry["almanac_version"],
        "job_id": history_entry.get("job_id"),
        "triggered_by": history_entry.get("triggered_by"),
        "created_at": history_entry.get("created_at"),
        "n_windows": sum(len(r.get("windows", []))
                         for r in history_entry.get("results", [])),
    }


def _public_result(doc: dict) -> dict:
    out = {k: v for k, v in doc.items() if k != "_id"}
    out["n_windows"] = sum(len(r["windows"]) for r in doc["results"])
    return out
