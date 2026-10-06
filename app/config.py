"""运行配置，全部来自环境变量，带默认值。"""
import os


class Config:
    MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
    MONGO_DB = os.environ.get("MONGO_DB", "gnss_planning")
    WORKERS = int(os.environ.get("WORKERS", "4"))
    # 共享可见性缓存最多保留多少个 (历书版本, 点位版本, 卫星, 秒) 结果
    CACHE_SIZE = int(os.environ.get("CACHE_SIZE", "200000"))
    # 时段搜索默认粗扫描步长（秒）
    SCAN_STEP = int(os.environ.get("SCAN_STEP", "60"))
    HOST = os.environ.get("HOST", "0.0.0.0")
    PORT = int(os.environ.get("PORT", "8080"))
