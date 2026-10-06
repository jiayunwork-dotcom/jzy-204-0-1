"""服务启动入口：python -m app.main（容器内由 waitress 拉起）。"""
from __future__ import annotations

import os
from .api import create_app
from .config import Config


def main():
    app = create_app(config=Config)
    if os.environ.get("FLASK_DEV_SERVER") == "1":
        app.run(host=Config.HOST, port=Config.PORT)
    else:
        import waitress
        waitress.serve(app, host=Config.HOST, port=Config.PORT,
                       threads=max(4, Config.WORKERS + 2))


if __name__ == "__main__":
    main()
