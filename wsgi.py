"""WSGI entry point for production deployments.

Exposes ``app`` (the Flask application) for any WSGI server.

Production servers
------------------
Windows / cross-platform (recommended, thread-pool based):

    waitress-serve --listen=127.0.0.1:5000 --threads=8 wsgi:app

Linux / macOS (multi-process + threads for horizontal CPU scaling):

    gunicorn -c gunicorn.conf.py wsgi:app

Note on this codebase's shared, single-session agent: the web API keeps one
MLAgent instance per process. To scale across processes, set
``WSGI_WORKERS>1`` (gunicorn) so each worker has its own independent agent
session, or deploy per-session isolation at the application layer.
"""
from ml_agent.logging_utils import configure_logging
from web_api import app as application

# `application` is the conventional name gunicorn/uWSGI look for by default.
app = application

if __name__ == "__main__":
    from web_api import main

    main()