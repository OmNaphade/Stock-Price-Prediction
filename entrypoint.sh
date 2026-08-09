#!/bin/sh
# Restores each SQLite store from its Litestream replica (if one exists
# and the local file doesn't — i.e. a fresh container after a redeploy),
# then runs the app. Replication only activates when LITESTREAM_BUCKET_URL
# is set; otherwise this is identical to running streamlit directly.
set -e

STREAMLIT_CMD="streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"

if [ -n "$LITESTREAM_BUCKET_URL" ] && command -v litestream >/dev/null 2>&1; then
    for db in users.db track_record.db monitoring.db; do
        if [ ! -f "/app/$db" ]; then
            litestream restore -if-replica-exists -config /app/litestream.yml "/app/$db" || true
        fi
    done

    # `litestream replicate -exec` refuses to start the wrapped command at
    # all if it can't reach the configured bucket (wrong/expired
    # credentials, IAM issue, transient network failure, ...) — verified
    # by actually running this against an unreachable bucket: the whole
    # container exits before the app ever starts. That's litestream's own
    # fail-fast design, but it turns "replication is optional and
    # additive" (true everywhere else in this app) into "a bucket problem
    # takes the whole app down," which it must not. So this isn't `exec`'d
    # directly — it's backgrounded, given a moment to fail fast, and if it
    # has already died we fall back to running the app without
    # replication rather than not running at all. A signal trap forwards
    # shutdown signals to whichever process ends up supervising the app,
    # since backgrounding (instead of exec) means this script — not
    # litestream — is PID 1 in the success path.
    litestream replicate -config /app/litestream.yml -exec "$STREAMLIT_CMD" &
    LITESTREAM_PID=$!
    trap 'kill -TERM "$LITESTREAM_PID" 2>/dev/null; wait "$LITESTREAM_PID" 2>/dev/null' TERM INT

    sleep 2
    if kill -0 "$LITESTREAM_PID" 2>/dev/null; then
        wait "$LITESTREAM_PID"
    else
        echo "WARNING: litestream failed to start (bucket unreachable or misconfigured) — continuing without replication" >&2
        exec $STREAMLIT_CMD
    fi
else
    exec $STREAMLIT_CMD
fi
