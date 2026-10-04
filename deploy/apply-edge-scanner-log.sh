#!/usr/bin/env bash
# Применение scanner-фильтра edge-nginx на VPS: только контейнер s21_agent_edge,
# агент и мониторинг не трогаются.
#
# Запуск (с ноутбука):
#   scp nginx.conf nginx-scanner-noise.conf nginx-entrypoint.sh docker-compose.agent.yml \
#       deploy/check-nginx-conf.sh deploy/s21-agent-edge-scanners.logrotate \
#       deploy/apply-edge-scanner-log.sh gabrielg@<host>:/tmp/edge-stage/
#   ssh gabrielg@<host> 'bash /tmp/edge-stage/apply-edge-scanner-log.sh'
set -euo pipefail

STAGE="${STAGE:-/tmp/edge-stage}"
AGENT_ROOT="${AGENT_ROOT:-$HOME/agent}"
SCANNER_LOG_DIR="${AGENT_EDGE_SCANNER_LOG_DIR:-/var/log/s21_agent_edge}"

compose() {
    if docker compose version >/dev/null 2>&1; then
        docker compose -f docker-compose.agent.yml "$@"
    else
        docker-compose -f docker-compose.agent.yml "$@"
    fi
}

echo "==> nginx -t в одноразовом контейнере"
docker run --rm -v "$STAGE:/test:ro" nginx:alpine sh /test/check-nginx-conf.sh

echo "==> logrotate для $SCANNER_LOG_DIR/scanners.log"
sudo -n mkdir -p "$SCANNER_LOG_DIR"
sudo -n install -m 644 -o root -g root \
    "$STAGE/s21-agent-edge-scanners.logrotate" /etc/logrotate.d/s21-agent-edge-scanners
sudo -n logrotate -d /etc/logrotate.d/s21-agent-edge-scanners >/dev/null

echo "==> конфиги в $AGENT_ROOT"
cp "$STAGE/nginx.conf" "$STAGE/nginx-scanner-noise.conf" \
   "$STAGE/nginx-entrypoint.sh" "$STAGE/docker-compose.agent.yml" "$AGENT_ROOT/"

echo "==> пересоздание только s21_agent_edge (env берём из работающего контейнера)"
cd "$AGENT_ROOT"
set -a
# shellcheck disable=SC2046
eval "$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' s21_agent_edge \
    | grep -E '^(AGENT_APP_HOST|AGENT_APP_PORT|TLS_DISABLED|TLS_CERT_PEM_BASE64|TLS_KEY_PEM_BASE64)=' \
    | sed "s/=\(.*\)$/='\1'/")"
AGENT_EDGE_PORT="$(docker inspect \
    -f '{{range $p, $conf := .NetworkSettings.Ports}}{{range $conf}}{{.HostPort}}{{end}}{{end}}' \
    s21_agent_edge)"
AGENT_EDGE_PORT="${AGENT_EDGE_PORT:-8000}"
set +a
compose up -d --force-recreate --no-deps --wait --timeout 90 s21_agent_edge

echo "==> проверка"
if [ "${TLS_DISABLED:-false}" = "true" ]; then
    BASE="http://127.0.0.1:${AGENT_EDGE_PORT}"
else
    BASE="https://127.0.0.1:${AGENT_EDGE_PORT}"
fi
curl -skf --max-time 10 "$BASE/healthz" >/dev/null && echo "healthz: ok"
curl -sk --max-time 10 -o /dev/null -w 'probe //wp2/wp-includes/wlwmanifest.xml -> %{http_code}\n' \
    "$BASE//wp2/wp-includes/wlwmanifest.xml" || true
sleep 1
echo "--- последние строки docker logs (без шума) ---"
docker logs --tail 5 s21_agent_edge 2>&1 || true
echo "--- последние строки $SCANNER_LOG_DIR/scanners.log ---"
sudo -n tail -n 5 "$SCANNER_LOG_DIR/scanners.log" 2>/dev/null || echo "(пусто)"
