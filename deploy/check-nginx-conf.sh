#!/bin/sh
# nginx -t для обоих вариантов edge-конфига (TLS и TLS_DISABLED).
# Запускается внутри одноразового контейнера, конфиги смонтированы в /test:
#   docker run --rm -v "$PWD:/test:ro" nginx:alpine sh /test/deploy/check-nginx-conf.sh
set -e

TEST_DIR="${TEST_DIR:-/test}"

mkdir -p /tmp/nginx-ssl /var/log/nginx/scanner

# Самоподписанный cert только чтобы nginx -t смог загрузить ssl_certificate.
command -v openssl >/dev/null 2>&1 || apk add --no-cache openssl >/dev/null
openssl req -x509 -newkey rsa:2048 -nodes -keyout /tmp/nginx-ssl/ssl.key \
    -out /tmp/nginx-ssl/ssl.crt -days 1 -subj /CN=test >/dev/null 2>&1

cp "$TEST_DIR/nginx-scanner-noise.conf" /etc/nginx/scanner-noise.conf

echo "--- TLS variant (nginx.conf) ---"
AGENT_APP_HOST=127.0.0.1 AGENT_APP_PORT=8000 \
    envsubst '${AGENT_APP_HOST} ${AGENT_APP_PORT}' < "$TEST_DIR/nginx.conf" > /tmp/nginx.conf
nginx -t -c /tmp/nginx.conf

echo "--- HTTP variant (nginx-entrypoint.sh, TLS_DISABLED=true) ---"
sed 's|^exec nginx .*|nginx -t -c /tmp/nginx.conf|' "$TEST_DIR/nginx-entrypoint.sh" > /tmp/e.sh
AGENT_APP_HOST=127.0.0.1 AGENT_APP_PORT=8000 TLS_DISABLED=true sh /tmp/e.sh
