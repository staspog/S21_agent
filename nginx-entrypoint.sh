#!/bin/sh

set -e

AGENT_APP_HOST="${AGENT_APP_HOST:-s21_agent}"
AGENT_APP_PORT="${AGENT_APP_PORT:-8000}"
TLS_CERT_PEM_BASE64="${TLS_CERT_PEM_BASE64:-}"
TLS_KEY_PEM_BASE64="${TLS_KEY_PEM_BASE64:-}"
export AGENT_APP_HOST AGENT_APP_PORT

SSL_DIR="/tmp/nginx-ssl"
mkdir -p "$SSL_DIR"

# Каталог отдельного access-лога сканеров (обычно проброшен с хоста).
mkdir -p /var/log/nginx/scanner

if [ -n "$TLS_CERT_PEM_BASE64" ] && [ -n "$TLS_KEY_PEM_BASE64" ]; then
    printf '%s' "$TLS_CERT_PEM_BASE64" | base64 -d > "$SSL_DIR/ssl.crt"
    printf '%s' "$TLS_KEY_PEM_BASE64" | base64 -d > "$SSL_DIR/ssl.key"
    chmod 600 "$SSL_DIR/ssl.key"
    chmod 644 "$SSL_DIR/ssl.crt"
elif [ -f /etc/nginx/ssl/ssl.crt ] && [ -f /etc/nginx/ssl/ssl.key ]; then
    cp /etc/nginx/ssl/ssl.crt "$SSL_DIR/ssl.crt"
    cp /etc/nginx/ssl/ssl.key "$SSL_DIR/ssl.key"
    chmod 600 "$SSL_DIR/ssl.key"
    chmod 644 "$SSL_DIR/ssl.crt"
fi

if [ "$TLS_DISABLED" = "true" ]; then
    cat > /tmp/nginx-agent-http.conf.template <<'EOF'
events {
    worker_connections 1024;
}

http {
    # log_format + $scanner_probe/$scanner_hit/$app_traffic
    include /etc/nginx/scanner-noise.conf;

    server {
        listen 8000;
        listen [::]:8000;
        server_name _;

        proxy_connect_timeout 120s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        client_max_body_size 10M;

        # Основной лог без scanner-шума; шум — в отдельный файл.
        access_log /var/log/nginx/access.log s21_main if=$app_traffic;
        access_log /var/log/nginx/scanner/scanners.log s21_main if=$scanner_hit;

        # Известные probe рубим до проксирования: приложение их не видит.
        if ($scanner_probe) {
            return 444;
        }

        location = /nginx-health {
            access_log off;
            return 200 "ok\n";
        }

        location /healthz {
            access_log off;
            proxy_pass http://${AGENT_APP_HOST}:${AGENT_APP_PORT}/health/live;
            proxy_connect_timeout 5s;
            proxy_read_timeout 5s;
        }

        location / {
            proxy_pass http://${AGENT_APP_HOST}:${AGENT_APP_PORT};
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_http_version 1.1;
        }
    }
}
EOF
    envsubst '${AGENT_APP_HOST} ${AGENT_APP_PORT}' \
        < /tmp/nginx-agent-http.conf.template > /tmp/nginx.conf
else
    if [ ! -f "$SSL_DIR/ssl.crt" ] || [ ! -f "$SSL_DIR/ssl.key" ]; then
        echo "ОШИБКА: TLS для agent edge не настроен"
        echo "Передайте TLS_CERT_PEM_BASE64 и TLS_KEY_PEM_BASE64 или включите TLS_DISABLED=true"
        exit 1
    fi
    envsubst '${AGENT_APP_HOST} ${AGENT_APP_PORT}' \
        < /etc/nginx/nginx.conf > /tmp/nginx.conf
fi

exec nginx -c /tmp/nginx.conf -g "daemon off;"
