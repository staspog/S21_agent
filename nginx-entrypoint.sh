#!/bin/sh

set -e

AGENT_APP_HOST="${AGENT_APP_HOST:-s21_agent}"
AGENT_APP_PORT="${AGENT_APP_PORT:-8000}"
TLS_CERT_PEM_BASE64="${TLS_CERT_PEM_BASE64:-}"
TLS_KEY_PEM_BASE64="${TLS_KEY_PEM_BASE64:-}"
export AGENT_APP_HOST AGENT_APP_PORT

SSL_DIR="/tmp/nginx-ssl"
mkdir -p "$SSL_DIR"

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
    server {
        listen 8000;
        listen [::]:8000;
        server_name _;

        proxy_connect_timeout 120s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        client_max_body_size 10M;

        location / {
            proxy_pass http://${AGENT_APP_HOST}:${AGENT_APP_PORT};
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_http_version 1.1;
        }

        location /healthz {
            access_log off;
            proxy_pass http://${AGENT_APP_HOST}:${AGENT_APP_PORT}/health;
            proxy_connect_timeout 5s;
            proxy_read_timeout 5s;
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
