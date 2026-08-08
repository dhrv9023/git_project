#!/usr/bin/env bash
# Script to generate local self-signed TLS certificates for Nginx testing

set -e

CERT_DIR="$(dirname "$0")/certs"
mkdir -p "$CERT_DIR"

if [ -f "$CERT_DIR/selfsigned.crt" ] && [ -f "$CERT_DIR/selfsigned.key" ]; then
    echo "✓ TLS certificates already exist in $CERT_DIR"
    exit 0
fi

echo "Generating self-signed TLS certificate for local SSL testing..."
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout "$CERT_DIR/selfsigned.key" \
    -out "$CERT_DIR/selfsigned.crt" \
    -subj "/C=US/ST=State/L=City/O=StockBuddy/OU=Quant/CN=localhost"

chmod 600 "$CERT_DIR/selfsigned.key"
chmod 644 "$CERT_DIR/selfsigned.crt"

echo "✓ TLS certificate generated successfully in $CERT_DIR"
