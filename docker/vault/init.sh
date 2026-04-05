#!/bin/sh
# ─────────────────────────────────────────────────────────────────────────────
# VAULT INIT SCRIPT — Solo para desarrollo local
# En producción Vault se configura con auto-unseal (AWS KMS, GCP Cloud KMS, etc.)
# ─────────────────────────────────────────────────────────────────────────────

set -e

VAULT_ADDR="http://127.0.0.1:8200"
export VAULT_ADDR

echo "Esperando a que Vault arranque..."
until vault status 2>/dev/null; do
  sleep 1
done

echo "Autenticando con Vault..."
vault auth -method=token token="${VAULT_DEV_ROOT_TOKEN_ID:-dev-root-token}"

echo "Habilitando KV v2 secrets engine..."
vault secrets enable -path=secret kv-v2 2>/dev/null || echo "KV ya habilitado"

echo "Escribiendo secretos de desarrollo..."
vault kv put secret/rag-chatbot \
  groq_api_key="${GROQ_API_KEY:-gsk_placeholder}" \
  secret_key="${SECRET_KEY:-dev-secret-key-change-in-production-32ch}" \
  database_url="postgresql://raguser:ragpassword@postgres:5432/ragdb" \
  redis_url="redis://redis:6379/0" \
  tavily_api_key="${TAVILY_API_KEY:-}"

echo "Configurando política de acceso..."
vault policy write rag-chatbot - <<POLICY
path "secret/data/rag-chatbot" {
  capabilities = ["read"]
}
path "secret/metadata/rag-chatbot" {
  capabilities = ["read", "list"]
}
POLICY

echo "Creando token de aplicación con la política..."
vault token create \
  -policy=rag-chatbot \
  -ttl=24h \
  -display-name=rag-chatbot-api \
  -format=json | tee /vault/config/app-token.json

APP_TOKEN=$(cat /vault/config/app-token.json | python3 -c "import sys,json; print(json.load(sys.stdin)['auth']['client_token'])")
echo ""
echo "==================================================="
echo "TOKEN DE APLICACIÓN:"
echo "$APP_TOKEN"
echo ""
echo "Añade al .env:"
echo "VAULT_ENABLED=true"
echo "VAULT_TOKEN=$APP_TOKEN"
echo "==================================================="
