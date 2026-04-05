#!/bin/sh
# ============================================================
# VAULT INIT SCRIPT — Solo para desarrollo local
# En producción Vault se inicializa de forma segura con
# unseal keys distribuidas entre múltiples responsables.
# ============================================================

set -e

echo "Esperando que Vault arranque..."
sleep 3

# Inicializar Vault (primera vez)
vault operator init -key-shares=1 -key-threshold=1 \
  -format=json > /vault/data/init.json 2>/dev/null || true

# Extraer unseal key y root token
UNSEAL_KEY=$(cat /vault/data/init.json | grep -o '"unseal_keys_b64":\["[^"]*"' | grep -o '"[^"]*"$' | tr -d '"')
ROOT_TOKEN=$(cat /vault/data/init.json | grep -o '"root_token":"[^"]*"' | grep -o '"[^"]*"$' | tr -d '"')

# Unsealar
vault operator unseal $UNSEAL_KEY

# Login con root token
vault login $ROOT_TOKEN

# Habilitar KV secrets engine v2
vault secrets enable -path=secret kv-v2 2>/dev/null || true

# Guardar secretos de la app
# En producción estos valores vendrían de tu sistema de gestión de secretos
vault kv put secret/rag-chatbot \
  groq_api_key="${GROQ_API_KEY:-your-groq-api-key-here}" \
  secret_key="$(openssl rand -hex 32)" \
  database_url="postgresql://raguser:ragpassword@postgres:5432/ragdb" \
  redis_url="redis://redis:6379/0" \
  tavily_api_key="${TAVILY_API_KEY:-}"

echo "Vault inicializado y secretos cargados."
echo "Root token: $ROOT_TOKEN"
echo "IMPORTANTE: En producción, distribuye las unseal keys de forma segura."
