storage "file" {
  path = "/vault/data"
}

listener "tcp" {
  address     = "0.0.0.0:8200"
  tls_disable = 1
  # tls_disable=1 solo en desarrollo.
  # En producción: tls_cert_file y tls_key_file obligatorios.
}

api_addr = "http://0.0.0.0:8200"
cluster_addr = "http://0.0.0.0:8201"

ui = true
# UI accesible en http://localhost:8200/ui
