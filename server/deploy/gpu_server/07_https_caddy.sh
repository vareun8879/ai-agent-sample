#!/usr/bin/env bash
# 7단계 (선택) — HTTPS 종단 (Caddy 리버스 프록시)
#
#   키오스크 → 오케스트레이터 사이는 X-API-Key 가 평문으로 흐른다. 인터넷을 지나면 반드시 HTTPS 여야 한다.
#   Caddy 는 도메인만 있으면 Let's Encrypt 인증서를 자동 발급·갱신한다.
#
#   전제: 도메인의 A 레코드가 이 서버 공인 IP 를 가리키고, ACG 에서 80/443 이 열려 있어야 한다.
#   사용법: bash 07_https_caddy.sh kiosk-api.example.com
#   결과:  https://kiosk-api.example.com/ask/text  →  127.0.0.1:8000
#          이후 ACG 에서 8000 을 닫고 443 만 열면 된다. 키오스크 App.config 의 OrchestratorUrl 을 https 주소로 교체.
source "$(dirname "$0")/lib.sh"
DOMAIN="${1:-}"
[ -n "$DOMAIN" ] || die "도메인을 인자로 주세요. 예: bash 07_https_caddy.sh kiosk-api.example.com"

if ! command -v caddy >/dev/null 2>&1; then
    log "Caddy 설치"
    sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
    sudo apt-get update -y && sudo apt-get install -y caddy
fi

log "Caddyfile 작성: ${DOMAIN} → 127.0.0.1:8000"
sudo tee /etc/caddy/Caddyfile >/dev/null <<EOF
${DOMAIN} {
    # 답변 생성이 길어질 수 있으므로 (RAG_TIMEOUT 330s) 프록시 타임아웃을 넉넉히
    reverse_proxy 127.0.0.1:8000 {
        transport http {
            response_header_timeout 360s
        }
    }
}
EOF
sudo systemctl enable --now caddy
sudo systemctl reload caddy

cat <<EOF

7단계 완료. 확인:
  curl -s https://${DOMAIN}/health
키오스크 App.config:
  VoiceQa.OrchestratorUrl = https://${DOMAIN}
ACG: 443 은 키오스크 IP 만 허용, 8000 은 닫기 (orchestrator 는 그대로 0.0.0.0:8000 이지만 ACG 가 막는다).
EOF
