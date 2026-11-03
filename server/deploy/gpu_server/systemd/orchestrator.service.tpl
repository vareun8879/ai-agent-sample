[Unit]
Description=Kiosk orchestrator (question gate, session, QA log) :8000
After=network-online.target rag_service.service
Wants=network-online.target

[Service]
Type=simple
User=__USER__
WorkingDirectory=__APP_ROOT__/orchestrator
EnvironmentFile=__ENV_FILE__
Environment=PYTHONUNBUFFERED=1
ExecStart=__APP_ROOT__/app-env/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --log-config log_config.json
Restart=always
RestartSec=5
KillSignal=SIGINT
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
