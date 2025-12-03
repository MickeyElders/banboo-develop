PY ?= python3
PIP ?= $(PY) -m pip
PREFIX ?= /opt/bamboo-vision
SERVICE ?= bamboo-vision.service
JETSON_PY ?= /usr/local/python

.PHONY: deps check-jetson run install service service-restart service-stop service-status logs clean-install redeploy

deps:
	$(PIP) install -r requirements.txt
	$(MAKE) check-jetson

check-jetson:
	@PYTHONPATH="$(JETSON_PY):$$PYTHONPATH" $(PY) - <<'PY'
import sys
try:
    import jetson.inference  # noqa: F401
    import jetson.utils      # noqa: F401
    print("jetson-inference bindings OK")
except Exception:
    sys.stderr.write("Missing jetson-inference Python bindings.\n")
    sys.exit(1)
PY

run:
	$(PY) -m bamboo_vision.app --config config/runtime.yaml

install: deps
	sudo mkdir -p "$(PREFIX)"
	sudo cp -r bamboo_vision.py bamboo_vision config models bamboo.html requirements.txt RUNNING.md "$(PREFIX)"
	sudo install -D -m644 deploy/systemd/$(SERVICE) /etc/systemd/system/$(SERVICE)
	sudo systemctl daemon-reload
	sudo systemctl enable --now $(SERVICE)

service: install

service-restart:
	sudo systemctl restart $(SERVICE)

service-stop:
	sudo systemctl stop $(SERVICE)

service-status:
	sudo systemctl status $(SERVICE) --no-pager

logs:
	sudo journalctl -u $(SERVICE) -n 200 -f

clean-install:
	sudo systemctl stop $(SERVICE) 2>/dev/null || true
	sudo rm -rf "$(PREFIX)"

redeploy: clean-install install
