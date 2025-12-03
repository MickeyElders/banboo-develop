SHELL := /bin/bash

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
	@if PYTHONPATH="$(JETSON_PY):$$PYTHONPATH" $(PY) -c "import jetson.inference, jetson.utils" >/dev/null 2>&1; then \
		echo "jetson-inference bindings OK"; \
	else \
		echo "jetson-inference Python bindings not found; attempting source install..." ; \
		$(MAKE) install-jetson; \
	fi

install-jetson:
	@set -e; \
	missing=""; \
	for pkg in python3-dev python3-numpy; do \
		dpkg -s $$pkg >/dev/null 2>&1 || missing="$$missing $$pkg"; \
	done; \
	if [ -n "$$missing" ]; then \
		echo "Installing missing build deps:$$missing"; \
		sudo apt-get update && sudo apt-get install -y $$missing; \
	else \
		echo "Build deps OK (python3-dev/python3-numpy)"; \
	fi; \
	NUMPY_INC=$$(python3 - <<'PY' 2>/dev/null || true\nimport numpy, os\nprint(numpy.get_include())\nPY); \
	NUMPY_LIBDIR=$$(python3 - <<'PY' 2>/dev/null || true\nimport numpy, os\nprint(os.path.join(os.path.dirname(numpy.__file__), 'core', 'lib'))\nPY); \
	if [ -z "$$NUMPY_INC" ] || [ -z "$$NUMPY_LIBDIR" ]; then echo "numpy not found, please install python3-numpy"; exit 1; fi; \
	if [ -f "$$NUMPY_LIBDIR/libnpymath.a" ]; then \
		for dst in /usr/lib/libnpymath.a /usr/lib/aarch64-linux-gnu/libnpymath.a; do \
			if [ ! -f $$dst ]; then sudo ln -sf "$$NUMPY_LIBDIR/libnpymath.a" $$dst; fi; \
		done; \
	else \
		echo "libnpymath.a not found under $$NUMPY_LIBDIR"; exit 1; \
	fi; \
	tmpdir=$$(mktemp -d); \
	echo "Cloning jetson-inference into $$tmpdir"; \
	cd $$tmpdir; \
	git clone --recursive https://github.com/dusty-nv/jetson-inference.git; \
	cd jetson-inference; \
	mkdir -p build && cd build; \
	cmake .. -DENABLE_PYTHON=ON -DNUMPY_INCLUDE_DIRS="$$NUMPY_INC" -DNUMPY_LIBRARIES="$$NUMPY_LIBDIR/libnpymath.a"; \
	make -j$$(nproc); \
	sudo make install; \
	sudo ldconfig; \
	rm -rf "$$tmpdir"; \
	echo "jetson-inference install completed"

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
