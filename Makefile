.PHONY: binary clean-binary install-pyinstaller

# Install PyInstaller if not present
install-pyinstaller:
	poetry add --group dev pyinstaller

# Build binary using PyInstaller
binary: install-pyinstaller
	poetry run pyinstaller \
		--onefile \
		--name jirascope \
		--console \
		--clean \
		--distpath ./dist \
		--workpath ./build \
		--specpath . \
		src/jirascope/cli/main.py
	@echo "✅ Binary built successfully!"
	@echo "📦 Binary location: ./dist/jirascope"

# Clean binary build artifacts
clean-binary:
	rm -rf build/ dist/ *.spec

# Build optimized binary (smaller size)
binary-optimized: install-pyinstaller
	poetry run pyinstaller \
		--onefile \
		--name jirascope \
		--console \
		--clean \
		--strip \
		--upx-dir=/usr/local/bin \
		--distpath ./dist \
		--workpath ./build \
		--specpath . \
		src/jirascope/cli/main.py
	@echo "✅ Optimized binary built successfully!"
	@echo "📦 Binary location: ./dist/jirascope" 