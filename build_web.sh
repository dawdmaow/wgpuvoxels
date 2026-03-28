#!/usr/bin/env bash

set -euxo pipefail

INITIAL_MEMORY_PAGES=1024
MAX_MEMORY_PAGES=8192

WASM_PAGE_BYTES=65536
INITIAL_MEMORY_BYTES=$(expr $INITIAL_MEMORY_PAGES \* $WASM_PAGE_BYTES)
MAX_MEMORY_BYTES=$(expr $MAX_MEMORY_PAGES \* $WASM_PAGE_BYTES)

ODIN_ROOT=$(odin root)
ODIN_JS="$ODIN_ROOT/core/sys/wasm/js/odin.js"
WGPU_JS="$ODIN_ROOT/vendor/wgpu/wgpu.js"

# odin build . -target:js_wasm32 -out:main.wasm -o:speed \
# 	-extra-linker-flags:"--export-table --import-memory --initial-memory=$INITIAL_MEMORY_BYTES --max-memory=$MAX_MEMORY_BYTES"

odin build . -target:js_wasm32 -out:main.wasm -o:speed -debug \
	-extra-linker-flags:"--export-table --import-memory --initial-memory=$INITIAL_MEMORY_BYTES --max-memory=$MAX_MEMORY_BYTES"

# cp $ODIN_JS odin.js
# cp $WGPU_JS wgpu.js

# Ensure we can call preventDefault() for `wheel` events in Chrome.
# Upstream Odin registers wheel listeners as passive, which blocks
# preventDefault() and triggers console warnings.
#
# Note: this patches the generated odin.js in-place after copying it.
# perl -0777 -pe 's/element\\.addEventListener\\(name, listener, \\!\\!use_capture\\);/element.addEventListener(name, listener, (name === \"wheel\") ? { capture: !!use_capture, passive: false } : (!!use_capture));/g' -i odin.js