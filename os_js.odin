package main

import "base:runtime"
import "core:sys/wasm/js"
import "vendor:wgpu"

OS :: struct {
	initialized:           bool,
	mouse_captured:        bool,
	wants_pointer_capture: bool,
	pointer_retry_after:   f64,
	should_quit:           bool,
	// Tab visibility + window focus; combined in sync_game_process_active.
	page_visible:          bool,
	window_focused:        bool,
}

os_init :: proc() {
	ok: bool
	ok = js.add_window_event_listener(.Resize, nil, size_callback)
	assert(ok)
	// Use capture phase so we can suppress browser-reserved shortcuts (e.g. F7
	// caret browsing) before default handlers run.
	ok = js.add_window_event_listener(.Key_Down, nil, key_down_callback, true)
	assert(ok)
	ok = js.add_window_event_listener(.Key_Up, nil, key_up_callback)
	assert(ok)
	ok = js.add_window_event_listener(.Mouse_Move, nil, mouse_move_callback)
	assert(ok)
	ok = js.add_window_event_listener(.Mouse_Down, nil, mouse_down_callback)
	assert(ok)
	ok = js.add_window_event_listener(.Mouse_Up, nil, mouse_up_callback)
	assert(ok)
	ok = js.add_window_event_listener(.Wheel, nil, wheel_callback, true)
	assert(ok)
	ok = js.add_document_event_listener(.Pointer_Lock_Change, nil, pointer_lock_change_callback)
	assert(ok)
	ok = js.add_document_event_listener(.Visibility_Change, nil, visibility_callback)
	assert(ok)
	ok = js.add_window_event_listener(.Focus, nil, window_focus_callback)
	assert(ok)
	ok = js.add_window_event_listener(.Blur, nil, window_blur_callback)
	assert(ok)
}

// NOTE: frame loop is done by the runtime.js repeatedly calling `step`.
os_run :: proc() {
	state.os.initialized = true
	state.os.wants_pointer_capture = true
	state.os.page_visible = true
	state.os.window_focused = true
}

sync_game_process_active :: proc() {
	state.game_process_active = state.os.page_visible && state.os.window_focused
}

@(private = "file", export)
step :: proc(dt: f32) -> bool {
	if !state.os.initialized {
		return true
	}
	if state.os.should_quit {
		return false
	}

	frame(dt)
	state.keys_just_pressed = {}
	state.mouse_delta = {}
	state.mouse_wheel_steps = 0
	return true
}

os_get_framebuffer_size :: proc() -> (width, height: u32) {
	rect := js.get_bounding_client_rect("body")
	dpi := js.device_pixel_ratio()
	return u32(f64(rect.width) * dpi), u32(f64(rect.height) * dpi)
}

os_get_surface :: proc(instance: wgpu.Instance) -> wgpu.Surface {
	return wgpu.InstanceCreateSurface(
		instance,
		&wgpu.SurfaceDescriptor {
			nextInChain = &wgpu.SurfaceSourceCanvasHTMLSelector {
				sType = .SurfaceSourceCanvasHTMLSelector,
				selector = "#wgpu-canvas",
			},
		},
	)
}

@(private = "file", fini)
os_fini :: proc "contextless" () {
	context = runtime.default_context()
	js.remove_window_event_listener(.Resize, nil, size_callback)
	js.remove_window_event_listener(.Key_Down, nil, key_down_callback)
	js.remove_window_event_listener(.Key_Up, nil, key_up_callback)
	js.remove_window_event_listener(.Mouse_Move, nil, mouse_move_callback)
	js.remove_window_event_listener(.Mouse_Down, nil, mouse_down_callback)
	js.remove_window_event_listener(.Mouse_Up, nil, mouse_up_callback)
	js.remove_window_event_listener(.Wheel, nil, wheel_callback, true)
	js.remove_document_event_listener(.Pointer_Lock_Change, nil, pointer_lock_change_callback)
	js.remove_document_event_listener(.Visibility_Change, nil, visibility_callback)
	js.remove_window_event_listener(.Focus, nil, window_focus_callback)
	js.remove_window_event_listener(.Blur, nil, window_blur_callback)

	finish()
}

@(private = "file")
visibility_callback :: proc(e: js.Event) {
	state.os.page_visible = e.visibility_change.is_visible
}

@(private = "file")
window_focus_callback :: proc(e: js.Event) {
	state.os.window_focused = true
}

@(private = "file")
window_blur_callback :: proc(e: js.Event) {
	state.os.window_focused = false
}

@(private = "file")
size_callback :: proc(e: js.Event) {
	resize()
}

@(private = "file")
translate_js_key :: proc(code: string) -> (Key, bool) {
	switch code {
	case "Digit1":
		return .Num1, true
	case "Digit2":
		return .Num2, true
	case "Digit3":
		return .Num3, true
	case "Digit4":
		return .Num4, true
	case "Digit5":
		return .Num5, true
	case "Digit6":
		return .Num6, true
	case "Digit7":
		return .Num7, true
	case "Digit8":
		return .Num8, true
	case "Digit9":
		return .Num9, true
	case "KeyA":
		return .A, true
	case "KeyD":
		return .D, true
	case "KeyW":
		return .W, true
	case "KeyS":
		return .S, true
	case "KeyG":
		return .G, true
	case "KeyB":
		return .B, true
	case "KeyR":
		return .R, true
	case "KeyL":
		return .L, true
	case "KeyC":
		return .C, true
	case "KeyF":
		return .F, true
	case "F1":
		return .F1, true
	case "F2":
		return .F2, true
	case "F3":
		return .F3, true
	case "F4":
		return .F4, true
	case "F5":
		return .F5, true
	case "F6":
		return .F6, true
	case "F7":
		return .F7, true
	case "F8":
		return .F8, true
	case "F9":
		return .F9, true
	case "F10":
		return .F10, true
	case "F11":
		return .F11, true
	case "F12":
		return .F12, true
	case "Space":
		return .Space, true
	case "ShiftLeft", "ShiftRight":
		return .Shift, true
	case "Escape":
		return .Escape, true
	}
	return {}, false
}

@(private = "file")
request_pointer_capture :: proc() {
	// Browsers enforce strict lock timing/document validity rules; swallowing promise
	// rejections keeps devtools clean when we intentionally probe eligibility.
	js.evaluate(
		`(() => {
		const canvas = document.getElementById("wgpu-canvas");
		if (!canvas || !canvas.isConnected || canvas.ownerDocument !== document) {
			return;
		}
		if (!document.hasFocus() || document.visibilityState !== "visible") {
			return;
		}
		if (document.pointerLockElement !== canvas) {
			try {
				const p = canvas.requestPointerLock();
				if (p && typeof p.catch === "function") {
					p.catch(() => {});
				}
			} catch (_) {}
		}
	})()`,
	)
}

@(private = "file")
release_pointer_capture :: proc() {
	js.evaluate(`{
		if (document.pointerLockElement) {
			document.exitPointerLock();
		}
	}`)
}

@(private = "file")
key_down_callback :: proc(e: js.Event) {
	// Firefox (and others) reserve F7 for caret browsing. Prevent the default
	// behavior so our `F7` hotkey can work in WASM.
	if e.key.code == "F7" {
		js.event_prevent_default()
		js.event_stop_immediate_propagation()
	}

	// When we have mouse capture, block browser default handlers for all keys
	// so reserved shortcuts don't interfere with in-game controls.
	if state.os.mouse_captured {
		js.event_prevent_default()
		js.event_stop_immediate_propagation()
	}

	if e.key.code == "Escape" {
		if e.key.repeat do return
		if state.os.mouse_captured {
			state.os.wants_pointer_capture = false
			release_pointer_capture()
		}
		return
	}

	if !state.os.mouse_captured || e.key.repeat {
		return
	}
	key, ok := translate_js_key(e.key.code)
	if !ok do return
	state.keys_down += {key}
	state.keys_just_pressed += {key}
}

@(private = "file")
key_up_callback :: proc(e: js.Event) {
	if !state.os.mouse_captured {
		return
	}
	key, ok := translate_js_key(e.key.code)
	if !ok do return
	state.keys_down -= {key}
}

@(private = "file")
mouse_move_callback :: proc(e: js.Event) {
	state.mouse_pos = {f32(e.mouse.client[0]), f32(e.mouse.client[1])}
	if state.os.mouse_captured {
		state.mouse_delta += {f32(e.mouse.movement[0]), f32(e.mouse.movement[1])}
	}
}

@(private = "file")
mouse_down_callback :: proc(e: js.Event) {
	switch e.mouse.button {
	case 0:
		if !state.os.mouse_captured {
			// Browsers reject immediate reacquire after ESC unlock; delay until the
			// next explicit click outside the cooldown window.
			if e.timestamp < state.os.pointer_retry_after do return
			// Mirror SDL: first click after release only reacquires mouse capture.
			state.os.wants_pointer_capture = true
			request_pointer_capture()
			return
		}
		state.keys_down += {.Left_Mouse_Button}
		state.keys_just_pressed += {.Left_Mouse_Button}
	case 2:
		if state.os.mouse_captured {
			state.keys_down += {.Right_Mouse_Button}
			state.keys_just_pressed += {.Right_Mouse_Button}
		}
	}
}

@(private = "file")
mouse_up_callback :: proc(e: js.Event) {
	if !state.os.mouse_captured {
		return
	}
	switch e.mouse.button {
	case 0:
		state.keys_down -= {.Left_Mouse_Button}
	case 2:
		state.keys_down -= {.Right_Mouse_Button}
	}
}

@(private = "file")
wheel_callback :: proc(e: js.Event) {
	// Only apply wheel selection while the game has pointer capture, so normal
	// browser scrolling works when we're not actively controlling the camera.
	if !state.os.mouse_captured {
		return
	}

	// Suppress browser handling (scroll/zoom/etc.) while we're controlling
	// the game camera. This only works if `wheel` is registered with
	// `{ passive: false }` (see `odin.js` patch).
	js.event_prevent_default()
	js.event_stop_immediate_propagation()

	dy := e.wheel.delta[1]
	if dy == 0 do return

	abs_d := dy
	if abs_d < 0 do abs_d = -abs_d

	// Convert the browser's delta units into rough "ticks" so trackpads and
	// wheel mice both feel usable. We'll cap per-event steps to avoid jumps.
	scale: f64
	switch e.wheel.delta_mode {
	case .Pixel:
		scale = 100.0
	case .Line, .Page:
		scale = 1.0
	}

	ticks: int = int(abs_d / scale + 0.5)
	if dy < 0 do ticks = -ticks

	// Avoid pathological deltas from producing huge selection jumps.
	if ticks > 5 do ticks = 5
	if ticks < -5 do ticks = -5
	if ticks == 0 do return

	state.mouse_wheel_steps += ticks
}

@(private = "file")
pointer_lock_change_callback :: proc(e: js.Event) {
	state.os.mouse_captured = !state.os.mouse_captured
	if !state.os.mouse_captured {
		// Match Chromium's post-unlock lockout so we don't spam rejected requests.
		state.os.pointer_retry_after = e.timestamp + 0.25
		// Prevent sticky movement/actions when capture is released outside our ESC flow.
		state.keys_down = {}
	}
}
