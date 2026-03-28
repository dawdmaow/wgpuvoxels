#+build !js
package main

import "core:fmt"
import SDL "vendor:sdl3"
import "vendor:wgpu"
import "vendor:wgpu/sdl3glue"

OS :: struct {
	window: ^SDL.Window,
}

os_init :: proc() {
	if !SDL.Init({.VIDEO}) {
		fmt.panicf("SDL.Init error: ", SDL.GetError())
	}

	state.os.window = SDL.CreateWindow(
		"WGPU Native Triangle",
		1280,
		720,
		{.RESIZABLE, .HIGH_PIXEL_DENSITY},
	)
	if state.os.window == nil {
		fmt.panicf("SDL.CreateWindow error: ", SDL.GetError())
	}

	// Start with relative mouse (captured); ESC releases, left-click captures again.
	if !SDL.SetWindowRelativeMouseMode(state.os.window, true) {
		fmt.panicf("SDL.SetWindowRelativeMouseMode error: ", SDL.GetError())
	}
}

@(private = "file")
translate_sdl_key :: proc(key: SDL.Keycode) -> (Key, bool) {
	switch key {
	case SDL.K_1:
		return .Num1, true
	case SDL.K_2:
		return .Num2, true
	case SDL.K_3:
		return .Num3, true
	case SDL.K_4:
		return .Num4, true
	case SDL.K_5:
		return .Num5, true
	case SDL.K_6:
		return .Num6, true
	case SDL.K_7:
		return .Num7, true
	case SDL.K_8:
		return .Num8, true
	case SDL.K_9:
		return .Num9, true
	case SDL.K_A:
		return .A, true
	case SDL.K_D:
		return .D, true
	case SDL.K_W:
		return .W, true
	case SDL.K_S:
		return .S, true
	case SDL.K_G:
		return .G, true
	case SDL.K_B:
		return .B, true
	case SDL.K_R:
		return .R, true
	case SDL.K_L:
		return .L, true
	case SDL.K_C:
		return .C, true
	case SDL.K_F:
		return .F, true
	case SDL.K_F1:
		return .F1, true
	case SDL.K_F2:
		return .F2, true
	case SDL.K_F3:
		return .F3, true
	case SDL.K_F4:
		return .F4, true
	case SDL.K_F5:
		return .F5, true
	case SDL.K_F6:
		return .F6, true
	case SDL.K_F7:
		return .F7, true
	case SDL.K_F8:
		return .F8, true
	case SDL.K_F9:
		return .F9, true
	case SDL.K_F10:
		return .F10, true
	case SDL.K_F11:
		return .F11, true
	case SDL.K_F12:
		return .F12, true
	case SDL.K_SPACE:
		return .Space, true
	case SDL.K_LSHIFT:
		return .Shift, true
	case SDL.K_ESCAPE:
		return .Escape, true
	}
	return {}, false
}

// Pauses world simulation when the window is hidden, minimized, or not receiving keyboard focus.
sync_game_process_active :: proc() {
	flags := SDL.GetWindowFlags(state.os.window)
	// NOTE: ` && (.INPUT_FOCUS in flags)` was preventing the window from focusing at launch in the first place.
	state.game_process_active = !(.MINIMIZED in flags || .HIDDEN in flags)
}

os_run :: proc() {
	now := SDL.GetPerformanceCounter()
	last: u64
	dt: f32
	main_loop: for {
		last = now
		now = SDL.GetPerformanceCounter()
		dt = f32(now - last) / f32(SDL.GetPerformanceFrequency())

		state.keys_just_pressed = {}
		state.mouse_delta = {}
		state.mouse_wheel_steps = 0

		e: SDL.Event
		for SDL.PollEvent(&e) {
			#partial switch (e.type) {
			case .KEY_DOWN:
				if e.key.key == SDL.K_ESCAPE {
					if e.key.repeat do continue
					if SDL.GetWindowRelativeMouseMode(state.os.window) {
						if !SDL.SetWindowRelativeMouseMode(state.os.window, false) {
							fmt.panicf("SDL.SetWindowRelativeMouseMode error: ", SDL.GetError())
						}
					} else {
						break main_loop
					}
					continue
				}

				if SDL.GetWindowRelativeMouseMode(state.os.window) {
					if e.key.repeat do continue
					key := translate_sdl_key(e.key.key) or_continue
					state.keys_down += {key}
					state.keys_just_pressed += {key}
				}

			case .KEY_UP:
				if SDL.GetWindowRelativeMouseMode(state.os.window) {
					key := translate_sdl_key(e.key.key) or_continue
					state.keys_down -= {key}
				}

			case .MOUSE_MOTION:
				state.mouse_pos = {f32(e.motion.x), f32(e.motion.y)}
				// Ignore pointer motion for camera look while mouse isn't captured.
				if SDL.GetWindowRelativeMouseMode(state.os.window) {
					state.mouse_delta += {f32(e.motion.xrel), f32(e.motion.yrel)}
				}

			case .MOUSE_BUTTON_DOWN:
				// Re-capture after ESC release; only left button toggles capture.
				switch e.button.button {
				case 0, 1:
					if !SDL.GetWindowRelativeMouseMode(state.os.window) {
						if !SDL.SetWindowRelativeMouseMode(state.os.window, true) {
							fmt.panicf("SDL.SetWindowRelativeMouseMode error: ", SDL.GetError())
						}
						// NOTE: We don't want to propagete the left click if it just activated the relative mouse mode.
						continue
					}
				}
				switch e.button.button {
				// SDL commonly reports 1=left, 2=middle, 3=right.
				// Keep 0 as a compatibility fallback for backends that normalize differently.
				case 0, 1:
					state.keys_down += {.Left_Mouse_Button}
					state.keys_just_pressed += {.Left_Mouse_Button}
				case 3:
					state.keys_down += {.Right_Mouse_Button}
					state.keys_just_pressed += {.Right_Mouse_Button}
				}

			case .MOUSE_BUTTON_UP:
				if SDL.GetWindowRelativeMouseMode(state.os.window) {
					switch e.button.button {
					case 0, 1:
						state.keys_down -= {.Left_Mouse_Button}
					case 3:
						state.keys_down -= {.Right_Mouse_Button}
					}
				}

			case .MOUSE_WHEEL:
				// Match the "only when controlling the camera" behavior of keyboard
				// hotbar selection (relative mouse mode implies the game is active).
				if SDL.GetWindowRelativeMouseMode(state.os.window) {
					ticks: i32
					ticks = e.wheel.integer_y
					if ticks == 0 {
						if e.wheel.y > 0 do ticks = 1
						if e.wheel.y < 0 do ticks = -1
					}
					if ticks != 0 {
						state.mouse_wheel_steps += int(ticks)
					}
				}

			case .QUIT:
				break main_loop

			case .WINDOW_RESIZED, .WINDOW_PIXEL_SIZE_CHANGED:
				resize()
			}
		}

		frame(dt)
	}

	finish()

	SDL.DestroyWindow(state.os.window)
	SDL.Quit()
}


os_get_framebuffer_size :: proc() -> (width, height: u32) {
	w, h: i32
	SDL.GetWindowSizeInPixels(state.os.window, &w, &h)
	return u32(w), u32(h)
}

os_get_surface :: proc(instance: wgpu.Instance) -> wgpu.Surface {
	return sdl3glue.GetSurface(instance, state.os.window)
}
