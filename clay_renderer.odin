package main

import clay "./clay-odin"
import "base:runtime"
import "core:c"
import "core:fmt"
import "core:math"
import "vendor:wgpu"

Clay_Image_Handle :: struct {
	view:        wgpu.TextureView,
	sampler:     wgpu.Sampler,
	use_uv_rect: bool,
	uv_min:      [2]f32,
	uv_max:      [2]f32,
}

Clay_Custom_Render_Proc :: proc(command: ^clay.RenderCommand, pass: wgpu.RenderPassEncoder)

Clay_Vertex :: struct {
	pos:           [2]f32,
	uv:            [2]f32,
	color:         [4]f32,
	use_tex:       f32,
	rect_size:     [2]f32,
	corner_radius: f32,
	// 0..1 across the quad; rounded-rect mask must use this, not uv (atlas images use uv_min..uv_max).
	uv_shape:      [2]f32,
	// Border widths are left/right/top/bottom in mask-space pixels.
	border:        [4]f32,
	border_only:   f32,
}

Clay_Renderer_Screen_Uniform :: struct #align (16) {
	screen_size: [2]f32,
	_pad:        [2]f32,
}

Clay_Scissor_Rect :: struct {
	x, y, w, h: u32,
}

Clay_Draw_Kind :: enum {
	Geometry,
	Custom,
}

Clay_Draw_Item :: struct {
	kind:               Clay_Draw_Kind,
	index_start:        u32,
	index_count:        u32,
	scissor:            Clay_Scissor_Rect,
	bind_group:         wgpu.BindGroup,
	release_bind_group: bool,
	custom_command:     ^clay.RenderCommand,
}

clay_renderer_state: struct {
	initialized:            bool,
	device:                 wgpu.Device,
	queue:                  wgpu.Queue,
	format:                 wgpu.TextureFormat,
	width:                  u32,
	height:                 u32,
	clay_ctx:               ^clay.Context,
	clay_memory:            []u8,
	module:                 wgpu.ShaderModule,
	bind_group_layout:      wgpu.BindGroupLayout,
	pipeline_layout:        wgpu.PipelineLayout,
	pipeline:               wgpu.RenderPipeline,
	uniform_buffer:         wgpu.Buffer,
	white_texture:          wgpu.Texture,
	white_texture_view:     wgpu.TextureView,
	white_sampler:          wgpu.Sampler,
	white_bind_group:       wgpu.BindGroup,
	vertex_buffer:          wgpu.Buffer,
	vertex_buffer_capacity: u64,
	index_buffer:           wgpu.Buffer,
	index_buffer_capacity:  u64,
	custom_render:          Clay_Custom_Render_Proc,
}

clay_renderer_set_custom_render_callback :: proc(callback: Clay_Custom_Render_Proc) {
	clay_renderer_state.custom_render = callback
}

@(private = "file")
clay_error_to_string :: proc(s: clay.String) -> string {
	if s.chars == nil || s.length <= 0 {
		return ""
	}
	return string(s.chars[:s.length])
}

@(private = "file")
clay_error_handler :: proc "c" (error_data: clay.ErrorData) {
	context = runtime.default_context()
	msg := clay_error_to_string(error_data.errorText)
	fmt.panicf("Clay error type=%v text=%v", error_data.errorType, msg)
}

@(private = "file")
clay_measure_text_unimplemented :: proc "c" (
	text: clay.StringSlice,
	config: ^clay.TextElementConfig,
	user_data: rawptr,
) -> clay.Dimensions {
	context = runtime.default_context()
	panic("unimplemented: Clay text measurement is disabled for now")
}

@(private = "file")
clay_color_to_rgba_f32 :: proc(color: clay.Color) -> [4]f32 {
	return {
		f32(color[0]) / 255.0,
		f32(color[1]) / 255.0,
		f32(color[2]) / 255.0,
		f32(color[3]) / 255.0,
	}
}

@(private = "file")
clay_color_is_zero :: proc(color: clay.Color) -> bool {
	return color[0] == 0 && color[1] == 0 && color[2] == 0 && color[3] == 0
}

@(private = "file")
next_pow2_u64_local :: proc(v: u64) -> u64 {
	if v <= 1 {
		return 1
	}
	v := v - 1
	v |= v >> 1
	v |= v >> 2
	v |= v >> 4
	v |= v >> 8
	v |= v >> 16
	v |= v >> 32
	return v + 1
}

@(private = "file")
clay_renderer_update_screen_uniform :: proc() {
	u := Clay_Renderer_Screen_Uniform {
		screen_size = {f32(clay_renderer_state.width), f32(clay_renderer_state.height)},
	}
	wgpu.QueueWriteBuffer(
		clay_renderer_state.queue,
		clay_renderer_state.uniform_buffer,
		0,
		&u,
		uint(size_of(Clay_Renderer_Screen_Uniform)),
	)
}

@(private = "file")
clay_renderer_vertex_buffer_ensure :: proc(min_bytes: u64) {
	if min_bytes <= clay_renderer_state.vertex_buffer_capacity &&
	   clay_renderer_state.vertex_buffer != nil {
		return
	}
	new_capacity := next_pow2_u64_local(max(min_bytes, 4096))
	if clay_renderer_state.vertex_buffer != nil {
		wgpu.BufferRelease(clay_renderer_state.vertex_buffer)
	}
	clay_renderer_state.vertex_buffer = wgpu.DeviceCreateBuffer(
		clay_renderer_state.device,
		&{usage = {.Vertex, .CopyDst}, size = new_capacity},
	)
	clay_renderer_state.vertex_buffer_capacity = new_capacity
}

@(private = "file")
clay_renderer_index_buffer_ensure :: proc(min_bytes: u64) {
	if min_bytes <= clay_renderer_state.index_buffer_capacity &&
	   clay_renderer_state.index_buffer != nil {
		return
	}
	new_capacity := next_pow2_u64_local(max(min_bytes, 4096))
	if clay_renderer_state.index_buffer != nil {
		wgpu.BufferRelease(clay_renderer_state.index_buffer)
	}
	clay_renderer_state.index_buffer = wgpu.DeviceCreateBuffer(
		clay_renderer_state.device,
		&{usage = {.Index, .CopyDst}, size = new_capacity},
	)
	clay_renderer_state.index_buffer_capacity = new_capacity
}

@(private = "file")
clay_renderer_make_image_bind_group :: proc(
	texture_view: wgpu.TextureView,
	sampler: wgpu.Sampler,
) -> (
	wgpu.BindGroup,
	bool,
) {
	if texture_view == nil || sampler == nil {
		return clay_renderer_state.white_bind_group, false
	}
	entries := [3]wgpu.BindGroupEntry {
		{
			binding = 0,
			buffer = clay_renderer_state.uniform_buffer,
			offset = 0,
			size = size_of(Clay_Renderer_Screen_Uniform),
		},
		{binding = 1, textureView = texture_view},
		{binding = 2, sampler = sampler},
	}
	bg := wgpu.DeviceCreateBindGroup(
		clay_renderer_state.device,
		&{
			layout = clay_renderer_state.bind_group_layout,
			entryCount = len(entries),
			entries = &entries[0],
		},
	)
	return bg, true
}

@(private = "file")
append_quad :: proc(
	vertices: ^[dynamic]Clay_Vertex,
	indices: ^[dynamic]u32,
	x, y, w, h: f32,
	color: [4]f32,
	uv_min, uv_max: [2]f32,
	use_tex: f32,
	corner_radius: f32,
	mask_size: [2]f32,
	uv_shape_min, uv_shape_max: [2]f32,
	border: [4]f32,
	border_only: f32,
) -> (
	u32,
	bool,
) {
	if w <= 0 || h <= 0 {
		return 0, false
	}
	base := u32(len(vertices^))
	append(
		vertices,
		Clay_Vertex {
			pos = {x, y},
			uv = {uv_min[0], uv_min[1]},
			color = color,
			use_tex = use_tex,
			rect_size = mask_size,
			corner_radius = corner_radius,
			uv_shape = {uv_shape_min[0], uv_shape_min[1]},
			border = border,
			border_only = border_only,
		},
		Clay_Vertex {
			pos = {x + w, y},
			uv = {uv_max[0], uv_min[1]},
			color = color,
			use_tex = use_tex,
			rect_size = mask_size,
			corner_radius = corner_radius,
			uv_shape = {uv_shape_max[0], uv_shape_min[1]},
			border = border,
			border_only = border_only,
		},
		Clay_Vertex {
			pos = {x + w, y + h},
			uv = {uv_max[0], uv_max[1]},
			color = color,
			use_tex = use_tex,
			rect_size = mask_size,
			corner_radius = corner_radius,
			uv_shape = {uv_shape_max[0], uv_shape_max[1]},
			border = border,
			border_only = border_only,
		},
		Clay_Vertex {
			pos = {x, y + h},
			uv = {uv_min[0], uv_max[1]},
			color = color,
			use_tex = use_tex,
			rect_size = mask_size,
			corner_radius = corner_radius,
			uv_shape = {uv_shape_min[0], uv_shape_max[1]},
			border = border,
			border_only = border_only,
		},
	)
	append(indices, base + 0, base + 1, base + 2, base + 0, base + 2, base + 3)
	return u32(len(indices^) - 6), true
}

@(private = "file")
clay_scissor_intersect :: proc(a, b: Clay_Scissor_Rect) -> Clay_Scissor_Rect {
	ax1 := i32(a.x) + i32(a.w)
	ay1 := i32(a.y) + i32(a.h)
	bx1 := i32(b.x) + i32(b.w)
	by1 := i32(b.y) + i32(b.h)
	x0 := max(i32(a.x), i32(b.x))
	y0 := max(i32(a.y), i32(b.y))
	x1 := min(ax1, bx1)
	y1 := min(ay1, by1)
	if x1 <= x0 || y1 <= y0 {
		return {}
	}
	return {x = u32(x0), y = u32(y0), w = u32(x1 - x0), h = u32(y1 - y0)}
}

@(private = "file")
clay_bounds_to_scissor :: proc(bounds: clay.BoundingBox) -> (Clay_Scissor_Rect, bool) {
	x0 := i32(math.floor(f64(bounds.x)))
	y0 := i32(math.floor(f64(bounds.y)))
	x1 := i32(math.ceil(f64(bounds.x + bounds.width)))
	y1 := i32(math.ceil(f64(bounds.y + bounds.height)))
	x0 = clamp(x0, 0, i32(clay_renderer_state.width))
	y0 = clamp(y0, 0, i32(clay_renderer_state.height))
	x1 = clamp(x1, 0, i32(clay_renderer_state.width))
	y1 = clamp(y1, 0, i32(clay_renderer_state.height))
	if x1 <= x0 || y1 <= y0 {
		return {}, false
	}
	return {x = u32(x0), y = u32(y0), w = u32(x1 - x0), h = u32(y1 - y0)}, true
}

@(private = "file")
draw_item_add_rectangle :: proc(
	vertices: ^[dynamic]Clay_Vertex,
	indices: ^[dynamic]u32,
	draw_items: ^[dynamic]Clay_Draw_Item,
	bounds: clay.BoundingBox,
	color: [4]f32,
	scissor: Clay_Scissor_Rect,
	bind_group: wgpu.BindGroup,
	release_bind_group: bool,
	uv_min, uv_max: [2]f32,
	use_tex: f32,
	corner_radius: f32,
	mask_size: [2]f32,
	uv_shape_min, uv_shape_max: [2]f32,
	border: [4]f32,
	border_only: f32,
) {
	index_start, ok := append_quad(
		vertices,
		indices,
		f32(bounds.x),
		f32(bounds.y),
		f32(bounds.width),
		f32(bounds.height),
		color,
		uv_min,
		uv_max,
		use_tex,
		corner_radius,
		mask_size,
		uv_shape_min,
		uv_shape_max,
		border,
		border_only,
	)
	if !ok {
		if release_bind_group && bind_group != nil {
			wgpu.BindGroupRelease(bind_group)
		}
		return
	}
	append(
		draw_items,
		Clay_Draw_Item {
			kind = .Geometry,
			index_start = index_start,
			index_count = 6,
			scissor = scissor,
			bind_group = bind_group,
			release_bind_group = release_bind_group,
		},
	)
}

@(private = "file")
clay_corner_radius_uniform :: proc(
	bounds: clay.BoundingBox,
	corner_radius: clay.CornerRadius,
) -> f32 {
	max_corner := max(
		max(f32(corner_radius.topLeft), f32(corner_radius.topRight)),
		max(f32(corner_radius.bottomLeft), f32(corner_radius.bottomRight)),
	)
	max_by_rect := min(f32(bounds.width), f32(bounds.height)) * 0.5
	return clamp(max_corner, 0, max_by_rect)
}

clay_renderer_init :: proc(
	device: wgpu.Device,
	queue: wgpu.Queue,
	surface_format: wgpu.TextureFormat,
	width, height: u32,
) {
	if clay_renderer_state.initialized {
		return
	}
	clay_renderer_state.device = device
	clay_renderer_state.queue = queue
	clay_renderer_state.format = surface_format
	clay_renderer_state.width = width
	clay_renderer_state.height = height

	clay_memory_size := int(clay.MinMemorySize())
	clay_renderer_state.clay_memory = make([]u8, clay_memory_size, context.allocator)
	arena := clay.CreateArenaWithCapacityAndMemory(
		c.size_t(len(clay_renderer_state.clay_memory)),
		([^]u8)(raw_data(clay_renderer_state.clay_memory[:])),
	)
	clay_renderer_state.clay_ctx = clay.Initialize(
		arena,
		{width = c.float(width), height = c.float(height)},
		{handler = clay_error_handler},
	)
	clay.SetCurrentContext(clay_renderer_state.clay_ctx)
	clay.SetMeasureTextFunction(clay_measure_text_unimplemented, nil)

	shader_source := #load("clay_renderer.wgsl", string)
	clay_renderer_state.module = wgpu.DeviceCreateShaderModule(
		device,
		&{nextInChain = &wgpu.ShaderSourceWGSL{sType = .ShaderSourceWGSL, code = shader_source}},
	)

	layout_entries := [3]wgpu.BindGroupLayoutEntry {
		{
			binding = 0,
			visibility = {.Vertex},
			buffer = {
				type = .Uniform,
				hasDynamicOffset = false,
				minBindingSize = size_of(Clay_Renderer_Screen_Uniform),
			},
		},
		{
			binding = 1,
			visibility = {.Fragment},
			texture = {sampleType = .Float, viewDimension = ._2D, multisampled = false},
		},
		{binding = 2, visibility = {.Fragment}, sampler = {type = .Filtering}},
	}
	clay_renderer_state.bind_group_layout = wgpu.DeviceCreateBindGroupLayout(
		device,
		&{entryCount = len(layout_entries), entries = &layout_entries[0]},
	)
	clay_renderer_state.pipeline_layout = wgpu.DeviceCreatePipelineLayout(
		device,
		&{bindGroupLayoutCount = 1, bindGroupLayouts = &clay_renderer_state.bind_group_layout},
	)

	clay_renderer_state.uniform_buffer = wgpu.DeviceCreateBuffer(
		device,
		&{usage = {.Uniform, .CopyDst}, size = size_of(Clay_Renderer_Screen_Uniform)},
	)

	clay_renderer_state.white_texture = wgpu.DeviceCreateTexture(
		device,
		&{
			label = "clay_white_texture",
			size = {width = 1, height = 1, depthOrArrayLayers = 1},
			mipLevelCount = 1,
			sampleCount = 1,
			dimension = ._2D,
			format = .RGBA8Unorm,
			usage = {.TextureBinding, .CopyDst},
		},
	)
	clay_renderer_state.white_texture_view = wgpu.TextureCreateView(
		clay_renderer_state.white_texture,
		nil,
	)
	clay_renderer_state.white_sampler = wgpu.DeviceCreateSampler(
		device,
		&{
			addressModeU = .ClampToEdge,
			addressModeV = .ClampToEdge,
			addressModeW = .ClampToEdge,
			magFilter = .Linear,
			minFilter = .Linear,
			mipmapFilter = .Nearest,
			lodMinClamp = 0,
			lodMaxClamp = 0,
			maxAnisotropy = 1,
		},
	)
	white_pixel := [4]u8{255, 255, 255, 255}
	wgpu.QueueWriteTexture(
		queue,
		&{
			texture = clay_renderer_state.white_texture,
			mipLevel = 0,
			origin = {x = 0, y = 0, z = 0},
			aspect = .All,
		},
		raw_data(white_pixel[:]),
		uint(len(white_pixel)),
		&{offset = 0, bytesPerRow = 4, rowsPerImage = 1},
		&{width = 1, height = 1, depthOrArrayLayers = 1},
	)

	white_entries := [3]wgpu.BindGroupEntry {
		{
			binding = 0,
			buffer = clay_renderer_state.uniform_buffer,
			offset = 0,
			size = size_of(Clay_Renderer_Screen_Uniform),
		},
		{binding = 1, textureView = clay_renderer_state.white_texture_view},
		{binding = 2, sampler = clay_renderer_state.white_sampler},
	}
	clay_renderer_state.white_bind_group = wgpu.DeviceCreateBindGroup(
		device,
		&{
			layout = clay_renderer_state.bind_group_layout,
			entryCount = len(white_entries),
			entries = &white_entries[0],
		},
	)

	vertex_attrs := [9]wgpu.VertexAttribute {
		{format = .Float32x2, offset = 0, shaderLocation = 0},
		{format = .Float32x2, offset = 8, shaderLocation = 1},
		{format = .Float32x4, offset = 16, shaderLocation = 2},
		{format = .Float32, offset = 32, shaderLocation = 3},
		{format = .Float32x2, offset = 36, shaderLocation = 4},
		{format = .Float32, offset = 44, shaderLocation = 5},
		{format = .Float32x2, offset = 48, shaderLocation = 6},
		{format = .Float32x4, offset = 56, shaderLocation = 7},
		{format = .Float32, offset = 72, shaderLocation = 8},
	}
	vertex_layout := wgpu.VertexBufferLayout {
		stepMode       = .Vertex,
		arrayStride    = size_of(Clay_Vertex),
		attributeCount = len(vertex_attrs),
		attributes     = &vertex_attrs[0],
	}
	blend_alpha := wgpu.BlendState {
		color = {operation = .Add, srcFactor = .SrcAlpha, dstFactor = .OneMinusSrcAlpha},
		alpha = {operation = .Add, srcFactor = .One, dstFactor = .OneMinusSrcAlpha},
	}
	clay_renderer_state.pipeline = wgpu.DeviceCreateRenderPipeline(
		device,
		&{
			layout = clay_renderer_state.pipeline_layout,
			vertex = {
				module = clay_renderer_state.module,
				entryPoint = "vs_main",
				bufferCount = 1,
				buffers = &vertex_layout,
			},
			fragment = &{
				module = clay_renderer_state.module,
				entryPoint = "fs_main",
				targetCount = 1,
				targets = &wgpu.ColorTargetState {
					format = surface_format,
					blend = &blend_alpha,
					writeMask = wgpu.ColorWriteMaskFlags_All,
				},
			},
			primitive = {
				topology = .TriangleList,
				stripIndexFormat = .Undefined,
				frontFace = .CCW,
				cullMode = .None,
			},
			multisample = {count = 1, mask = 0xFFFFFFFF},
		},
	)

	clay_renderer_vertex_buffer_ensure(4096)
	clay_renderer_index_buffer_ensure(4096)
	clay_renderer_update_screen_uniform()
	clay_renderer_state.initialized = true
}

clay_renderer_resize :: proc(width, height: u32) {
	if !clay_renderer_state.initialized {
		return
	}
	clay_renderer_state.width = width
	clay_renderer_state.height = height
	clay.SetCurrentContext(clay_renderer_state.clay_ctx)
	clay.SetLayoutDimensions({width = c.float(width), height = c.float(height)})
	clay_renderer_update_screen_uniform()
}

clay_renderer_begin_frame :: proc(dt: f32, mouse_pos: [2]f32, mouse_down: bool) {
	if !clay_renderer_state.initialized {
		return
	}
	clay.SetCurrentContext(clay_renderer_state.clay_ctx)
	clay.SetPointerState({c.float(mouse_pos[0]), c.float(mouse_pos[1])}, mouse_down)
	clay.UpdateScrollContainers(true, {0, 0}, c.float(dt))
	clay.BeginLayout()
}

clay_renderer_end_frame_and_record :: proc(
	encoder: wgpu.CommandEncoder,
	target_view: wgpu.TextureView,
	allocator := context.temp_allocator,
) {
	if !clay_renderer_state.initialized {
		return
	}
	clay.SetCurrentContext(clay_renderer_state.clay_ctx)
	render_commands := clay.EndLayout()

	vertices := make([dynamic]Clay_Vertex, 0, render_commands.length * 4, allocator)
	indices := make([dynamic]u32, 0, render_commands.length * 6, allocator)
	draw_items := make([dynamic]Clay_Draw_Item, 0, render_commands.length, allocator)

	full_scissor := Clay_Scissor_Rect {
		x = 0,
		y = 0,
		w = clay_renderer_state.width,
		h = clay_renderer_state.height,
	}
	scissor_stack := make([dynamic]Clay_Scissor_Rect, 0, 8, allocator)
	append(&scissor_stack, full_scissor)

	current_scissor := full_scissor
	for i in 0 ..< render_commands.length {
		command := clay.RenderCommandArray_Get(&render_commands, i)
		bounds := command.boundingBox

		switch command.commandType {
		case .None:
		// Skip

		case .ScissorStart:
			scissor_rect, ok := clay_bounds_to_scissor(bounds)
			if !ok {
				current_scissor = {}
				append(&scissor_stack, current_scissor)
				continue
			}
			current_scissor = clay_scissor_intersect(current_scissor, scissor_rect)
			append(&scissor_stack, current_scissor)

		case .ScissorEnd:
			if len(scissor_stack) > 1 {
				_ = pop(&scissor_stack)
			}
			current_scissor = scissor_stack[len(scissor_stack) - 1]

		case .Rectangle:
			config := command.renderData.rectangle
			draw_item_add_rectangle(
				&vertices,
				&indices,
				&draw_items,
				bounds,
				clay_color_to_rgba_f32(config.backgroundColor),
				current_scissor,
				clay_renderer_state.white_bind_group,
				false,
				{0, 0},
				{1, 1},
				0,
				clay_corner_radius_uniform(bounds, config.cornerRadius),
				{f32(bounds.width), f32(bounds.height)},
				{0, 0},
				{1, 1},
				{0, 0, 0, 0},
				0,
			)

		case .Border:
			config := command.renderData.border
			color := clay_color_to_rgba_f32(config.color)
			left_w := f32(config.width.left)
			right_w := f32(config.width.right)
			top_w := f32(config.width.top)
			bottom_w := f32(config.width.bottom)
			outer_w := f32(bounds.width)
			outer_h := f32(bounds.height)
			if outer_w <= 0 || outer_h <= 0 {
				continue
			}
			corner_radius := clay_corner_radius_uniform(bounds, config.cornerRadius)
			// A single quad with border-only mask allows large radii to form true rings/circles.
			draw_item_add_rectangle(
				&vertices,
				&indices,
				&draw_items,
				bounds,
				color,
				current_scissor,
				clay_renderer_state.white_bind_group,
				false,
				{0, 0},
				{1, 1},
				0,
				corner_radius,
				{outer_w, outer_h},
				{0, 0},
				{1, 1},
				{left_w, right_w, top_w, bottom_w},
				1,
			)

		case .Image:
			config := command.renderData.image
			image_handle := (^Clay_Image_Handle)(config.imageData)
			texture_view := clay_renderer_state.white_texture_view
			sampler := clay_renderer_state.white_sampler
			if image_handle != nil && image_handle.view != nil && image_handle.sampler != nil {
				texture_view = image_handle.view
				sampler = image_handle.sampler
			}
			bind_group, release_bind_group := clay_renderer_make_image_bind_group(
				texture_view,
				sampler,
			)
			uv_min := [2]f32{0, 0}
			uv_max := [2]f32{1, 1}
			if image_handle != nil && image_handle.use_uv_rect {
				uv_min = image_handle.uv_min
				uv_max = image_handle.uv_max
			}
			color := [4]f32{1, 1, 1, 1}
			if !clay_color_is_zero(config.backgroundColor) {
				color = clay_color_to_rgba_f32(config.backgroundColor)
			}
			draw_item_add_rectangle(
				&vertices,
				&indices,
				&draw_items,
				bounds,
				color,
				current_scissor,
				bind_group,
				release_bind_group,
				uv_min,
				uv_max,
				1,
				clay_corner_radius_uniform(bounds, config.cornerRadius),
				{f32(bounds.width), f32(bounds.height)},
				{0, 0},
				{1, 1},
				{0, 0, 0, 0},
				0,
			)

		case .Text:
			panic("unimplemented: Clay text rendering is disabled for now")

		case .Custom:
			if clay_renderer_state.custom_render != nil {
				append(
					&draw_items,
					Clay_Draw_Item {
						kind = .Custom,
						scissor = current_scissor,
						custom_command = command,
					},
				)
			}
		}
	}

	if len(draw_items) == 0 {
		return
	}

	vertex_bytes := u64(len(vertices)) * u64(size_of(Clay_Vertex))
	index_bytes := u64(len(indices)) * u64(size_of(u32))

	if len(vertices) > 0 {
		clay_renderer_vertex_buffer_ensure(vertex_bytes)
		clay_renderer_index_buffer_ensure(index_bytes)
		wgpu.QueueWriteBuffer(
			clay_renderer_state.queue,
			clay_renderer_state.vertex_buffer,
			0,
			raw_data(vertices[:]),
			uint(vertex_bytes),
		)
		wgpu.QueueWriteBuffer(
			clay_renderer_state.queue,
			clay_renderer_state.index_buffer,
			0,
			raw_data(indices[:]),
			uint(index_bytes),
		)
	}

	pass := wgpu.CommandEncoderBeginRenderPass(
		encoder,
		&{
			colorAttachmentCount = 1,
			colorAttachments = &wgpu.RenderPassColorAttachment {
				view = target_view,
				loadOp = .Load,
				storeOp = .Store,
				depthSlice = wgpu.DEPTH_SLICE_UNDEFINED,
			},
		},
	)
	wgpu.RenderPassEncoderSetPipeline(pass, clay_renderer_state.pipeline)
	if len(vertices) > 0 {
		wgpu.RenderPassEncoderSetVertexBuffer(
			pass,
			0,
			clay_renderer_state.vertex_buffer,
			0,
			vertex_bytes,
		)
		wgpu.RenderPassEncoderSetIndexBuffer(
			pass,
			clay_renderer_state.index_buffer,
			.Uint32,
			0,
			index_bytes,
		)
	}

	for item in draw_items {
		wgpu.RenderPassEncoderSetScissorRect(
			pass,
			item.scissor.x,
			item.scissor.y,
			item.scissor.w,
			item.scissor.h,
		)
		switch item.kind {
		case .Geometry:
			wgpu.RenderPassEncoderSetBindGroup(pass, 0, item.bind_group)
			wgpu.RenderPassEncoderDrawIndexed(pass, item.index_count, 1, item.index_start, 0, 0)
			if item.release_bind_group && item.bind_group != nil {
				wgpu.BindGroupRelease(item.bind_group)
			}
		case .Custom:
			clay_renderer_state.custom_render(item.custom_command, pass)
		}
	}

	wgpu.RenderPassEncoderEnd(pass)
	wgpu.RenderPassEncoderRelease(pass)
}

clay_renderer_destroy :: proc() {
	if !clay_renderer_state.initialized {
		return
	}

	if clay_renderer_state.index_buffer != nil {
		wgpu.BufferRelease(clay_renderer_state.index_buffer)
		clay_renderer_state.index_buffer = nil
	}
	if clay_renderer_state.vertex_buffer != nil {
		wgpu.BufferRelease(clay_renderer_state.vertex_buffer)
		clay_renderer_state.vertex_buffer = nil
	}
	if clay_renderer_state.white_bind_group != nil {
		wgpu.BindGroupRelease(clay_renderer_state.white_bind_group)
		clay_renderer_state.white_bind_group = nil
	}
	if clay_renderer_state.white_sampler != nil {
		wgpu.SamplerRelease(clay_renderer_state.white_sampler)
		clay_renderer_state.white_sampler = nil
	}
	if clay_renderer_state.white_texture_view != nil {
		wgpu.TextureViewRelease(clay_renderer_state.white_texture_view)
		clay_renderer_state.white_texture_view = nil
	}
	if clay_renderer_state.white_texture != nil {
		wgpu.TextureRelease(clay_renderer_state.white_texture)
		clay_renderer_state.white_texture = nil
	}
	if clay_renderer_state.uniform_buffer != nil {
		wgpu.BufferRelease(clay_renderer_state.uniform_buffer)
		clay_renderer_state.uniform_buffer = nil
	}
	if clay_renderer_state.pipeline != nil {
		wgpu.RenderPipelineRelease(clay_renderer_state.pipeline)
		clay_renderer_state.pipeline = nil
	}
	if clay_renderer_state.pipeline_layout != nil {
		wgpu.PipelineLayoutRelease(clay_renderer_state.pipeline_layout)
		clay_renderer_state.pipeline_layout = nil
	}
	if clay_renderer_state.bind_group_layout != nil {
		wgpu.BindGroupLayoutRelease(clay_renderer_state.bind_group_layout)
		clay_renderer_state.bind_group_layout = nil
	}
	if clay_renderer_state.module != nil {
		wgpu.ShaderModuleRelease(clay_renderer_state.module)
		clay_renderer_state.module = nil
	}
	if clay_renderer_state.clay_memory != nil {
		delete(clay_renderer_state.clay_memory, context.allocator)
		clay_renderer_state.clay_memory = nil
	}

	clay_renderer_state = {}
}
