package main

import clay "./clay-odin"
import "base:runtime"
import "core:fmt"
import "core:image"
import _ "core:image/png"
import "core:log"
import "core:math"
import "core:math/linalg"
import "core:mem"
import "vendor:wgpu"

FOG_NEAR_FACTOR :: 0.4

Key :: enum u8 {
	Left_Mouse_Button,
	Right_Mouse_Button,
	Num1,
	Num2,
	Num3,
	Num4,
	Num5,
	Num6,
	Num7,
	Num8,
	Num9,
	W,
	A,
	S,
	D,
	G,
	B,
	R,
	L,
	C,
	F,
	F1,
	F2,
	F3,
	F4,
	F5,
	F6,
	F7,
	F8,
	F9,
	F10,
	F11,
	F12,
	Shift,
	Space,
	Escape,
}

WORLD_RIGHT :: [3]f32{1, 0, 0}
WORLD_UP :: [3]f32{0, 1, 0}
WORLD_FORWARD :: [3]f32{0, 0, 1}

SKY_CLEAR_COLOR :: wgpu.Color{0.12, 0.14, 0.22, 1}

RAIN_STREAK_COUNT :: 12288

Shared_Frame_Uniform :: struct #align (16) {
	mvp:          matrix[4, 4]f32,
	camera_pos:   [3]f32,
	elapsed_time: f32,
}

Shared_Fog_Uniform :: struct #align (16) {
	fog_color_near: [4]f32,
	fog_far:        f32,
	fog_enabled:    b32,
	_:              [2]f32,
}

Scene_Uniform :: struct #align (16) {
	reflection_view_proj: matrix[4, 4]f32,
	reflection_plane_y:   f32,
	render_mode:          f32,
}

Cloud_Uniform :: struct #align (16) {
	// .x = min world Y, .y = max world Y across active cloud slices (vertical fade in cloud.wgsl).
	cloud_y_bounds:    [2]f32,
	// Layer world Y packed low to high (instances 0..N-1); fixed order keeps alpha stacking stable with pitch.
	cloud_sorted_y_lo: [4]f32,
	cloud_sorted_y_hi: [4]f32,
}

Chunk_Bounds_Vertex :: struct {
	pos:   [3]f32,
	_:     f32,
	color: [4]f32,
}

Transparent_Chunk_Draw :: struct {
	vertex_buffer: wgpu.Buffer,
	index_buffer:  wgpu.Buffer,
	vertex_bytes:  u64,
	index_bytes:   u64,
	index_count:   u32,
	dist2:         f32,
}

Post_Uniform :: struct #align (16) {
	camera_elapsed:      [4]f32,
	underwater_strength: f32,
	bloom_strength:      f32,
	resolution:          [2]f32,
	inv_view_proj:       matrix[4, 4]f32,
	fxaa_enabled:        f32,
	_:                   [3]f32,
	dof_start:           f32,
	dof_end:             f32,
	dof_strength:        f32,
	_:                   f32,
}

POST_UNIFORM_SIZE :: size_of(Post_Uniform)

// Must match BloomUniform in bloom.wgsl.
Bloom_Uniform :: struct #align (16) {
	threshold_knee: [4]f32,
	texel_dir:      [4]f32,
}

BLOOM_UNIFORM_SIZE :: size_of(Bloom_Uniform)

// Tuned for LDR scene (lit terrain rarely exceeds ~0.5-0.8); high thresholds yield invisible bloom.
BLOOM_THRESHOLD :: 0.12
BLOOM_KNEE :: 0.75
BLOOM_STRENGTH :: 0.3
DOF_START_DISTANCE :: 30.0
DOF_END_DISTANCE :: DOF_START_DISTANCE + 10.0
DOF_STRENGTH :: 1.25
// Underwater: narrow depth of field near the camera (murky water / eye accommodation).
DOF_UNDERWATER_START_DISTANCE :: 1.0
DOF_UNDERWATER_END_DISTANCE :: DOF_UNDERWATER_START_DISTANCE + 2.0
DOF_UNDERWATER_STRENGTH :: 5

// More slices = thicker stacks; each slice is cheap (single fBm in cloud.wgsl).
CLOUD_LAYER_COUNT :: 5
CAMERA_FOV_DEG :: 70.0
// Tighter clip range reduces depth precision artifacts that show up as edge leaks when very close to blocks.
// Keep far high enough for current cloud quad extent and streamed terrain radius.
CAMERA_NEAR_PLANE :: 0.01
CAMERA_FAR_PLANE :: 500.0

// Horizontal slices follow the camera in XZ. Layer Y is fixed ascending world space (see offsets).
sort_cloud_layers :: proc(globals: ^Cloud_Uniform) {
	cloud_layer_y_offsets := [CLOUD_LAYER_COUNT]f32{14.3, 21.6, 28.5, 36.1, 44.2} - 10
	cloud_base_y := f32(SEA_LEVEL)
	ys: [CLOUD_LAYER_COUNT]f32
	for i in 0 ..< CLOUD_LAYER_COUNT {
		ys[i] = cloud_base_y + cloud_layer_y_offsets[i]
	}
	min_y := ys[0]
	max_y := ys[0]
	for i in 1 ..< CLOUD_LAYER_COUNT {
		min_y = min(min_y, ys[i])
		max_y = max(max_y, ys[i])
	}
	globals.cloud_y_bounds[0] = min_y
	globals.cloud_y_bounds[1] = max_y
	for i in 0 ..< CLOUD_LAYER_COUNT {
		y := ys[i]
		if i < 4 {
			globals.cloud_sorted_y_lo[i] = y
		} else {
			globals.cloud_sorted_y_hi[i - 4] = y
		}
	}
}

transparent_chunk_dist2_from_camera :: proc(camera_pos: [3]f32, chunk_pos: Chunk_Coords) -> f32 {
	center_x := f32(chunk_pos.x) * CHUNK_WIDTH + f32(CHUNK_WIDTH) * 0.5
	center_y := f32(CHUNK_HEIGHT) * 0.5
	center_z := f32(chunk_pos.y) * CHUNK_WIDTH + f32(CHUNK_WIDTH) * 0.5
	dx := center_x - camera_pos.x
	dy := center_y - camera_pos.y
	dz := center_z - camera_pos.z
	return dx * dx + dy * dy + dz * dz
}

sort_transparent_chunk_draws_back_to_front :: proc(draws: ^[dynamic]Transparent_Chunk_Draw) {
	// Transparent blend with depth-write off requires far-to-near submission.
	for i in 1 ..< len(draws^) {
		key_item := draws^[i]
		j := i - 1
		for j >= 0 && draws^[j].dist2 < key_item.dist2 {
			draws^[j + 1] = draws^[j]
			j -= 1
		}
		draws^[j + 1] = key_item
	}
}
GHOST_PREVIEW_MATERIAL_MARKER :: -1
// Full opaque cube: 6 faces * 4 verts * 11 f32 (voxel_preview_append_face).
GHOST_PREVIEW_VERTEX_BUFFER_MAX_BYTES :: u64(6 * 4 * 11 * size_of(f32))
// Two triangles per face, 6 indices each.
GHOST_PREVIEW_INDEX_BUFFER_MAX_BYTES :: u64(6 * 6 * size_of(u32))
ARMED_TNT_MATERIAL_MARKER :: -2

Terrain_Batch_Uniform :: struct #align (16) {
	chunk_count: u32,
	noise_scale: f32,
	seed:        f32,
	_:           u32,
}

TERRAIN_CHUNK_VOXEL_BYTES :: CHUNK_VOXEL_COUNT * size_of(Cube_Kind)
TERRAIN_MAX_BATCH_CHUNKS :: 512
TERRAIN_CHUNK_PARAMS_BUFFER_SIZE :: u64(TERRAIN_MAX_BATCH_CHUNKS) * u64(size_of([2]i32))
TERRAIN_BATCH_VOXEL_BUFFER_SIZE :: u64(TERRAIN_MAX_BATCH_CHUNKS) * TERRAIN_CHUNK_VOXEL_BYTES

Terrain_Gen_Phase :: enum u8 {
	Idle,
	Waiting_Work_Done,
	Waiting_Map,
}

state: struct {
	ctx:                            runtime.Context,
	os:                             OS,
	instance:                       wgpu.Instance,
	surface:                        wgpu.Surface,
	adapter:                        wgpu.Adapter,
	device:                         wgpu.Device,
	config:                         wgpu.SurfaceConfiguration,
	queue:                          wgpu.Queue,
	module:                         wgpu.ShaderModule,
	bind_group_layout:              wgpu.BindGroupLayout,
	pipeline_layout:                wgpu.PipelineLayout,
	pipeline:                       wgpu.RenderPipeline,
	reflection_pipeline:            wgpu.RenderPipeline,
	pipeline_flower_cutout:         wgpu.RenderPipeline,
	pipeline_transparent:           wgpu.RenderPipeline,
	line_module:                    wgpu.ShaderModule,
	line_pipeline:                  wgpu.RenderPipeline,
	rain_module:                    wgpu.ShaderModule,
	rain_pipeline:                  wgpu.RenderPipeline,
	cloud_module:                   wgpu.ShaderModule,
	cloud_pipeline:                 wgpu.RenderPipeline,
	shared_frame_uniform_buffer:    wgpu.Buffer,
	shared_fog_uniform_buffer:      wgpu.Buffer,
	scene_uniform_buffer:           wgpu.Buffer,
	cloud_uniform_buffer:           wgpu.Buffer,
	shared_frame_bind_group_layout: wgpu.BindGroupLayout,
	shared_frame_pipeline_layout:   wgpu.PipelineLayout,
	rain_bind_group_layout:         wgpu.BindGroupLayout,
	cloud_bind_group_layout:        wgpu.BindGroupLayout,
	line_bind_group_layout:         wgpu.BindGroupLayout,
	rain_pipeline_layout:           wgpu.PipelineLayout,
	cloud_pipeline_layout:          wgpu.PipelineLayout,
	line_pipeline_layout:           wgpu.PipelineLayout,
	ghost_preview_vertex_buffer:    wgpu.Buffer,
	ghost_preview_index_buffer:     wgpu.Buffer,
	chunk_bounds_vertex_buffer:     wgpu.Buffer,
	chunk_bounds_vertex_capacity:   u64,
	atlas_texture:                  wgpu.Texture,
	atlas_view:                     wgpu.TextureView,
	atlas_sampler:                  wgpu.Sampler,
	reflection_texture:             wgpu.Texture,
	reflection_view:                wgpu.TextureView,
	reflection_sampler:             wgpu.Sampler,
	reflection_depth_texture:       wgpu.Texture,
	reflection_depth_view:          wgpu.TextureView,
	bind_group:                     wgpu.BindGroup,
	bind_group_reflection_only:     wgpu.BindGroup,
	rain_bind_group:                wgpu.BindGroup,
	cloud_bind_group:               wgpu.BindGroup,
	line_bind_group:                wgpu.BindGroup,
	depth_texture:                  wgpu.Texture,
	depth_view:                     wgpu.TextureView,
	depth_sample_sampler:           wgpu.Sampler,
	scene_color_texture:            wgpu.Texture,
	scene_color_view:               wgpu.TextureView,
	post_composite_texture:         wgpu.Texture,
	post_composite_view:            wgpu.TextureView,
	post_bind_group_layout:         wgpu.BindGroupLayout,
	post_pipeline_layout:           wgpu.PipelineLayout,
	post_effects_module:            wgpu.ShaderModule,
	post_pipeline:                  wgpu.RenderPipeline,
	post_uniform_buffer:            wgpu.Buffer,
	post_bind_group:                wgpu.BindGroup,
	bloom_texture_a:                wgpu.Texture,
	bloom_view_a:                   wgpu.TextureView,
	bloom_texture_b:                wgpu.Texture,
	bloom_view_b:                   wgpu.TextureView,
	bloom_sampler:                  wgpu.Sampler,
	bloom_uniform_buffer:           wgpu.Buffer,
	bloom_bind_group_layout:        wgpu.BindGroupLayout,
	bloom_pipeline_layout:          wgpu.PipelineLayout,
	bloom_module:                   wgpu.ShaderModule,
	bloom_extract_pipeline:         wgpu.RenderPipeline,
	bloom_blur_pipeline:            wgpu.RenderPipeline,
	bloom_extract_bind_group:       wgpu.BindGroup,
	bloom_blur_bind_group_a:        wgpu.BindGroup,
	bloom_blur_bind_group_b:        wgpu.BindGroup,
	dof_texture_a:                  wgpu.Texture,
	dof_view_a:                     wgpu.TextureView,
	dof_texture_b:                  wgpu.Texture,
	dof_view_b:                     wgpu.TextureView,
	dof_extract_pipeline:           wgpu.RenderPipeline,
	dof_extract_bind_group:         wgpu.BindGroup,
	dof_blur_bind_group_a:          wgpu.BindGroup,
	dof_blur_bind_group_b:          wgpu.BindGroup,
	dof_module:                     wgpu.ShaderModule,
	dof_final_bind_group_layout:    wgpu.BindGroupLayout,
	dof_final_pipeline_layout:      wgpu.PipelineLayout,
	dof_final_pipeline:             wgpu.RenderPipeline,
	dof_final_bind_group:           wgpu.BindGroup,
	chunks:                         map[Chunk_Coords]Chunk,
	player_pos:                     [3]f32,
	player_vel:                     [3]f32,
	player_rotation:                quaternion128,
	player_pitch:                   f32,
	player_noclip:                  bool,
	player_on_ground:               bool,
	debug_chunk_bounds:             bool,
	rain_enabled:                   bool,
	bloom_enabled:                  bool,
	dof_enabled:                    bool,
	clouds_enabled:                 bool,
	fog_enabled:                    bool,
	fxaa_enabled:                   bool,
	water_fluid_queue:              [dynamic]Water_Fluid_Key,
	water_fluid_seen:               map[Water_Fluid_Key]struct{},
	block_change_queue:             [dynamic]Block_Change,
	pending_remesh_chunks:          [dynamic]Chunk_Coords,
	pending_remesh_seen:            map[Chunk_Coords]struct{},
	light_increase_queue:           [dynamic]Light_Key,
	light_decrease_queue:           [dynamic]Light_Key,
	light_increase_seen:            map[Light_Key]struct{},
	light_decrease_seen:            map[Light_Key]struct{},
	water_fluid_wave_timer:         f32,
	elapsed_time:                   f32,
	// When false, skip simulation and chunk streaming (minimized tab, unfocused window, etc.).
	game_process_active:            bool,
	keys_down:                      bit_set[Key],
	keys_just_pressed:              bit_set[Key],
	mouse_pos:                      [2]f32,
	mouse_delta:                    [2]f32,
	// Signed wheel steps accumulated since last frame; positive means "scroll down".
	mouse_wheel_steps:              int,
	hotbar_slot_kinds:              [9]Cube_Kind,
	hotbar_selected_slot:           int,
	ghost_preview_world_pos:        [3]f32,
	ghost_preview_kind:             Cube_Kind,
	ghost_preview_visible:          bool,
	dig_highlight_world_pos:        [3]f32,
	dig_highlight_visible:          bool,
	armed_tnt:                      [dynamic]Armed_TNT,
	tnt_flash_highlight_world_pos:  [dynamic][3]f32,
	hotbar_image_handles:           [9]Clay_Image_Handle,
	terrain_module:                 wgpu.ShaderModule,
	terrain_bind_group_layout:      wgpu.BindGroupLayout,
	terrain_pipeline_layout:        wgpu.PipelineLayout,
	terrain_compute_pipeline:       wgpu.ComputePipeline,
	terrain_bind_group:             wgpu.BindGroup,
	terrain_batch_uniform_buffer:   wgpu.Buffer,
	terrain_chunk_params_buffer:    wgpu.Buffer,
	terrain_voxel_buffer:           wgpu.Buffer,
	terrain_staging_buffer:         wgpu.Buffer,
	terrain_gen_phase:              Terrain_Gen_Phase,
	terrain_gen_pending:            [dynamic]Chunk_Coords,
	terrain_gen_batch_n:            int,
	terrain_gen_copy_bytes:         u64,
	terrain_gen_added:              [dynamic]Chunk_Coords,
	terrain_gen_removed:            [dynamic]Chunk_Coords,
	terrain_gen_pool_counters:      [3]u64,
	chunk_vertex_buffer_pool:       map[u64][dynamic]wgpu.Buffer,
	chunk_index_buffer_pool:        map[u64][dynamic]wgpu.Buffer,
	pool_hit_count:                 u64,
	pool_miss_count:                u64,
	pool_fallback_hit_count:        u64,
}

voxel_preview_append_face :: proc(
	vertices: ^[dynamic]f32,
	indices: ^[dynamic]u32,
	p0, p1, p2, p3: [3]f32,
	face: Cube_Face,
	kind: Cube_Kind,
	rotation: Cube_Yaw_Rotation,
	material_marker: f32,
	voxel_light: f32 = 1,
) {
	base := u32(len(vertices^) / 11)
	normal: [3]f32
	switch face {
	case .PosX:
		normal = {1, 0, 0}
	case .NegX:
		normal = {-1, 0, 0}
	case .PosY:
		normal = {0, 1, 0}
	case .NegY:
		normal = {0, -1, 0}
	case .PosZ:
		normal = {0, 0, 1}
	case .NegZ:
		normal = {0, 0, -1}
	}
	tile := atlas_coords_for_face(kind, face, rotation)
	uv_tl := atlas_uv_for_corner(tile, {0, 0})
	uv_bl := atlas_uv_for_corner(tile, {0, 1})
	uv_br := atlas_uv_for_corner(tile, {1, 1})
	uv_tr := atlas_uv_for_corner(tile, {1, 0})
	uv_quad: [4][2]f32
	// Face-specific UV orientation keeps top/side textures aligned exactly like chunk meshing.
	switch face {
	case .PosX, .NegX:
		uv_quad = {uv_bl, uv_tl, uv_tr, uv_br}
	case .PosZ, .NegZ:
		uv_quad = {uv_bl, uv_br, uv_tr, uv_tl}
	case .PosY:
		uv_quad = {uv_tl, uv_bl, uv_br, uv_tr}
	case .NegY:
		uv_quad = {uv_tl, uv_tr, uv_br, uv_bl}
	}
	quad := [4][3]f32{p0, p1, p2, p3}
	for p, i in quad {
		uv := uv_quad[i]
		append(
			vertices,
			p.x,
			p.y,
			p.z,
			normal.x,
			normal.y,
			normal.z,
			uv.x,
			uv.y,
			f32(1),
			material_marker,
			voxel_light,
		)
	}
	append(indices, base + 0, base + 1, base + 2, base + 0, base + 2, base + 3)
}

ghost_preview_build_mesh :: proc(
	world_min: [3]f32,
	kind: Cube_Kind,
	vertices: ^[dynamic]f32,
	indices: ^[dynamic]u32,
) {
	rotation := placed_cube_rotation_from_player_forward(
		linalg.quaternion_mul_vector3(state.player_rotation, WORLD_FORWARD),
	)
	voxel_preview_append_face(
		vertices,
		indices,
		world_min + {1, 0, 0},
		world_min + {1, 1, 0},
		world_min + {1, 1, 1},
		world_min + {1, 0, 1},
		.PosX,
		kind,
		rotation,
		GHOST_PREVIEW_MATERIAL_MARKER,
	)
	voxel_preview_append_face(
		vertices,
		indices,
		world_min + {0, 0, 1},
		world_min + {0, 1, 1},
		world_min + {0, 1, 0},
		world_min,
		.NegX,
		kind,
		rotation,
		GHOST_PREVIEW_MATERIAL_MARKER,
	)
	voxel_preview_append_face(
		vertices,
		indices,
		world_min + {0, 1, 0},
		world_min + {0, 1, 1},
		world_min + {1, 1, 1},
		world_min + {1, 1, 0},
		.PosY,
		kind,
		rotation,
		GHOST_PREVIEW_MATERIAL_MARKER,
	)
	voxel_preview_append_face(
		vertices,
		indices,
		world_min,
		world_min + {1, 0, 0},
		world_min + {1, 0, 1},
		world_min + {0, 0, 1},
		.NegY,
		kind,
		rotation,
		GHOST_PREVIEW_MATERIAL_MARKER,
	)
	voxel_preview_append_face(
		vertices,
		indices,
		world_min + {0, 0, 1},
		world_min + {1, 0, 1},
		world_min + {1, 1, 1},
		world_min + {0, 1, 1},
		.PosZ,
		kind,
		rotation,
		GHOST_PREVIEW_MATERIAL_MARKER,
	)
	voxel_preview_append_face(
		vertices,
		indices,
		world_min + {1, 0, 0},
		world_min,
		world_min + {0, 1, 0},
		world_min + {1, 1, 0},
		.NegZ,
		kind,
		rotation,
		GHOST_PREVIEW_MATERIAL_MARKER,
	)
}

armed_tnt_build_mesh :: proc(vertices: ^[dynamic]f32, indices: ^[dynamic]u32) {
	for armed in state.armed_tnt {
		voxel_preview_append_face(
			vertices,
			indices,
			armed.pos + {1, 0, 0},
			armed.pos + {1, 1, 0},
			armed.pos + {1, 1, 1},
			armed.pos + {1, 0, 1},
			.PosX,
			.TNT,
			.R0,
			ARMED_TNT_MATERIAL_MARKER,
		)
		voxel_preview_append_face(
			vertices,
			indices,
			armed.pos + {0, 0, 1},
			armed.pos + {0, 1, 1},
			armed.pos + {0, 1, 0},
			armed.pos,
			.NegX,
			.TNT,
			.R0,
			ARMED_TNT_MATERIAL_MARKER,
		)
		voxel_preview_append_face(
			vertices,
			indices,
			armed.pos + {0, 1, 0},
			armed.pos + {0, 1, 1},
			armed.pos + {1, 1, 1},
			armed.pos + {1, 1, 0},
			.PosY,
			.TNT,
			.R0,
			ARMED_TNT_MATERIAL_MARKER,
		)
		voxel_preview_append_face(
			vertices,
			indices,
			armed.pos,
			armed.pos + {1, 0, 0},
			armed.pos + {1, 0, 1},
			armed.pos + {0, 0, 1},
			.NegY,
			.TNT,
			.R0,
			ARMED_TNT_MATERIAL_MARKER,
		)
		voxel_preview_append_face(
			vertices,
			indices,
			armed.pos + {0, 0, 1},
			armed.pos + {1, 0, 1},
			armed.pos + {1, 1, 1},
			armed.pos + {0, 1, 1},
			.PosZ,
			.TNT,
			.R0,
			ARMED_TNT_MATERIAL_MARKER,
		)
		voxel_preview_append_face(
			vertices,
			indices,
			armed.pos + {1, 0, 0},
			armed.pos,
			armed.pos + {0, 1, 0},
			armed.pos + {1, 1, 0},
			.NegZ,
			.TNT,
			.R0,
			ARMED_TNT_MATERIAL_MARKER,
		)
	}
}

atlas_resources_create :: proc() {
	img, err := image.load_from_bytes(#load("atlas.png", []byte))
	if err != nil || img == nil {
		fmt.panicf("atlas load failed err=%v", err)
	}
	defer image.destroy(img)

	assert(img.width == 512)
	assert(img.height == 256)
	assert(img.channels == 4)
	assert(img.depth == 8)

	state.atlas_texture = wgpu.DeviceCreateTexture(
		state.device,
		&{
			label = "atlas",
			size = {width = u32(img.width), height = u32(img.height), depthOrArrayLayers = 1},
			mipLevelCount = 1,
			sampleCount = 1,
			dimension = ._2D,
			format = .RGBA8Unorm,
			usage = {.TextureBinding, .CopyDst},
		},
	)
	state.atlas_view = wgpu.TextureCreateView(state.atlas_texture, nil)
	state.atlas_sampler = wgpu.DeviceCreateSampler(
		state.device,
		&{
			label = "atlas_sampler",
			addressModeU = .ClampToEdge,
			addressModeV = .ClampToEdge,
			addressModeW = .ClampToEdge,
			magFilter = .Nearest,
			minFilter = .Nearest,
			mipmapFilter = .Nearest,
			lodMinClamp = 0,
			lodMaxClamp = 0,
			maxAnisotropy = 1,
		},
	)

	pixels := img.pixels.buf[:]
	row_stride := u32(img.width * 4)
	wgpu.QueueWriteTexture(
		state.queue,
		&{
			texture = state.atlas_texture,
			mipLevel = 0,
			origin = {x = 0, y = 0, z = 0},
			aspect = .All,
		},
		raw_data(pixels),
		uint(len(pixels)),
		&{offset = 0, bytesPerRow = row_stride, rowsPerImage = u32(img.height)},
		&{width = u32(img.width), height = u32(img.height), depthOrArrayLayers = 1},
	)
}

atlas_resources_destroy :: proc() {
	if state.atlas_sampler != nil {
		wgpu.SamplerRelease(state.atlas_sampler)
		state.atlas_sampler = nil
	}
	if state.atlas_view != nil {
		wgpu.TextureViewRelease(state.atlas_view)
		state.atlas_view = nil
	}
	if state.atlas_texture != nil {
		wgpu.TextureRelease(state.atlas_texture)
		state.atlas_texture = nil
	}
}

reflection_resources_create :: proc() {
	if state.reflection_sampler == nil {
		state.reflection_sampler = wgpu.DeviceCreateSampler(
			state.device,
			&{
				label = "reflection_sampler",
				addressModeU = .ClampToEdge,
				addressModeV = .ClampToEdge,
				addressModeW = .ClampToEdge,
				magFilter = .Linear,
				minFilter = .Linear,
				mipmapFilter = .Linear,
				lodMinClamp = 0,
				lodMaxClamp = 0,
				maxAnisotropy = 1,
			},
		)
	}
	if state.reflection_view != nil {
		wgpu.TextureViewRelease(state.reflection_view)
		state.reflection_view = nil
	}
	if state.reflection_texture != nil {
		wgpu.TextureRelease(state.reflection_texture)
		state.reflection_texture = nil
	}
	if state.reflection_depth_view != nil {
		wgpu.TextureViewRelease(state.reflection_depth_view)
		state.reflection_depth_view = nil
	}
	if state.reflection_depth_texture != nil {
		wgpu.TextureRelease(state.reflection_depth_texture)
		state.reflection_depth_texture = nil
	}
	w := state.config.width
	h := state.config.height
	if w == 0 || h == 0 {
		return
	}
	state.reflection_texture = wgpu.DeviceCreateTexture(
		state.device,
		&{
			label = "reflection_color",
			size = {width = w, height = h, depthOrArrayLayers = 1},
			mipLevelCount = 1,
			sampleCount = 1,
			dimension = ._2D,
			format = .BGRA8Unorm,
			usage = {.RenderAttachment, .TextureBinding},
		},
	)
	state.reflection_view = wgpu.TextureCreateView(state.reflection_texture, nil)
	state.reflection_depth_texture = wgpu.DeviceCreateTexture(
		state.device,
		&{
			label = "reflection_depth",
			size = {width = w, height = h, depthOrArrayLayers = 1},
			mipLevelCount = 1,
			sampleCount = 1,
			dimension = ._2D,
			format = .Depth32Float,
			usage = {.RenderAttachment},
		},
	)
	state.reflection_depth_view = wgpu.TextureCreateView(state.reflection_depth_texture, nil)
}

reflection_resources_destroy :: proc() {
	if state.reflection_depth_view != nil {
		wgpu.TextureViewRelease(state.reflection_depth_view)
		state.reflection_depth_view = nil
	}
	if state.reflection_depth_texture != nil {
		wgpu.TextureRelease(state.reflection_depth_texture)
		state.reflection_depth_texture = nil
	}
	if state.reflection_view != nil {
		wgpu.TextureViewRelease(state.reflection_view)
		state.reflection_view = nil
	}
	if state.reflection_texture != nil {
		wgpu.TextureRelease(state.reflection_texture)
		state.reflection_texture = nil
	}
	if state.reflection_sampler != nil {
		wgpu.SamplerRelease(state.reflection_sampler)
		state.reflection_sampler = nil
	}
}

scene_bind_groups_rebuild :: proc() {
	if state.bind_group != nil {
		wgpu.BindGroupRelease(state.bind_group)
		state.bind_group = nil
	}
	if state.bind_group_reflection_only != nil {
		wgpu.BindGroupRelease(state.bind_group_reflection_only)
		state.bind_group_reflection_only = nil
	}
	if state.rain_bind_group != nil {
		wgpu.BindGroupRelease(state.rain_bind_group)
		state.rain_bind_group = nil
	}
	if state.cloud_bind_group != nil {
		wgpu.BindGroupRelease(state.cloud_bind_group)
		state.cloud_bind_group = nil
	}
	if state.line_bind_group != nil {
		wgpu.BindGroupRelease(state.line_bind_group)
		state.line_bind_group = nil
	}
	bind_entries := [7]wgpu.BindGroupEntry {
		{
			binding = 0,
			buffer = state.shared_frame_uniform_buffer,
			offset = 0,
			size = size_of(Shared_Frame_Uniform),
		},
		{
			binding = 1,
			buffer = state.scene_uniform_buffer,
			offset = 0,
			size = size_of(Scene_Uniform),
		},
		{
			binding = 2,
			buffer = state.shared_fog_uniform_buffer,
			offset = 0,
			size = size_of(Shared_Fog_Uniform),
		},
		{binding = 3, textureView = state.atlas_view},
		{binding = 4, sampler = state.atlas_sampler},
		{binding = 5, textureView = state.reflection_view},
		{binding = 6, sampler = state.reflection_sampler},
	}
	state.bind_group = wgpu.DeviceCreateBindGroup(
		state.device,
		&{layout = state.bind_group_layout, entryCount = 7, entries = &bind_entries[0]},
	)
	reflection_entries := [7]wgpu.BindGroupEntry {
		{
			binding = 0,
			buffer = state.shared_frame_uniform_buffer,
			offset = 0,
			size = size_of(Shared_Frame_Uniform),
		},
		{
			binding = 1,
			buffer = state.scene_uniform_buffer,
			offset = 0,
			size = size_of(Scene_Uniform),
		},
		{
			binding = 2,
			buffer = state.shared_fog_uniform_buffer,
			offset = 0,
			size = size_of(Shared_Fog_Uniform),
		},
		{binding = 3, textureView = state.atlas_view},
		{binding = 4, sampler = state.atlas_sampler},
		// Reflection pass renders into reflection_texture, so bind atlas here.
		{binding = 5, textureView = state.atlas_view},
		{binding = 6, sampler = state.atlas_sampler},
	}
	state.bind_group_reflection_only = wgpu.DeviceCreateBindGroup(
		state.device,
		&{layout = state.bind_group_layout, entryCount = 7, entries = &reflection_entries[0]},
	)
	rain_entries := [1]wgpu.BindGroupEntry {
		{
			binding = 0,
			buffer = state.shared_frame_uniform_buffer,
			offset = 0,
			size = size_of(Shared_Frame_Uniform),
		},
	}
	state.rain_bind_group = wgpu.DeviceCreateBindGroup(
		state.device,
		&{layout = state.rain_bind_group_layout, entryCount = 1, entries = &rain_entries[0]},
	)
	cloud_entries := [3]wgpu.BindGroupEntry {
		{
			binding = 0,
			buffer = state.shared_frame_uniform_buffer,
			offset = 0,
			size = size_of(Shared_Frame_Uniform),
		},
		{
			binding = 1,
			buffer = state.cloud_uniform_buffer,
			offset = 0,
			size = size_of(Cloud_Uniform),
		},
		{
			binding = 2,
			buffer = state.shared_fog_uniform_buffer,
			offset = 0,
			size = size_of(Shared_Fog_Uniform),
		},
	}
	state.cloud_bind_group = wgpu.DeviceCreateBindGroup(
		state.device,
		&{layout = state.cloud_bind_group_layout, entryCount = 3, entries = &cloud_entries[0]},
	)
	line_entries := [1]wgpu.BindGroupEntry {
		{
			binding = 0,
			buffer = state.shared_frame_uniform_buffer,
			offset = 0,
			size = size_of(Shared_Frame_Uniform),
		},
	}
	state.line_bind_group = wgpu.DeviceCreateBindGroup(
		state.device,
		&{layout = state.line_bind_group_layout, entryCount = 1, entries = &line_entries[0]},
	)
}

next_pow2_u64 :: proc(v: u64) -> u64 {
	if v <= 1 {
		return 1
	}
	v := v
	v -= 1
	v |= v >> 1
	v |= v >> 2
	v |= v >> 4
	v |= v >> 8
	v |= v >> 16
	v |= v >> 32
	return v + 1
}

// Max free buffers per pow2 bucket; beyond this we release on return so pools cannot grow without bound.
CHUNK_GPU_POOL_MAX_PER_BUCKET :: 8

chunk_gpu_pool_take_vertex :: proc(
	required_bytes: u64,
) -> (
	buffer: wgpu.Buffer,
	bucket_bytes: u64,
	reused: bool,
) {
	bucket := next_pow2_u64(required_bytes)
	list, ok := state.chunk_vertex_buffer_pool[bucket]
	if ok && len(list) > 0 {
		buffer = pop(&list)
		if len(list) == 0 {
			delete_key(&state.chunk_vertex_buffer_pool, bucket)
		} else {
			state.chunk_vertex_buffer_pool[bucket] = list
		}
		state.pool_hit_count += 1
		return buffer, bucket, true
	}
	buffer = wgpu.DeviceCreateBuffer(state.device, &{usage = {.Vertex, .CopyDst}, size = bucket})
	state.pool_miss_count += 1
	log.infof("creating vertex buffer size_mib=%v", f64(bucket) / (1024.0 * 1024.0))
	return buffer, bucket, false
}

chunk_gpu_pool_take_index :: proc(
	required_bytes: u64,
) -> (
	buffer: wgpu.Buffer,
	bucket_bytes: u64,
	reused: bool,
) {
	bucket := next_pow2_u64(required_bytes)
	list, ok := state.chunk_index_buffer_pool[bucket]
	if ok && len(list) > 0 {
		buffer = pop(&list)
		if len(list) == 0 {
			delete_key(&state.chunk_index_buffer_pool, bucket)
		} else {
			state.chunk_index_buffer_pool[bucket] = list
		}
		state.pool_hit_count += 1
		return buffer, bucket, true
	}
	buffer = wgpu.DeviceCreateBuffer(state.device, &{usage = {.Index, .CopyDst}, size = bucket})
	state.pool_miss_count += 1
	log.infof("creating index buffer size_mib=%v", f64(bucket) / (1024.0 * 1024.0))
	return buffer, bucket, false
}

chunk_gpu_pool_put_vertex :: proc(buffer: wgpu.Buffer, buffer_bytes: u64) {
	if buffer == nil || buffer_bytes == 0 {
		return
	}
	// Keep buffers alive between chunk unload/load to smooth allocator/driver spikes.
	bucket := next_pow2_u64(buffer_bytes)
	list := state.chunk_vertex_buffer_pool[bucket]
	if len(list) >= CHUNK_GPU_POOL_MAX_PER_BUCKET {
		wgpu.BufferRelease(buffer)
		return
	}
	append(&list, buffer)
	state.chunk_vertex_buffer_pool[bucket] = list
}

chunk_gpu_pool_put_index :: proc(buffer: wgpu.Buffer, buffer_bytes: u64) {
	if buffer == nil || buffer_bytes == 0 {
		return
	}
	// Keep buffers alive between chunk unload/load to smooth allocator/driver spikes.
	bucket := next_pow2_u64(buffer_bytes)
	list := state.chunk_index_buffer_pool[bucket]
	if len(list) >= CHUNK_GPU_POOL_MAX_PER_BUCKET {
		wgpu.BufferRelease(buffer)
		return
	}
	append(&list, buffer)
	state.chunk_index_buffer_pool[bucket] = list
}

chunk_gpu_pool_destroy :: proc() {
	for _, list in state.chunk_vertex_buffer_pool {
		for buffer in list {
			if buffer != nil {
				wgpu.BufferRelease(buffer)
				// log.info("releasing vertex buffer")
			}
		}
	}
	clear(&state.chunk_vertex_buffer_pool)
	for _, list in state.chunk_index_buffer_pool {
		for buffer in list {
			if buffer != nil {
				wgpu.BufferRelease(buffer)
				// log.info("releasing index buffer")
			}
		}
	}
	clear(&state.chunk_index_buffer_pool)
}

chunk_return_buffers_to_pool :: proc(chunk: ^Chunk) {
	if chunk.gpu_vertex_opaque != nil {
		chunk_gpu_pool_put_vertex(chunk.gpu_vertex_opaque, chunk.gpu_vertex_opaque_bytes)
		chunk.gpu_vertex_opaque = nil
		chunk.gpu_vertex_opaque_bytes = 0
	}
	if chunk.gpu_index_opaque != nil {
		chunk_gpu_pool_put_index(chunk.gpu_index_opaque, chunk.gpu_index_opaque_bytes)
		chunk.gpu_index_opaque = nil
		chunk.gpu_index_opaque_bytes = 0
	}
	if chunk.gpu_vertex_transparent != nil {
		chunk_gpu_pool_put_vertex(chunk.gpu_vertex_transparent, chunk.gpu_vertex_transparent_bytes)
		chunk.gpu_vertex_transparent = nil
		chunk.gpu_vertex_transparent_bytes = 0
	}
	if chunk.gpu_index_transparent != nil {
		chunk_gpu_pool_put_index(chunk.gpu_index_transparent, chunk.gpu_index_transparent_bytes)
		chunk.gpu_index_transparent = nil
		chunk.gpu_index_transparent_bytes = 0
	}
	if chunk.gpu_vertex_flower != nil {
		chunk_gpu_pool_put_vertex(chunk.gpu_vertex_flower, chunk.gpu_vertex_flower_bytes)
		chunk.gpu_vertex_flower = nil
		chunk.gpu_vertex_flower_bytes = 0
	}
	if chunk.gpu_index_flower != nil {
		chunk_gpu_pool_put_index(chunk.gpu_index_flower, chunk.gpu_index_flower_bytes)
		chunk.gpu_index_flower = nil
		chunk.gpu_index_flower_bytes = 0
	}
	chunk.gpu_index_opaque_count = 0
	chunk.gpu_index_transparent_count = 0
	chunk.gpu_index_flower_count = 0
}

chunk_gpu_sync_one :: proc(
	chunk_pos: Chunk_Coords,
	vertices: []f32,
	indices: []u32,
	vertex_buffer: ^wgpu.Buffer,
	vertex_bytes: ^u64,
	index_buffer: ^wgpu.Buffer,
	index_bytes: ^u64,
	index_count: ^u32,
) {
	if len(indices) == 0 {
		if vertex_buffer^ != nil {
			chunk_gpu_pool_put_vertex(vertex_buffer^, vertex_bytes^)
			vertex_buffer^ = nil
			vertex_bytes^ = 0
		}
		if index_buffer^ != nil {
			chunk_gpu_pool_put_index(index_buffer^, index_bytes^)
			index_buffer^ = nil
			index_bytes^ = 0
		}
		index_count^ = 0
		return
	}
	vertices_buffer_size := u64(len(vertices) * size_of(f32))
	indices_buffer_size := u64(len(indices) * size_of(u32))

	x := f32(chunk_pos[0]) * CHUNK_WIDTH
	z := f32(chunk_pos[1]) * CHUNK_WIDTH

	verts_upload := make([]f32, len(vertices), context.temp_allocator)
	copy(verts_upload, vertices)
	for i := 0; i < len(verts_upload); i += 11 {
		verts_upload[i + 0] += x
		verts_upload[i + 2] += z
	}

	if vertex_buffer^ == nil || vertex_bytes^ < vertices_buffer_size {
		if vertex_buffer^ != nil {
			chunk_gpu_pool_put_vertex(vertex_buffer^, vertex_bytes^)
		}
		vertex_buffer^, vertex_bytes^, _ = chunk_gpu_pool_take_vertex(vertices_buffer_size)
	}
	if index_buffer^ == nil || index_bytes^ < indices_buffer_size {
		if index_buffer^ != nil {
			chunk_gpu_pool_put_index(index_buffer^, index_bytes^)
		}
		index_buffer^, index_bytes^, _ = chunk_gpu_pool_take_index(indices_buffer_size)
	}
	wgpu.QueueWriteBuffer(
		state.queue,
		vertex_buffer^,
		0,
		raw_data(verts_upload),
		uint(vertices_buffer_size),
	)
	wgpu.QueueWriteBuffer(
		state.queue,
		index_buffer^,
		0,
		raw_data(indices),
		uint(indices_buffer_size),
	)
	index_count^ = u32(len(indices))
}

chunk_gpu_sync :: proc(
	chunk_pos: Chunk_Coords,
	chunk: ^Chunk,
	vertices_opaque: []f32,
	indices_opaque: []u32,
	vertices_transparent: []f32,
	indices_transparent: []u32,
	vertices_flower: []f32,
	indices_flower: []u32,
) {
	chunk_gpu_sync_one(
		chunk_pos,
		vertices_opaque,
		indices_opaque,
		&chunk.gpu_vertex_opaque,
		&chunk.gpu_vertex_opaque_bytes,
		&chunk.gpu_index_opaque,
		&chunk.gpu_index_opaque_bytes,
		&chunk.gpu_index_opaque_count,
	)
	chunk_gpu_sync_one(
		chunk_pos,
		vertices_transparent,
		indices_transparent,
		&chunk.gpu_vertex_transparent,
		&chunk.gpu_vertex_transparent_bytes,
		&chunk.gpu_index_transparent,
		&chunk.gpu_index_transparent_bytes,
		&chunk.gpu_index_transparent_count,
	)
	chunk_gpu_sync_one(
		chunk_pos,
		vertices_flower,
		indices_flower,
		&chunk.gpu_vertex_flower,
		&chunk.gpu_vertex_flower_bytes,
		&chunk.gpu_index_flower,
		&chunk.gpu_index_flower_bytes,
		&chunk.gpu_index_flower_count,
	)
}

chunk_bounds_vertex_buffer_ensure :: proc(required_bytes: u64) {
	if required_bytes == 0 {
		return
	}
	if state.chunk_bounds_vertex_buffer != nil &&
	   state.chunk_bounds_vertex_capacity >= required_bytes {
		return
	}
	if state.chunk_bounds_vertex_buffer != nil {
		wgpu.BufferRelease(state.chunk_bounds_vertex_buffer)
	}
	state.chunk_bounds_vertex_capacity = next_pow2_u64(required_bytes)
	state.chunk_bounds_vertex_buffer = wgpu.DeviceCreateBuffer(
		state.device,
		&{usage = {.Vertex, .CopyDst}, size = state.chunk_bounds_vertex_capacity},
	)
}

chunk_bounds_append_line :: proc(
	vertices: ^[dynamic]Chunk_Bounds_Vertex,
	a, b: [3]f32,
	color: [4]f32,
) {
	append(vertices, Chunk_Bounds_Vertex{pos = a, color = color})
	append(vertices, Chunk_Bounds_Vertex{pos = b, color = color})
}

chunk_bounds_append_box :: proc(
	vertices: ^[dynamic]Chunk_Bounds_Vertex,
	min, max: [3]f32,
	color: [4]f32,
) {
	p000 := [3]f32{min.x, min.y, min.z}
	p001 := [3]f32{min.x, min.y, max.z}
	p010 := [3]f32{min.x, max.y, min.z}
	p011 := [3]f32{min.x, max.y, max.z}
	p100 := [3]f32{max.x, min.y, min.z}
	p101 := [3]f32{max.x, min.y, max.z}
	p110 := [3]f32{max.x, max.y, min.z}
	p111 := [3]f32{max.x, max.y, max.z}

	chunk_bounds_append_line(vertices, p000, p100, color)
	chunk_bounds_append_line(vertices, p001, p101, color)
	chunk_bounds_append_line(vertices, p010, p110, color)
	chunk_bounds_append_line(vertices, p011, p111, color)

	chunk_bounds_append_line(vertices, p000, p001, color)
	chunk_bounds_append_line(vertices, p100, p101, color)
	chunk_bounds_append_line(vertices, p010, p011, color)
	chunk_bounds_append_line(vertices, p110, p111, color)

	chunk_bounds_append_line(vertices, p000, p010, color)
	chunk_bounds_append_line(vertices, p001, p011, color)
	chunk_bounds_append_line(vertices, p100, p110, color)
	chunk_bounds_append_line(vertices, p101, p111, color)
}

terrain_gpu_create :: proc() {
	terrain_source :: #load("terrain.wgsl", string)
	state.terrain_module = wgpu.DeviceCreateShaderModule(
		state.device,
		&{nextInChain = &wgpu.ShaderSourceWGSL{sType = .ShaderSourceWGSL, code = terrain_source}},
	)

	terrain_layout_entries := [3]wgpu.BindGroupLayoutEntry {
		{
			binding = 0,
			visibility = {.Compute},
			buffer = {
				type = .Uniform,
				hasDynamicOffset = false,
				minBindingSize = size_of(Terrain_Batch_Uniform),
			},
		},
		{
			binding = 1,
			visibility = {.Compute},
			buffer = {
				type = .ReadOnlyStorage,
				hasDynamicOffset = false,
				minBindingSize = TERRAIN_CHUNK_PARAMS_BUFFER_SIZE,
			},
		},
		{
			binding = 2,
			visibility = {.Compute},
			buffer = {
				type = .Storage,
				hasDynamicOffset = false,
				minBindingSize = TERRAIN_BATCH_VOXEL_BUFFER_SIZE,
			},
		},
	}
	state.terrain_bind_group_layout = wgpu.DeviceCreateBindGroupLayout(
		state.device,
		&{entryCount = 3, entries = &terrain_layout_entries[0]},
	)
	state.terrain_pipeline_layout = wgpu.DeviceCreatePipelineLayout(
		state.device,
		&{bindGroupLayoutCount = 1, bindGroupLayouts = &state.terrain_bind_group_layout},
	)

	state.terrain_batch_uniform_buffer = wgpu.DeviceCreateBuffer(
		state.device,
		&{usage = {.Uniform, .CopyDst}, size = size_of(Terrain_Batch_Uniform)},
	)
	state.terrain_chunk_params_buffer = wgpu.DeviceCreateBuffer(
		state.device,
		&{usage = {.Storage, .CopyDst}, size = TERRAIN_CHUNK_PARAMS_BUFFER_SIZE},
	)
	state.terrain_voxel_buffer = wgpu.DeviceCreateBuffer(
		state.device,
		&{usage = {.Storage, .CopySrc, .CopyDst}, size = TERRAIN_BATCH_VOXEL_BUFFER_SIZE},
	)
	state.terrain_staging_buffer = wgpu.DeviceCreateBuffer(
		state.device,
		&{usage = {.MapRead, .CopyDst}, size = TERRAIN_BATCH_VOXEL_BUFFER_SIZE},
	)

	state.terrain_compute_pipeline = wgpu.DeviceCreateComputePipeline(
		state.device,
		&{
			layout = state.terrain_pipeline_layout,
			compute = {module = state.terrain_module, entryPoint = "terrain_main"},
		},
	)

	terrain_bg_entries := [3]wgpu.BindGroupEntry {
		{
			binding = 0,
			buffer = state.terrain_batch_uniform_buffer,
			offset = 0,
			size = size_of(Terrain_Batch_Uniform),
		},
		{
			binding = 1,
			buffer = state.terrain_chunk_params_buffer,
			offset = 0,
			size = TERRAIN_CHUNK_PARAMS_BUFFER_SIZE,
		},
		{
			binding = 2,
			buffer = state.terrain_voxel_buffer,
			offset = 0,
			size = TERRAIN_BATCH_VOXEL_BUFFER_SIZE,
		},
	}
	state.terrain_bind_group = wgpu.DeviceCreateBindGroup(
		state.device,
		&{
			layout = state.terrain_bind_group_layout,
			entryCount = 3,
			entries = &terrain_bg_entries[0],
		},
	)
}

terrain_gen_kick :: proc(
	positions: []Chunk_Coords,
	added: []Chunk_Coords,
	removed: []Chunk_Coords,
) {
	assert(state.terrain_gen_phase == .Idle)

	clear(&state.terrain_gen_pending)
	append(&state.terrain_gen_pending, ..positions)

	clear(&state.terrain_gen_added)
	append(&state.terrain_gen_added, ..added)

	clear(&state.terrain_gen_removed)
	append(&state.terrain_gen_removed, ..removed)

	state.terrain_gen_pool_counters = {
		state.pool_hit_count,
		state.pool_miss_count,
		state.pool_fallback_hit_count,
	}

	terrain_gen_submit_next_batch()
}

terrain_gen_submit_next_batch :: proc() {
	if len(state.terrain_gen_pending) == 0 {
		terrain_gen_finalize()
		return
	}

	n := min(TERRAIN_MAX_BATCH_CHUNKS, len(state.terrain_gen_pending))
	sub := state.terrain_gen_pending[:n]
	state.terrain_gen_batch_n = n

	u: Terrain_Batch_Uniform
	u.chunk_count = u32(n)
	u.noise_scale = 0.04
	u.seed = 0.0

	wgpu.QueueWriteBuffer(state.queue, state.terrain_batch_uniform_buffer, 0, &u, uint(size_of(u)))

	coords := make([][2]i32, n, context.temp_allocator)
	for p, i in sub {
		coords[i] = {i32(p.x), i32(p.y)}
	}
	wgpu.QueueWriteBuffer(
		state.queue,
		state.terrain_chunk_params_buffer,
		0,
		raw_data(coords),
		uint(u64(n) * u64(size_of(coords[0]))),
	)

	encoder := wgpu.DeviceCreateCommandEncoder(state.device, nil)
	pass := wgpu.CommandEncoderBeginComputePass(encoder, nil)
	wgpu.ComputePassEncoderSetPipeline(pass, state.terrain_compute_pipeline)
	wgpu.ComputePassEncoderSetBindGroup(pass, 0, state.terrain_bind_group)
	// NOTE: X=ceil(16/8), Y=ceil(48/8), Z stacks chunks: 4 workgroups per chunk in Z
	wgpu.ComputePassEncoderDispatchWorkgroups(pass, 2, 6, 4 * u32(n))
	wgpu.ComputePassEncoderEnd(pass)
	wgpu.ComputePassEncoderRelease(pass)

	state.terrain_gen_copy_bytes = u64(n) * TERRAIN_CHUNK_VOXEL_BYTES
	wgpu.CommandEncoderCopyBufferToBuffer(
		encoder,
		state.terrain_voxel_buffer,
		0,
		state.terrain_staging_buffer,
		0,
		state.terrain_gen_copy_bytes,
	)

	cmd := wgpu.CommandEncoderFinish(encoder, nil)
	wgpu.CommandEncoderRelease(encoder)

	wgpu.QueueSubmit(state.queue, {cmd})
	wgpu.CommandBufferRelease(cmd)

	state.terrain_gen_phase = .Waiting_Work_Done
	wgpu.QueueOnSubmittedWorkDone(state.queue, {
		callback = proc "c" (status: wgpu.QueueWorkDoneStatus, u1, u2: rawptr) {
			context = state.ctx
			terrain_gen_on_work_done()
		},
	})
}

terrain_gen_on_work_done :: proc() {
	state.terrain_gen_phase = .Waiting_Map
	wgpu.BufferMapAsync(
		state.terrain_staging_buffer,
		{.Read},
		0,
		uint(state.terrain_gen_copy_bytes),
		{
			callback = proc "c" (status: wgpu.MapAsyncStatus, message: string, u1, u2: rawptr) {
				context = state.ctx
				if status != .Success {
					fmt.panicf("terrain staging map failed: [%v]", status)
				}
				terrain_gen_on_map_done()
			},
		},
	)
}

terrain_gen_on_map_done :: proc() {
	n := state.terrain_gen_batch_n
	sub := state.terrain_gen_pending[:n]

	mapped := wgpu.BufferGetConstMappedRangeSlice(
		state.terrain_staging_buffer,
		0,
		uint(n * CHUNK_VOXEL_COUNT),
		u32,
	)
	spawn_world_x := int(math.floor(state.player_pos.x))
	spawn_world_z := int(math.floor(state.player_pos.z))
	for ci in 0 ..< n {
		chunk := &state.chunks[sub[ci]]
		base := ci * CHUNK_VOXEL_COUNT
		mem.copy_non_overlapping(
			raw_data(chunk.kinds[:]),
			raw_data(mapped[base:][:CHUNK_VOXEL_COUNT]),
			TERRAIN_CHUNK_VOXEL_BYTES,
		)
		chunk_min_x := sub[ci].x * CHUNK_WIDTH
		chunk_min_z := sub[ci].y * CHUNK_WIDTH
		spawn_local_x := spawn_world_x - chunk_min_x
		spawn_local_z := spawn_world_z - chunk_min_z
		if spawn_local_x >= 0 &&
		   spawn_local_x < CHUNK_WIDTH &&
		   spawn_local_z >= 0 &&
		   spawn_local_z < CHUNK_WIDTH {
			// Give the spawn landing column a deterministic marker block so the
			// first touchdown point is easy to spot during terrain generation tweaks.
			for y := CHUNK_HEIGHT - 1; y >= 0; y -= 1 {
				idx := cube_index(Local_Pos{spawn_local_x, y, spawn_local_z})
				if chunk.kinds[idx] == .None {
					continue
				}
				chunk.kinds[idx] = .Brick
				break
			}
		}
		water_fluid_seed_chunk(sub[ci])
	}
	wgpu.BufferUnmap(state.terrain_staging_buffer)

	remove_range(&state.terrain_gen_pending, 0, n)
	terrain_gen_submit_next_batch()
}

terrain_gen_finalize :: proc() {
	state.terrain_gen_phase = .Idle

	added := state.terrain_gen_added[:]
	removed := state.terrain_gen_removed[:]

	dirty := make(map[Chunk_Coords]struct{}, context.temp_allocator)
	light_changed_count := 0

	for cc in removed {
		for offset in CHUNK_CARDINAL_OFFSETS {
			neighbor := cc + offset
			if neighbor in state.chunks {
				dirty[neighbor] = {}
			}
		}
	}

	for cc in added {
		dirty[cc] = {}
		// Geometry seam fix: when a new chunk appears, neighboring already-loaded chunks
		// must be remeshed so border faces can be culled against the new voxels.
		for offset in CHUNK_CARDINAL_OFFSETS {
			neighbor := cc + offset
			if neighbor in state.chunks {
				dirty[neighbor] = {}
			}
		}
	}

	if len(added) > 0 {
		light_changed := make(map[Chunk_Coords]struct{}, context.temp_allocator)
		lighting_rebuild_for_chunks(added, &light_changed)
		light_changed_count = len(light_changed)
		for cc in light_changed {
			dirty[cc] = {}
		}
	}

	if len(dirty) > 0 {
		remesh_queue_enqueue_dirty_map(dirty)
	}

	pool_hit_delta := state.pool_hit_count - state.terrain_gen_pool_counters[0]
	pool_miss_delta := state.pool_miss_count - state.terrain_gen_pool_counters[1]
	pool_fallback_hit_delta := state.pool_fallback_hit_count - state.terrain_gen_pool_counters[2]
	if len(added) > 0 ||
	   len(removed) > 0 ||
	   pool_hit_delta > 0 ||
	   pool_miss_delta > 0 ||
	   pool_fallback_hit_delta > 0 {
		log.infof(
			"stream update added_count=%v dirty_count=%v light_changed_count=%v pool_hit=%v pool_miss=%v pool_fallback_hit=%v",
			len(added),
			len(dirty),
			light_changed_count,
			pool_hit_delta,
			pool_miss_delta,
			pool_fallback_hit_delta,
		)
	}
}

destroy_chunk :: proc(chunk: ^Chunk, pos: Chunk_Coords) {
	chunk_return_buffers_to_pool(chunk)
	delete(chunk.rotation_by_index)
	delete_key(&state.chunks, pos)
}

// Load/unload chunks by squared Euclidean distance from the camera chunk; remesh+upload when the set changes.
// Terrain generation is async (callback-driven), so when new chunks are added the post-processing
// (lighting, remeshing) is deferred to terrain_gen_finalize which fires after the GPU readback completes.
chunks_stream_update :: proc() {
	// GPU terrain buffers are shared -- can't overlap two generations.
	if state.terrain_gen_phase != .Idle {
		return
	}

	center := chunk_coords(state.player_pos)
	load_radius := CHUNK_STREAM_LOAD_RADIUS
	unload_radius := CHUNK_STREAM_UNLOAD_RADIUS
	assert(load_radius < unload_radius)
	load_r2 := load_radius * load_radius
	unload_r2 := unload_radius * unload_radius

	to_remove := make([dynamic]Chunk_Coords, context.temp_allocator)

	for pos, _ in state.chunks {
		dx := pos.x - center.x
		dz := pos.y - center.y
		if dx * dx + dz * dz > unload_r2 {
			append(&to_remove, pos)
		}
	}

	changed := false
	for chunk_coords in to_remove {
		destroy_chunk(&state.chunks[chunk_coords], chunk_coords)
		changed = true
	}

	added := make([dynamic]Chunk_Coords, context.temp_allocator)

	for dz in -load_radius ..= load_radius {
		for dx in -load_radius ..= load_radius {
			if dx * dx + dz * dz > load_r2 {
				continue
			}
			chunk_coords := Chunk_Coords{center[0] + dx, center[1] + dz}
			if _, ok := state.chunks[chunk_coords]; ok {
				continue
			}
			chunk := Chunk{}
			chunk.rotation_by_index = make(map[Cube_Index]Cube_Yaw_Rotation)
			state.chunks[chunk_coords] = chunk
			append(&added, chunk_coords)
			changed = true
		}
	}

	if len(added) > 0 {
		// Kicks off async GPU terrain generation; terrain_gen_finalize will handle
		// dirty marking, lighting, and remeshing once the readback completes.
		terrain_gen_kick(added[:], added[:], to_remove[:])
		return
	}

	if !changed {
		return
	}

	// Removal-only path: no GPU work needed, just remesh neighbors of removed chunks.
	dirty := make(map[Chunk_Coords]struct{}, context.temp_allocator)
	for cc in to_remove {
		for offset in CHUNK_CARDINAL_OFFSETS {
			neighbor := cc + offset
			if neighbor in state.chunks {
				dirty[neighbor] = {}
			}
		}
	}
	if len(dirty) > 0 {
		remesh_queue_enqueue_dirty_map(dirty)
	}
}

terrain_gpu_destroy :: proc() {
	if state.terrain_bind_group != nil {
		wgpu.BindGroupRelease(state.terrain_bind_group)
		state.terrain_bind_group = nil
	}
	if state.terrain_staging_buffer != nil {
		wgpu.BufferRelease(state.terrain_staging_buffer)
		state.terrain_staging_buffer = nil
	}
	if state.terrain_voxel_buffer != nil {
		wgpu.BufferRelease(state.terrain_voxel_buffer)
		state.terrain_voxel_buffer = nil
	}
	if state.terrain_chunk_params_buffer != nil {
		wgpu.BufferRelease(state.terrain_chunk_params_buffer)
		state.terrain_chunk_params_buffer = nil
	}
	if state.terrain_batch_uniform_buffer != nil {
		wgpu.BufferRelease(state.terrain_batch_uniform_buffer)
		state.terrain_batch_uniform_buffer = nil
	}
	if state.terrain_compute_pipeline != nil {
		wgpu.ComputePipelineRelease(state.terrain_compute_pipeline)
		state.terrain_compute_pipeline = nil
	}
	if state.terrain_pipeline_layout != nil {
		wgpu.PipelineLayoutRelease(state.terrain_pipeline_layout)
		state.terrain_pipeline_layout = nil
	}
	if state.terrain_bind_group_layout != nil {
		wgpu.BindGroupLayoutRelease(state.terrain_bind_group_layout)
		state.terrain_bind_group_layout = nil
	}
	if state.terrain_module != nil {
		wgpu.ShaderModuleRelease(state.terrain_module)
		state.terrain_module = nil
	}
}

depth_resources_create :: proc() {
	if state.depth_texture != nil {
		wgpu.TextureViewRelease(state.depth_view)
		wgpu.TextureRelease(state.depth_texture)
		state.depth_view = nil
		state.depth_texture = nil
	}
	w := state.config.width
	h := state.config.height
	if w == 0 || h == 0 {
		return
	}
	state.depth_texture = wgpu.DeviceCreateTexture(
		state.device,
		&{
			label = "depth",
			size = {width = w, height = h, depthOrArrayLayers = 1},
			mipLevelCount = 1,
			sampleCount = 1,
			dimension = ._2D,
			format = .Depth32Float,
			usage = {.RenderAttachment, .TextureBinding},
		},
	)
	state.depth_view = wgpu.TextureCreateView(state.depth_texture, nil)
}

depth_resources_destroy :: proc() {
	if state.depth_view != nil {
		wgpu.TextureViewRelease(state.depth_view)
		state.depth_view = nil
	}
	if state.depth_texture != nil {
		wgpu.TextureRelease(state.depth_texture)
		state.depth_texture = nil
	}
}

post_bind_group_rebuild :: proc() {
	if state.post_bind_group != nil {
		wgpu.BindGroupRelease(state.post_bind_group)
		state.post_bind_group = nil
	}
	if state.scene_color_view == nil ||
	   state.depth_view == nil ||
	   state.bloom_view_a == nil ||
	   state.post_bind_group_layout == nil ||
	   state.post_uniform_buffer == nil {
		return
	}
	post_entries := [6]wgpu.BindGroupEntry {
		{binding = 0, buffer = state.post_uniform_buffer, offset = 0, size = POST_UNIFORM_SIZE},
		{binding = 1, textureView = state.scene_color_view},
		{binding = 2, sampler = state.reflection_sampler},
		{binding = 3, textureView = state.depth_view},
		{binding = 4, sampler = state.depth_sample_sampler},
		{binding = 5, textureView = state.bloom_view_a},
	}
	state.post_bind_group = wgpu.DeviceCreateBindGroup(
		state.device,
		&{layout = state.post_bind_group_layout, entryCount = 6, entries = &post_entries[0]},
	)
}

bloom_bind_groups_rebuild :: proc() {
	if state.bloom_extract_bind_group != nil {
		wgpu.BindGroupRelease(state.bloom_extract_bind_group)
		state.bloom_extract_bind_group = nil
	}
	if state.bloom_blur_bind_group_a != nil {
		wgpu.BindGroupRelease(state.bloom_blur_bind_group_a)
		state.bloom_blur_bind_group_a = nil
	}
	if state.bloom_blur_bind_group_b != nil {
		wgpu.BindGroupRelease(state.bloom_blur_bind_group_b)
		state.bloom_blur_bind_group_b = nil
	}
	if state.scene_color_view == nil ||
	   state.bloom_view_a == nil ||
	   state.bloom_view_b == nil ||
	   state.bloom_bind_group_layout == nil ||
	   state.bloom_uniform_buffer == nil {
		return
	}
	bloom_entries := [3]wgpu.BindGroupEntry {
		{binding = 0, buffer = state.bloom_uniform_buffer, offset = 0, size = BLOOM_UNIFORM_SIZE},
		{binding = 1, textureView = state.scene_color_view},
		{binding = 2, sampler = state.bloom_sampler},
	}
	state.bloom_extract_bind_group = wgpu.DeviceCreateBindGroup(
		state.device,
		&{layout = state.bloom_bind_group_layout, entryCount = 3, entries = &bloom_entries[0]},
	)
	blur_a_entries := [3]wgpu.BindGroupEntry {
		{binding = 0, buffer = state.bloom_uniform_buffer, offset = 0, size = BLOOM_UNIFORM_SIZE},
		{binding = 1, textureView = state.bloom_view_a},
		{binding = 2, sampler = state.bloom_sampler},
	}
	state.bloom_blur_bind_group_a = wgpu.DeviceCreateBindGroup(
		state.device,
		&{layout = state.bloom_bind_group_layout, entryCount = 3, entries = &blur_a_entries[0]},
	)
	blur_b_entries := [3]wgpu.BindGroupEntry {
		{binding = 0, buffer = state.bloom_uniform_buffer, offset = 0, size = BLOOM_UNIFORM_SIZE},
		{binding = 1, textureView = state.bloom_view_b},
		{binding = 2, sampler = state.bloom_sampler},
	}
	state.bloom_blur_bind_group_b = wgpu.DeviceCreateBindGroup(
		state.device,
		&{layout = state.bloom_bind_group_layout, entryCount = 3, entries = &blur_b_entries[0]},
	)
}

dof_bind_groups_rebuild :: proc() {
	if state.dof_extract_bind_group != nil {
		wgpu.BindGroupRelease(state.dof_extract_bind_group)
		state.dof_extract_bind_group = nil
	}
	if state.dof_blur_bind_group_a != nil {
		wgpu.BindGroupRelease(state.dof_blur_bind_group_a)
		state.dof_blur_bind_group_a = nil
	}
	if state.dof_blur_bind_group_b != nil {
		wgpu.BindGroupRelease(state.dof_blur_bind_group_b)
		state.dof_blur_bind_group_b = nil
	}
	if state.post_composite_view == nil ||
	   state.dof_view_a == nil ||
	   state.dof_view_b == nil ||
	   state.bloom_bind_group_layout == nil ||
	   state.bloom_uniform_buffer == nil {
		return
	}
	dof_extract_entries := [3]wgpu.BindGroupEntry {
		{binding = 0, buffer = state.bloom_uniform_buffer, offset = 0, size = BLOOM_UNIFORM_SIZE},
		{binding = 1, textureView = state.post_composite_view},
		{binding = 2, sampler = state.bloom_sampler},
	}
	state.dof_extract_bind_group = wgpu.DeviceCreateBindGroup(
		state.device,
		&{
			layout = state.bloom_bind_group_layout,
			entryCount = 3,
			entries = &dof_extract_entries[0],
		},
	)
	dof_blur_a_entries := [3]wgpu.BindGroupEntry {
		{binding = 0, buffer = state.bloom_uniform_buffer, offset = 0, size = BLOOM_UNIFORM_SIZE},
		{binding = 1, textureView = state.dof_view_a},
		{binding = 2, sampler = state.bloom_sampler},
	}
	state.dof_blur_bind_group_a = wgpu.DeviceCreateBindGroup(
		state.device,
		&{
			layout = state.bloom_bind_group_layout,
			entryCount = 3,
			entries = &dof_blur_a_entries[0],
		},
	)
	dof_blur_b_entries := [3]wgpu.BindGroupEntry {
		{binding = 0, buffer = state.bloom_uniform_buffer, offset = 0, size = BLOOM_UNIFORM_SIZE},
		{binding = 1, textureView = state.dof_view_b},
		{binding = 2, sampler = state.bloom_sampler},
	}
	state.dof_blur_bind_group_b = wgpu.DeviceCreateBindGroup(
		state.device,
		&{
			layout = state.bloom_bind_group_layout,
			entryCount = 3,
			entries = &dof_blur_b_entries[0],
		},
	)
}

dof_final_bind_group_rebuild :: proc() {
	if state.dof_final_bind_group != nil {
		wgpu.BindGroupRelease(state.dof_final_bind_group)
		state.dof_final_bind_group = nil
	}
	if state.post_composite_view == nil ||
	   state.dof_view_a == nil ||
	   state.depth_view == nil ||
	   state.dof_final_bind_group_layout == nil ||
	   state.post_uniform_buffer == nil {
		return
	}
	dof_entries := [6]wgpu.BindGroupEntry {
		{binding = 0, buffer = state.post_uniform_buffer, offset = 0, size = POST_UNIFORM_SIZE},
		{binding = 1, textureView = state.post_composite_view},
		{binding = 2, sampler = state.reflection_sampler},
		{binding = 3, textureView = state.dof_view_a},
		{binding = 4, textureView = state.depth_view},
		{binding = 5, sampler = state.depth_sample_sampler},
	}
	state.dof_final_bind_group = wgpu.DeviceCreateBindGroup(
		state.device,
		&{layout = state.dof_final_bind_group_layout, entryCount = 6, entries = &dof_entries[0]},
	)
}

bloom_resources_destroy :: proc() {
	if state.bloom_extract_bind_group != nil {
		wgpu.BindGroupRelease(state.bloom_extract_bind_group)
		state.bloom_extract_bind_group = nil
	}
	if state.bloom_blur_bind_group_a != nil {
		wgpu.BindGroupRelease(state.bloom_blur_bind_group_a)
		state.bloom_blur_bind_group_a = nil
	}
	if state.bloom_blur_bind_group_b != nil {
		wgpu.BindGroupRelease(state.bloom_blur_bind_group_b)
		state.bloom_blur_bind_group_b = nil
	}
	if state.bloom_view_a != nil {
		wgpu.TextureViewRelease(state.bloom_view_a)
		state.bloom_view_a = nil
	}
	if state.bloom_view_b != nil {
		wgpu.TextureViewRelease(state.bloom_view_b)
		state.bloom_view_b = nil
	}
	if state.bloom_texture_a != nil {
		wgpu.TextureRelease(state.bloom_texture_a)
		state.bloom_texture_a = nil
	}
	if state.bloom_texture_b != nil {
		wgpu.TextureRelease(state.bloom_texture_b)
		state.bloom_texture_b = nil
	}
}

// DOF blur runs in half-res ping-pong textures to keep cost close to bloom.
dof_resources_destroy :: proc() {
	if state.dof_final_bind_group != nil {
		wgpu.BindGroupRelease(state.dof_final_bind_group)
		state.dof_final_bind_group = nil
	}
	if state.dof_extract_bind_group != nil {
		wgpu.BindGroupRelease(state.dof_extract_bind_group)
		state.dof_extract_bind_group = nil
	}
	if state.dof_blur_bind_group_a != nil {
		wgpu.BindGroupRelease(state.dof_blur_bind_group_a)
		state.dof_blur_bind_group_a = nil
	}
	if state.dof_blur_bind_group_b != nil {
		wgpu.BindGroupRelease(state.dof_blur_bind_group_b)
		state.dof_blur_bind_group_b = nil
	}
	if state.dof_view_a != nil {
		wgpu.TextureViewRelease(state.dof_view_a)
		state.dof_view_a = nil
	}
	if state.dof_view_b != nil {
		wgpu.TextureViewRelease(state.dof_view_b)
		state.dof_view_b = nil
	}
	if state.dof_texture_a != nil {
		wgpu.TextureRelease(state.dof_texture_a)
		state.dof_texture_a = nil
	}
	if state.dof_texture_b != nil {
		wgpu.TextureRelease(state.dof_texture_b)
		state.dof_texture_b = nil
	}
}

bloom_resources_create :: proc() {
	bloom_resources_destroy()
	w := state.config.width
	h := state.config.height
	hw := max(1, w / 2)
	hh := max(1, h / 2)
	if w == 0 || h == 0 {
		return
	}
	state.bloom_texture_a = wgpu.DeviceCreateTexture(
		state.device,
		&{
			label = "bloom_a",
			size = {width = hw, height = hh, depthOrArrayLayers = 1},
			mipLevelCount = 1,
			sampleCount = 1,
			dimension = ._2D,
			format = .BGRA8Unorm,
			usage = {.RenderAttachment, .TextureBinding},
		},
	)
	state.bloom_view_a = wgpu.TextureCreateView(state.bloom_texture_a, nil)
	state.bloom_texture_b = wgpu.DeviceCreateTexture(
		state.device,
		&{
			label = "bloom_b",
			size = {width = hw, height = hh, depthOrArrayLayers = 1},
			mipLevelCount = 1,
			sampleCount = 1,
			dimension = ._2D,
			format = .BGRA8Unorm,
			usage = {.RenderAttachment, .TextureBinding},
		},
	)
	state.bloom_view_b = wgpu.TextureCreateView(state.bloom_texture_b, nil)
	bloom_bind_groups_rebuild()
}

dof_resources_create :: proc() {
	dof_resources_destroy()
	w := state.config.width
	h := state.config.height
	hw := max(1, w / 2)
	hh := max(1, h / 2)
	if w == 0 || h == 0 {
		return
	}
	state.dof_texture_a = wgpu.DeviceCreateTexture(
		state.device,
		&{
			label = "dof_a",
			size = {width = hw, height = hh, depthOrArrayLayers = 1},
			mipLevelCount = 1,
			sampleCount = 1,
			dimension = ._2D,
			format = .BGRA8Unorm,
			usage = {.RenderAttachment, .TextureBinding},
		},
	)
	state.dof_view_a = wgpu.TextureCreateView(state.dof_texture_a, nil)
	state.dof_texture_b = wgpu.DeviceCreateTexture(
		state.device,
		&{
			label = "dof_b",
			size = {width = hw, height = hh, depthOrArrayLayers = 1},
			mipLevelCount = 1,
			sampleCount = 1,
			dimension = ._2D,
			format = .BGRA8Unorm,
			usage = {.RenderAttachment, .TextureBinding},
		},
	)
	state.dof_view_b = wgpu.TextureCreateView(state.dof_texture_b, nil)
	dof_bind_groups_rebuild()
}

scene_color_resources_create :: proc() {
	if state.post_bind_group != nil {
		wgpu.BindGroupRelease(state.post_bind_group)
		state.post_bind_group = nil
	}
	bloom_resources_destroy()
	dof_resources_destroy()
	if state.scene_color_view != nil {
		wgpu.TextureViewRelease(state.scene_color_view)
		state.scene_color_view = nil
	}
	if state.post_composite_view != nil {
		wgpu.TextureViewRelease(state.post_composite_view)
		state.post_composite_view = nil
	}
	if state.scene_color_texture != nil {
		wgpu.TextureRelease(state.scene_color_texture)
		state.scene_color_texture = nil
	}
	if state.post_composite_texture != nil {
		wgpu.TextureRelease(state.post_composite_texture)
		state.post_composite_texture = nil
	}
	w := state.config.width
	h := state.config.height
	if w == 0 || h == 0 {
		return
	}
	state.scene_color_texture = wgpu.DeviceCreateTexture(
		state.device,
		&{
			label = "scene_color",
			size = {width = w, height = h, depthOrArrayLayers = 1},
			mipLevelCount = 1,
			sampleCount = 1,
			dimension = ._2D,
			format = .BGRA8Unorm,
			usage = {.RenderAttachment, .TextureBinding},
		},
	)
	state.scene_color_view = wgpu.TextureCreateView(state.scene_color_texture, nil)
	state.post_composite_texture = wgpu.DeviceCreateTexture(
		state.device,
		&{
			label = "post_composite",
			size = {width = w, height = h, depthOrArrayLayers = 1},
			mipLevelCount = 1,
			sampleCount = 1,
			dimension = ._2D,
			format = .BGRA8Unorm,
			usage = {.RenderAttachment, .TextureBinding},
		},
	)
	state.post_composite_view = wgpu.TextureCreateView(state.post_composite_texture, nil)
	bloom_resources_create()
	dof_resources_create()
	post_bind_group_rebuild()
	dof_final_bind_group_rebuild()
}

scene_color_resources_destroy :: proc() {
	if state.post_bind_group != nil {
		wgpu.BindGroupRelease(state.post_bind_group)
		state.post_bind_group = nil
	}
	bloom_resources_destroy()
	dof_resources_destroy()
	if state.scene_color_view != nil {
		wgpu.TextureViewRelease(state.scene_color_view)
		state.scene_color_view = nil
	}
	if state.post_composite_view != nil {
		wgpu.TextureViewRelease(state.post_composite_view)
		state.post_composite_view = nil
	}
	if state.scene_color_texture != nil {
		wgpu.TextureRelease(state.scene_color_texture)
		state.scene_color_texture = nil
	}
	if state.post_composite_texture != nil {
		wgpu.TextureRelease(state.post_composite_texture)
		state.post_composite_texture = nil
	}
}

main :: proc() {
	context.logger = log.create_console_logger()
	state.ctx = context
	state.game_process_active = true

	os_init()

	// Game init
	{
		state.player_pitch = 0
		state.player_rotation = quaternion(x = 0, y = 0, z = 0, w = 1)
		state.player_pos = {8, 40, 8}
		state.player_vel = {}
		state.player_noclip = false
		state.player_on_ground = false
		state.debug_chunk_bounds = false
		state.rain_enabled = true
		state.bloom_enabled = true
		state.dof_enabled = true
		state.clouds_enabled = true
		state.fog_enabled = true
		state.fxaa_enabled = true
		state.hotbar_slot_kinds = [HOTBAR_SLOT_COUNT]Cube_Kind {
			.None,
			.Dirt,
			.Grass,
			.Stone,
			.Wood,
			.Cobblestone,
			.Pumpkin,
			.Brick,
			.TNT,
		}
		state.hotbar_selected_slot = 0
		state.ghost_preview_world_pos = {}
		state.ghost_preview_kind = .None
		state.ghost_preview_visible = false
		state.dig_highlight_world_pos = {}
		state.dig_highlight_visible = false
		if state.water_fluid_seen == nil {
			state.water_fluid_seen = make(map[Water_Fluid_Key]struct{})
		}
		clear(&state.water_fluid_queue)
		clear(&state.water_fluid_seen)
		state.water_fluid_wave_timer = 0
		clear(&state.block_change_queue)
		if state.pending_remesh_seen == nil {
			state.pending_remesh_seen = make(map[Chunk_Coords]struct{})
		}
		clear(&state.pending_remesh_chunks)
		clear(&state.pending_remesh_seen)
		if state.light_increase_seen == nil {
			state.light_increase_seen = make(map[Light_Key]struct{})
		}
		if state.light_decrease_seen == nil {
			state.light_decrease_seen = make(map[Light_Key]struct{})
		}
		clear(&state.light_increase_seen)
		clear(&state.light_decrease_seen)
		clear(&state.armed_tnt)
		clear(&state.tnt_flash_highlight_world_pos)
	}

	state.instance = wgpu.CreateInstance(nil)
	if state.instance == nil {
		panic("WebGPU is not supported")
	}
	state.surface = os_get_surface(state.instance)

	wgpu.InstanceRequestAdapter(
		state.instance,
		&{compatibleSurface = state.surface},
		{callback = on_adapter},
	)

	on_adapter :: proc "c" (
		status: wgpu.RequestAdapterStatus,
		adapter: wgpu.Adapter,
		message: string,
		userdata1, userdata2: rawptr,
	) {
		context = state.ctx
		if status != .Success || adapter == nil {
			fmt.panicf("request adapter failure: [%v] %s", status, message)
		}
		state.adapter = adapter
		wgpu.AdapterRequestDevice(adapter, nil, {callback = on_device})
	}

	on_device :: proc "c" (
		status: wgpu.RequestDeviceStatus,
		device: wgpu.Device,
		message: string,
		userdata1, userdata2: rawptr,
	) {
		context = state.ctx
		if status != .Success || device == nil {
			fmt.panicf("request device failure: [%v] %s", status, message)
		}
		state.device = device

		width, height := os_get_framebuffer_size()

		state.config = wgpu.SurfaceConfiguration {
			device      = state.device,
			usage       = {.RenderAttachment},
			format      = .BGRA8Unorm,
			width       = width,
			height      = height,
			presentMode = .Fifo,
			alphaMode   = .Opaque,
		}
		wgpu.SurfaceConfigure(state.surface, &state.config)

		state.queue = wgpu.DeviceGetQueue(state.device)
		state.chunk_vertex_buffer_pool = make(map[u64][dynamic]wgpu.Buffer)
		state.chunk_index_buffer_pool = make(map[u64][dynamic]wgpu.Buffer)

		shader :: #load("shader.wgsl", string)

		state.module = wgpu.DeviceCreateShaderModule(
			state.device,
			&{nextInChain = &wgpu.ShaderSourceWGSL{sType = .ShaderSourceWGSL, code = shader}},
		)

		bind_layout_entries := [7]wgpu.BindGroupLayoutEntry {
			{
				binding = 0,
				visibility = {.Vertex, .Fragment},
				buffer = {
					type = .Uniform,
					hasDynamicOffset = false,
					minBindingSize = size_of(Shared_Frame_Uniform),
				},
			},
			{
				binding = 1,
				visibility = {.Vertex, .Fragment},
				buffer = {
					type = .Uniform,
					hasDynamicOffset = false,
					minBindingSize = size_of(Scene_Uniform),
				},
			},
			{
				binding = 2,
				visibility = {.Fragment},
				buffer = {
					type = .Uniform,
					hasDynamicOffset = false,
					minBindingSize = size_of(Shared_Fog_Uniform),
				},
			},
			{
				binding = 3,
				visibility = {.Fragment},
				texture = {sampleType = .Float, viewDimension = ._2D, multisampled = false},
			},
			{binding = 4, visibility = {.Fragment}, sampler = {type = .Filtering}},
			{
				binding = 5,
				visibility = {.Fragment},
				texture = {sampleType = .Float, viewDimension = ._2D, multisampled = false},
			},
			{binding = 6, visibility = {.Fragment}, sampler = {type = .Filtering}},
		}
		state.bind_group_layout = wgpu.DeviceCreateBindGroupLayout(
			state.device,
			&{entryCount = 7, entries = &bind_layout_entries[0]},
		)
		shared_frame_bind_layout_entries := [1]wgpu.BindGroupLayoutEntry {
			{
				binding = 0,
				visibility = {.Vertex, .Fragment},
				buffer = {
					type = .Uniform,
					hasDynamicOffset = false,
					minBindingSize = size_of(Shared_Frame_Uniform),
				},
			},
		}
		state.shared_frame_bind_group_layout = wgpu.DeviceCreateBindGroupLayout(
			state.device,
			&{entryCount = 1, entries = &shared_frame_bind_layout_entries[0]},
		)
		state.rain_bind_group_layout = state.shared_frame_bind_group_layout
		cloud_bind_layout_entries := [3]wgpu.BindGroupLayoutEntry {
			{
				binding = 0,
				visibility = {.Vertex, .Fragment},
				buffer = {
					type = .Uniform,
					hasDynamicOffset = false,
					minBindingSize = size_of(Shared_Frame_Uniform),
				},
			},
			{
				binding = 1,
				visibility = {.Vertex, .Fragment},
				buffer = {
					type = .Uniform,
					hasDynamicOffset = false,
					minBindingSize = size_of(Cloud_Uniform),
				},
			},
			{
				binding = 2,
				visibility = {.Fragment},
				buffer = {
					type = .Uniform,
					hasDynamicOffset = false,
					minBindingSize = size_of(Shared_Fog_Uniform),
				},
			},
		}
		state.cloud_bind_group_layout = wgpu.DeviceCreateBindGroupLayout(
			state.device,
			&{entryCount = 3, entries = &cloud_bind_layout_entries[0]},
		)
		state.line_bind_group_layout = state.shared_frame_bind_group_layout

		pipeline_layout_desc := wgpu.PipelineLayoutDescriptor {
			bindGroupLayoutCount = 1,
			bindGroupLayouts     = &state.bind_group_layout,
		}
		state.pipeline_layout = wgpu.DeviceCreatePipelineLayout(
			state.device,
			&pipeline_layout_desc,
		)
		state.shared_frame_pipeline_layout = wgpu.DeviceCreatePipelineLayout(
			state.device,
			&{bindGroupLayoutCount = 1, bindGroupLayouts = &state.shared_frame_bind_group_layout},
		)
		state.rain_pipeline_layout = state.shared_frame_pipeline_layout
		state.cloud_pipeline_layout = wgpu.DeviceCreatePipelineLayout(
			state.device,
			&{bindGroupLayoutCount = 1, bindGroupLayouts = &state.cloud_bind_group_layout},
		)
		state.line_pipeline_layout = state.shared_frame_pipeline_layout

		state.shared_frame_uniform_buffer = wgpu.DeviceCreateBuffer(
			state.device,
			&{usage = {.Uniform, .CopyDst}, size = size_of(Shared_Frame_Uniform)},
		)
		state.shared_fog_uniform_buffer = wgpu.DeviceCreateBuffer(
			state.device,
			&{usage = {.Uniform, .CopyDst}, size = size_of(Shared_Fog_Uniform)},
		)
		state.scene_uniform_buffer = wgpu.DeviceCreateBuffer(
			state.device,
			&{usage = {.Uniform, .CopyDst}, size = size_of(Scene_Uniform)},
		)
		state.cloud_uniform_buffer = wgpu.DeviceCreateBuffer(
			state.device,
			&{usage = {.Uniform, .CopyDst}, size = size_of(Cloud_Uniform)},
		)
		state.ghost_preview_vertex_buffer = wgpu.DeviceCreateBuffer(
			state.device,
			&{usage = {.Vertex, .CopyDst}, size = GHOST_PREVIEW_VERTEX_BUFFER_MAX_BYTES},
		)
		state.ghost_preview_index_buffer = wgpu.DeviceCreateBuffer(
			state.device,
			&{usage = {.Index, .CopyDst}, size = GHOST_PREVIEW_INDEX_BUFFER_MAX_BYTES},
		)
		atlas_resources_create()
		reflection_resources_create()

		scene_bind_groups_rebuild()

		depth_resources_create()

		vertex_attrs := [6]wgpu.VertexAttribute {
			{format = .Float32x3, offset = 0, shaderLocation = 0},
			{format = .Float32x3, offset = 12, shaderLocation = 1},
			{format = .Float32x2, offset = 24, shaderLocation = 2},
			{format = .Float32, offset = 32, shaderLocation = 3},
			{format = .Float32, offset = 36, shaderLocation = 4},
			{format = .Float32, offset = 40, shaderLocation = 5},
		}
		vertex_buffer_layout := wgpu.VertexBufferLayout {
			stepMode       = .Vertex,
			arrayStride    = 44,
			attributeCount = 6,
			attributes     = &vertex_attrs[0],
		}

		stencil_face := wgpu.StencilFaceState {
			compare     = .Always,
			failOp      = .Keep,
			depthFailOp = .Keep,
			passOp      = .Keep,
		}
		depth_stencil := wgpu.DepthStencilState {
			format            = .Depth32Float,
			depthWriteEnabled = .True,
			depthCompare      = .Less,
			stencilFront      = stencil_face,
			stencilBack       = stencil_face,
			stencilReadMask   = 0xFFFFFFFF,
			stencilWriteMask  = 0xFFFFFFFF,
		}

		state.pipeline = wgpu.DeviceCreateRenderPipeline(
			state.device,
			&{
				layout = state.pipeline_layout,
				vertex = {
					module = state.module,
					entryPoint = "vs_main",
					bufferCount = 1,
					buffers = &vertex_buffer_layout,
				},
				fragment = &{
					module = state.module,
					entryPoint = "fs_main",
					targetCount = 1,
					targets = &wgpu.ColorTargetState {
						format = .BGRA8Unorm,
						writeMask = wgpu.ColorWriteMaskFlags_All,
					},
				},
				primitive = {
					topology = .TriangleList,
					stripIndexFormat = .Undefined,
					frontFace = .CCW,
					cullMode = .Back,
				},
				depthStencil = &depth_stencil,
				multisample = {count = 1, mask = 0xFFFFFFFF},
			},
		)
		state.reflection_pipeline = wgpu.DeviceCreateRenderPipeline(
			state.device,
			&{
				layout = state.pipeline_layout,
				vertex = {
					module = state.module,
					entryPoint = "vs_main",
					bufferCount = 1,
					buffers = &vertex_buffer_layout,
				},
				fragment = &{
					module = state.module,
					entryPoint = "fs_main",
					targetCount = 1,
					targets = &wgpu.ColorTargetState {
						format = .BGRA8Unorm,
						writeMask = wgpu.ColorWriteMaskFlags_All,
					},
				},
				primitive = {
					topology         = .TriangleList,
					stripIndexFormat = .Undefined,
					frontFace        = .CCW,
					// Mirrored reflections can invert winding; disable culling for safety.
					cullMode         = .None,
				},
				depthStencil = &depth_stencil,
				multisample = {count = 1, mask = 0xFFFFFFFF},
			},
		)
		flower_cutout_depth_stencil := depth_stencil
		flower_cutout_depth_stencil.depthWriteEnabled = .True
		state.pipeline_flower_cutout = wgpu.DeviceCreateRenderPipeline(
			state.device,
			&{
				layout = state.pipeline_layout,
				vertex = {
					module = state.module,
					entryPoint = "vs_main",
					bufferCount = 1,
					buffers = &vertex_buffer_layout,
				},
				fragment = &{
					module = state.module,
					entryPoint = "fs_main",
					targetCount = 1,
					targets = &wgpu.ColorTargetState {
						format = .BGRA8Unorm,
						writeMask = wgpu.ColorWriteMaskFlags_All,
					},
				},
				primitive = {
					topology = .TriangleList,
					stripIndexFormat = .Undefined,
					frontFace = .CCW,
					cullMode = .Back,
				},
				depthStencil = &flower_cutout_depth_stencil,
				multisample = {count = 1, mask = 0xFFFFFFFF},
			},
		)
		blend_alpha := wgpu.BlendState {
			color = {operation = .Add, srcFactor = .SrcAlpha, dstFactor = .OneMinusSrcAlpha},
			alpha = {operation = .Add, srcFactor = .One, dstFactor = .OneMinusSrcAlpha},
		}
		transparent_depth_stencil := depth_stencil
		transparent_depth_stencil.depthWriteEnabled = .False
		state.pipeline_transparent = wgpu.DeviceCreateRenderPipeline(
			state.device,
			&{
				layout = state.pipeline_layout,
				vertex = {
					module = state.module,
					entryPoint = "vs_main",
					bufferCount = 1,
					buffers = &vertex_buffer_layout,
				},
				fragment = &{
					module = state.module,
					entryPoint = "fs_main",
					targetCount = 1,
					targets = &wgpu.ColorTargetState {
						format = .BGRA8Unorm,
						blend = &blend_alpha,
						writeMask = wgpu.ColorWriteMaskFlags_All,
					},
				},
				primitive = {
					topology = .TriangleList,
					stripIndexFormat = .Undefined,
					frontFace = .CCW,
					cullMode = .Back,
				},
				depthStencil = &transparent_depth_stencil,
				multisample = {count = 1, mask = 0xFFFFFFFF},
			},
		)

		line_shader :: #load("line_shader.wgsl", string)
		state.line_module = wgpu.DeviceCreateShaderModule(
			state.device,
			&{nextInChain = &wgpu.ShaderSourceWGSL{sType = .ShaderSourceWGSL, code = line_shader}},
		)
		line_vertex_attrs := [2]wgpu.VertexAttribute {
			{format = .Float32x3, offset = 0, shaderLocation = 0},
			{format = .Float32x4, offset = 16, shaderLocation = 1},
		}
		line_vertex_layout := wgpu.VertexBufferLayout {
			stepMode       = .Vertex,
			arrayStride    = size_of(Chunk_Bounds_Vertex),
			attributeCount = 2,
			attributes     = &line_vertex_attrs[0],
		}
		line_depth_stencil := wgpu.DepthStencilState {
			format            = .Depth32Float,
			depthWriteEnabled = .False,
			depthCompare      = .LessEqual,
			stencilFront      = stencil_face,
			stencilBack       = stencil_face,
			stencilReadMask   = 0xFFFFFFFF,
			stencilWriteMask  = 0xFFFFFFFF,
		}
		state.line_pipeline = wgpu.DeviceCreateRenderPipeline(
			state.device,
			&{
				layout = state.line_pipeline_layout,
				vertex = {
					module = state.line_module,
					entryPoint = "vs_main",
					bufferCount = 1,
					buffers = &line_vertex_layout,
				},
				fragment = &{
					module = state.line_module,
					entryPoint = "fs_main",
					targetCount = 1,
					targets = &wgpu.ColorTargetState {
						format = .BGRA8Unorm,
						blend = &wgpu.BlendState {
							color = {
								operation = .Add,
								srcFactor = .SrcAlpha,
								dstFactor = .OneMinusSrcAlpha,
							},
							alpha = {
								operation = .Add,
								srcFactor = .One,
								dstFactor = .OneMinusSrcAlpha,
							},
						},
						writeMask = wgpu.ColorWriteMaskFlags_All,
					},
				},
				primitive = {
					topology = .LineList,
					stripIndexFormat = .Undefined,
					frontFace = .CCW,
					cullMode = .None,
				},
				depthStencil = &line_depth_stencil,
				multisample = {count = 1, mask = 0xFFFFFFFF},
			},
		)

		rain_shader :: #load("rain.wgsl", string)
		state.rain_module = wgpu.DeviceCreateShaderModule(
			state.device,
			&{nextInChain = &wgpu.ShaderSourceWGSL{sType = .ShaderSourceWGSL, code = rain_shader}},
		)
		state.rain_pipeline = wgpu.DeviceCreateRenderPipeline(
			state.device,
			&{
				layout = state.rain_pipeline_layout,
				vertex = {
					module = state.rain_module,
					entryPoint = "vs_main",
					bufferCount = 0,
					buffers = nil,
				},
				fragment = &{
					module = state.rain_module,
					entryPoint = "fs_main",
					targetCount = 1,
					targets = &wgpu.ColorTargetState {
						format = .BGRA8Unorm,
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
				depthStencil = &line_depth_stencil,
				multisample = {count = 1, mask = 0xFFFFFFFF},
			},
		)

		cloud_shader :: #load("cloud.wgsl", string)
		state.cloud_module = wgpu.DeviceCreateShaderModule(
			state.device,
			&{
				nextInChain = &wgpu.ShaderSourceWGSL {
					sType = .ShaderSourceWGSL,
					code = cloud_shader,
				},
			},
		)
		state.cloud_pipeline = wgpu.DeviceCreateRenderPipeline(
			state.device,
			&{
				layout = state.cloud_pipeline_layout,
				vertex = {
					module = state.cloud_module,
					entryPoint = "vs_main",
					bufferCount = 0,
					buffers = nil,
				},
				fragment = &{
					module = state.cloud_module,
					entryPoint = "fs_main",
					targetCount = 1,
					targets = &wgpu.ColorTargetState {
						format = .BGRA8Unorm,
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
				depthStencil = &line_depth_stencil,
				multisample = {count = 1, mask = 0xFFFFFFFF},
			},
		)

		state.depth_sample_sampler = wgpu.DeviceCreateSampler(
			state.device,
			&{
				label = "depth_sample_sampler",
				addressModeU = .ClampToEdge,
				addressModeV = .ClampToEdge,
				addressModeW = .ClampToEdge,
				magFilter = .Linear,
				minFilter = .Linear,
				mipmapFilter = .Nearest,
				lodMinClamp = 0,
				lodMaxClamp = 32,
				maxAnisotropy = 1,
			},
		)

		post_layout_entries := [6]wgpu.BindGroupLayoutEntry {
			{
				binding = 0,
				visibility = {.Fragment},
				buffer = {
					type = .Uniform,
					hasDynamicOffset = false,
					minBindingSize = POST_UNIFORM_SIZE,
				},
			},
			{
				binding = 1,
				visibility = {.Fragment},
				texture = {sampleType = .Float, viewDimension = ._2D, multisampled = false},
			},
			{binding = 2, visibility = {.Fragment}, sampler = {type = .Filtering}},
			{
				binding = 3,
				visibility = {.Fragment},
				texture = {sampleType = .Depth, viewDimension = ._2D, multisampled = false},
			},
			{binding = 4, visibility = {.Fragment}, sampler = {type = .Filtering}},
			{
				binding = 5,
				visibility = {.Fragment},
				texture = {sampleType = .Float, viewDimension = ._2D, multisampled = false},
			},
		}
		state.post_bind_group_layout = wgpu.DeviceCreateBindGroupLayout(
			state.device,
			&{entryCount = 6, entries = &post_layout_entries[0]},
		)
		state.post_pipeline_layout = wgpu.DeviceCreatePipelineLayout(
			state.device,
			&{bindGroupLayoutCount = 1, bindGroupLayouts = &state.post_bind_group_layout},
		)

		post_effects_shader :: #load("post_effects.wgsl", string)
		state.post_effects_module = wgpu.DeviceCreateShaderModule(
			state.device,
			&{
				nextInChain = &wgpu.ShaderSourceWGSL {
					sType = .ShaderSourceWGSL,
					code = post_effects_shader,
				},
			},
		)

		state.post_uniform_buffer = wgpu.DeviceCreateBuffer(
			state.device,
			&{usage = {.Uniform, .CopyDst}, size = POST_UNIFORM_SIZE},
		)

		state.post_pipeline = wgpu.DeviceCreateRenderPipeline(
			state.device,
			&{
				layout = state.post_pipeline_layout,
				vertex = {
					module = state.post_effects_module,
					entryPoint = "vs_main",
					bufferCount = 0,
					buffers = nil,
				},
				fragment = &{
					module = state.post_effects_module,
					entryPoint = "fs_main",
					targetCount = 1,
					targets = &wgpu.ColorTargetState {
						format = .BGRA8Unorm,
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
		dof_layout_entries := [6]wgpu.BindGroupLayoutEntry {
			{
				binding = 0,
				visibility = {.Fragment},
				buffer = {
					type = .Uniform,
					hasDynamicOffset = false,
					minBindingSize = POST_UNIFORM_SIZE,
				},
			},
			{
				binding = 1,
				visibility = {.Fragment},
				texture = {sampleType = .Float, viewDimension = ._2D, multisampled = false},
			},
			{binding = 2, visibility = {.Fragment}, sampler = {type = .Filtering}},
			{
				binding = 3,
				visibility = {.Fragment},
				texture = {sampleType = .Float, viewDimension = ._2D, multisampled = false},
			},
			{
				binding = 4,
				visibility = {.Fragment},
				texture = {sampleType = .Depth, viewDimension = ._2D, multisampled = false},
			},
			{binding = 5, visibility = {.Fragment}, sampler = {type = .Filtering}},
		}
		state.dof_final_bind_group_layout = wgpu.DeviceCreateBindGroupLayout(
			state.device,
			&{entryCount = 6, entries = &dof_layout_entries[0]},
		)
		state.dof_final_pipeline_layout = wgpu.DeviceCreatePipelineLayout(
			state.device,
			&{bindGroupLayoutCount = 1, bindGroupLayouts = &state.dof_final_bind_group_layout},
		)
		dof_shader :: #load("dof.wgsl", string)
		state.dof_module = wgpu.DeviceCreateShaderModule(
			state.device,
			&{nextInChain = &wgpu.ShaderSourceWGSL{sType = .ShaderSourceWGSL, code = dof_shader}},
		)
		state.dof_final_pipeline = wgpu.DeviceCreateRenderPipeline(
			state.device,
			&{
				layout = state.dof_final_pipeline_layout,
				vertex = {
					module = state.dof_module,
					entryPoint = "vs_main",
					bufferCount = 0,
					buffers = nil,
				},
				fragment = &{
					module = state.dof_module,
					entryPoint = "fs_main",
					targetCount = 1,
					targets = &wgpu.ColorTargetState {
						format = .BGRA8Unorm,
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

		bloom_shader :: #load("bloom.wgsl", string)
		state.bloom_module = wgpu.DeviceCreateShaderModule(
			state.device,
			&{
				nextInChain = &wgpu.ShaderSourceWGSL {
					sType = .ShaderSourceWGSL,
					code = bloom_shader,
				},
			},
		)
		state.bloom_sampler = wgpu.DeviceCreateSampler(
			state.device,
			&{
				label = "bloom_sampler",
				addressModeU = .ClampToEdge,
				addressModeV = .ClampToEdge,
				addressModeW = .ClampToEdge,
				magFilter = .Linear,
				minFilter = .Linear,
				mipmapFilter = .Nearest,
				lodMinClamp = 0,
				lodMaxClamp = 32,
				maxAnisotropy = 1,
			},
		)
		state.bloom_uniform_buffer = wgpu.DeviceCreateBuffer(
			state.device,
			&{usage = {.Uniform, .CopyDst}, size = BLOOM_UNIFORM_SIZE},
		)
		bloom_layout_entries := [3]wgpu.BindGroupLayoutEntry {
			{
				binding = 0,
				visibility = {.Fragment},
				buffer = {
					type = .Uniform,
					hasDynamicOffset = false,
					minBindingSize = BLOOM_UNIFORM_SIZE,
				},
			},
			{
				binding = 1,
				visibility = {.Fragment},
				texture = {sampleType = .Float, viewDimension = ._2D, multisampled = false},
			},
			{binding = 2, visibility = {.Fragment}, sampler = {type = .Filtering}},
		}
		state.bloom_bind_group_layout = wgpu.DeviceCreateBindGroupLayout(
			state.device,
			&{entryCount = 3, entries = &bloom_layout_entries[0]},
		)
		state.bloom_pipeline_layout = wgpu.DeviceCreatePipelineLayout(
			state.device,
			&{bindGroupLayoutCount = 1, bindGroupLayouts = &state.bloom_bind_group_layout},
		)
		state.bloom_extract_pipeline = wgpu.DeviceCreateRenderPipeline(
			state.device,
			&{
				layout = state.bloom_pipeline_layout,
				vertex = {
					module = state.bloom_module,
					entryPoint = "vs_main",
					bufferCount = 0,
					buffers = nil,
				},
				fragment = &{
					module = state.bloom_module,
					entryPoint = "fs_extract",
					targetCount = 1,
					targets = &wgpu.ColorTargetState {
						format = .BGRA8Unorm,
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
		state.bloom_blur_pipeline = wgpu.DeviceCreateRenderPipeline(
			state.device,
			&{
				layout = state.bloom_pipeline_layout,
				vertex = {
					module = state.bloom_module,
					entryPoint = "vs_main",
					bufferCount = 0,
					buffers = nil,
				},
				fragment = &{
					module = state.bloom_module,
					entryPoint = "fs_blur",
					targetCount = 1,
					targets = &wgpu.ColorTargetState {
						format = .BGRA8Unorm,
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
		state.dof_extract_pipeline = wgpu.DeviceCreateRenderPipeline(
			state.device,
			&{
				layout = state.bloom_pipeline_layout,
				vertex = {
					module = state.bloom_module,
					entryPoint = "vs_main",
					bufferCount = 0,
					buffers = nil,
				},
				fragment = &{
					module = state.bloom_module,
					entryPoint = "fs_copy",
					targetCount = 1,
					targets = &wgpu.ColorTargetState {
						format = .BGRA8Unorm,
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

		scene_color_resources_create()

		terrain_gpu_create()
		chunks_stream_update()
		clay_renderer_init(
			state.device,
			state.queue,
			state.config.format,
			state.config.width,
			state.config.height,
		)

		os_run()
	}
}

resize :: proc "c" () {
	context = state.ctx

	state.config.width, state.config.height = os_get_framebuffer_size()
	wgpu.SurfaceConfigure(state.surface, &state.config)
	clay_renderer_resize(state.config.width, state.config.height)
	depth_resources_create()
	reflection_resources_create()
	scene_color_resources_create()
	scene_bind_groups_rebuild()
}

build_clay_ui :: proc() {
	clay.SetCurrentContext(clay_renderer_state.clay_ctx)
	cell_count :: HOTBAR_SLOT_COUNT
	cell_size: u16 : 120
	cell_gap: u16 : 20
	bar_padding: u16 : 20
	cell_inner_padding: u16 : 12
	bottom_margin: f32 : 18

	corner_radius := clay.CornerRadiusAll(25)

	if clay.UI(clay.ID("DemoBarRoot"))(
	{
		layout = {
			sizing = {width = clay.SizingGrow(), height = clay.SizingGrow()},
			childAlignment = {x = .Center, y = .Bottom},
		},
	},
	) {
		if clay.UI(clay.ID("DemoBar"))(
		{
			layout = {
				sizing = {width = clay.SizingFit(), height = clay.SizingFit()},
				padding = clay.PaddingAll(bar_padding),
				childGap = cell_gap,
				layoutDirection = .LeftToRight,
				childAlignment = {x = .Left, y = .Center},
			},
			// Pin the bar to the bottom independent of other root children.
			floating = {
				offset = {0, -bottom_margin},
				attachment = {element = .CenterBottom, parent = .CenterBottom},
				attachTo = .Root,
			},
			backgroundColor = {18, 26, 38, 230},
			cornerRadius = corner_radius,
			// border = {color = {90, 130, 220, 255}, width = clay.BorderOutside(bar_border)},
		},
		) {
			for i in 0 ..< cell_count {
				slot_kind := state.hotbar_slot_kinds[i]
				is_selected := state.hotbar_selected_slot == i
				cell_bg := clay.Color{28, 38, 52, 240}
				cell_border_color := clay.Color{76, 94, 118, 255}
				cell_border_width := u16(2)
				if is_selected {
					cell_bg = {54, 74, 102, 255}
					cell_border_color = {205, 223, 255, 255}
					cell_border_width = 4
				}

				if clay.UI(clay.ID("DemoBarCell", u32(i)))(
				{
					layout = {
						sizing = {
							width = clay.SizingFixed(f32(cell_size)),
							height = clay.SizingFixed(f32(cell_size)),
						},
						// padding = clay.PaddingAll(cell_inner_padding),
					},
					backgroundColor = cell_bg,
					cornerRadius = corner_radius,
					border = {
						color = cell_border_color,
						width = clay.BorderOutside(cell_border_width),
					},
				},
				) {
					if slot_kind != .None {
						tile := atlas_coords_for_face(slot_kind, .NegZ, for_hotbar = true)
						uv_min := atlas_uv_for_corner(tile, {0, 0})
						uv_max := atlas_uv_for_corner(tile, {1, 1})
						state.hotbar_image_handles[i] = {
							view        = state.atlas_view,
							sampler     = state.atlas_sampler,
							use_uv_rect = true,
							uv_min      = uv_min,
							uv_max      = uv_max,
						}
						if clay.UI(clay.ID("DemoBarCellImage", u32(i)))(
						{
							layout = {
								sizing = {width = clay.SizingGrow(), height = clay.SizingGrow()},
							},
							image = {imageData = &state.hotbar_image_handles[i]},
							cornerRadius = corner_radius,
						},
						) {
						}
					}
				}
			}
		}
	}
}

frame :: proc "c" (dt: f32) {
	context = state.ctx
	free_all(context.temp_allocator)
	sync_game_process_active()

	// Game step
	if state.game_process_active {
		state.elapsed_time += dt
		if .F1 in state.keys_just_pressed {
			state.player_noclip = !state.player_noclip
			if state.player_noclip {
				state.player_vel.y = 0
				state.player_on_ground = false
			}
		}
		if .F2 in state.keys_just_pressed {
			state.debug_chunk_bounds = !state.debug_chunk_bounds
		}
		if .F3 in state.keys_just_pressed {
			state.rain_enabled = !state.rain_enabled
		}
		if .F4 in state.keys_just_pressed {
			state.bloom_enabled = !state.bloom_enabled
		}
		if .F5 in state.keys_just_pressed {
			state.clouds_enabled = !state.clouds_enabled
		}
		if .F6 in state.keys_just_pressed {
			state.fog_enabled = !state.fog_enabled
		}
		if .F7 in state.keys_just_pressed {
			state.fxaa_enabled = !state.fxaa_enabled
		}
		if .F8 in state.keys_just_pressed {
			state.dof_enabled = !state.dof_enabled
		}

		// Hotbar selection via mouse wheel: positive scroll moves to the next slot.
		HOTBAR_SCROLL_MAX_STEPS_PER_FRAME :: 3
		if state.mouse_wheel_steps != 0 {
			steps := state.mouse_wheel_steps
			if steps > HOTBAR_SCROLL_MAX_STEPS_PER_FRAME do steps = HOTBAR_SCROLL_MAX_STEPS_PER_FRAME
			if steps < -HOTBAR_SCROLL_MAX_STEPS_PER_FRAME do steps = -HOTBAR_SCROLL_MAX_STEPS_PER_FRAME

			slot := state.hotbar_selected_slot + steps
			slot %= HOTBAR_SLOT_COUNT
			if slot < 0 do slot += HOTBAR_SLOT_COUNT
			state.hotbar_selected_slot = slot
		}

		if .Num1 in state.keys_just_pressed do state.hotbar_selected_slot = 0
		if .Num2 in state.keys_just_pressed do state.hotbar_selected_slot = 1
		if .Num3 in state.keys_just_pressed do state.hotbar_selected_slot = 2
		if .Num4 in state.keys_just_pressed do state.hotbar_selected_slot = 3
		if .Num5 in state.keys_just_pressed do state.hotbar_selected_slot = 4
		if .Num6 in state.keys_just_pressed do state.hotbar_selected_slot = 5
		if .Num7 in state.keys_just_pressed do state.hotbar_selected_slot = 6
		if .Num8 in state.keys_just_pressed do state.hotbar_selected_slot = 7
		if .Num9 in state.keys_just_pressed do state.hotbar_selected_slot = 8

		water_fluid_process_waves(dt)

		// Player rotation
		{
			MOUSE_SENS :: 0.002
			MAX_PITCH :: math.PI / 2 - 0.01
			angle_radians := state.mouse_delta * MOUSE_SENS

			yaw_quat := linalg.quaternion_angle_axis_f32(-angle_radians.x, WORLD_UP)

			state.player_rotation = linalg.normalize(yaw_quat * state.player_rotation)

			right := linalg.quaternion_mul_vector3(state.player_rotation, WORLD_RIGHT)

			target_pitch := math.clamp(state.player_pitch + angle_radians.y, -MAX_PITCH, MAX_PITCH)
			pitch_delta := target_pitch - state.player_pitch
			state.player_pitch = target_pitch

			// Keep pitch away from +-90deg so controls never flip.
			pitch_quat := linalg.quaternion_angle_axis_f32(pitch_delta, right)

			state.player_rotation = linalg.normalize(pitch_quat * state.player_rotation)
		}

		forward := linalg.quaternion_mul_vector3(state.player_rotation, WORLD_FORWARD)
		right := linalg.normalize(linalg.cross(forward, WORLD_UP))

		// Player movement
		{
			if state.player_noclip {
				move: [3]f32

				if .W in state.keys_down do move += forward
				if .S in state.keys_down do move -= forward
				if .A in state.keys_down do move -= right
				if .D in state.keys_down do move += right
				if .Shift in state.keys_down do move -= WORLD_UP
				if .Space in state.keys_down do move += WORLD_UP

				if move != {} {
					move = linalg.normalize(move)
				}

				NOCLIP_SPEED :: 12.0
				NOCLIP_ACCEL :: 14.0
				NOCLIP_DECEL :: 10.0
				desired_vel := move * NOCLIP_SPEED
				if move != {} {
					accel_alpha := math.clamp(NOCLIP_ACCEL * dt, 0, 1)
					state.player_vel += (desired_vel - state.player_vel) * accel_alpha
				} else {
					decel_alpha := math.clamp(NOCLIP_DECEL * dt, 0, 1)
					state.player_vel += (desired_vel - state.player_vel) * decel_alpha
				}
				state.player_pos += state.player_vel * dt
			} else {
				WALK_SPEED :: 4.3
				GROUND_ACCEL :: 20.0
				GROUND_DECEL :: 14.0
				AIR_ACCEL :: 3.0
				AIR_DECEL :: 1.0
				GRAVITY :: -20.0
				TERMINAL_FALL_SPEED :: -55.0
				JUMP_VELOCITY :: 8.5
				SWIM_SPEED :: 3.0
				WATER_ACCEL :: 12.0
				WATER_DECEL :: 9.0
				// Net upward acceleration so idle players drift toward the surface; Shift fights this.
				WATER_BUOYANCY :: 1.15
				// Deep/enclosed water: gentle sink so neutral buoyancy isn't "hovering in place".
				WATER_NO_BUOYANCY_DOWN_ACCEL :: -2
				WATER_TERMINAL_Y :: 4.2

				in_water := player_in_water()

				if in_water {
					move: [3]f32
					if .W in state.keys_down do move += forward
					if .S in state.keys_down do move -= forward
					if .A in state.keys_down do move -= right
					if .D in state.keys_down do move += right
					if .Shift in state.keys_down do move -= WORLD_UP
					if .Space in state.keys_down do move += WORLD_UP
					if move != {} {
						move = linalg.normalize(move)
					}
					desired_vel := move * SWIM_SPEED
					if move != {} {
						accel_alpha := math.clamp(WATER_ACCEL * dt, 0, 1)
						// desired_vel includes pitch (W/S along look); buoyancy still applied below.
						state.player_vel.x += (desired_vel.x - state.player_vel.x) * accel_alpha
						state.player_vel.y += (desired_vel.y - state.player_vel.y) * accel_alpha
						state.player_vel.z += (desired_vel.z - state.player_vel.z) * accel_alpha
					} else {
						decel_alpha := math.clamp(WATER_DECEL * dt, 0, 1)
						state.player_vel.x += (0 - state.player_vel.x) * decel_alpha
						state.player_vel.z += (0 - state.player_vel.z) * decel_alpha
						state.player_vel.y += (0 - state.player_vel.y) * decel_alpha * 0.2
					}
					near_surface, buoyancy_factor := player_water_buoyancy_sample()
					if near_surface {
						state.player_vel.y += WATER_BUOYANCY * buoyancy_factor * dt
					} else {
						state.player_vel.y += WATER_NO_BUOYANCY_DOWN_ACCEL * dt
					}
					state.player_vel.y = math.clamp(
						state.player_vel.y,
						-WATER_TERMINAL_Y,
						WATER_TERMINAL_Y,
					)

					// Head above water but torso still in swim mode: jump against a wall to clear a ledge / lip.
					if !player_eye_in_water() &&
					   player_hugging_wall() &&
					   .Space in state.keys_down {
						state.player_vel.y = JUMP_VELOCITY
					}

					delta := state.player_vel * dt
					collider_pos := state.player_pos
					collider_pos.y -= PLAYER_EYE_OFFSET_FROM_CENTER_Y
					pos, blocked_x := move_axis_with_collision(
						collider_pos,
						delta.x,
						0,
						PLAYER_HALF_EXTENTS,
					)
					blocked_z: bool
					pos, blocked_z = move_axis_with_collision(pos, delta.z, 2, PLAYER_HALF_EXTENTS)
					blocked_y: bool
					pos, blocked_y = move_axis_with_collision(pos, delta.y, 1, PLAYER_HALF_EXTENTS)

					state.player_on_ground = blocked_y && delta.y < 0
					if blocked_y {
						state.player_vel.y = 0
					}
					if blocked_x {
						state.player_vel.x = 0
					}
					if blocked_z {
						state.player_vel.z = 0
					}

					state.player_pos = pos
					state.player_pos.y += PLAYER_EYE_OFFSET_FROM_CENTER_Y
				} else {
					horizontal_forward := linalg.normalize0([3]f32{forward.x, 0, forward.z})
					horizontal_right := linalg.normalize0([3]f32{right.x, 0, right.z})

					move: [3]f32
					if .W in state.keys_down do move += horizontal_forward
					if .S in state.keys_down do move -= horizontal_forward
					if .A in state.keys_down do move -= horizontal_right
					if .D in state.keys_down do move += horizontal_right
					if move != {} {
						move = linalg.normalize(move)
					}

					if state.player_on_ground && .Space in state.keys_down && !in_water {
						// Gate jump on edge-triggered input so holding Space does not auto-bhop every landing.
						state.player_vel.y = JUMP_VELOCITY
						state.player_on_ground = false
					}

					desired_horizontal_vel := move * WALK_SPEED
					horizontal_vel := [3]f32{state.player_vel.x, 0, state.player_vel.z}
					accel: f32 = AIR_ACCEL
					decel: f32 = AIR_DECEL
					if state.player_on_ground {
						// Minecraft-like control: sharp ground response, softer air steering.
						accel = GROUND_ACCEL
						decel = GROUND_DECEL
					}
					if move != {} {
						accel_alpha := math.clamp(accel * dt, 0, 1)
						horizontal_vel += (desired_horizontal_vel - horizontal_vel) * accel_alpha
					} else {
						decel_alpha := math.clamp(decel * dt, 0, 1)
						horizontal_vel += (desired_horizontal_vel - horizontal_vel) * decel_alpha
					}

					state.player_vel.y += GRAVITY * dt
					state.player_vel.y = math.max(state.player_vel.y, TERMINAL_FALL_SPEED)

					horizontal_delta := horizontal_vel * dt
					vertical_delta := state.player_vel.y * dt

					start := state.player_pos
					collider_pos := start
					// Camera position is eye-level, but collision should resolve against the body volume below it.
					collider_pos.y -= PLAYER_EYE_OFFSET_FROM_CENTER_Y
					pos, blocked_x := move_axis_with_collision(
						collider_pos,
						horizontal_delta.x,
						0,
						PLAYER_HALF_EXTENTS,
					)
					blocked_z: bool
					pos, blocked_z = move_axis_with_collision(
						pos,
						horizontal_delta.z,
						2,
						PLAYER_HALF_EXTENTS,
					)
					blocked_y: bool
					pos, blocked_y = move_axis_with_collision(
						pos,
						vertical_delta,
						1,
						PLAYER_HALF_EXTENTS,
					)

					state.player_on_ground = blocked_y && vertical_delta < 0
					if blocked_y {
						state.player_vel.y = 0
					}
					if blocked_x {
						horizontal_vel.x = 0
					}
					if blocked_z {
						horizontal_vel.z = 0
					}

					state.player_pos = pos
					state.player_pos.y += PLAYER_EYE_OFFSET_FROM_CENTER_Y
					state.player_vel.x = horizontal_vel.x
					state.player_vel.z = horizontal_vel.z
				}
			}
		}

		// Player digging
		{
			state.ghost_preview_visible = false
			state.ghost_preview_kind = .None
			state.dig_highlight_visible = false
			kind_to_place := state.hotbar_slot_kinds[state.hotbar_selected_slot]
			if kind_to_place != .None {
				place_pos, place_ok := place_raycast_last_empty_before_solid(
					state.player_pos,
					forward,
					DIG_MAX_REACH,
				)
				if place_ok {
					_, _, can_place := resolve_placeable_voxel_at_world(place_pos)
					if can_place {
						state.ghost_preview_world_pos = {
							f32(int(math.floor(place_pos.x))),
							f32(int(math.floor(place_pos.y))),
							f32(int(math.floor(place_pos.z))),
						}
						state.ghost_preview_kind = kind_to_place
						state.ghost_preview_visible = true
					}
				}
			}

			hit_pos, hit_ok := dig_raycast_first_solid(state.player_pos, forward, DIG_MAX_REACH)
			cube_kind := Cube_Kind.None
			if hit_ok {
				cube_kind, _ = cube_at(hit_pos)
				if cube_kind != .None && cube_kind != .Bedrock {
					state.dig_highlight_world_pos = {
						f32(int(math.floor(hit_pos.x))),
						f32(int(math.floor(hit_pos.y))),
						f32(int(math.floor(hit_pos.z))),
					}
					state.dig_highlight_visible = true
				}
			}

			if .Left_Mouse_Button in state.keys_just_pressed {
				if hit_ok {
					if state.dig_highlight_visible {
						if cube_kind == .TNT {
							push_dir := [3]f32{forward.x, 0, forward.z}
							if push_dir == {} {
								push_dir = WORLD_FORWARD
							} else {
								push_dir = linalg.normalize(push_dir)
							}
							arm_vel := push_dir * TNT_ARM_KICK_SPEED
							arm_vel.y = TNT_ARM_UPWARD_KICK
							_ = tnt_arm_if_needed(hit_pos, arm_vel)
						} else {
							_, _, _ = remove_voxel_at_world_internal(hit_pos)
						}
					}
				}
			}
			if .Right_Mouse_Button in state.keys_just_pressed {
				place_pos, place_ok := place_raycast_last_empty_before_solid(
					state.player_pos,
					forward,
					DIG_MAX_REACH,
				)
				if place_ok {
					if kind_to_place != .None {
						chunk_pos, local, ok := resolve_placeable_voxel_at_world(place_pos)
						if ok {
							_ = block_change_enqueue(chunk_pos, local, kind_to_place)
						}
					}
				}
			}
		}
		block_change_apply_batch()

		tnt_update_fuses(dt)

		// Falling sand
		{
			// Sand can move many voxels in one tick, so route all edits through one batched lighting/remesh commit.
			for chunk_coords, &chunk in state.chunks {
				for cube_kind, src_idx in chunk.kinds {
					if cube_kind != .Sand do continue

					src_local := local_from_index(Cube_Index(src_idx))
					dst_local := src_local - {0, 1, 0}

					{
						below_kind := cube_kind_at_local(chunk, dst_local) or_continue
						if below_kind != .None && below_kind != .Water do continue
					}

					dst_idx := cube_index(dst_local)

					if chunk.kinds[dst_idx] != .None && chunk.kinds[dst_idx] != .Water do continue
					_ = block_change_enqueue(chunk_coords, src_local, .None)
					_ = block_change_enqueue(chunk_coords, dst_local, .Sand)
				}
			}
			block_change_apply_batch()
		}
		remesh_queue_process_budgeted(REMESH_CHUNKS_PER_FRAME)
		chunks_stream_update()
	}

	surface_texture := wgpu.SurfaceGetCurrentTexture(state.surface)
	switch surface_texture.status {
	case .SuccessOptimal, .SuccessSuboptimal:
	// All good, could handle suboptimal here.
	case .Timeout, .Outdated, .Lost:
		// Skip this frame, and re-configure surface.
		if surface_texture.texture != nil {
			wgpu.TextureRelease(surface_texture.texture)
		}
		resize()
		return
	case .OutOfMemory, .DeviceLost, .Error:
		// Fatal error
		fmt.panicf("get_current_texture status=%v", surface_texture.status)
	}
	defer wgpu.TextureRelease(surface_texture.texture)

	frame_view := wgpu.TextureCreateView(surface_texture.texture, nil)
	defer wgpu.TextureViewRelease(frame_view)

	if state.depth_view == nil {
		return
	}
	if state.scene_color_view == nil {
		return
	}
	if state.bloom_view_a == nil ||
	   state.bloom_extract_bind_group == nil ||
	   state.post_composite_view == nil ||
	   state.dof_view_a == nil ||
	   state.dof_extract_bind_group == nil ||
	   state.dof_extract_pipeline == nil ||
	   state.dof_final_bind_group == nil ||
	   state.dof_final_pipeline == nil ||
	   state.post_bind_group == nil {
		return
	}
	if state.reflection_view == nil || state.reflection_depth_view == nil {
		return
	}
	clay_renderer_begin_frame(dt, state.mouse_pos, .Left_Mouse_Button in state.keys_down)
	build_clay_ui()

	view_proj: matrix[4, 4]f32
	forward := linalg.quaternion128_mul_vector3(state.player_rotation, WORLD_FORWARD)
	aspect := f32(state.config.width) / max(f32(state.config.height), 1)
	proj := linalg.matrix4_perspective_f32(
		math.to_radians_f32(CAMERA_FOV_DEG),
		aspect,
		CAMERA_NEAR_PLANE,
		CAMERA_FAR_PLANE,
		true,
	)
	{
		forward = linalg.normalize0(forward)
		centre := state.player_pos + forward
		view := linalg.matrix4_look_at_f32(state.player_pos, centre, WORLD_UP, true)
		view_proj = linalg.mul(proj, view)
	}

	shared_frame_uniform := Shared_Frame_Uniform {
		mvp          = view_proj,
		camera_pos   = state.player_pos,
		elapsed_time = state.elapsed_time,
	}
	shared_fog_uniform: Shared_Fog_Uniform
	scene_uniform: Scene_Uniform
	sea_plane_y := f32(SEA_LEVEL) + WATER_SURFACE_Y_OFFSET
	mirrored_pos := state.player_pos
	mirrored_pos.y = 2.0 * sea_plane_y - mirrored_pos.y
	mirrored_forward := forward
	mirrored_forward.y = -mirrored_forward.y
	mirrored_view := linalg.matrix4_look_at_f32(
		mirrored_pos,
		mirrored_pos + mirrored_forward,
		WORLD_UP,
		true,
	)
	reflection_view_proj := linalg.mul(proj, mirrored_view)
	scene_uniform.reflection_view_proj = reflection_view_proj
	scene_uniform.reflection_plane_y = sea_plane_y
	fog_far_world := f32(CHUNK_STREAM_LOAD_RADIUS) * f32(CHUNK_WIDTH) * 1
	fog_near_world := FOG_NEAR_FACTOR * fog_far_world
	shared_fog_uniform.fog_color_near = {
		f32(SKY_CLEAR_COLOR[0]),
		f32(SKY_CLEAR_COLOR[1]),
		f32(SKY_CLEAR_COLOR[2]),
		fog_near_world,
	}
	// Keep fog params hot in the same vec4; .y is a branch-free runtime toggle for terrain shading.
	shared_fog_uniform.fog_far = fog_far_world
	shared_fog_uniform.fog_enabled = b32(state.fog_enabled)

	scene_uniform.render_mode = 1.0
	shared_frame_uniform.mvp = reflection_view_proj
	wgpu.QueueWriteBuffer(
		state.queue,
		state.shared_frame_uniform_buffer,
		0,
		&shared_frame_uniform,
		uint(size_of(shared_frame_uniform)),
	)
	wgpu.QueueWriteBuffer(
		state.queue,
		state.shared_fog_uniform_buffer,
		0,
		&shared_fog_uniform,
		uint(size_of(shared_fog_uniform)),
	)
	wgpu.QueueWriteBuffer(
		state.queue,
		state.scene_uniform_buffer,
		0,
		&scene_uniform,
		uint(size_of(scene_uniform)),
	)
	reflection_encoder := wgpu.DeviceCreateCommandEncoder(state.device, nil)
	reflection_pass := wgpu.CommandEncoderBeginRenderPass(
		reflection_encoder,
		&{
			colorAttachmentCount = 1,
			colorAttachments = &wgpu.RenderPassColorAttachment {
				view = state.reflection_view,
				loadOp = .Clear,
				storeOp = .Store,
				depthSlice = wgpu.DEPTH_SLICE_UNDEFINED,
				clearValue = SKY_CLEAR_COLOR,
			},
			depthStencilAttachment = &wgpu.RenderPassDepthStencilAttachment {
				view = state.reflection_depth_view,
				depthLoadOp = .Clear,
				depthStoreOp = .Store,
				depthClearValue = 1,
				depthReadOnly = false,
				stencilLoadOp = .Undefined,
				stencilStoreOp = .Undefined,
				stencilClearValue = 0,
				stencilReadOnly = true,
			},
		},
	)
	wgpu.RenderPassEncoderSetPipeline(reflection_pass, state.reflection_pipeline)
	wgpu.RenderPassEncoderSetBindGroup(reflection_pass, 0, state.bind_group_reflection_only)
	for _, &chunk in state.chunks {
		if chunk.gpu_index_opaque_count == 0 ||
		   chunk.gpu_vertex_opaque == nil ||
		   chunk.gpu_index_opaque == nil ||
		   chunk.gpu_vertex_opaque_bytes == 0 ||
		   chunk.gpu_index_opaque_bytes == 0 {
			continue
		}
		wgpu.RenderPassEncoderSetVertexBuffer(
			reflection_pass,
			0,
			chunk.gpu_vertex_opaque,
			0,
			chunk.gpu_vertex_opaque_bytes,
		)
		wgpu.RenderPassEncoderSetIndexBuffer(
			reflection_pass,
			chunk.gpu_index_opaque,
			.Uint32,
			0,
			chunk.gpu_index_opaque_bytes,
		)
		wgpu.RenderPassEncoderDrawIndexed(
			reflection_pass,
			chunk.gpu_index_opaque_count,
			1,
			0,
			0,
			0,
		)
	}
	wgpu.RenderPassEncoderEnd(reflection_pass)
	wgpu.RenderPassEncoderRelease(reflection_pass)
	reflection_cmd := wgpu.CommandEncoderFinish(reflection_encoder, nil)
	wgpu.CommandEncoderRelease(reflection_encoder)
	wgpu.QueueSubmit(state.queue, {reflection_cmd})
	wgpu.CommandBufferRelease(reflection_cmd)

	scene_uniform.render_mode = 0.0
	shared_frame_uniform.mvp = view_proj
	wgpu.QueueWriteBuffer(
		state.queue,
		state.shared_frame_uniform_buffer,
		0,
		&shared_frame_uniform,
		uint(size_of(shared_frame_uniform)),
	)
	wgpu.QueueWriteBuffer(
		state.queue,
		state.scene_uniform_buffer,
		0,
		&scene_uniform,
		uint(size_of(scene_uniform)),
	)
	opaque_encoder := wgpu.DeviceCreateCommandEncoder(state.device, nil)
	opaque_pass := wgpu.CommandEncoderBeginRenderPass(
		opaque_encoder,
		&{
			colorAttachmentCount = 1,
			colorAttachments = &wgpu.RenderPassColorAttachment {
				view = state.scene_color_view,
				loadOp = .Clear,
				storeOp = .Store,
				depthSlice = wgpu.DEPTH_SLICE_UNDEFINED,
				clearValue = SKY_CLEAR_COLOR,
			},
			depthStencilAttachment = &wgpu.RenderPassDepthStencilAttachment {
				view = state.depth_view,
				depthLoadOp = .Clear,
				depthStoreOp = .Store,
				depthClearValue = 1,
				depthReadOnly = false,
				stencilLoadOp = .Undefined,
				stencilStoreOp = .Undefined,
				stencilClearValue = 0,
				stencilReadOnly = true,
			},
		},
	)
	wgpu.RenderPassEncoderSetPipeline(opaque_pass, state.pipeline)
	wgpu.RenderPassEncoderSetBindGroup(opaque_pass, 0, state.bind_group)
	for _, &chunk in state.chunks {
		if chunk.gpu_index_opaque_count == 0 ||
		   chunk.gpu_vertex_opaque == nil ||
		   chunk.gpu_index_opaque == nil ||
		   chunk.gpu_vertex_opaque_bytes == 0 ||
		   chunk.gpu_index_opaque_bytes == 0 {
			continue
		}
		wgpu.RenderPassEncoderSetVertexBuffer(
			opaque_pass,
			0,
			chunk.gpu_vertex_opaque,
			0,
			chunk.gpu_vertex_opaque_bytes,
		)
		wgpu.RenderPassEncoderSetIndexBuffer(
			opaque_pass,
			chunk.gpu_index_opaque,
			.Uint32,
			0,
			chunk.gpu_index_opaque_bytes,
		)
		wgpu.RenderPassEncoderDrawIndexed(opaque_pass, chunk.gpu_index_opaque_count, 1, 0, 0, 0)
	}
	if len(state.armed_tnt) > 0 {
		tnt_vertices := make([dynamic]f32, 0, context.temp_allocator)
		tnt_indices := make([dynamic]u32, 0, context.temp_allocator)
		armed_tnt_build_mesh(&tnt_vertices, &tnt_indices)
		if len(tnt_indices) > 0 {
			tnt_vertex_bytes := u64(len(tnt_vertices)) * u64(size_of(f32))
			tnt_index_bytes := u64(len(tnt_indices)) * u64(size_of(u32))
			tnt_vertex_buffer := wgpu.DeviceCreateBuffer(
				state.device,
				&{usage = {.Vertex, .CopyDst}, size = tnt_vertex_bytes},
			)
			tnt_index_buffer := wgpu.DeviceCreateBuffer(
				state.device,
				&{usage = {.Index, .CopyDst}, size = tnt_index_bytes},
			)
			wgpu.QueueWriteBuffer(
				state.queue,
				tnt_vertex_buffer,
				0,
				raw_data(tnt_vertices[:]),
				uint(tnt_vertex_bytes),
			)
			wgpu.QueueWriteBuffer(
				state.queue,
				tnt_index_buffer,
				0,
				raw_data(tnt_indices[:]),
				uint(tnt_index_bytes),
			)
			// Armed TNT should write depth as solid geometry so submerged TNT stays visible through water blend.
			wgpu.RenderPassEncoderSetPipeline(opaque_pass, state.pipeline)
			wgpu.RenderPassEncoderSetBindGroup(opaque_pass, 0, state.bind_group)
			wgpu.RenderPassEncoderSetVertexBuffer(
				opaque_pass,
				0,
				tnt_vertex_buffer,
				0,
				tnt_vertex_bytes,
			)
			wgpu.RenderPassEncoderSetIndexBuffer(
				opaque_pass,
				tnt_index_buffer,
				.Uint32,
				0,
				tnt_index_bytes,
			)
			wgpu.RenderPassEncoderDrawIndexed(opaque_pass, u32(len(tnt_indices)), 1, 0, 0, 0)
			wgpu.BufferRelease(tnt_index_buffer)
			wgpu.BufferRelease(tnt_vertex_buffer)
		}
	}
	wgpu.RenderPassEncoderEnd(opaque_pass)
	wgpu.RenderPassEncoderRelease(opaque_pass)
	opaque_cmd := wgpu.CommandEncoderFinish(opaque_encoder, nil)
	wgpu.CommandEncoderRelease(opaque_encoder)
	wgpu.QueueSubmit(state.queue, {opaque_cmd})
	wgpu.CommandBufferRelease(opaque_cmd)

	scene_uniform.render_mode = 2.0
	wgpu.QueueWriteBuffer(
		state.queue,
		state.scene_uniform_buffer,
		0,
		&scene_uniform,
		uint(size_of(scene_uniform)),
	)
	cloud_uniform := Cloud_Uniform{}
	sort_cloud_layers(&cloud_uniform)
	wgpu.QueueWriteBuffer(
		state.queue,
		state.cloud_uniform_buffer,
		0,
		&cloud_uniform,
		uint(size_of(cloud_uniform)),
	)
	transparent_encoder := wgpu.DeviceCreateCommandEncoder(state.device, nil)
	flower_cutout_pass := wgpu.CommandEncoderBeginRenderPass(
		transparent_encoder,
		&{
			colorAttachmentCount = 1,
			colorAttachments = &wgpu.RenderPassColorAttachment {
				view = state.scene_color_view,
				loadOp = .Load,
				storeOp = .Store,
				depthSlice = wgpu.DEPTH_SLICE_UNDEFINED,
			},
			depthStencilAttachment = &wgpu.RenderPassDepthStencilAttachment {
				view = state.depth_view,
				depthLoadOp = .Load,
				depthStoreOp = .Store,
				depthClearValue = 1,
				depthReadOnly = false,
				stencilLoadOp = .Undefined,
				stencilStoreOp = .Undefined,
				stencilClearValue = 0,
				stencilReadOnly = true,
			},
		},
	)
	wgpu.RenderPassEncoderSetPipeline(flower_cutout_pass, state.pipeline_flower_cutout)
	wgpu.RenderPassEncoderSetBindGroup(flower_cutout_pass, 0, state.bind_group)
	for _, &chunk in state.chunks {
		if chunk.gpu_index_flower_count == 0 ||
		   chunk.gpu_vertex_flower == nil ||
		   chunk.gpu_index_flower == nil ||
		   chunk.gpu_vertex_flower_bytes == 0 ||
		   chunk.gpu_index_flower_bytes == 0 {
			continue
		}
		wgpu.RenderPassEncoderSetVertexBuffer(
			flower_cutout_pass,
			0,
			chunk.gpu_vertex_flower,
			0,
			chunk.gpu_vertex_flower_bytes,
		)
		wgpu.RenderPassEncoderSetIndexBuffer(
			flower_cutout_pass,
			chunk.gpu_index_flower,
			.Uint32,
			0,
			chunk.gpu_index_flower_bytes,
		)
		wgpu.RenderPassEncoderDrawIndexed(
			flower_cutout_pass,
			chunk.gpu_index_flower_count,
			1,
			0,
			0,
			0,
		)
	}
	wgpu.RenderPassEncoderEnd(flower_cutout_pass)
	wgpu.RenderPassEncoderRelease(flower_cutout_pass)

	transparent_draws := make(
		[dynamic]Transparent_Chunk_Draw,
		0,
		len(state.chunks),
		context.temp_allocator,
	)
	for chunk_pos, &chunk in state.chunks {
		if chunk.gpu_index_transparent_count == 0 ||
		   chunk.gpu_vertex_transparent == nil ||
		   chunk.gpu_index_transparent == nil ||
		   chunk.gpu_vertex_transparent_bytes == 0 ||
		   chunk.gpu_index_transparent_bytes == 0 {
			continue
		}
		append(
			&transparent_draws,
			Transparent_Chunk_Draw {
				vertex_buffer = chunk.gpu_vertex_transparent,
				index_buffer = chunk.gpu_index_transparent,
				vertex_bytes = chunk.gpu_vertex_transparent_bytes,
				index_bytes = chunk.gpu_index_transparent_bytes,
				index_count = chunk.gpu_index_transparent_count,
				dist2 = transparent_chunk_dist2_from_camera(state.player_pos, chunk_pos),
			},
		)
	}
	sort_transparent_chunk_draws_back_to_front(&transparent_draws)
	transparent_pass := wgpu.CommandEncoderBeginRenderPass(
		transparent_encoder,
		&{
			colorAttachmentCount = 1,
			colorAttachments = &wgpu.RenderPassColorAttachment {
				view = state.scene_color_view,
				loadOp = .Load,
				storeOp = .Store,
				depthSlice = wgpu.DEPTH_SLICE_UNDEFINED,
			},
			depthStencilAttachment = &wgpu.RenderPassDepthStencilAttachment {
				view = state.depth_view,
				depthLoadOp = .Load,
				depthStoreOp = .Store,
				depthClearValue = 1,
				depthReadOnly = false,
				stencilLoadOp = .Undefined,
				stencilStoreOp = .Undefined,
				stencilClearValue = 0,
				stencilReadOnly = true,
			},
		},
	)
	wgpu.RenderPassEncoderSetPipeline(transparent_pass, state.pipeline_transparent)
	wgpu.RenderPassEncoderSetBindGroup(transparent_pass, 0, state.bind_group)
	for draw in transparent_draws {
		wgpu.RenderPassEncoderSetVertexBuffer(
			transparent_pass,
			0,
			draw.vertex_buffer,
			0,
			draw.vertex_bytes,
		)
		wgpu.RenderPassEncoderSetIndexBuffer(
			transparent_pass,
			draw.index_buffer,
			.Uint32,
			0,
			draw.index_bytes,
		)
		wgpu.RenderPassEncoderDrawIndexed(transparent_pass, draw.index_count, 1, 0, 0, 0)
	}
	wgpu.RenderPassEncoderEnd(transparent_pass)
	wgpu.RenderPassEncoderRelease(transparent_pass)

	if state.ghost_preview_visible && state.ghost_preview_kind != .None {
		ghost_vertices := make([dynamic]f32, 0, context.temp_allocator)
		ghost_indices := make([dynamic]u32, 0, context.temp_allocator)
		ghost_preview_build_mesh(
			state.ghost_preview_world_pos,
			state.ghost_preview_kind,
			&ghost_vertices,
			&ghost_indices,
		)
		if len(ghost_indices) > 0 {
			ghost_vertex_bytes := u64(len(ghost_vertices)) * u64(size_of(f32))
			ghost_index_bytes := u64(len(ghost_indices)) * u64(size_of(u32))
			assert(ghost_vertex_bytes <= GHOST_PREVIEW_VERTEX_BUFFER_MAX_BYTES)
			assert(ghost_index_bytes <= GHOST_PREVIEW_INDEX_BUFFER_MAX_BYTES)
			wgpu.QueueWriteBuffer(
				state.queue,
				state.ghost_preview_vertex_buffer,
				0,
				raw_data(ghost_vertices[:]),
				uint(ghost_vertex_bytes),
			)
			wgpu.QueueWriteBuffer(
				state.queue,
				state.ghost_preview_index_buffer,
				0,
				raw_data(ghost_indices[:]),
				uint(ghost_index_bytes),
			)
			ghost_pass := wgpu.CommandEncoderBeginRenderPass(
				transparent_encoder,
				&{
					colorAttachmentCount = 1,
					colorAttachments = &wgpu.RenderPassColorAttachment {
						view = state.scene_color_view,
						loadOp = .Load,
						storeOp = .Store,
						depthSlice = wgpu.DEPTH_SLICE_UNDEFINED,
					},
					depthStencilAttachment = &wgpu.RenderPassDepthStencilAttachment {
						view = state.depth_view,
						depthLoadOp = .Load,
						depthStoreOp = .Store,
						depthClearValue = 1,
						depthReadOnly = false,
						stencilLoadOp = .Undefined,
						stencilStoreOp = .Undefined,
						stencilClearValue = 0,
						stencilReadOnly = true,
					},
				},
			)
			wgpu.RenderPassEncoderSetPipeline(ghost_pass, state.pipeline_transparent)
			wgpu.RenderPassEncoderSetBindGroup(ghost_pass, 0, state.bind_group)
			wgpu.RenderPassEncoderSetVertexBuffer(
				ghost_pass,
				0,
				state.ghost_preview_vertex_buffer,
				0,
				ghost_vertex_bytes,
			)
			wgpu.RenderPassEncoderSetIndexBuffer(
				ghost_pass,
				state.ghost_preview_index_buffer,
				.Uint32,
				0,
				ghost_index_bytes,
			)
			wgpu.RenderPassEncoderDrawIndexed(ghost_pass, u32(len(ghost_indices)), 1, 0, 0, 0)
			wgpu.RenderPassEncoderEnd(ghost_pass)
			wgpu.RenderPassEncoderRelease(ghost_pass)
		}
	}

	if state.clouds_enabled {
		cloud_pass := wgpu.CommandEncoderBeginRenderPass(
			transparent_encoder,
			&{
				colorAttachmentCount = 1,
				colorAttachments = &wgpu.RenderPassColorAttachment {
					view = state.scene_color_view,
					loadOp = .Load,
					storeOp = .Store,
					depthSlice = wgpu.DEPTH_SLICE_UNDEFINED,
				},
				depthStencilAttachment = &wgpu.RenderPassDepthStencilAttachment {
					view = state.depth_view,
					depthLoadOp = .Load,
					depthStoreOp = .Store,
					depthClearValue = 1,
					depthReadOnly = false,
					stencilLoadOp = .Undefined,
					stencilStoreOp = .Undefined,
					stencilClearValue = 0,
					stencilReadOnly = true,
				},
			},
		)
		wgpu.RenderPassEncoderSetPipeline(cloud_pass, state.cloud_pipeline)
		wgpu.RenderPassEncoderSetBindGroup(cloud_pass, 0, state.cloud_bind_group)
		wgpu.RenderPassEncoderDraw(cloud_pass, 6, CLOUD_LAYER_COUNT, 0, 0)
		wgpu.RenderPassEncoderEnd(cloud_pass)
		wgpu.RenderPassEncoderRelease(cloud_pass)
	}

	// Rain is screen-space above the camera; skip underwater and when a roof/cave blocks the sky column.
	rain := state.rain_enabled && !player_eye_in_water() && player_has_open_sky_above()
	// fmt.printfln("rain: %v", rain)
	// fmt.printfln("player_eye_in_water: %v", player_eye_in_water())
	// fmt.printfln("player_has_open_sky_above: %v", player_has_open_sky_above())
	if rain {
		rain_pass := wgpu.CommandEncoderBeginRenderPass(
			transparent_encoder,
			&{
				colorAttachmentCount   = 1,
				colorAttachments       = &wgpu.RenderPassColorAttachment {
					view = state.scene_color_view,
					loadOp = .Load,
					storeOp = .Store,
					depthSlice = wgpu.DEPTH_SLICE_UNDEFINED,
				},
				depthStencilAttachment = &wgpu.RenderPassDepthStencilAttachment {
					view              = state.depth_view,
					depthLoadOp       = .Load,
					depthStoreOp      = .Store,
					depthClearValue   = 1,
					// Same as transparent pass: attachment is writable, but rain_pipeline has depthWriteEnabled = false so streaks never write depth.
					depthReadOnly     = false,
					stencilLoadOp     = .Undefined,
					stencilStoreOp    = .Undefined,
					stencilClearValue = 0,
					stencilReadOnly   = true,
				},
			},
		)
		wgpu.RenderPassEncoderSetPipeline(rain_pass, state.rain_pipeline)
		wgpu.RenderPassEncoderSetBindGroup(rain_pass, 0, state.rain_bind_group)
		wgpu.RenderPassEncoderDraw(rain_pass, 6, RAIN_STREAK_COUNT, 0, 0)
		wgpu.RenderPassEncoderEnd(rain_pass)
		wgpu.RenderPassEncoderRelease(rain_pass)
	}

	if state.debug_chunk_bounds || state.dig_highlight_visible {
		bounds_vertices := make([dynamic]Chunk_Bounds_Vertex, context.temp_allocator)
		CHUNK_BOUNDS_COLOR :: [4]f32{0.95, 0.17, 0.28, 1}
		if state.debug_chunk_bounds {
			for chunk_pos, _ in state.chunks {
				min := [3]f32{f32(chunk_pos.x) * CHUNK_WIDTH, 0, f32(chunk_pos.y) * CHUNK_WIDTH}
				max := [3]f32{min.x + CHUNK_WIDTH, CHUNK_HEIGHT, min.z + CHUNK_WIDTH}
				chunk_bounds_append_box(&bounds_vertices, min, max, CHUNK_BOUNDS_COLOR)
			}
		}
		if state.dig_highlight_visible {
			DIG_HIGHLIGHT_COLOR :: [4]f32{1, 1, 1, 0.3}
			// Nudge the wireframe slightly outside block faces to avoid coplanar depth fighting.
			DIG_HIGHLIGHT_OUTSET :: f32(0.0025)
			world_min :=
				state.dig_highlight_world_pos -
				[3]f32{DIG_HIGHLIGHT_OUTSET, DIG_HIGHLIGHT_OUTSET, DIG_HIGHLIGHT_OUTSET}
			world_max :=
				state.dig_highlight_world_pos +
				[3]f32 {
						1 + DIG_HIGHLIGHT_OUTSET,
						1 + DIG_HIGHLIGHT_OUTSET,
						1 + DIG_HIGHLIGHT_OUTSET,
					}
			chunk_bounds_append_box(&bounds_vertices, world_min, world_max, DIG_HIGHLIGHT_COLOR)
		}
		if len(bounds_vertices) > 0 {
			vertex_bytes := u64(len(bounds_vertices)) * u64(size_of(bounds_vertices[0]))
			chunk_bounds_vertex_buffer_ensure(vertex_bytes)
			wgpu.QueueWriteBuffer(
				state.queue,
				state.chunk_bounds_vertex_buffer,
				0,
				raw_data(bounds_vertices[:]),
				uint(vertex_bytes),
			)
			line_pass := wgpu.CommandEncoderBeginRenderPass(
				transparent_encoder,
				&{
					colorAttachmentCount = 1,
					colorAttachments = &wgpu.RenderPassColorAttachment {
						view = state.scene_color_view,
						loadOp = .Load,
						storeOp = .Store,
						depthSlice = wgpu.DEPTH_SLICE_UNDEFINED,
					},
					depthStencilAttachment = &wgpu.RenderPassDepthStencilAttachment {
						view = state.depth_view,
						depthLoadOp = .Load,
						depthStoreOp = .Store,
						depthClearValue = 1,
						depthReadOnly = false,
						stencilLoadOp = .Undefined,
						stencilStoreOp = .Undefined,
						stencilClearValue = 0,
						stencilReadOnly = true,
					},
				},
			)
			wgpu.RenderPassEncoderSetPipeline(line_pass, state.line_pipeline)
			wgpu.RenderPassEncoderSetBindGroup(line_pass, 0, state.line_bind_group)
			wgpu.RenderPassEncoderSetVertexBuffer(
				line_pass,
				0,
				state.chunk_bounds_vertex_buffer,
				0,
				vertex_bytes,
			)
			wgpu.RenderPassEncoderDraw(line_pass, u32(len(bounds_vertices)), 1, 0, 0)
			wgpu.RenderPassEncoderEnd(line_pass)
			wgpu.RenderPassEncoderRelease(line_pass)
		}
	}
	transparent_cmd := wgpu.CommandEncoderFinish(transparent_encoder, nil)
	wgpu.CommandEncoderRelease(transparent_encoder)
	wgpu.QueueSubmit(state.queue, {transparent_cmd})
	wgpu.CommandBufferRelease(transparent_cmd)

	if state.bloom_enabled {
		hw := max(1, i32(state.config.width) / 2)
		hh := max(1, i32(state.config.height) / 2)
		bloom_u: Bloom_Uniform
		bloom_u.threshold_knee = {BLOOM_THRESHOLD, BLOOM_KNEE, 0, 0}
		bloom_u.texel_dir = {0, 0, 0, 0}
		wgpu.QueueWriteBuffer(
			state.queue,
			state.bloom_uniform_buffer,
			0,
			&bloom_u,
			uint(BLOOM_UNIFORM_SIZE),
		)
		bloom_encoder := wgpu.DeviceCreateCommandEncoder(state.device, nil)
		extract_pass := wgpu.CommandEncoderBeginRenderPass(
			bloom_encoder,
			&{
				colorAttachmentCount = 1,
				colorAttachments = &wgpu.RenderPassColorAttachment {
					view = state.bloom_view_a,
					loadOp = .Clear,
					storeOp = .Store,
					depthSlice = wgpu.DEPTH_SLICE_UNDEFINED,
					clearValue = wgpu.Color{0, 0, 0, 1},
				},
			},
		)
		wgpu.RenderPassEncoderSetPipeline(extract_pass, state.bloom_extract_pipeline)
		wgpu.RenderPassEncoderSetBindGroup(extract_pass, 0, state.bloom_extract_bind_group)
		wgpu.RenderPassEncoderDraw(extract_pass, 3, 1, 0, 0)
		wgpu.RenderPassEncoderEnd(extract_pass)
		wgpu.RenderPassEncoderRelease(extract_pass)

		bloom_u.texel_dir = {1.0 / f32(hw), 1.0 / f32(hh), 1, 0}
		wgpu.QueueWriteBuffer(
			state.queue,
			state.bloom_uniform_buffer,
			0,
			&bloom_u,
			uint(BLOOM_UNIFORM_SIZE),
		)
		blur_h_pass := wgpu.CommandEncoderBeginRenderPass(
			bloom_encoder,
			&{
				colorAttachmentCount = 1,
				colorAttachments = &wgpu.RenderPassColorAttachment {
					view = state.bloom_view_b,
					loadOp = .Clear,
					storeOp = .Store,
					depthSlice = wgpu.DEPTH_SLICE_UNDEFINED,
					clearValue = wgpu.Color{0, 0, 0, 1},
				},
			},
		)
		wgpu.RenderPassEncoderSetPipeline(blur_h_pass, state.bloom_blur_pipeline)
		wgpu.RenderPassEncoderSetBindGroup(blur_h_pass, 0, state.bloom_blur_bind_group_a)
		wgpu.RenderPassEncoderDraw(blur_h_pass, 3, 1, 0, 0)
		wgpu.RenderPassEncoderEnd(blur_h_pass)
		wgpu.RenderPassEncoderRelease(blur_h_pass)

		bloom_u.texel_dir = {1.0 / f32(hw), 1.0 / f32(hh), 0, 1}
		wgpu.QueueWriteBuffer(
			state.queue,
			state.bloom_uniform_buffer,
			0,
			&bloom_u,
			uint(BLOOM_UNIFORM_SIZE),
		)
		blur_v_pass := wgpu.CommandEncoderBeginRenderPass(
			bloom_encoder,
			&{
				colorAttachmentCount = 1,
				colorAttachments = &wgpu.RenderPassColorAttachment {
					view = state.bloom_view_a,
					loadOp = .Clear,
					storeOp = .Store,
					depthSlice = wgpu.DEPTH_SLICE_UNDEFINED,
					clearValue = wgpu.Color{0, 0, 0, 1},
				},
			},
		)
		wgpu.RenderPassEncoderSetPipeline(blur_v_pass, state.bloom_blur_pipeline)
		wgpu.RenderPassEncoderSetBindGroup(blur_v_pass, 0, state.bloom_blur_bind_group_b)
		wgpu.RenderPassEncoderDraw(blur_v_pass, 3, 1, 0, 0)
		wgpu.RenderPassEncoderEnd(blur_v_pass)
		wgpu.RenderPassEncoderRelease(blur_v_pass)

		bloom_cmd := wgpu.CommandEncoderFinish(bloom_encoder, nil)
		wgpu.CommandEncoderRelease(bloom_encoder)
		wgpu.QueueSubmit(state.queue, {bloom_cmd})
		wgpu.CommandBufferRelease(bloom_cmd)
	}

	if state.dof_enabled {
		hw := max(1, i32(state.config.width) / 2)
		hh := max(1, i32(state.config.height) / 2)
		dof_u: Bloom_Uniform
		dof_u.threshold_knee = {0, 0, 0, 0}
		dof_u.texel_dir = {0, 0, 0, 0}
		wgpu.QueueWriteBuffer(
			state.queue,
			state.bloom_uniform_buffer,
			0,
			&dof_u,
			uint(BLOOM_UNIFORM_SIZE),
		)
		dof_encoder := wgpu.DeviceCreateCommandEncoder(state.device, nil)
		dof_extract_pass := wgpu.CommandEncoderBeginRenderPass(
			dof_encoder,
			&{
				colorAttachmentCount = 1,
				colorAttachments = &wgpu.RenderPassColorAttachment {
					view = state.dof_view_a,
					loadOp = .Clear,
					storeOp = .Store,
					depthSlice = wgpu.DEPTH_SLICE_UNDEFINED,
					clearValue = wgpu.Color{0, 0, 0, 1},
				},
			},
		)
		wgpu.RenderPassEncoderSetPipeline(dof_extract_pass, state.dof_extract_pipeline)
		wgpu.RenderPassEncoderSetBindGroup(dof_extract_pass, 0, state.dof_extract_bind_group)
		wgpu.RenderPassEncoderDraw(dof_extract_pass, 3, 1, 0, 0)
		wgpu.RenderPassEncoderEnd(dof_extract_pass)
		wgpu.RenderPassEncoderRelease(dof_extract_pass)

		dof_u.texel_dir = {1.0 / f32(hw), 1.0 / f32(hh), 1, 0}
		wgpu.QueueWriteBuffer(
			state.queue,
			state.bloom_uniform_buffer,
			0,
			&dof_u,
			uint(BLOOM_UNIFORM_SIZE),
		)
		dof_blur_h_pass := wgpu.CommandEncoderBeginRenderPass(
			dof_encoder,
			&{
				colorAttachmentCount = 1,
				colorAttachments = &wgpu.RenderPassColorAttachment {
					view = state.dof_view_b,
					loadOp = .Clear,
					storeOp = .Store,
					depthSlice = wgpu.DEPTH_SLICE_UNDEFINED,
					clearValue = wgpu.Color{0, 0, 0, 1},
				},
			},
		)
		wgpu.RenderPassEncoderSetPipeline(dof_blur_h_pass, state.bloom_blur_pipeline)
		wgpu.RenderPassEncoderSetBindGroup(dof_blur_h_pass, 0, state.dof_blur_bind_group_a)
		wgpu.RenderPassEncoderDraw(dof_blur_h_pass, 3, 1, 0, 0)
		wgpu.RenderPassEncoderEnd(dof_blur_h_pass)
		wgpu.RenderPassEncoderRelease(dof_blur_h_pass)

		dof_u.texel_dir = {1.0 / f32(hw), 1.0 / f32(hh), 0, 1}
		wgpu.QueueWriteBuffer(
			state.queue,
			state.bloom_uniform_buffer,
			0,
			&dof_u,
			uint(BLOOM_UNIFORM_SIZE),
		)
		dof_blur_v_pass := wgpu.CommandEncoderBeginRenderPass(
			dof_encoder,
			&{
				colorAttachmentCount = 1,
				colorAttachments = &wgpu.RenderPassColorAttachment {
					view = state.dof_view_a,
					loadOp = .Clear,
					storeOp = .Store,
					depthSlice = wgpu.DEPTH_SLICE_UNDEFINED,
					clearValue = wgpu.Color{0, 0, 0, 1},
				},
			},
		)
		wgpu.RenderPassEncoderSetPipeline(dof_blur_v_pass, state.bloom_blur_pipeline)
		wgpu.RenderPassEncoderSetBindGroup(dof_blur_v_pass, 0, state.dof_blur_bind_group_b)
		wgpu.RenderPassEncoderDraw(dof_blur_v_pass, 3, 1, 0, 0)
		wgpu.RenderPassEncoderEnd(dof_blur_v_pass)
		wgpu.RenderPassEncoderRelease(dof_blur_v_pass)

		dof_cmd := wgpu.CommandEncoderFinish(dof_encoder, nil)
		wgpu.CommandEncoderRelease(dof_encoder)
		wgpu.QueueSubmit(state.queue, {dof_cmd})
		wgpu.CommandBufferRelease(dof_cmd)
	}

	post_u: Post_Uniform
	post_u.camera_elapsed = {
		state.player_pos.x,
		state.player_pos.y,
		state.player_pos.z,
		state.elapsed_time,
	}
	uw: f32 = 0
	if player_eye_in_water() {
		uw = 1
	}
	post_u.underwater_strength = uw
	post_u.bloom_strength = BLOOM_STRENGTH if state.bloom_enabled else 0
	post_u.resolution = {f32(state.config.width), f32(state.config.height)}
	post_u.inv_view_proj = linalg.inverse(view_proj)
	post_u.fxaa_enabled = 1 if state.fxaa_enabled else 0
	dof_start: f32 = DOF_START_DISTANCE
	dof_end: f32 = DOF_END_DISTANCE
	dof_strength: f32 = DOF_STRENGTH if state.dof_enabled else 0
	if state.dof_enabled && uw > 0 {
		dof_start = DOF_UNDERWATER_START_DISTANCE
		dof_end = DOF_UNDERWATER_END_DISTANCE
		dof_strength = DOF_UNDERWATER_STRENGTH
	}
	post_u.dof_start = dof_start
	post_u.dof_end = dof_end
	post_u.dof_strength = dof_strength
	wgpu.QueueWriteBuffer(
		state.queue,
		state.post_uniform_buffer,
		0,
		&post_u,
		uint(POST_UNIFORM_SIZE),
	)

	composite_encoder := wgpu.DeviceCreateCommandEncoder(state.device, nil)
	composite_pass := wgpu.CommandEncoderBeginRenderPass(
		composite_encoder,
		&{
			colorAttachmentCount = 1,
			colorAttachments = &wgpu.RenderPassColorAttachment {
				view = state.post_composite_view,
				loadOp = .Clear,
				storeOp = .Store,
				depthSlice = wgpu.DEPTH_SLICE_UNDEFINED,
				clearValue = SKY_CLEAR_COLOR,
			},
		},
	)
	wgpu.RenderPassEncoderSetPipeline(composite_pass, state.post_pipeline)
	wgpu.RenderPassEncoderSetBindGroup(composite_pass, 0, state.post_bind_group)
	wgpu.RenderPassEncoderDraw(composite_pass, 3, 1, 0, 0)
	wgpu.RenderPassEncoderEnd(composite_pass)
	wgpu.RenderPassEncoderRelease(composite_pass)
	composite_cmd := wgpu.CommandEncoderFinish(composite_encoder, nil)
	wgpu.CommandEncoderRelease(composite_encoder)
	wgpu.QueueSubmit(state.queue, {composite_cmd})
	wgpu.CommandBufferRelease(composite_cmd)

	dof_final_encoder := wgpu.DeviceCreateCommandEncoder(state.device, nil)
	dof_final_pass := wgpu.CommandEncoderBeginRenderPass(
		dof_final_encoder,
		&{
			colorAttachmentCount = 1,
			colorAttachments = &wgpu.RenderPassColorAttachment {
				view = frame_view,
				loadOp = .Clear,
				storeOp = .Store,
				depthSlice = wgpu.DEPTH_SLICE_UNDEFINED,
				clearValue = SKY_CLEAR_COLOR,
			},
		},
	)
	wgpu.RenderPassEncoderSetPipeline(dof_final_pass, state.dof_final_pipeline)
	wgpu.RenderPassEncoderSetBindGroup(dof_final_pass, 0, state.dof_final_bind_group)
	wgpu.RenderPassEncoderDraw(dof_final_pass, 3, 1, 0, 0)
	wgpu.RenderPassEncoderEnd(dof_final_pass)
	wgpu.RenderPassEncoderRelease(dof_final_pass)
	clay_renderer_end_frame_and_record(dof_final_encoder, frame_view)
	dof_final_cmd := wgpu.CommandEncoderFinish(dof_final_encoder, nil)
	wgpu.CommandEncoderRelease(dof_final_encoder)
	wgpu.QueueSubmit(state.queue, {dof_final_cmd})
	wgpu.CommandBufferRelease(dof_final_cmd)

	wgpu.SurfacePresent(state.surface)
}

finish :: proc() {
	clay_renderer_destroy()
	{
		total_vertex_bytes: u64
		total_index_bytes: u64
		total_vertex_buffers: int
		total_index_buffers: int
		for bucket, list in state.chunk_vertex_buffer_pool {
			total_vertex_bytes += bucket * u64(len(list))
			total_vertex_buffers += len(list)
		}
		for bucket, list in state.chunk_index_buffer_pool {
			total_index_bytes += bucket * u64(len(list))
			total_index_buffers += len(list)
		}
		total_bytes := total_vertex_bytes + total_index_bytes
		log.infof(
			"chunk pool usage total_mib=%v vertex_mib=%v index_mib=%v vertex_buffers=%v index_buffers=%v",
			f64(total_bytes) / (1024.0 * 1024.0),
			f64(total_vertex_bytes) / (1024.0 * 1024.0),
			f64(total_index_bytes) / (1024.0 * 1024.0),
			total_vertex_buffers,
			total_index_buffers,
		)
	}
	for pos, &chunk in state.chunks {
		destroy_chunk(&chunk, pos)
	}
	chunk_gpu_pool_destroy()

	terrain_gpu_destroy()

	scene_color_resources_destroy()
	if state.post_pipeline != nil {
		wgpu.RenderPipelineRelease(state.post_pipeline)
		state.post_pipeline = nil
	}
	if state.post_pipeline_layout != nil {
		wgpu.PipelineLayoutRelease(state.post_pipeline_layout)
		state.post_pipeline_layout = nil
	}
	if state.post_bind_group_layout != nil {
		wgpu.BindGroupLayoutRelease(state.post_bind_group_layout)
		state.post_bind_group_layout = nil
	}
	if state.post_effects_module != nil {
		wgpu.ShaderModuleRelease(state.post_effects_module)
		state.post_effects_module = nil
	}
	if state.post_uniform_buffer != nil {
		wgpu.BufferRelease(state.post_uniform_buffer)
		state.post_uniform_buffer = nil
	}
	if state.bloom_extract_pipeline != nil {
		wgpu.RenderPipelineRelease(state.bloom_extract_pipeline)
		state.bloom_extract_pipeline = nil
	}
	if state.bloom_blur_pipeline != nil {
		wgpu.RenderPipelineRelease(state.bloom_blur_pipeline)
		state.bloom_blur_pipeline = nil
	}
	if state.dof_extract_pipeline != nil {
		wgpu.RenderPipelineRelease(state.dof_extract_pipeline)
		state.dof_extract_pipeline = nil
	}
	if state.dof_final_pipeline != nil {
		wgpu.RenderPipelineRelease(state.dof_final_pipeline)
		state.dof_final_pipeline = nil
	}
	if state.dof_final_pipeline_layout != nil {
		wgpu.PipelineLayoutRelease(state.dof_final_pipeline_layout)
		state.dof_final_pipeline_layout = nil
	}
	if state.dof_final_bind_group_layout != nil {
		wgpu.BindGroupLayoutRelease(state.dof_final_bind_group_layout)
		state.dof_final_bind_group_layout = nil
	}
	if state.dof_module != nil {
		wgpu.ShaderModuleRelease(state.dof_module)
		state.dof_module = nil
	}
	if state.bloom_pipeline_layout != nil {
		wgpu.PipelineLayoutRelease(state.bloom_pipeline_layout)
		state.bloom_pipeline_layout = nil
	}
	if state.bloom_bind_group_layout != nil {
		wgpu.BindGroupLayoutRelease(state.bloom_bind_group_layout)
		state.bloom_bind_group_layout = nil
	}
	if state.bloom_module != nil {
		wgpu.ShaderModuleRelease(state.bloom_module)
		state.bloom_module = nil
	}
	if state.bloom_uniform_buffer != nil {
		wgpu.BufferRelease(state.bloom_uniform_buffer)
		state.bloom_uniform_buffer = nil
	}
	if state.bloom_sampler != nil {
		wgpu.SamplerRelease(state.bloom_sampler)
		state.bloom_sampler = nil
	}
	if state.depth_sample_sampler != nil {
		wgpu.SamplerRelease(state.depth_sample_sampler)
		state.depth_sample_sampler = nil
	}

	depth_resources_destroy()
	reflection_resources_destroy()

	if state.bind_group != nil {
		wgpu.BindGroupRelease(state.bind_group)
	}
	if state.bind_group_reflection_only != nil {
		wgpu.BindGroupRelease(state.bind_group_reflection_only)
	}
	if state.rain_bind_group != nil {
		wgpu.BindGroupRelease(state.rain_bind_group)
	}
	if state.cloud_bind_group != nil {
		wgpu.BindGroupRelease(state.cloud_bind_group)
	}
	if state.line_bind_group != nil {
		wgpu.BindGroupRelease(state.line_bind_group)
	}
	atlas_resources_destroy()
	if state.shared_frame_uniform_buffer != nil {
		wgpu.BufferRelease(state.shared_frame_uniform_buffer)
	}
	if state.shared_fog_uniform_buffer != nil {
		wgpu.BufferRelease(state.shared_fog_uniform_buffer)
	}
	if state.scene_uniform_buffer != nil {
		wgpu.BufferRelease(state.scene_uniform_buffer)
	}
	if state.cloud_uniform_buffer != nil {
		wgpu.BufferRelease(state.cloud_uniform_buffer)
	}
	if state.ghost_preview_vertex_buffer != nil {
		wgpu.BufferRelease(state.ghost_preview_vertex_buffer)
		state.ghost_preview_vertex_buffer = nil
	}
	if state.ghost_preview_index_buffer != nil {
		wgpu.BufferRelease(state.ghost_preview_index_buffer)
		state.ghost_preview_index_buffer = nil
	}
	if state.bind_group_layout != nil {
		wgpu.BindGroupLayoutRelease(state.bind_group_layout)
	}
	if state.shared_frame_bind_group_layout != nil {
		wgpu.BindGroupLayoutRelease(state.shared_frame_bind_group_layout)
	}
	if state.cloud_bind_group_layout != nil {
		wgpu.BindGroupLayoutRelease(state.cloud_bind_group_layout)
	}
	if state.chunk_bounds_vertex_buffer != nil {
		wgpu.BufferRelease(state.chunk_bounds_vertex_buffer)
	}
	if state.rain_pipeline != nil {
		wgpu.RenderPipelineRelease(state.rain_pipeline)
	}
	if state.rain_module != nil {
		wgpu.ShaderModuleRelease(state.rain_module)
	}
	if state.cloud_pipeline != nil {
		wgpu.RenderPipelineRelease(state.cloud_pipeline)
	}
	if state.cloud_module != nil {
		wgpu.ShaderModuleRelease(state.cloud_module)
	}
	if state.line_pipeline != nil {
		wgpu.RenderPipelineRelease(state.line_pipeline)
	}
	if state.pipeline_flower_cutout != nil {
		wgpu.RenderPipelineRelease(state.pipeline_flower_cutout)
	}
	if state.pipeline_transparent != nil {
		wgpu.RenderPipelineRelease(state.pipeline_transparent)
	}
	if state.reflection_pipeline != nil {
		wgpu.RenderPipelineRelease(state.reflection_pipeline)
	}
	if state.line_module != nil {
		wgpu.ShaderModuleRelease(state.line_module)
	}

	wgpu.RenderPipelineRelease(state.pipeline)
	wgpu.PipelineLayoutRelease(state.pipeline_layout)
	if state.shared_frame_pipeline_layout != nil {
		wgpu.PipelineLayoutRelease(state.shared_frame_pipeline_layout)
	}
	if state.cloud_pipeline_layout != nil {
		wgpu.PipelineLayoutRelease(state.cloud_pipeline_layout)
	}
	wgpu.ShaderModuleRelease(state.module)
	wgpu.QueueRelease(state.queue)
	wgpu.DeviceRelease(state.device)
	wgpu.AdapterRelease(state.adapter)
	wgpu.SurfaceRelease(state.surface)
	wgpu.InstanceRelease(state.instance)
}


Cube_Kind :: enum u32 {
	None        = 0,
	Grass       = 1,
	Dirt        = 2,
	Stone       = 3,
	Bedrock     = 4,
	Water       = 5,
	Sand        = 6,
	Flower1     = 7,
	Flower2     = 8,
	Flower3     = 9,
	Flower4     = 10,
	Flower5     = 11,
	Flower6     = 12,
	Flower7     = 13,
	Flower8     = 14,
	Flower9     = 15,
	Flower10    = 16,
	Flower11    = 17,
	Flower12    = 18,
	Wood        = 19,
	Cobblestone = 20,
	Pumpkin     = 21,
	Brick       = 22,
	TNT         = 23,
	Ore_Diamond = 24,
	Ore_Gold    = 25,
	Ore_Iron    = 26,
	Ore_Green   = 27,
	Ore_Red     = 28,
	Ore_Blue    = 29,
	Ore_Coal    = 30,
}

FLOWER_KINDS_FIRST :: Cube_Kind.Flower1
FLOWER_KINDS_LAST :: Cube_Kind.Flower12

// @(rodata)
// cube_kind_flowers: bit_set[Cube_Kind] = {
// 	.Flower1,
// 	.Flower2,
// 	.Flower3,
// 	.Flower4,
// 	.Flower5,
// 	.Flower6,
// 	.Flower7,
// 	.Flower8,
// 	.Flower9,
// 	.Flower10,
// 	.Flower11,
// 	.Flower12,
// 	.Flower13,
// 	.Flower14,
// 	.Flower15,
// }

cube_kind_is_flower :: proc(kind: Cube_Kind) -> bool {
	return kind >= FLOWER_KINDS_FIRST && kind <= FLOWER_KINDS_LAST
}

CHUNK_HEIGHT :: 48
CHUNK_WIDTH :: 16
CHUNK_VOXEL_COUNT :: CHUNK_WIDTH * CHUNK_HEIGHT * CHUNK_WIDTH
HOTBAR_SLOT_COUNT :: 9

Chunk :: struct {
	// TODO: should be indexed by Cube_Index
	kinds:                        [CHUNK_WIDTH * CHUNK_HEIGHT * CHUNK_WIDTH]Cube_Kind,
	// TODO: should be indexed by Cube_Index
	light:                        [CHUNK_WIDTH * CHUNK_HEIGHT * CHUNK_WIDTH]u8,
	// Sparse by design: only blocks with directional variants (pumpkin for now) allocate entries.
	// TODO: this should be a small-array/[dynamic;N] array in order to avoid allocations. linear search is fine.
	rotation_by_index:            map[Cube_Index]Cube_Yaw_Rotation,
	gpu_vertex_opaque:            wgpu.Buffer,
	gpu_vertex_opaque_bytes:      u64,
	gpu_index_opaque:             wgpu.Buffer,
	gpu_index_opaque_bytes:       u64,
	gpu_index_opaque_count:       u32,
	gpu_vertex_transparent:       wgpu.Buffer,
	gpu_vertex_transparent_bytes: u64,
	gpu_index_transparent:        wgpu.Buffer,
	gpu_index_transparent_bytes:  u64,
	gpu_index_transparent_count:  u32,
	gpu_vertex_flower:            wgpu.Buffer,
	gpu_vertex_flower_bytes:      u64,
	gpu_index_flower:             wgpu.Buffer,
	gpu_index_flower_bytes:       u64,
	gpu_index_flower_count:       u32,
}

Chunk_Mesh :: struct {
	vertices_opaque:      [dynamic]f32,
	indices_opaque:       [dynamic]u32,
	vertices_transparent: [dynamic]f32,
	indices_transparent:  [dynamic]u32,
	vertices_flower:      [dynamic]f32,
	indices_flower:       [dynamic]u32,
}

Cube_Face :: enum u8 {
	PosX,
	NegX,
	PosY,
	NegY,
	PosZ,
	NegZ,
}

Cube_Yaw_Rotation :: enum u8 {
	R0,
	R90,
	R180,
	R270,
}

ATLAS_WIDTH_PX :: 512.0
ATLAS_HEIGHT_PX :: 256.0
ATLAS_TILE_SIZE_PX :: 16.0
ATLAS_TILE_INSET_PX :: 0.25
ATLAS_TILE_STRIDE_U :: ATLAS_TILE_SIZE_PX / ATLAS_WIDTH_PX
ATLAS_TILE_STRIDE_V :: ATLAS_TILE_SIZE_PX / ATLAS_HEIGHT_PX
ATLAS_TILE_INSET_U :: ATLAS_TILE_INSET_PX / ATLAS_WIDTH_PX
ATLAS_TILE_INSET_V :: ATLAS_TILE_INSET_PX / ATLAS_HEIGHT_PX

Atlas_Tile :: enum {
	Sand,
	Grass_Top,
	Grass_Side,
	Dirt,
	Stone,
	Bedrock,
	Water_0,
	Water_1,
	Water_2,
	Water_3,
	Flower1,
	Flower2,
	Flower3,
	Flower4,
	Flower5,
	Flower6,
	Flower7,
	Flower8,
	Flower9,
	Flower10,
	Flower11,
	Flower12,
	Wood,
	Cobblestone,
	Pumpkin_Top,
	Pumpkin_Side,
	Pumpkin_Front,
	Brick,
	TNT_Top,
	TNT_Side,
	TNT_Bottom,
	Ore_Diamond,
	Ore_Gold,
	Ore_Iron,
	Ore_Green,
	Ore_Red,
	Ore_Blue,
	Ore_Coal,
}

Atlas_Coords :: distinct [2]int

rotate_horizontal_face_yaw :: proc(face: Cube_Face, rotation: Cube_Yaw_Rotation) -> Cube_Face {
	if face == .PosY || face == .NegY do return face

	turns := int(rotation)
	rotated := face
	for _ in 0 ..< turns {
		switch rotated {
		case .NegZ:
			rotated = .PosX
		case .PosX:
			rotated = .PosZ
		case .PosZ:
			rotated = .NegX
		case .NegX:
			rotated = .NegZ
		case .PosY, .NegY:
			unreachable()
		}
	}
	return rotated
}

inverse_cube_yaw_rotation :: proc(rotation: Cube_Yaw_Rotation) -> Cube_Yaw_Rotation {
	switch rotation {
	case .R0:
		return .R0
	case .R90:
		return .R270
	case .R180:
		return .R180
	case .R270:
		return .R90
	}
	unreachable()
}

placed_cube_rotation_from_player_forward :: proc(forward: [3]f32) -> Cube_Yaw_Rotation {
	// Pumpkin front should face the player, so use the opposite of look direction.
	toward_player := -forward
	// Keep placement deterministic near diagonals by snapping to the dominant horizontal axis.
	if math.abs(toward_player.z) >= math.abs(toward_player.x) {
		if toward_player.z < 0 do return .R0
		return .R180
	}
	if toward_player.x >= 0 do return .R90
	return .R270
}

chunk_cube_rotation_get :: proc(chunk: ^Chunk, idx: Cube_Index) -> Cube_Yaw_Rotation {
	rotation, ok := chunk.rotation_by_index[idx]
	if ok do return rotation
	return .R0
}

atlas_coords_for_face :: proc(
	kind: Cube_Kind,
	face: Cube_Face,
	rotation: Cube_Yaw_Rotation = .R0,
	for_hotbar: bool = false,
) -> Atlas_Coords {
	atlas_tile: Atlas_Tile

	effective_face := face
	if !for_hotbar {
		if kind == .Pumpkin {
			// Convert world face into unrotated model-space face before tile selection.
			effective_face = rotate_horizontal_face_yaw(face, inverse_cube_yaw_rotation(rotation))
		}
	}

	switch kind {
	case .Ore_Diamond:
		atlas_tile = .Ore_Diamond
	case .Ore_Gold:
		atlas_tile = .Ore_Gold
	case .Ore_Iron:
		atlas_tile = .Ore_Iron
	case .Ore_Green:
		atlas_tile = .Ore_Green
	case .Ore_Red:
		atlas_tile = .Ore_Red
	case .Ore_Blue:
		atlas_tile = .Ore_Blue
	case .Ore_Coal:
		atlas_tile = .Ore_Coal
	case .Wood:
		atlas_tile = .Wood
	case .Cobblestone:
		atlas_tile = .Cobblestone
	case .Pumpkin:
		switch effective_face {
		case .PosY, .NegY:
			atlas_tile = .Pumpkin_Top
		case .NegZ:
			atlas_tile = .Pumpkin_Front
		case .PosX, .NegX, .PosZ:
			atlas_tile = .Pumpkin_Side
		}
	case .Brick:
		atlas_tile = .Brick
	case .TNT:
		switch face {
		case .PosY:
			atlas_tile = .TNT_Top
		case .NegY:
			atlas_tile = .TNT_Bottom
		case .PosX, .NegX, .PosZ, .NegZ:
			atlas_tile = .TNT_Side
		}
	case .Flower1:
		atlas_tile = .Flower1
	case .Flower2:
		atlas_tile = .Flower2
	case .Flower3:
		atlas_tile = .Flower3
	case .Flower4:
		atlas_tile = .Flower4
	case .Flower5:
		atlas_tile = .Flower5
	case .Flower6:
		atlas_tile = .Flower6
	case .Flower7:
		atlas_tile = .Flower7
	case .Flower8:
		atlas_tile = .Flower8
	case .Flower9:
		atlas_tile = .Flower9
	case .Flower10:
		atlas_tile = .Flower10
	case .Flower11:
		atlas_tile = .Flower11
	case .Flower12:
		atlas_tile = .Flower12
	case .Sand:
		atlas_tile = .Sand
	case .Grass:
		switch face {
		case .PosY:
			atlas_tile = .Grass_Top
		case .NegY:
			atlas_tile = .Dirt
		case .PosX, .NegX, .PosZ, .NegZ:
			atlas_tile = .Grass_Side
		}
	case .Dirt:
		atlas_tile = .Dirt
	case .Stone:
		atlas_tile = .Stone
	case .Bedrock:
		atlas_tile = .Bedrock
	case .Water:
		atlas_tile = .Water_0
	case .None:
		unreachable()
	}

	switch atlas_tile {
	case .Ore_Diamond:
		return {8, 3}
	case .Ore_Gold:
		return {15, 9}
	case .Ore_Iron:
		return {15, 11}
	case .Ore_Green:
		return {11, 7}
	case .Ore_Red:
		return {18, 4}
	case .Ore_Blue:
		return {6, 12}
	case .Ore_Coal:
		return {1, 5}
	case .Grass_Top:
		return {13, 10}
	case .Grass_Side:
		return {0, 10}
	case .Dirt:
		return {8, 4}
	case .Stone:
		return {19, 10}
	case .Bedrock:
		return {4, 3}
	case .Water_0:
		return {2, 0}
	case .Water_1:
		return {3, 0}
	case .Water_2:
		return {2, 1}
	case .Water_3:
		return {3, 1}
	case .Sand:
		return {18, 15}
	case .Flower1:
		return {11, 1}
	case .Flower2:
		return {14, 0}
	case .Flower3:
		return {14, 2}
	case .Flower4:
		return {14, 3}
	case .Flower5:
		return {14, 4}
	case .Flower6:
		return {14, 5}
	case .Flower7:
		return {14, 6}
	case .Flower8:
		return {13, 6}
	case .Flower9:
		return {13, 5}
	case .Flower10:
		return {13, 4}
	case .Flower11:
		return {13, 7}
	case .Flower12:
		return {11, 0}
	case .Wood:
		return {14, 15}
	case .Cobblestone:
		return {2, 5}
	case .Brick:
		return {5, 3}
	case .Pumpkin_Front:
		return {16, 7}
	case .Pumpkin_Top:
		return {16, 11}
	case .Pumpkin_Side:
		return {16, 8}
	case .TNT_Top:
		return {20, 10}
	case .TNT_Side:
		return {20, 9}
	case .TNT_Bottom:
		return {20, 8}
	}
	unreachable()
}

atlas_uv_for_corner :: proc(tile: Atlas_Coords, corner: [2]f32) -> [2]f32 {
	// Inset each tile so float precision/filtering never pulls color from neighboring tiles.
	u0 := f32(tile.x) * ATLAS_TILE_STRIDE_U + ATLAS_TILE_INSET_U
	v0 := f32(tile.y) * ATLAS_TILE_STRIDE_V + ATLAS_TILE_INSET_V
	span_u := ATLAS_TILE_STRIDE_U - f32(2.0) * ATLAS_TILE_INSET_U
	span_v := ATLAS_TILE_STRIDE_V - f32(2.0) * ATLAS_TILE_INSET_V
	return {u0 + corner.x * span_u, v0 + corner.y * span_v}
}

chunk_append_quad :: proc(
	chunk: ^Chunk,
	voxel_idx: Cube_Index,
	mesh: ^Chunk_Mesh,
	is_transparent: bool,
	p0, p1, p2, p3: [3]f32,
	normal: [3]f32,
	face: Cube_Face,
	kind: Cube_Kind,
	ao_quad: [4]f32,
	light_factor: f32,
) {
	vertices := &mesh.vertices_opaque
	indices := &mesh.indices_opaque
	if is_transparent {
		vertices = &mesh.vertices_transparent
		indices = &mesh.indices_transparent
	}
	base := u32(len(vertices^) / 11)
	quad := [4][3]f32{p0, p1, p2, p3}
	rotation := chunk_cube_rotation_get(chunk, voxel_idx)
	tile := atlas_coords_for_face(kind, face, rotation)
	uv_tl := atlas_uv_for_corner(tile, {0, 0})
	uv_bl := atlas_uv_for_corner(tile, {0, 1})
	uv_br := atlas_uv_for_corner(tile, {1, 1})
	uv_tr := atlas_uv_for_corner(tile, {1, 0})
	uv_quad: [4][2]f32
	// Each face has a different vertex winding in world space; one shared UV order
	// rotates/flips textures on some faces, so we map UVs per face orientation.
	switch face {
	case .PosX, .NegX:
		uv_quad = {uv_bl, uv_tl, uv_tr, uv_br}
	case .PosZ, .NegZ:
		uv_quad = {uv_bl, uv_br, uv_tr, uv_tl}
	case .PosY:
		uv_quad = {uv_tl, uv_bl, uv_br, uv_tr}
	case .NegY:
		uv_quad = {uv_tl, uv_tr, uv_br, uv_bl}
	}
	vertex_light_factor := light_factor
	if kind == .Pumpkin {
		// Emissive blocks should stay visually self-lit; neighbors still use propagated light.
		// TODO: maybe unnecessary, doesnt seemt o change anything.
		vertex_light_factor = 1
	}
	for p, i in quad {
		uv := uv_quad[i]
		material_marker: f32 = 0
		if kind == .Water do material_marker = 1
		append(
			vertices,
			p.x,
			p.y,
			p.z,
			normal.x,
			normal.y,
			normal.z,
			uv.x,
			uv.y,
			ao_quad[i],
			material_marker,
			vertex_light_factor,
		)
	}
	// Pick the diagonal that better matches opposite-corner occlusion so AO gradients stay smooth.
	if ao_quad[0] + ao_quad[2] > ao_quad[1] + ao_quad[3] {
		append(indices, base + 0, base + 1, base + 2, base + 0, base + 2, base + 3)
	} else {
		append(indices, base + 0, base + 1, base + 3, base + 1, base + 2, base + 3)
	}
}

FLOWER_BILLBOARD_WIDTH :: 0.55
FLOWER_BILLBOARD_HEIGHT :: 0.75
FLOWER_MATERIAL_MARKER :: 2

// Flowers render in a depth-writing cutout pass so vegetation self-occludes correctly.
chunk_append_flower_billboard :: proc(
	mesh: ^Chunk_Mesh,
	voxel_min: [3]f32,
	light_factor: f32,
	kind: Cube_Kind,
) {
	vertices := &mesh.vertices_flower
	indices := &mesh.indices_flower
	base := u32(len(vertices^) / 11)
	cx := voxel_min.x + 0.5
	cz := voxel_min.z + 0.5
	half_w := f32(FLOWER_BILLBOARD_WIDTH) * 0.5
	h := f32(FLOWER_BILLBOARD_HEIGHT)
	tile := atlas_coords_for_face(kind, .PosY)
	uv_tl := atlas_uv_for_corner(tile, {0, 0})
	uv_bl := atlas_uv_for_corner(tile, {0, 1})
	uv_br := atlas_uv_for_corner(tile, {1, 1})
	uv_tr := atlas_uv_for_corner(tile, {1, 0})
	quad_pos := [4][3]f32 {
		{cx, voxel_min.y, cz},
		{cx, voxel_min.y, cz},
		{cx, voxel_min.y, cz},
		{cx, voxel_min.y, cz},
	}
	// For flower sprites, `normal` carries local billboard offsets that the vertex shader
	// expands using camera basis vectors so the quad always faces the camera.
	quad_local_offsets := [4][3]f32 {
		{-half_w, 0, 0},
		{-half_w, h, 0},
		{half_w, h, 0},
		{half_w, 0, 0},
	}
	uv_quad := [4][2]f32{uv_bl, uv_tl, uv_tr, uv_br}
	for p, i in quad_pos {
		append(
			vertices,
			p.x,
			p.y,
			p.z,
			quad_local_offsets[i].x,
			quad_local_offsets[i].y,
			quad_local_offsets[i].z,
			uv_quad[i].x,
			uv_quad[i].y,
			f32(1),
			FLOWER_MATERIAL_MARKER,
			light_factor,
		)
	}
	// Keep sprite visible from both sides without adding a second quad.
	append(
		indices,
		base + 0,
		base + 1,
		base + 2,
		base + 0,
		base + 2,
		base + 3,
		base + 0,
		base + 2,
		base + 1,
		base + 0,
		base + 3,
		base + 2,
	)
}

// Open water top is +Y; back-face culling hides it from underwater - duplicate with -Y and flipped winding.
chunk_append_water_top_underface :: proc(
	mesh: ^Chunk_Mesh,
	is_transparent: bool,
	p0, p1, p2, p3: [3]f32,
	kind: Cube_Kind,
	ao_quad: [4]f32,
	light_factor: f32,
) {
	vertices := &mesh.vertices_opaque
	indices := &mesh.indices_opaque
	if is_transparent {
		vertices = &mesh.vertices_transparent
		indices = &mesh.indices_transparent
	}
	base := u32(len(vertices^) / 11)
	quad := [4][3]f32{p0, p1, p2, p3}
	tile := atlas_coords_for_face(kind, .PosY)
	uv_tl := atlas_uv_for_corner(tile, {0, 0})
	uv_bl := atlas_uv_for_corner(tile, {0, 1})
	uv_br := atlas_uv_for_corner(tile, {1, 1})
	uv_tr := atlas_uv_for_corner(tile, {1, 0})
	uv_quad := [4][2]f32{uv_tl, uv_bl, uv_br, uv_tr}
	normal := [3]f32{0, -1, 0}
	for p, i in quad {
		append(
			vertices,
			p.x,
			p.y,
			p.z,
			normal.x,
			normal.y,
			normal.z,
			uv_quad[i].x,
			uv_quad[i].y,
			ao_quad[i],
			f32(1),
			light_factor,
		)
	}
	if ao_quad[0] + ao_quad[2] > ao_quad[1] + ao_quad[3] {
		append(indices, base + 0, base + 2, base + 1, base + 0, base + 3, base + 2)
	} else {
		append(indices, base + 0, base + 3, base + 1, base + 1, base + 3, base + 2)
	}
}

is_transparent_kind :: proc(kind: Cube_Kind) -> bool {
	return kind == .Water || cube_kind_is_flower(kind)
}

face_occluded_by_neighbor :: proc(kind, neighbor_kind: Cube_Kind) -> bool {
	if neighbor_kind == .None {
		return false
	}
	if is_transparent_kind(kind) {
		// Transparent voxels only need boundary faces against air; interior interfaces
		// against solids/other transparent voxels add overdraw without adding visibility.
		return true
	}
	// Opaque faces should still render behind transparent surfaces.
	return !is_transparent_kind(neighbor_kind)
}

chunk_coords :: proc(pos: [3]f32) -> Chunk_Coords {
	// Floor division so negatives map to the correct chunk (truncation toward zero does not).
	w := f32(CHUNK_WIDTH)
	return Chunk_Coords{int(math.floor(pos.x / w)), int(math.floor(pos.z / w))}
}

local_pos :: proc(pos: [3]f32, chunk_pos: Chunk_Coords) -> Local_Pos {
	// Match chunk_coords: floor world position into voxel cells so locals stay in 0..W at boundaries.
	wx := int(math.floor(pos.x))
	wy := int(math.floor(pos.y))
	wz := int(math.floor(pos.z))
	return Local_Pos{wx - chunk_pos.x * CHUNK_WIDTH, wy, wz - chunk_pos.y * CHUNK_WIDTH}
}

chunk_world_pos :: proc(chunk_coords: Chunk_Coords, local: Local_Pos) -> [3]f32 {
	return {
		f32(chunk_coords.x) * CHUNK_WIDTH + f32(local.x),
		f32(local.y),
		f32(chunk_coords.y) * CHUNK_WIDTH + f32(local.z),
	}
}

@(rodata)
cube_default: Cube_Kind

COLLISION_DEBUG_LOGS :: false

DIG_MAX_REACH :: 8.0
DIG_RAY_STEP :: 0.05
TNT_FLASH_PHASE_SEC :: 0.2
TNT_FLASH_TOGGLE_COUNT :: 8
TNT_EXPLOSION_RADIUS :: 5
TNT_ARM_KICK_SPEED :: 3.1
TNT_ARM_UPWARD_KICK :: 2.0
TNT_CHAIN_ARM_UPWARD_KICK :: 1.2
TNT_GRAVITY :: -20.0
TNT_TERMINAL_FALL_SPEED :: -45.0
TNT_GROUND_FRICTION :: 11.0
TNT_AABB_HALF_EXTENTS :: [3]f32{0.45, 0.45, 0.45}

Armed_TNT :: struct {
	pos:              [3]f32,
	vel:              [3]f32,
	phase_elapsed:    f32,
	phase_toggle_cnt: int,
}

world_voxel_origin :: proc(pos: [3]f32) -> [3]f32 {
	return {f32(int(math.floor(pos.x))), f32(int(math.floor(pos.y))), f32(int(math.floor(pos.z)))}
}

cube_at :: proc(pos: [3]f32) -> (result: Cube_Kind, ok: bool) {
	// Chunk_Coords only tiles X/Z; a single chunk owns y in [0, CHUNK_HEIGHT). No vertical neighbor to recurse to.
	wy := int(math.floor(pos.y))
	if wy < 0 || wy >= CHUNK_HEIGHT {
		return cube_default, false
	}
	chunk_pos := chunk_coords(pos)
	chunk, chunk_ok := state.chunks[chunk_pos]
	if !chunk_ok do return cube_default, false
	return cube_at_chunk(chunk_pos, chunk, pos)
}

cube_at_chunk :: proc(
	chunk_pos: Chunk_Coords,
	chunk: Chunk,
	pos: [3]f32, // TODO: passing floats here is weird, but Local is (in theory) only for within-chunk coords. maybe pass local here anyway since it's being constructed in the proc body anyway. (then remove `cube_at_local` maybe)
) -> (
	result: Cube_Kind,
	ok: bool,
) {
	local := local_pos(pos, chunk_pos)
	if local.y < 0 || local.y >= CHUNK_HEIGHT {
		return cube_default, false
	}
	if local.x < 0 || local.x >= CHUNK_WIDTH || local.z < 0 || local.z >= CHUNK_WIDTH {
		return cube_at(pos)
	}
	idx := cube_index(local)
	return chunk.kinds[idx], true
}

local_within_chunk :: proc(local: Local_Pos) -> bool {
	return(
		local.x >= 0 &&
		local.x < CHUNK_WIDTH &&
		local.z >= 0 &&
		local.z < CHUNK_WIDTH &&
		local.y >= 0 &&
		local.y < CHUNK_HEIGHT \
	)
}

cube_kind_at_local :: proc(chunk: Chunk, local: Local_Pos) -> (Cube_Kind, bool) {
	if !local_within_chunk(local) {
		return .None, false
	}
	idx := cube_index(local)
	return chunk.kinds[idx], true
}

Chunk_Neighborhood :: struct {
	center_pos: Chunk_Coords,
	center:     ^Chunk,
}

chunk_coords_for_local :: proc(
	nb: ^Chunk_Neighborhood,
	local: [3]int,
) -> (
	chunk_pos: Chunk_Coords,
	coords: Local_Pos,
	ok: bool,
) {
	if local.y < 0 || local.y >= CHUNK_HEIGHT {
		return {}, {}, false
	}

	chunk_dx := 0
	chunk_dz := 0
	local_x := local.x
	local_z := local.z
	if local_x < 0 {
		chunk_dx = -1
		local_x += CHUNK_WIDTH
	} else if local_x >= CHUNK_WIDTH {
		chunk_dx = 1
		local_x -= CHUNK_WIDTH
	}
	if local_z < 0 {
		chunk_dz = -1
		local_z += CHUNK_WIDTH
	} else if local_z >= CHUNK_WIDTH {
		chunk_dz = 1
		local_z -= CHUNK_WIDTH
	}

	chunk_pos = Chunk_Coords{nb.center_pos.x + chunk_dx, nb.center_pos.y + chunk_dz}
	return chunk_pos, Local_Pos{local_x, local.y, local_z}, true
}

// Resolves neighbor voxel across chunk XZ boundaries; y must stay in-chunk (same as chunk_coords_for_local).
chunk_neighbor_from_offset :: proc(
	center_chunk: Chunk_Coords,
	local: [3]int,
	delta: [3]int,
) -> (
	chunk_pos: Chunk_Coords,
	coords: Local_Pos,
	ok: bool,
) {
	dummy: Chunk
	nb := Chunk_Neighborhood {
		center_pos = center_chunk,
		center     = &dummy,
	}
	return chunk_coords_for_local(&nb, local + delta)
}

// TODO: not a very descirptive name!
is_solid_local_for_remesh :: proc(nb: ^Chunk_Neighborhood, local: [3]int) -> bool {
	kind, ok := cube_kind_local_for_remesh(nb, local)
	// Sprite voxels (flowers) should not darken neighboring block corners.
	return ok && kind != .None && !cube_kind_is_flower(kind) && kind != .Pumpkin
}

cube_kind_local_for_remesh :: proc(
	nb: ^Chunk_Neighborhood,
	local: [3]int,
) -> (
	kind: Cube_Kind,
	ok: bool,
) {
	chunk_pos, coords, coords_ok := chunk_coords_for_local(nb, local)
	if !coords_ok do return .None, false
	if chunk_pos == nb.center_pos {
		return nb.center.kinds[cube_index(coords)], true
	}
	neighbor, neighbor_ok := state.chunks[chunk_pos]
	if !neighbor_ok do return .None, false
	return neighbor.kinds[cube_index(coords)], true
}

light_local_for_remesh :: proc(nb: ^Chunk_Neighborhood, local: [3]int) -> (packed: u8, ok: bool) {
	chunk_pos, coords, coords_ok := chunk_coords_for_local(nb, local)
	if !coords_ok do return 0, false
	if chunk_pos == nb.center_pos {
		return nb.center.light[cube_index(coords)], true
	}
	neighbor, neighbor_ok := state.chunks[chunk_pos]
	if !neighbor_ok do return 0, false
	return neighbor.light[cube_index(coords)], true
}

face_light_factor :: proc(nb: ^Chunk_Neighborhood, voxel_local_min, delta: [3]int) -> f32 {
	packed, ok := light_local_for_remesh(nb, voxel_local_min + delta)
	if !ok {
		// Missing neighbor chunks should not create dark seams at the stream frontier.
		return light_level_to_factor(SKY_LIGHT_SOURCE_LEVEL)
	}
	return light_level_to_factor(light_level_max_get(packed))
}

is_solid_offset_for_ao_local :: proc(
	nb: ^Chunk_Neighborhood,
	voxel_local_min: [3]int,
	offset: [3]int,
) -> bool {
	return is_solid_local_for_remesh(nb, voxel_local_min + offset)
}

vertex_ao :: proc(
	nb: ^Chunk_Neighborhood,
	voxel_local_min: [3]int,
	normal: [3]int,
	tangent_a: [3]int,
	tangent_b: [3]int,
) -> f32 {
	side_a := is_solid_offset_for_ao_local(nb, voxel_local_min, normal + tangent_a)
	side_b := is_solid_offset_for_ao_local(nb, voxel_local_min, normal + tangent_b)
	corner := is_solid_offset_for_ao_local(nb, voxel_local_min, normal + tangent_a + tangent_b)
	if side_a && side_b {
		return 0
	}
	occlusion := 0
	if side_a do occlusion += 1
	if side_b do occlusion += 1
	if corner do occlusion += 1
	return 1.0 - f32(occlusion) / 3.0
}

voxel_face_ao_quad :: proc(
	face: Cube_Face,
	nb: ^Chunk_Neighborhood,
	voxel_local_min: [3]int,
) -> [4]f32 {
	switch face {
	case .PosX:
		return {
			vertex_ao(nb, voxel_local_min, {1, 0, 0}, {0, -1, 0}, {0, 0, -1}),
			vertex_ao(nb, voxel_local_min, {1, 0, 0}, {0, 1, 0}, {0, 0, -1}),
			vertex_ao(nb, voxel_local_min, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}),
			vertex_ao(nb, voxel_local_min, {1, 0, 0}, {0, -1, 0}, {0, 0, 1}),
		}
	case .NegX:
		return {
			vertex_ao(nb, voxel_local_min, {-1, 0, 0}, {0, -1, 0}, {0, 0, 1}),
			vertex_ao(nb, voxel_local_min, {-1, 0, 0}, {0, 1, 0}, {0, 0, 1}),
			vertex_ao(nb, voxel_local_min, {-1, 0, 0}, {0, 1, 0}, {0, 0, -1}),
			vertex_ao(nb, voxel_local_min, {-1, 0, 0}, {0, -1, 0}, {0, 0, -1}),
		}
	case .PosY:
		return {
			vertex_ao(nb, voxel_local_min, {0, 1, 0}, {-1, 0, 0}, {0, 0, -1}),
			vertex_ao(nb, voxel_local_min, {0, 1, 0}, {-1, 0, 0}, {0, 0, 1}),
			vertex_ao(nb, voxel_local_min, {0, 1, 0}, {1, 0, 0}, {0, 0, 1}),
			vertex_ao(nb, voxel_local_min, {0, 1, 0}, {1, 0, 0}, {0, 0, -1}),
		}
	case .NegY:
		return {
			vertex_ao(nb, voxel_local_min, {0, -1, 0}, {-1, 0, 0}, {0, 0, -1}),
			vertex_ao(nb, voxel_local_min, {0, -1, 0}, {1, 0, 0}, {0, 0, -1}),
			vertex_ao(nb, voxel_local_min, {0, -1, 0}, {1, 0, 0}, {0, 0, 1}),
			vertex_ao(nb, voxel_local_min, {0, -1, 0}, {-1, 0, 0}, {0, 0, 1}),
		}
	case .PosZ:
		return {
			vertex_ao(nb, voxel_local_min, {0, 0, 1}, {-1, 0, 0}, {0, -1, 0}),
			vertex_ao(nb, voxel_local_min, {0, 0, 1}, {1, 0, 0}, {0, -1, 0}),
			vertex_ao(nb, voxel_local_min, {0, 0, 1}, {1, 0, 0}, {0, 1, 0}),
			vertex_ao(nb, voxel_local_min, {0, 0, 1}, {-1, 0, 0}, {0, 1, 0}),
		}
	case .NegZ:
		return {
			vertex_ao(nb, voxel_local_min, {0, 0, -1}, {1, 0, 0}, {0, -1, 0}),
			vertex_ao(nb, voxel_local_min, {0, 0, -1}, {-1, 0, 0}, {0, -1, 0}),
			vertex_ao(nb, voxel_local_min, {0, 0, -1}, {-1, 0, 0}, {0, 1, 0}),
			vertex_ao(nb, voxel_local_min, {0, 0, -1}, {1, 0, 0}, {0, 1, 0}),
		}
	}
	unreachable()
}

// FIXME: we are making things way more complicated than they need to be by not doing queue_remesh and remeshing once at the end/beginning of the main loop.
// TODO: once we do this, we can vastly simplify sand falling too - it will be able to just call `remove_voxel_at_world` since it that proc will no longer cause a remesh on every call.
chunk_remesh :: proc(chunk_pos: Chunk_Coords, chunk: ^Chunk) {
	neighborhood := Chunk_Neighborhood {
		center_pos = chunk_pos,
		center     = chunk,
	}
	mesh := Chunk_Mesh {
		vertices_opaque      = make([dynamic]f32, 0, context.temp_allocator),
		indices_opaque       = make([dynamic]u32, 0, context.temp_allocator),
		vertices_transparent = make([dynamic]f32, 0, context.temp_allocator),
		indices_transparent  = make([dynamic]u32, 0, context.temp_allocator),
		vertices_flower      = make([dynamic]f32, 0, context.temp_allocator),
		indices_flower       = make([dynamic]u32, 0, context.temp_allocator),
	}
	// Pass 1: blocks + water.
	for kind, i in chunk.kinds {
		if kind == .None || cube_kind_is_flower(kind) do continue
		coords := cast([3]int)(local_from_index(Cube_Index(i)))
		f := [3]f32{f32(coords.x), f32(coords.y), f32(coords.z)}
		if kind == .TNT &&
		   tnt_is_armed_at(chunk_world_pos(chunk_pos, local_from_index(Cube_Index(i)))) {
			continue
		}
		is_transparent := is_transparent_kind(kind)
		neighbor_posx, _ := cube_kind_local_for_remesh(&neighborhood, coords + [3]int{1, 0, 0})
		neighbor_negx, _ := cube_kind_local_for_remesh(&neighborhood, coords + [3]int{-1, 0, 0})
		neighbor_posy, _ := cube_kind_local_for_remesh(&neighborhood, coords + [3]int{0, 1, 0})
		neighbor_negy, _ := cube_kind_local_for_remesh(&neighborhood, coords + [3]int{0, -1, 0})
		neighbor_posz, _ := cube_kind_local_for_remesh(&neighborhood, coords + [3]int{0, 0, 1})
		neighbor_negz, _ := cube_kind_local_for_remesh(&neighborhood, coords + [3]int{0, 0, -1})
		is_surface_water := kind == .Water && neighbor_posy != .Water
		top_y := f.y + 1
		if is_surface_water do top_y = f.y + WATER_SURFACE_Y_OFFSET

		{
			if !face_occluded_by_neighbor(kind, neighbor_posx) {
				ao_quad := voxel_face_ao_quad(.PosX, &neighborhood, coords)
				light_factor := face_light_factor(&neighborhood, coords, {1, 0, 0})
				chunk_append_quad(
					chunk,
					Cube_Index(i),
					&mesh,
					is_transparent,
					{f.x + 1, f.y, f.z},
					{f.x + 1, top_y, f.z},
					{f.x + 1, top_y, f.z + 1},
					{f.x + 1, f.y, f.z + 1},
					{1, 0, 0},
					.PosX,
					kind,
					ao_quad,
					light_factor,
				)
			}
		}
		{
			if !face_occluded_by_neighbor(kind, neighbor_negx) {
				ao_quad := voxel_face_ao_quad(.NegX, &neighborhood, coords)
				light_factor := face_light_factor(&neighborhood, coords, {-1, 0, 0})
				chunk_append_quad(
					chunk,
					Cube_Index(i),
					&mesh,
					is_transparent,
					{f.x, f.y, f.z + 1},
					{f.x, top_y, f.z + 1},
					{f.x, top_y, f.z},
					{f.x, f.y, f.z},
					{-1, 0, 0},
					.NegX,
					kind,
					ao_quad,
					light_factor,
				)
			}
		}
		{
			if !face_occluded_by_neighbor(kind, neighbor_posy) {
				ao_quad := voxel_face_ao_quad(.PosY, &neighborhood, coords)
				light_factor := face_light_factor(&neighborhood, coords, {0, 1, 0})
				chunk_append_quad(
					chunk,
					Cube_Index(i),
					&mesh,
					is_transparent,
					{f.x, top_y, f.z},
					{f.x, top_y, f.z + 1},
					{f.x + 1, top_y, f.z + 1},
					{f.x + 1, top_y, f.z},
					{0, 1, 0},
					.PosY,
					kind,
					ao_quad,
					light_factor,
				)
				if is_surface_water {
					chunk_append_water_top_underface(
						&mesh,
						is_transparent,
						{f.x, top_y, f.z},
						{f.x, top_y, f.z + 1},
						{f.x + 1, top_y, f.z + 1},
						{f.x + 1, top_y, f.z},
						kind,
						ao_quad,
						light_factor,
					)
				}
			}
		}
		{
			// World floor (y == 0): bottom face is never visible from below.
			if coords.y > 0 && !face_occluded_by_neighbor(kind, neighbor_negy) {
				ao_quad := voxel_face_ao_quad(.NegY, &neighborhood, coords)
				light_factor := face_light_factor(&neighborhood, coords, {0, -1, 0})
				chunk_append_quad(
					chunk,
					Cube_Index(i),
					&mesh,
					is_transparent,
					{f.x, f.y, f.z},
					{f.x + 1, f.y, f.z},
					{f.x + 1, f.y, f.z + 1},
					{f.x, f.y, f.z + 1},
					{0, -1, 0},
					.NegY,
					kind,
					ao_quad,
					light_factor,
				)
			}
		}
		{
			if !face_occluded_by_neighbor(kind, neighbor_posz) {
				ao_quad := voxel_face_ao_quad(.PosZ, &neighborhood, coords)
				light_factor := face_light_factor(&neighborhood, coords, {0, 0, 1})
				chunk_append_quad(
					chunk,
					Cube_Index(i),
					&mesh,
					is_transparent,
					{f.x, f.y, f.z + 1},
					{f.x + 1, f.y, f.z + 1},
					{f.x + 1, top_y, f.z + 1},
					{f.x, top_y, f.z + 1},
					{0, 0, 1},
					.PosZ,
					kind,
					ao_quad,
					light_factor,
				)
			}
		}
		{
			if !face_occluded_by_neighbor(kind, neighbor_negz) {
				ao_quad := voxel_face_ao_quad(.NegZ, &neighborhood, coords)
				light_factor := face_light_factor(&neighborhood, coords, {0, 0, -1})
				chunk_append_quad(
					chunk,
					Cube_Index(i),
					&mesh,
					is_transparent,
					{f.x + 1, f.y, f.z},
					{f.x, f.y, f.z},
					{f.x, top_y, f.z},
					{f.x + 1, top_y, f.z},
					{0, 0, -1},
					.NegZ,
					kind,
					ao_quad,
					light_factor,
				)
			}
		}
	}
	// Pass 2: flower sprites in a dedicated cutout mesh (no blending; depth write enabled).
	for kind, i in chunk.kinds {
		if !cube_kind_is_flower(kind) do continue
		coords := cast([3]int)(local_from_index(Cube_Index(i)))
		f := [3]f32{f32(coords.x), f32(coords.y), f32(coords.z)}
		light_factor := face_light_factor(&neighborhood, coords, {0, 1, 0})
		chunk_append_flower_billboard(&mesh, f, light_factor, kind)
	}
	chunk_gpu_sync(
		chunk_pos,
		chunk,
		mesh.vertices_opaque[:],
		mesh.indices_opaque[:],
		mesh.vertices_transparent[:],
		mesh.indices_transparent[:],
		mesh.vertices_flower[:],
		mesh.indices_flower[:],
	)
}

Chunk_Coords :: distinct [2]int
Cube_Index :: distinct int
Local_Pos :: distinct [3]int

Water_Fluid_Key :: struct {
	chunk: Chunk_Coords,
	local: Local_Pos,
}

Light_Key :: struct {
	chunk: Chunk_Coords,
	local: Local_Pos,
}

Block_Change :: struct {
	chunk:    Chunk_Coords,
	local:    Local_Pos,
	old_kind: Cube_Kind,
	new_kind: Cube_Kind,
}

// Squared Euclidean distance in chunk grid; keep unload radius larger than load radius to avoid boundary thrash.
CHUNK_STREAM_LOAD_RADIUS :: 3
CHUNK_STREAM_UNLOAD_RADIUS :: CHUNK_STREAM_LOAD_RADIUS + 2
REMESH_CHUNKS_PER_FRAME :: 6

// XZ chunk grid: neighbors that share a full voxel face across a chunk boundary (used for incremental remesh).
CHUNK_CARDINAL_OFFSETS :: [4]Chunk_Coords{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
// Must match SEA_LEVEL in terrain.wgsl so reflected water aligns with generated oceans.
SEA_LEVEL :: 12
WATER_SURFACE_Y_OFFSET :: 0.9
WATER_OVERFLOW_OFFSETS :: [5][3]int{{1, 0, 0}, {-1, 0, 0}, {0, 0, 1}, {0, 0, -1}, {0, -1, 0}}
NEIGHBOR_OFFSETS_6 :: [6][3]int {
	{1, 0, 0},
	{-1, 0, 0},
	{0, 1, 0},
	{0, -1, 0},
	{0, 0, 1},
	{0, 0, -1},
}
// One spread wave per this many real seconds; snapshot queue at wave start so new water waits for the next wave.
WATER_FLUID_WAVE_INTERVAL_SEC :: f32(1.0)
// Keep worst-case fluid work bounded so mass terrain edits cannot stall a single frame.
WATER_FLUID_KEYS_PER_WAVE_MAX :: 4096
LIGHT_MAX :: u8(15)
// Drive sky light below full range so exteriors read like evening instead of noon.
SKY_LIGHT_SOURCE_LEVEL :: u8(10)
LIGHT_NIBBLE_MASK :: u8(0x0F)
LIGHT_SKY_SHIFT :: u8(4)
// Keep pumpkin glow below max so cave contrast survives while still reading as a clear light source.
PUMPKIN_EMISSION_LEVEL :: u8(11)

// Eye is above the collision AABB center; must stay in sync with movement/collision in game_step.
PLAYER_EYE_OFFSET_FROM_CENTER_Y :: 0.72
PLAYER_HALF_EXTENTS :: [3]f32{0.3, 0.9, 0.3}

cube_index :: proc(pos: Local_Pos) -> Cube_Index {
	return Cube_Index(pos.x + pos.y * CHUNK_WIDTH + pos.z * CHUNK_WIDTH * CHUNK_HEIGHT)
}

light_sky_get :: proc(packed: u8) -> u8 {
	return (packed >> LIGHT_SKY_SHIFT) & LIGHT_NIBBLE_MASK
}

light_block_get :: proc(packed: u8) -> u8 {
	return packed & LIGHT_NIBBLE_MASK
}

light_sky_set :: proc(packed: ^u8, value: u8) {
	v := value
	if v > LIGHT_MAX do v = LIGHT_MAX
	packed^ = (packed^ & 0x0F) | (v << LIGHT_SKY_SHIFT)
}

light_block_set :: proc(packed: ^u8, value: u8) {
	v := value
	if v > LIGHT_MAX do v = LIGHT_MAX
	packed^ = (packed^ & 0xF0) | (v & LIGHT_NIBBLE_MASK)
}

light_level_max_get :: proc(packed: u8) -> u8 {
	sky := light_sky_get(packed)
	block := light_block_get(packed)
	if sky > block {
		return sky
	}
	return block
}

light_level_to_factor :: proc(level: u8) -> f32 {
	// Keep a little ambient floor so deep caves still read as shapes instead of flat black silhouettes.
	return 0.08 + 0.92 * f32(level) / f32(LIGHT_MAX)
}

cube_opacity :: proc(kind: Cube_Kind) -> u8 {
	switch kind {
	case .None, FLOWER_KINDS_FIRST ..= FLOWER_KINDS_LAST:
		return 0
	case .Water:
		return 3
	case .Grass,
	     .Dirt,
	     .Stone,
	     .Bedrock,
	     .Sand,
	     .Wood,
	     .Cobblestone,
	     .Pumpkin,
	     .Brick,
	     .TNT,
	     .Ore_Diamond,
	     .Ore_Gold,
	     .Ore_Iron,
	     .Ore_Green,
	     .Ore_Red,
	     .Ore_Blue,
	     .Ore_Coal:
		return LIGHT_MAX
	}
	unreachable()
}

cube_emission :: proc(kind: Cube_Kind) -> u8 {
	switch kind {
	case .Pumpkin:
		return PUMPKIN_EMISSION_LEVEL
	case .None,
	     .Grass,
	     .Dirt,
	     .Stone,
	     .Bedrock,
	     .Water,
	     .Sand,
	     .Wood,
	     .Cobblestone,
	     .Brick,
	     .TNT,
	     .Ore_Diamond,
	     .Ore_Gold,
	     .Ore_Iron,
	     .Ore_Green,
	     .Ore_Red,
	     .Ore_Blue,
	     .Ore_Coal,
	     FLOWER_KINDS_FIRST ..=
	     FLOWER_KINDS_LAST:
		return 0
	}
	unreachable()
}

light_queues_reset :: proc() {
	clear(&state.light_decrease_queue)
	clear(&state.light_increase_queue)
	clear(&state.light_decrease_seen)
	clear(&state.light_increase_seen)
}

light_queue_enqueue_increase :: proc(chunk: Chunk_Coords, local: Local_Pos) {
	if !(chunk in state.chunks) do return
	key := Light_Key{chunk, local}
	if key in state.light_increase_seen do return
	state.light_increase_seen[key] = {}
	append(&state.light_increase_queue, key)
}

light_queue_enqueue_decrease :: proc(chunk: Chunk_Coords, local: Local_Pos) {
	if !(chunk in state.chunks) do return
	key := Light_Key{chunk, local}
	if key in state.light_decrease_seen do return
	state.light_decrease_seen[key] = {}
	append(&state.light_decrease_queue, key)
}

lighting_column_recompute :: proc(
	chunk_pos: Chunk_Coords,
	local_x, local_z: int,
	dirty: ^map[Chunk_Coords]struct{},
) {
	if !(chunk_pos in state.chunks) do return
	chunk := &state.chunks[chunk_pos]
	sky := SKY_LIGHT_SOURCE_LEVEL
	for y := CHUNK_HEIGHT - 1; y >= 0; y -= 1 {
		coords := Local_Pos{local_x, y, local_z}
		idx := cube_index(coords)
		kind := chunk.kinds[idx]
		opacity := cube_opacity(kind)
		new_sky := sky
		if opacity >= LIGHT_MAX {
			new_sky = 0
		} else if opacity > 0 {
			if new_sky > opacity {
				new_sky -= opacity
			} else {
				new_sky = 0
			}
		}

		packed := chunk.light[idx]
		old_sky := light_sky_get(packed)
		if old_sky != new_sky {
			if new_sky > old_sky {
				light_queue_enqueue_increase(chunk_pos, coords)
			} else {
				light_queue_enqueue_decrease(chunk_pos, coords)
			}
			light_sky_set(&packed, new_sky)
			chunk.light[idx] = packed
			dirty[chunk_pos] = {}
		}

		emit := cube_emission(kind)
		if light_block_get(packed) != emit {
			light_block_set(&packed, emit)
			chunk.light[idx] = packed
			dirty[chunk_pos] = {}
			if emit > 0 {
				light_queue_enqueue_increase(chunk_pos, coords)
			}
		}
		sky = new_sky
	}
}

lighting_propagate_decrease :: proc(dirty: ^map[Chunk_Coords]struct{}, include_sky: bool) {
	head := 0
	for head < len(state.light_decrease_queue) {
		key := state.light_decrease_queue[head]
		head += 1
		// Dedupe should only suppress duplicate pending entries; once popped, allow re-enqueue.
		delete_key(&state.light_decrease_seen, key)
		if !(key.chunk in state.chunks) do continue
		chunk := &state.chunks[key.chunk]
		idx := cube_index(key.local)
		cur := chunk.light[idx]
		cur_sky: u8
		if include_sky do cur_sky = light_sky_get(cur)
		cur_block := light_block_get(cur)
		local := [3]int{key.local.x, key.local.y, key.local.z}
		for d in NEIGHBOR_OFFSETS_6 {
			ncp, ncl, ok := chunk_neighbor_from_offset(key.chunk, local, d)
			if !ok || !(ncp in state.chunks) do continue
			nchunk := &state.chunks[ncp]
			nidx := cube_index(ncl)
			np := nchunk.light[nidx]
			ns: u8
			if include_sky do ns = light_sky_get(np)
			nb := light_block_get(np)
			if ns == 0 && nb == 0 do continue

			// If a neighbor can still be lit by direct emission, keep it as a refill seed.
			emit := cube_emission(nchunk.kinds[nidx])
			if emit > 0 {
				light_queue_enqueue_increase(ncp, ncl)
				continue
			}

			atten := cube_opacity(nchunk.kinds[nidx])
			if atten < 1 do atten = 1
			expected_sky: u8
			expected_block: u8
			if cur_sky > atten do expected_sky = cur_sky - atten
			if cur_block > atten do expected_block = cur_block - atten

			// Stale block light can survive emitter removal if we only enqueue increase.
			// Invalidate block-light locally, but avoid touching sky light here to keep chunk streaming fast.
			if nb > expected_block {
				light_block_set(&np, emit)
				nchunk.light[nidx] = np
				dirty[ncp] = {}
				light_queue_enqueue_decrease(ncp, ncl)
				light_queue_enqueue_increase(ncp, ncl)
				// This cell was just invalidated, so it no longer carries light to refill itself.
				// Seed neighboring cells too, so alternate paths (e.g. around a removed blocker)
				// can immediately repopulate the gap instead of leaving a dark scar.
				nlocal := [3]int{ncl.x, ncl.y, ncl.z}
				for nd in NEIGHBOR_OFFSETS_6 {
					rp, rl, rok := chunk_neighbor_from_offset(ncp, nlocal, nd)
					if !rok do continue
					light_queue_enqueue_increase(rp, rl)
				}
				continue
			}
			if include_sky {
				// Sky light frequently has alternate support paths while chunks stream in/out.
				// Let increase propagation re-validate it without forcing an expensive local reset.
				if ns > expected_sky {
					light_queue_enqueue_increase(ncp, ncl)
					continue
				}

				light_sky_set(&np, 0)
				light_block_set(&np, 0)
				nchunk.light[nidx] = np
				dirty[ncp] = {}
				light_queue_enqueue_decrease(ncp, ncl)
				light_queue_enqueue_increase(ncp, ncl)
				continue
			}
		}
	}
	clear(&state.light_decrease_queue)
	clear(&state.light_decrease_seen)
}

lighting_propagate_increase :: proc(dirty: ^map[Chunk_Coords]struct{}, include_sky: bool) {
	head := 0
	for head < len(state.light_increase_queue) {
		key := state.light_increase_queue[head]
		head += 1
		// A stronger path discovered later in the pass must be able to queue this cell again.
		delete_key(&state.light_increase_seen, key)
		if !(key.chunk in state.chunks) do continue
		chunk := &state.chunks[key.chunk]
		idx := cube_index(key.local)
		cur := chunk.light[idx]
		cur_sky: u8
		if include_sky do cur_sky = light_sky_get(cur)
		cur_block := light_block_get(cur)
		if include_sky {
			if cur_sky <= 1 && cur_block <= 1 do continue
		} else if cur_block <= 1 {
			continue
		}

		local := [3]int{key.local.x, key.local.y, key.local.z}
		for d in NEIGHBOR_OFFSETS_6 {
			ncp, ncl, ok := chunk_neighbor_from_offset(key.chunk, local, d)
			if !ok || !(ncp in state.chunks) do continue
			nchunk := &state.chunks[ncp]
			nidx := cube_index(ncl)
			np := nchunk.light[nidx]
			opacity := cube_opacity(nchunk.kinds[nidx])
			atten := opacity
			if atten < 1 do atten = 1

			next_sky: u8
			next_block: u8
			if cur_sky > atten do next_sky = cur_sky - atten
			if cur_block > atten do next_block = cur_block - atten
			emit := cube_emission(nchunk.kinds[nidx])
			if emit > next_block do next_block = emit

			old_sky := light_sky_get(np)
			old_block := light_block_get(np)
			if include_sky {
				if next_sky <= old_sky && next_block <= old_block do continue
			} else if next_block <= old_block {
				continue
			}
			if include_sky && next_sky > old_sky do light_sky_set(&np, next_sky)
			if next_block > old_block do light_block_set(&np, next_block)
			nchunk.light[nidx] = np
			dirty[ncp] = {}
			light_queue_enqueue_increase(ncp, ncl)
		}
	}
	clear(&state.light_increase_queue)
	clear(&state.light_increase_seen)
}

World_Pos :: distinct [3]f32

lighting_seed_after_block_change :: proc(
	chunk_pos: Chunk_Coords,
	local: Local_Pos,
	old_kind, new_kind: Cube_Kind,
	dirty: ^map[Chunk_Coords]struct{},
) {
	old_opacity := cube_opacity(old_kind)
	new_opacity := cube_opacity(new_kind)
	old_emission := cube_emission(old_kind)
	new_emission := cube_emission(new_kind)
	if old_opacity == new_opacity && old_emission == new_emission {
		return
	}

	// Sky column recompute is only needed when opacity changes (occluders added/removed).
	if old_opacity != new_opacity {
		lighting_column_recompute(chunk_pos, local.x, local.z, dirty)
	}
	light_queue_enqueue_increase(chunk_pos, local)

	if new_opacity > old_opacity || new_emission < old_emission {
		light_queue_enqueue_decrease(chunk_pos, local)
	}

	local_i := [3]int{local.x, local.y, local.z}
	for d in NEIGHBOR_OFFSETS_6 {
		ncp, ncl, ok := chunk_neighbor_from_offset(chunk_pos, local_i, d)
		if !ok do continue
		light_queue_enqueue_increase(ncp, ncl)
		if new_opacity > old_opacity || new_emission < old_emission {
			light_queue_enqueue_decrease(ncp, ncl)
		}
	}
}

lighting_update_after_block_change :: proc(
	chunk_pos: Chunk_Coords,
	local: Local_Pos,
	old_kind, new_kind: Cube_Kind,
	dirty: ^map[Chunk_Coords]struct{},
) {
	light_queues_reset()
	lighting_seed_after_block_change(chunk_pos, local, old_kind, new_kind, dirty)
	lighting_propagate_decrease(dirty, true)
	lighting_propagate_increase(dirty, true)
}

lighting_rebuild_for_chunks :: proc(chunks: []Chunk_Coords, changed: ^map[Chunk_Coords]struct{}) {
	added := make(map[Chunk_Coords]struct{}, context.temp_allocator)
	for chunk_pos in chunks {
		if chunk_pos in state.chunks {
			added[chunk_pos] = {}
		}
	}

	light_queues_reset()

	for chunk_pos in added {
		chunk := &state.chunks[chunk_pos]
		for i in 0 ..< CHUNK_VOXEL_COUNT {
			packed := chunk.light[i]
			light_sky_set(&packed, 0)
			light_block_set(&packed, cube_emission(chunk.kinds[i]))
			chunk.light[i] = packed
		}
		for z in 0 ..< CHUNK_WIDTH {
			for x in 0 ..< CHUNK_WIDTH {
				lighting_column_recompute(chunk_pos, x, z, changed)
			}
		}
	}

	for chunk_pos in added {
		chunk := &state.chunks[chunk_pos]
		for i in 0 ..< CHUNK_VOXEL_COUNT {
			coords := local_from_index(Cube_Index(i))
			packed := chunk.light[i]
			if light_sky_get(packed) > 1 || light_block_get(packed) > 1 {
				light_queue_enqueue_increase(chunk_pos, coords)
			}
		}
	}

	// Boundary-only handoff: seed propagation from newly loaded chunk border voxels
	// into already-loaded neighboring chunks.
	for chunk_pos in added {
		chunk := &state.chunks[chunk_pos]
		for y in 0 ..< CHUNK_HEIGHT {
			for z in 0 ..< CHUNK_WIDTH {
				coords := Local_Pos{0, y, z}
				packed := chunk.light[cube_index(coords)]
				if light_level_max_get(packed) > 1 {
					neighbor := chunk_pos + Chunk_Coords{-1, 0}
					if neighbor in state.chunks && !(neighbor in added) {
						light_queue_enqueue_increase(chunk_pos, coords)
					}
				}

				coords = Local_Pos{CHUNK_WIDTH - 1, y, z}
				packed = chunk.light[cube_index(coords)]
				if light_level_max_get(packed) > 1 {
					neighbor := chunk_pos + Chunk_Coords{1, 0}
					if neighbor in state.chunks && !(neighbor in added) {
						light_queue_enqueue_increase(chunk_pos, coords)
					}
				}
			}

			for x in 0 ..< CHUNK_WIDTH {
				coords := Local_Pos{x, y, 0}
				packed := chunk.light[cube_index(coords)]
				if light_level_max_get(packed) > 1 {
					neighbor := chunk_pos + Chunk_Coords{0, -1}
					if neighbor in state.chunks && !(neighbor in added) {
						light_queue_enqueue_increase(chunk_pos, coords)
					}
				}

				coords = Local_Pos{x, y, CHUNK_WIDTH - 1}
				packed = chunk.light[cube_index(coords)]
				if light_level_max_get(packed) > 1 {
					neighbor := chunk_pos + Chunk_Coords{0, 1}
					if neighbor in state.chunks && !(neighbor in added) {
						light_queue_enqueue_increase(chunk_pos, coords)
					}
				}
			}
		}
	}

	lighting_propagate_increase(changed, true)
}

local_from_index :: proc(index: Cube_Index) -> Local_Pos {
	return {
		int(index) % CHUNK_WIDTH,
		int(index) / CHUNK_WIDTH % CHUNK_HEIGHT,
		int(index) / CHUNK_WIDTH / CHUNK_HEIGHT,
	}
}

init_chunk_flat :: proc(chunk: ^Chunk) {
	chunk.kinds = {}
	for i in 0 ..< len(chunk.kinds) {
		pos := local_from_index(Cube_Index(i))
		if pos.y == 0 {
			chunk.kinds[i] = .Bedrock
		} else if pos.y == CHUNK_HEIGHT - 1 {
			chunk.kinds[i] = .Grass
		} else {
			chunk.kinds[i] = .Stone
		}
	}
}

is_solid_voxel :: proc(pos: [3]f32) -> bool {
	kind := cube_at(pos) or_return
	if kind == .None do return false
	if kind == .Water do return false
	if cube_kind_is_flower(kind) do return false
	return true
}

player_in_water :: proc() -> bool {
	eye := state.player_pos
	center := state.player_pos
	center.y -= PLAYER_EYE_OFFSET_FROM_CENTER_Y
	kind_eye, ok_eye := cube_at(eye)
	if ok_eye && kind_eye == .Water {
		return true
	}
	kind_center, ok_center := cube_at(center)
	return ok_center && kind_center == .Water
}

player_eye_in_water :: proc() -> bool {
	kind, ok := cube_at(state.player_pos)
	return ok && kind == .Water
}

// Open air from the eye to the world ceiling - rain/sky effects only make sense when nothing solid caps this column.
player_has_open_sky_above :: proc() -> bool {
	col_x := int(math.floor(state.player_pos.x))
	col_z := int(math.floor(state.player_pos.z))
	for wy in int(math.floor(state.player_pos.y)) + 1 ..< CHUNK_HEIGHT {
		if is_solid_voxel({f32(col_x) + 0.5, f32(wy) + 0.5, f32(col_z) + 0.5}) {
			return false
		}
	}
	return true
}

// Buoyancy only near an open surface: same (x,z) column must reach air (not solid) above the water we're in,
// and the body must be within BUOYANCY_MAX_DEPTH_M of that surface (downward in water).
// factor ramps 0 to 1 from the bottom of that band to the surface so a tap of swim near the lake floor
// doesn't instantly hit full WATER_BUOYANCY + terminal upward speed for the whole column.
player_water_buoyancy_sample :: proc() -> (near_open_surface: bool, factor: f32) {
	BUOYANCY_MAX_DEPTH_M :: 1.35
	center := state.player_pos
	center.y -= PLAYER_EYE_OFFSET_FROM_CENTER_Y
	col_x: int
	col_z: int
	kind_eye, ok_eye := cube_at(state.player_pos)
	if ok_eye && kind_eye == .Water {
		col_x = int(math.floor(state.player_pos.x))
		col_z = int(math.floor(state.player_pos.z))
	} else {
		col_x = int(math.floor(center.x))
		col_z = int(math.floor(center.z))
	}
	wy_start := int(math.floor(center.y)) + 1
	for wy in wy_start ..< CHUNK_HEIGHT {
		kind, ok := cube_at({f32(col_x) + 0.5, f32(wy) + 0.5, f32(col_z) + 0.5})
		if !ok {
			return false, 0
		}
		if kind == .Water {
			continue
		}
		if kind != .None {
			// Roof or wall before any air - no free surface above this water pocket.
			return false, 0
		}
		surface_y := f32(wy)
		depth := surface_y - center.y
		if depth < 0 || depth > BUOYANCY_MAX_DEPTH_M {
			return false, 0
		}
		f := 1 - depth / BUOYANCY_MAX_DEPTH_M
		return true, math.clamp(f, 0, 1)
	}
	return false, 0
}

// True when the body AABB sits against solid in +/-X or +/-Z (ledge / pool wall), for climb-out jumps at the surface.
player_hugging_wall :: proc() -> bool {
	c := state.player_pos
	c.y -= PLAYER_EYE_OFFSET_FROM_CENTER_Y
	h := PLAYER_HALF_EXTENTS
	PROBE :: 0.22
	// A few heights so thin ledges and deep walls still register.
	y_offs := [3]f32{-0.45, 0, 0.45}
	dirs := [4][3]f32 {
		{h.x + PROBE, 0, 0},
		{-h.x - PROBE, 0, 0},
		{0, 0, h.z + PROBE},
		{0, 0, -h.z - PROBE},
	}
	for d in dirs {
		for yo in y_offs {
			if is_solid_voxel(c + d + {0, yo, 0}) {
				return true
			}
		}
	}
	return false
}

aabb_collides :: proc(center, half_extents: [3]f32) -> bool {
	EPS :: 0.0001
	min_x := int(math.floor(center.x - half_extents.x))
	max_x := int(math.floor(center.x + half_extents.x - EPS))
	min_y := int(math.floor(center.y - half_extents.y))
	max_y := int(math.floor(center.y + half_extents.y - EPS))
	min_z := int(math.floor(center.z - half_extents.z))
	max_z := int(math.floor(center.z + half_extents.z - EPS))

	for z in min_z ..= max_z {
		for y in min_y ..= max_y {
			for x in min_x ..= max_x {
				if is_solid_voxel({f32(x) + 0.5, f32(y) + 0.5, f32(z) + 0.5}) {
					return true
				}
			}
		}
	}
	return false
}

move_axis_with_collision :: proc(
	pos: [3]f32,
	delta: f32,
	axis: int,
	half_extents: [3]f32,
) -> (
	new_pos: [3]f32,
	blocked: bool,
) {
	new_pos = pos
	if delta == 0 {
		return
	}

	MAX_STEP :: 0.1
	steps := int(math.max(1, math.ceil(math.abs(delta) / MAX_STEP)))
	step := delta / f32(steps)
	for _ in 0 ..< steps {
		candidate := new_pos
		candidate[axis] += step
		if aabb_collides(candidate, half_extents) {
			return new_pos, true
		}
		new_pos = candidate
	}
	return
}

tnt_resolve_body_collisions :: proc() {
	if len(state.armed_tnt) < 2 do return

	full_extents := TNT_AABB_HALF_EXTENTS * 2
	// A couple of passes keeps chain overlaps from leaving bodies interpenetrating.
	for _ in 0 ..< 2 {
		for i in 0 ..< len(state.armed_tnt) - 1 {
			for j in i + 1 ..< len(state.armed_tnt) {
				a := &state.armed_tnt[i]
				b := &state.armed_tnt[j]
				a_center := a.pos + [3]f32{0.5, 0.5, 0.5}
				b_center := b.pos + [3]f32{0.5, 0.5, 0.5}
				diff := b_center - a_center

				overlap_x := full_extents.x - math.abs(diff.x)
				overlap_y := full_extents.y - math.abs(diff.y)
				overlap_z := full_extents.z - math.abs(diff.z)
				if overlap_x <= 0 || overlap_y <= 0 || overlap_z <= 0 do continue

				resolve_axis := 0
				resolve_overlap := overlap_x
				if overlap_y < resolve_overlap {
					resolve_axis = 1
					resolve_overlap = overlap_y
				}
				if overlap_z < resolve_overlap {
					resolve_axis = 2
					resolve_overlap = overlap_z
				}

				dir := f32(1)
				if diff[resolve_axis] < 0 do dir = -1
				half_push := resolve_overlap * 0.5

				a_candidate := a_center
				b_candidate := b_center
				a_candidate[resolve_axis] -= dir * half_push
				b_candidate[resolve_axis] += dir * half_push

				// Keep separation stable without shoving TNT into terrain when packed in tight spots.
				if !aabb_collides(a_candidate, TNT_AABB_HALF_EXTENTS) {
					a_center = a_candidate
				}
				if !aabb_collides(b_candidate, TNT_AABB_HALF_EXTENTS) {
					b_center = b_candidate
				}
				a.pos = a_center - [3]f32{0.5, 0.5, 0.5}
				b.pos = b_center - [3]f32{0.5, 0.5, 0.5}

				// Only damp relative velocity when moving toward each other on the resolved axis.
				relative_speed := (b.vel[resolve_axis] - a.vel[resolve_axis]) * dir
				if relative_speed < 0 {
					shared_speed := (a.vel[resolve_axis] + b.vel[resolve_axis]) * 0.5
					a.vel[resolve_axis] = shared_speed
					b.vel[resolve_axis] = shared_speed
				}
			}
		}
	}
}

dig_raycast_first_solid :: proc(
	origin: [3]f32,
	dir: [3]f32,
	max_reach: f32,
) -> (
	hit_pos: [3]f32,
	ok: bool,
) {
	if dir == {} {
		return {}, false
	}
	ray_dir := linalg.normalize(dir)
	t: f32
	for t <= max_reach {
		sample := origin + ray_dir * t
		kind, kind_ok := cube_at(sample)
		if kind_ok && kind != .None && kind != .Water {
			return sample, true
		}
		t += DIG_RAY_STEP
	}
	return {}, false
}

place_raycast_last_empty_before_solid :: proc(
	origin: [3]f32,
	dir: [3]f32,
	max_reach: f32,
) -> (
	place_pos: [3]f32,
	ok: bool,
) {
	if dir == {} {
		return {}, false
	}
	ray_dir := linalg.normalize(dir)
	last_empty: [3]f32
	have_last_empty := false
	t: f32
	for t <= max_reach {
		sample := origin + ray_dir * t
		kind, kind_ok := cube_at(sample)
		if !kind_ok {
			t += DIG_RAY_STEP
			continue
		}
		// Treat water like empty space for build targeting so we can place blocks underwater.
		if kind == .None || kind == .Water {
			last_empty = sample
			have_last_empty = true
			t += DIG_RAY_STEP
			continue
		}
		// Place into the last empty voxel in front of the hit solid voxel.
		if kind != .Water && have_last_empty {
			return last_empty, true
		}
		return {}, false
	}
	return {}, false
}

player_aabb_overlaps_voxel :: proc(world_pos: [3]f32) -> bool {
	voxel_center := [3]f32 {
		f32(int(math.floor(world_pos.x))) + 0.5,
		f32(int(math.floor(world_pos.y))) + 0.5,
		f32(int(math.floor(world_pos.z))) + 0.5,
	}
	player_center := state.player_pos
	player_center.y -= PLAYER_EYE_OFFSET_FROM_CENTER_Y

	dx := math.abs(voxel_center.x - player_center.x)
	dy := math.abs(voxel_center.y - player_center.y)
	dz := math.abs(voxel_center.z - player_center.z)

	return(
		dx < PLAYER_HALF_EXTENTS.x + 0.5 &&
		dy < PLAYER_HALF_EXTENTS.y + 0.5 &&
		dz < PLAYER_HALF_EXTENTS.z + 0.5 \
	)
}

resolve_placeable_voxel_at_world :: proc(
	world_pos: [3]f32,
) -> (
	chunk_pos: Chunk_Coords,
	local: Local_Pos,
	ok: bool,
) {
	wy := int(math.floor(world_pos.y))
	if wy < 0 || wy >= CHUNK_HEIGHT do return {}, {}, false

	if player_aabb_overlaps_voxel(world_pos) do return {}, {}, false

	chunk_pos = chunk_coords(world_pos)
	if !(chunk_pos in state.chunks) do return {}, {}, false

	local = local_pos(world_pos, chunk_pos)
	if !local_within_chunk(local) do return {}, {}, false

	chunk := state.chunks[chunk_pos]
	idx := cube_index(local)
	current_kind := chunk.kinds[idx]
	if current_kind != .None && current_kind != .Water do return {}, {}, false

	return chunk_pos, local, true
}

block_change_enqueue :: proc(
	chunk_pos: Chunk_Coords,
	local: Local_Pos,
	new_kind: Cube_Kind,
) -> bool {
	if !(chunk_pos in state.chunks) do return false
	chunk := &state.chunks[chunk_pos]
	idx := cube_index(local)
	old_kind := chunk.kinds[idx]
	if old_kind == new_kind do return false
	append(
		&state.block_change_queue,
		Block_Change{chunk = chunk_pos, local = local, old_kind = old_kind, new_kind = new_kind},
	)
	return true
}

block_change_apply_batch :: proc() {
	if len(state.block_change_queue) == 0 do return
	dirty := make(map[Chunk_Coords]struct{}, context.temp_allocator)
	light_queues_reset()
	any_opacity_change := false
	for change in state.block_change_queue {
		if !(change.chunk in state.chunks) do continue
		chunk := &state.chunks[change.chunk]
		idx := cube_index(change.local)
		old_kind := chunk.kinds[idx]
		if old_kind == change.new_kind do continue
		if cube_opacity(old_kind) != cube_opacity(change.new_kind) {
			any_opacity_change = true
		}
		chunk.kinds[idx] = change.new_kind
		delete_key(&chunk.rotation_by_index, idx)
		if change.new_kind == .Pumpkin {
			// Batch placement still needs deterministic pumpkin-facing metadata.
			chunk.rotation_by_index[idx] = placed_cube_rotation_from_player_forward(
				linalg.quaternion_mul_vector3(state.player_rotation, WORLD_FORWARD),
			)
		}
		packed := chunk.light[idx]
		light_block_set(&packed, cube_emission(change.new_kind))
		chunk.light[idx] = packed
		chunk_remesh_find_dirty_neighbors_based_on_modified_voxel(
			&dirty,
			change.chunk,
			change.local,
		)
		lighting_seed_after_block_change(
			change.chunk,
			change.local,
			old_kind,
			change.new_kind,
			&dirty,
		)
		if old_kind != .None && change.new_kind == .None {
			water_fluid_schedule_after_voxel_removal(change.chunk, change.local)
		}
	}
	clear(&state.block_change_queue)
	lighting_propagate_decrease(&dirty, any_opacity_change)
	lighting_propagate_increase(&dirty, any_opacity_change)
	remesh_queue_enqueue_dirty_map(dirty)
}

place_voxel_at_world :: proc(
	world_pos: [3]f32,
	kind_to_place: Cube_Kind,
) -> (
	chunk_pos: Chunk_Coords,
	local: Local_Pos,
	placed: bool,
) {
	ok: bool
	chunk_pos, local, ok = resolve_placeable_voxel_at_world(world_pos)
	if !ok do return {}, {}, false
	if !block_change_enqueue(chunk_pos, local, kind_to_place) do return {}, {}, false
	block_change_apply_batch()
	return chunk_pos, local, true
}

resolve_removable_voxel_at_world :: proc(
	world_pos: [3]f32,
) -> (
	chunk_pos: Chunk_Coords,
	local: Local_Pos,
	kind: Cube_Kind,
	ok: bool,
) {
	{
		wy := int(math.floor(world_pos.y))
		if wy < 0 || wy >= CHUNK_HEIGHT do return {}, {}, .None, false
	}

	chunk_pos = chunk_coords(world_pos)
	if !(chunk_pos in state.chunks) do return {}, {}, .None, false

	local = local_pos(world_pos, chunk_pos)
	if !local_within_chunk(local) do return {}, {}, .None, false

	chunk := &state.chunks[chunk_pos]
	idx := cube_index(local)
	kind = chunk.kinds[idx]
	if kind == .None do return {}, {}, .None, false
	return chunk_pos, local, kind, true
}

remove_voxel_at_world_internal :: proc(
	world_pos: [3]f32,
) -> (
	chunk_pos: Chunk_Coords,
	local: Local_Pos,
	removed: bool,
) {
	old_kind: Cube_Kind
	ok: bool
	chunk_pos, local, old_kind, ok = resolve_removable_voxel_at_world(world_pos)
	if !ok do return {}, {}, false
	_ = old_kind
	if !block_change_enqueue(chunk_pos, local, .None) do return {}, {}, false
	above := local + {0, 1, 0}
	if above.y < CHUNK_HEIGHT && chunk_pos in state.chunks {
		chunk := &state.chunks[chunk_pos]
		above_idx := cube_index(above)
		if cube_kind_is_flower(chunk.kinds[above_idx]) {
			_ = block_change_enqueue(chunk_pos, above, .None)
		}
	}
	return chunk_pos, local, true
}

remove_voxel_at_world :: proc(
	world_pos: [3]f32,
) -> (
	chunk_pos: Chunk_Coords,
	local: Local_Pos,
	removed: bool,
) {
	chunk_pos, local, removed = remove_voxel_at_world_internal(world_pos)
	if !removed do return {}, {}, false
	block_change_apply_batch()
	return chunk_pos, local, true
}

tnt_arm_if_needed :: proc(world_pos: [3]f32, initial_vel: [3]f32 = {}) -> bool {
	voxel_pos := world_voxel_origin(world_pos)
	for i in 0 ..< len(state.armed_tnt) {
		if world_voxel_origin(state.armed_tnt[i].pos) == voxel_pos {
			return false
		}
	}
	// Keep gameplay/rendering ownership single-source: if removal fails, do not create an armed overlay copy.
	// Otherwise the terrain TNT may remain and incorrectly occlude neighbors while a dynamic TNT also exists.
	_, _, removed := remove_voxel_at_world_internal(voxel_pos)
	if !removed {
		return false
	}
	block_change_apply_batch()
	append(&state.armed_tnt, Armed_TNT{pos = voxel_pos, vel = initial_vel})
	return true
}

tnt_is_armed_at :: proc(world_pos: [3]f32) -> bool {
	voxel_pos := world_voxel_origin(world_pos)
	for i in 0 ..< len(state.armed_tnt) {
		if world_voxel_origin(state.armed_tnt[i].pos) == voxel_pos {
			return true
		}
	}
	return false
}

tnt_explode :: proc(center_world_pos: [3]f32) {
	center := world_voxel_origin(center_world_pos)
	chain_arms := make([dynamic][3]f32, 0, context.temp_allocator)
	radius2 := TNT_EXPLOSION_RADIUS * TNT_EXPLOSION_RADIUS

	for dy in -TNT_EXPLOSION_RADIUS ..= TNT_EXPLOSION_RADIUS {
		for dz in -TNT_EXPLOSION_RADIUS ..= TNT_EXPLOSION_RADIUS {
			for dx in -TNT_EXPLOSION_RADIUS ..= TNT_EXPLOSION_RADIUS {
				dist2 := dx * dx + dy * dy + dz * dz
				if dist2 > radius2 do continue

				target := center + [3]f32{f32(dx), f32(dy), f32(dz)}
				kind, ok := cube_at(target)
				if !ok || kind == .None || kind == .Bedrock do continue

				if kind == .TNT && target != center {
					// Arm in a dedicated pass below. Do not remove here, otherwise arming sees no voxel
					// and bails out, which breaks chain reactions.
					append(&chain_arms, target)
					continue
				}

				_, _, _ = remove_voxel_at_world_internal(target)
			}
		}
	}
	block_change_apply_batch()
	for chain_pos in chain_arms {
		// Explosions often arm multiple TNT in the same frame; seed with both world position and
		// elapsed time so chain reactions spread with believable variation instead of a fixed radial push.
		random_seed :=
			chain_pos.x * 12.9898 +
			chain_pos.y * 78.233 +
			chain_pos.z * 37.719 +
			state.elapsed_time * 17.171
		random01 := math.abs(math.sin(random_seed) * 43758.5453)
		random01 -= math.floor(random01)
		random_angle := random01 * (2.0 * math.PI)
		kick_dir := [3]f32{math.cos(random_angle), 0, math.sin(random_angle)}
		chain_vel := kick_dir * (TNT_ARM_KICK_SPEED * 0.35)
		chain_vel.y = TNT_CHAIN_ARM_UPWARD_KICK
		_ = tnt_arm_if_needed(chain_pos, chain_vel)
	}
}

tnt_update_fuses :: proc(dt: f32) {
	clear(&state.tnt_flash_highlight_world_pos)
	detonations := make([dynamic][3]f32, context.temp_allocator)
	i := 0
	for i < len(state.armed_tnt) {
		armed := &state.armed_tnt[i]
		armed.vel.y += TNT_GRAVITY * dt
		armed.vel.y = math.max(armed.vel.y, TNT_TERMINAL_FALL_SPEED)
		armed_center := armed.pos + [3]f32{0.5, 0.5, 0.5}
		delta := armed.vel * dt
		next_center, blocked_x := move_axis_with_collision(
			armed_center,
			delta.x,
			0,
			TNT_AABB_HALF_EXTENTS,
		)
		blocked_z: bool
		next_center, blocked_z = move_axis_with_collision(
			next_center,
			delta.z,
			2,
			TNT_AABB_HALF_EXTENTS,
		)
		blocked_y: bool
		next_center, blocked_y = move_axis_with_collision(
			next_center,
			delta.y,
			1,
			TNT_AABB_HALF_EXTENTS,
		)
		if blocked_x {
			armed.vel.x = 0
		}
		if blocked_z {
			armed.vel.z = 0
		}
		if blocked_y {
			armed.vel.y = 0
			if delta.y < 0 {
				// Extra ground damping keeps armed TNT from drifting forever on tiny contact errors.
				friction_alpha := math.clamp(TNT_GROUND_FRICTION * dt, 0, 1)
				armed.vel.x += (0 - armed.vel.x) * friction_alpha
				armed.vel.z += (0 - armed.vel.z) * friction_alpha
			}
		}
		armed.pos = next_center - [3]f32{0.5, 0.5, 0.5}
		armed.phase_elapsed += dt
		for armed.phase_elapsed >= TNT_FLASH_PHASE_SEC &&
		    armed.phase_toggle_cnt < TNT_FLASH_TOGGLE_COUNT {
			armed.phase_elapsed -= TNT_FLASH_PHASE_SEC
			armed.phase_toggle_cnt += 1
		}

		if armed.phase_toggle_cnt >= TNT_FLASH_TOGGLE_COUNT {
			append(&detonations, world_voxel_origin(armed.pos))
			last_idx := len(state.armed_tnt) - 1
			state.armed_tnt[i] = state.armed_tnt[last_idx]
			_ = pop(&state.armed_tnt)
			continue
		}

		if armed.phase_toggle_cnt % 2 == 0 {
			append(&state.tnt_flash_highlight_world_pos, world_voxel_origin(armed.pos))
		}
		i += 1
	}
	tnt_resolve_body_collisions()

	for detonation_pos in detonations {
		tnt_explode(detonation_pos)
	}
}

// TODO: change the name so its more obvious its related to lighting/light propagation?
chunk_remesh_find_dirty_neighbors_based_on_modified_voxel :: proc(
	dirty: ^map[Chunk_Coords]struct{},
	chunk_pos: Chunk_Coords,
	local: Local_Pos,
) {
	dirty[chunk_pos] = {}
	if local.x == 0 {
		neighbor := chunk_pos + Chunk_Coords{-1, 0}
		if neighbor in state.chunks do dirty[neighbor] = {}
	}
	if local.x == CHUNK_WIDTH - 1 {
		neighbor := chunk_pos + Chunk_Coords{1, 0}
		if neighbor in state.chunks do dirty[neighbor] = {}
	}
	if local.z == 0 {
		neighbor := chunk_pos + Chunk_Coords{0, -1}
		if neighbor in state.chunks do dirty[neighbor] = {}
	}
	if local.z == CHUNK_WIDTH - 1 {
		neighbor := chunk_pos + Chunk_Coords{0, 1}
		if neighbor in state.chunks do dirty[neighbor] = {}
	}
}

remesh_queue_enqueue :: proc(chunk_pos: Chunk_Coords) {
	if !(chunk_pos in state.chunks) do return
	if chunk_pos in state.pending_remesh_seen do return
	state.pending_remesh_seen[chunk_pos] = {}
	append(&state.pending_remesh_chunks, chunk_pos)
}

remesh_queue_enqueue_dirty_map :: proc(dirty: map[Chunk_Coords]struct{}) {
	for chunk_pos in dirty {
		remesh_queue_enqueue(chunk_pos)
	}
}

remesh_queue_process_budgeted :: proc(max_chunks: int) {
	if max_chunks <= 0 do return
	processed := 0
	for processed < max_chunks && len(state.pending_remesh_chunks) > 0 {
		best_idx := 0
		best_dist2 := math.inf_f32(1)
		for queued, i in state.pending_remesh_chunks {
			if !(queued in state.chunks) do continue
			chunk_center := [3]f32 {
				(f32(queued[0]) + 0.5) * f32(CHUNK_WIDTH),
				state.player_pos.y,
				(f32(queued[1]) + 0.5) * f32(CHUNK_WIDTH),
			}
			delta := chunk_center - state.player_pos
			dist2 := delta.x * delta.x + delta.z * delta.z
			if dist2 < best_dist2 {
				best_dist2 = dist2
				best_idx = i
			}
		}
		chunk_pos := state.pending_remesh_chunks[best_idx]
		last_idx := len(state.pending_remesh_chunks) - 1
		state.pending_remesh_chunks[best_idx] = state.pending_remesh_chunks[last_idx]
		_ = pop(&state.pending_remesh_chunks)
		delete_key(&state.pending_remesh_seen, chunk_pos)
		if chunk_pos in state.chunks {
			chunk_remesh(chunk_pos, &state.chunks[chunk_pos])
		}
		processed += 1
	}
}

water_fluid_enqueue :: proc(chunk_pos: Chunk_Coords, local: Local_Pos) {
	if !(chunk_pos in state.chunks) {
		return
	}
	key := Water_Fluid_Key{chunk_pos, local}
	if key in state.water_fluid_seen {
		return
	}
	state.water_fluid_seen[key] = {}
	append(&state.water_fluid_queue, key)
}

water_fluid_process_waves :: proc(dt: f32) {
	state.water_fluid_wave_timer += dt
	if state.water_fluid_wave_timer < WATER_FLUID_WAVE_INTERVAL_SEC {
		return
	}
	state.water_fluid_wave_timer -= WATER_FLUID_WAVE_INTERVAL_SEC

	n := len(state.water_fluid_queue)
	wave := make([]Water_Fluid_Key, n, context.temp_allocator)
	for i in 0 ..< n {
		wave[i] = state.water_fluid_queue[i]
		delete_key(&state.water_fluid_seen, wave[i])
	}
	clear(&state.water_fluid_queue)

	processed := n
	if processed > WATER_FLUID_KEYS_PER_WAVE_MAX {
		processed = WATER_FLUID_KEYS_PER_WAVE_MAX
	}
	// Coalesce multiple sources trying to flood the same destination in the same wave.
	spawned_this_wave := make(map[Water_Fluid_Key]struct{}, context.temp_allocator)
	for i in 0 ..< processed {
		key := wave[i]
		if !(key.chunk in state.chunks) do continue
		chunk_ptr := &state.chunks[key.chunk]
		idx := int(cube_index(key.local))
		if chunk_ptr.kinds[idx] != .Water do continue
		local := [3]int{key.local.x, key.local.y, key.local.z}
		for d in WATER_OVERFLOW_OFFSETS {
			ncp, ncl, ok := chunk_neighbor_from_offset(key.chunk, local, d)
			if !ok do continue
			if !(ncp in state.chunks) do continue
			tgt := &state.chunks[ncp]
			nidx := cube_index(ncl)
			if tgt.kinds[nidx] != .None do continue
			target_key := Water_Fluid_Key{ncp, ncl}
			if target_key in spawned_this_wave do continue
			spawned_this_wave[target_key] = {}
			_ = block_change_enqueue(ncp, ncl, .Water)
			water_fluid_enqueue(ncp, ncl)
		}
	}
	for i in processed ..< n {
		water_fluid_enqueue(wave[i].chunk, wave[i].local)
	}
	block_change_apply_batch()
}

water_fluid_seed_chunk :: proc(chunk_pos: Chunk_Coords) {
	if !(chunk_pos in state.chunks) do return
	chunk := &state.chunks[chunk_pos]
	for i in 0 ..< CHUNK_VOXEL_COUNT {
		if chunk.kinds[i] != .Water do continue
		coords := local_from_index(Cube_Index(i))
		local := [3]int{coords.x, coords.y, coords.z}
		for d in WATER_OVERFLOW_OFFSETS {
			ncp, ncl, ok := chunk_neighbor_from_offset(chunk_pos, local, d)
			if !ok do continue
			if !(ncp in state.chunks) do continue
			if state.chunks[ncp].kinds[cube_index(ncl)] == .None {
				water_fluid_enqueue(chunk_pos, coords)
				break
			}
		}
	}
}

water_fluid_schedule_after_voxel_removal :: proc(chunk_pos: Chunk_Coords, local: Local_Pos) {
	li := [3]int{local.x, local.y, local.z}
	for d in NEIGHBOR_OFFSETS_6 {
		ncp, ncl, ok := chunk_neighbor_from_offset(chunk_pos, li, d)
		if !ok do continue
		if !(ncp in state.chunks) do continue
		if state.chunks[ncp].kinds[cube_index(ncl)] == .Water {
			water_fluid_enqueue(ncp, ncl)
		}
	}
}

dig_remesh_dirty_chunks :: proc(chunk_pos: Chunk_Coords, local_coords: Local_Pos) {
	dirty := make(map[Chunk_Coords]struct{}, context.temp_allocator)
	chunk_remesh_find_dirty_neighbors_based_on_modified_voxel(&dirty, chunk_pos, local_coords)
	remesh_queue_enqueue_dirty_map(dirty)
}
