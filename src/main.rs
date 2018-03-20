// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// Welcome to the glTF example!
//
// This file contains the `main` function that creates the window and performs the drawing, but
// the interesting part is in the `gltf_system` module.

extern crate cgmath;
extern crate engine;
extern crate gltf;
extern crate gltf_importer;
extern crate image;
#[macro_use]
extern crate vulkano;
extern crate vulkano_win;
extern crate winit;

use engine::gltf_system;
use vulkano_win::VkSurfaceBuild;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::device::Device;
use vulkano::framebuffer::Framebuffer;
use vulkano::framebuffer::Subpass;
use vulkano::instance::Instance;
use vulkano::swapchain;
use vulkano::swapchain::PresentMode;
use vulkano::swapchain::SurfaceTransform;
use vulkano::swapchain::Swapchain;
use vulkano::swapchain::AcquireError;
use vulkano::swapchain::SwapchainCreationError;
use vulkano::sync::now;
use vulkano::sync::GpuFuture;
use cgmath::Vector3;
use cgmath::Point3;
use std::env;
use std::path::Path;
use std::sync::Arc;
use std::mem;
use std::f32;
use std::time::Instant;

fn main() {
	// These initialization steps are common to all examples. See the `triangle` example if you
	// want explanations.
	let instance = {
		let extensions = vulkano_win::required_extensions();
		Instance::new(None, &extensions, None).expect("failed to create Vulkan instance")
	};
	let physical = vulkano::instance::PhysicalDevice::enumerate(&instance)
		.next()
		.expect("no device available");

	let mut events_loop = winit::EventsLoop::new();

	for (num, monitor) in events_loop.get_available_monitors().enumerate() {
		println!("Monitor #{}: {:?}", num, monitor.get_name());
	}

	let surface = winit::WindowBuilder::new()
		// .with_fullscreen(Some(events_loop.get_available_monitors().nth(0).unwrap()))
		.build_vk_surface(&events_loop, instance.clone())
		.unwrap();

	let queue = physical
		.queue_families()
		.find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
		.expect("couldn't find a graphical queue family");

	let (device, mut queues) = {
		let device_ext = vulkano::device::DeviceExtensions {
			khr_swapchain: true,
			..vulkano::device::DeviceExtensions::none()
		};

		Device::new(
			physical,
			physical.supported_features(),
			&device_ext,
			[(queue, 0.5)].iter().cloned(),
		).expect("failed to create device")
	};

	let queue = queues.next().unwrap();

	let (mut swapchain, mut images) = {
		let caps = surface
			.capabilities(physical)
			.expect("failed to get surface capabilities");
		let alpha = caps.supported_composite_alpha.iter().next().unwrap();
		let format = caps.supported_formats[0].0;
		let dims = surface.window().get_inner_size().unwrap();
		Swapchain::new(
			device.clone(),
			surface.clone(),
			caps.min_image_count,
			format,
			[dims.0, dims.1],
			1,
			caps.supported_usage_flags,
			&queue,
			SurfaceTransform::Identity,
			alpha,
			PresentMode::Fifo,
			true,
			None,
		).expect("failed to create swapchain")
	};

	let render_pass = Arc::new(
		single_pass_renderpass!(device.clone(),
				attachments: {
						color: {
								load: Clear,
								store: Store,
								format: swapchain.format(),
								samples: 1,
						}
				},
				pass: {
						color: [color],
						depth_stencil: {}
				}
		).unwrap(),
	);

	// This is where the glTF-specific code starts.

	// Try loading our glTF model.
	let args: Vec<_> = env::args().collect();
	let path = args
		.get(1)
		.map(|s| s.as_str())
		.unwrap_or("assets/Duck.gltf");
	let base = Path::new(path).parent().unwrap_or(Path::new("assets"));
	let (gltf, buffers) = gltf_importer::import(path).expect("Error while loading glTF file");

	// Upload everything to get ready for drawing.
	let model = gltf_system::GltfModel::new(
		gltf,
		&buffers,
		base,
		queue.clone(),
		Subpass::from(render_pass.clone(), 0).unwrap(),
	);

	let mut recreate_swapchain = false;
	let mut previous_frame_end = Box::new(now(device.clone())) as Box<GpuFuture>;

	let move_speed = 0.008;
	let mouse_speed = 0.0015;

	let mut pos = Point3::new(0.0, 0.0, 5.0);

	let mut horizontal_angle = f32::consts::PI;
	let mut vertical_angle: f32 = 0.0;

	let mut dir = Vector3::new(
		vertical_angle.cos() * horizontal_angle.sin(),
		vertical_angle.sin(),
		vertical_angle.cos() * horizontal_angle.cos(),
	);

	let mut right = Vector3::new(
		(horizontal_angle - f32::consts::FRAC_PI_2).sin(),
		0.0,
		(horizontal_angle - f32::consts::FRAC_PI_2).cos(),
	);

	let mut up = right.cross(dir);
	up.y = -up.y;

	let mut last_x: f32 = 0.0;
	let mut last_y: f32 = 0.0;

	let mut last_time_delta = Instant::now();

	let (mut w, mut a, mut s, mut d) = (false, false, false, false);

	loop {
		let time_duration = last_time_delta.elapsed();
		let time_delta = (time_duration.as_secs() * 1_000) + ((time_duration.subsec_nanos() / 1_000_000) as u64);
		last_time_delta = Instant::now();
		// println!("time_delta: {}", time_delta as f32);

		previous_frame_end.cleanup_finished();

		let dimensions = {
			let (new_width, new_height) = surface.window().get_inner_size().unwrap();
			[new_width, new_height]
		};

		if recreate_swapchain {
			let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
				Ok(r) => r,
				Err(SwapchainCreationError::UnsupportedDimensions) => {
					continue;
				}
				Err(err) => panic!("{:?}", err),
			};

			mem::replace(&mut swapchain, new_swapchain);
			mem::replace(&mut images, new_images);
			recreate_swapchain = false;
		}

		let (image_num, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(), None) {
			Ok(r) => r,
			Err(AcquireError::OutOfDate) => {
				recreate_swapchain = true;
				continue;
			}
			Err(err) => panic!("{:?}", err),
		};

		let framebuffer = Arc::new(
			Framebuffer::start(render_pass.clone())
				.add(images[image_num].clone())
				.unwrap()
				.build()
				.unwrap(),
		);

		let mut builder =
			AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family())
				.unwrap()
				.begin_render_pass(
					framebuffer.clone(),
					false,
					vec![[0.0, 0.0, 1.0, 1.0].into()],
				)
				.unwrap();

		builder = model.draw_default_scene(dimensions, pos, dir, up, builder);

		let command_buffer = builder.end_render_pass().unwrap().build().unwrap();

		let future = previous_frame_end
			.join(acquire_future)
			.then_execute(queue.clone(), command_buffer)
			.unwrap()
			.then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
			.then_signal_fence_and_flush()
			.unwrap();
		previous_frame_end = Box::new(future) as Box<_>;

		let mut done = false;
		events_loop.poll_events(|ev| {
			match ev {
				winit::Event::WindowEvent {
					event: winit::WindowEvent::Closed,
					..
				} => done = true,
				winit::Event::WindowEvent {
					event: winit::WindowEvent::KeyboardInput { input, .. },
					..
				} => {
					// println!("key input: {:?}", input);
					match input {
						winit::KeyboardInput {
							state,
							virtual_keycode,
							..
						} => match virtual_keycode {
							Some(winit::VirtualKeyCode::Escape) => done = true,

							Some(winit::VirtualKeyCode::W) => {
								if state == winit::ElementState::Pressed {
									w = true;
								} else if state == winit::ElementState::Released {
									w = false;
								}
							}
							Some(winit::VirtualKeyCode::S) => {
								if state == winit::ElementState::Pressed {
									s = true;
								} else if state == winit::ElementState::Released {
									s = false;
								}
							}
							Some(winit::VirtualKeyCode::A) => {
								if state == winit::ElementState::Pressed {
									a = true;
								} else if state == winit::ElementState::Released {
									a = false;
								}
							}
							Some(winit::VirtualKeyCode::D) => {
								if state == winit::ElementState::Pressed {
									d = true;
								} else if state == winit::ElementState::Released {
									d = false;
								}
							}
							_ => println!("couldn't detect which key was pressed/released"),
						},
					}
				}
				winit::Event::WindowEvent {
					event: winit::WindowEvent::CursorMoved { position, .. },
					..
				} => {
					let diff_x = position.0 as f32 - last_x;
					let diff_y = position.1 as f32 - last_y;

					// // println!("horizontal_angle {}", horizontal_angle);
					if diff_x > 0.0 {
						horizontal_angle += time_delta as f32 * mouse_speed;
					} else if diff_x < 0.0 {
						horizontal_angle -= time_delta as f32 * mouse_speed;
					}
					if diff_x != 0.0 {
						if horizontal_angle.to_degrees() > 360.0 {
							horizontal_angle -= 360.0f32.to_radians();
						} else if horizontal_angle.to_degrees() < 0.0 {
							horizontal_angle += 360.0f32.to_radians();
						}
					}

					// println!("vertical_angle {}", vertical_angle);
					if diff_y > 0.0 {
						vertical_angle -= time_delta as f32 * mouse_speed;
					} else if diff_y < 0.0 {
						vertical_angle += time_delta as f32 * mouse_speed;
					}
					if diff_y != 0.0 {
						vertical_angle = vertical_angle.max(-90.0f32.to_radians()).min(90.0f32.to_radians());
					}

					if diff_x != 0.0 || diff_y != 0.0 {
						dir.x = vertical_angle.cos() * horizontal_angle.sin();
						dir.y = vertical_angle.sin();
						dir.z = vertical_angle.cos() * horizontal_angle.cos();
						right.x = (horizontal_angle - f32::consts::FRAC_PI_2).sin();
						right.y = 0.0;
						right.z = (horizontal_angle - f32::consts::FRAC_PI_2).cos();
						up = right.cross(dir);
						up.y = -up.y;
					}

					last_x = position.0 as f32;
					last_y = position.1 as f32;
				}
				_ => (),
			}
		});

		if w {
			pos += dir * time_delta as f32 * move_speed;
			pos.y = 0.0;
		}
		if s {
			pos -= dir * time_delta as f32 * move_speed;
			pos.y = 0.0;
		}
		if a {
			pos += right * time_delta as f32 * move_speed;
		}
		if d {
			pos -= right * time_delta as f32 * move_speed;
		}

		if done {
			return;
		}
	}
}
