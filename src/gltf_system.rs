// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// Welcome to the glTF example!

use vulkano::buffer::BufferAccess;
use vulkano::buffer::BufferSlice;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuBufferPool;
use vulkano::buffer::ImmutableBuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::DynamicState;
use vulkano::descriptor::descriptor::ShaderStages;
use vulkano::descriptor::descriptor_set::DescriptorSet;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::pipeline_layout::PipelineLayoutAbstract;
use vulkano::descriptor::pipeline_layout::PipelineLayoutDesc;
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::framebuffer::RenderPassAbstract;
use vulkano::framebuffer::Subpass;
use vulkano::image::Dimensions;
use vulkano::image::immutable::ImmutableImage;
use vulkano::pipeline::shader::ShaderInterfaceDef;
use vulkano::pipeline::vertex::AttributeInfo;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::GraphicsPipelineAbstract;
use vulkano::pipeline::input_assembly::PrimitiveTopology;
use vulkano::pipeline::vertex::IncompatibleVertexDefinitionError;
use vulkano::pipeline::vertex::InputRate;
use vulkano::pipeline::vertex::VertexDefinition;
use vulkano::pipeline::vertex::VertexSource;
use vulkano::pipeline::viewport::Viewport;
use vulkano::sampler::Sampler;
use vulkano::sync::GpuFuture;
use vulkano::sync::now;
use cgmath::Matrix4;
use cgmath::perspective;
use cgmath::Rad;
use cgmath::Point3;
use cgmath::Vector3;
use gltf;
use gltf_importer;
use image;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::vec::IntoIter as VecIntoIter;
use std::f32;

/// Represents a fully-loaded glTF model, ready to be drawn.
pub struct GltfModel {
	// The main glTF document.
	gltf: gltf::Gltf,
	// Each mesh of the glTF scene is made of one or more primitives.
	gltf_meshes: Vec<Vec<PrimitiveInfo>>,
	// Buffer used to upload `InstanceParams` when drawing.
	instance_params_upload: CpuBufferPool<vs::ty::InstanceParams>,
	// Pipeline layout common to all the graphics pipeline of all the primitives.
	pipeline_layout: Arc<PipelineLayoutAbstract + Send + Sync>,
}

// Information about a primitive.
struct PrimitiveInfo {
	// The graphics pipeline used to draw the primitive.
	pipeline: Arc<GraphicsPipelineAbstract + Send + Sync>,
	// List of vertex buffers to bind when drawing.
	vertex_buffers: Vec<Arc<BufferAccess + Sync + Send>>,
	// If `Some`, contains the buffer that contains the indices of the primitive.
	index_buffer: Option<BufferSlice<[u16], Arc<ImmutableBuffer<[u8]>>>>,
	// Descriptor set to bind to slot #0 when drawing.
	material: Arc<DescriptorSet + Send + Sync>,
}

fn load_gltf_image_data(
	data: &gltf::image::Data,
	buffers: &gltf_importer::Buffers,
	base: &Path,
) -> (Dimensions, Format, Vec<u8>) {
	use image::DynamicImage::*;
	use image::ImageFormat::{JPEG as Jpeg, PNG as Png};
	let image = match *data {
		gltf::image::Data::View {
			ref view,
			mime_type,
		} => {
			let format = match mime_type {
				"image/png" => Png,
				"image/jpeg" => Jpeg,
				_ => unreachable!(),
			};
			let data = buffers.view(&view).unwrap();
			if data.starts_with(b"data:") {
				// TODO: Data URI decoding for images must be handled by the user
				unimplemented!()
			} else {
				image::load_from_memory_with_format(&data, format)
			}
		}
		gltf::image::Data::Uri { uri, mime_type } => {
			let path: PathBuf = base.join(uri);
			if let Some(ty) = mime_type {
				let format = match ty {
					"image/png" => Png,
					"image/jpeg" => Jpeg,
					_ => unreachable!(),
				};
				let file = File::open(&path).unwrap();
				let reader = BufReader::new(file);
				image::load(reader, format)
			} else {
				image::open(&path)
			}
		}
	}.expect("image decoding failed");

	match image {
		ImageLuma8(buf) => {
			let dimensions = Dimensions::Dim2d {
				width: buf.width(),
				height: buf.height(),
			};
			(dimensions, Format::R8Srgb, buf.into_raw())
		}
		ImageLumaA8(buf) => {
			let dimensions = Dimensions::Dim2d {
				width: buf.width(),
				height: buf.height(),
			};
			(dimensions, Format::R8G8Srgb, buf.into_raw())
		}
		ImageRgb8(_) => {
			// Since RGB is often not supported by Vulkan, convert to RGBA instead.
			let rgba = image.to_rgba();
			let dimensions = Dimensions::Dim2d {
				width: rgba.width(),
				height: rgba.height(),
			};
			(dimensions, Format::R8G8B8A8Srgb, rgba.into_raw())
		}
		ImageRgba8(buf) => {
			let dimensions = Dimensions::Dim2d {
				width: buf.width(),
				height: buf.height(),
			};
			(dimensions, Format::R8G8B8A8Srgb, buf.into_raw())
		}
	}
}

impl GltfModel {
	/// Loads all the resources necessary to draw `gltf`.
	///
	/// The `queue` parameter is the queue that will be used to submit data transfer commands as
	/// part of the loading.
	///
	/// The `subpass` parameter is the render pass subpass that we will need to be in when drawing.
	pub fn new<R>(
		gltf: gltf::Gltf,
		buffers: &gltf_importer::Buffers,
		base: &Path,
		queue: Arc<Queue>,
		subpass: Subpass<R>,
	) -> GltfModel
	where
		R: RenderPassAbstract + Clone + Send + Sync + 'static,
	{
		// This variable will be modified during the function, and will correspond to when the
		// transfer commands are finished.
		let mut final_future = Box::new(now(queue.device().clone())) as Box<GpuFuture>;

		// The first step is to go through all the glTF buffer definitions and load them as
		// `ImmutableBuffer`.
		let gltf_buffers: Vec<Arc<ImmutableBuffer<[u8]>>> = {
			let mut gpu_buffers = Vec::new();
			for buffer in gltf.buffers() {
				let data = buffers.buffer(&buffer).unwrap();
				let (buf, future) = {
					ImmutableBuffer::from_iter(data.iter().cloned(), BufferUsage::all(), queue.clone())
						.expect("Failed to create immutable buffer")
				};

				final_future = Box::new(final_future.join(future));
				gpu_buffers.push(buf);
			}
			gpu_buffers
		};

		// Then we go through each glTF texture and load them.
		let gltf_textures = {
			// TODO: use the sampler defined by the JSON struct
			let sampler = Sampler::simple_repeat_linear(queue.device().clone());

			let mut textures = Vec::new();
			for texture in gltf.textures() {
				let data = texture.source().data();
				let (dimensions, format, raw_pixels) = load_gltf_image_data(&data, buffers, base);
				let (img, future) = {
					ImmutableImage::from_iter(raw_pixels.into_iter(), dimensions, format, queue.clone())
						.expect("Failed to create immutable image")
				};
				final_future = Box::new(final_future.join(future));
				textures.push((img, sampler.clone()));
			}
			textures
		};

		// Usually in vulkano we build a graphics pipeline first, and build descriptor sets that
		// are based on it.
		// However in this situation it is more convenient to build the *pipeline layout object*
		// ahead of time. This object is normally automatically built by vulkano at the same time
		// as the graphics pipeline, but here we create it immediately and will pass it when
		// building the pipelines.
		let pipeline_layout = {
			let vs = vs::Layout(ShaderStages {
				vertex: true,
				..ShaderStages::none()
			});
			let fs = fs::Layout(ShaderStages {
				fragment: true,
				..ShaderStages::none()
			});
			Arc::new(vs.union(fs).build(queue.device().clone()).unwrap())
		};

		// We are going to build a descriptor set for each material defined in the glTF file.
		let gltf_materials: Vec<Arc<DescriptorSet + Send + Sync>> = {
			// TODO: meh, we want some device-local thing here
			let params_buffer = CpuBufferPool::new(queue.device().clone(), BufferUsage::uniform_buffer());

			// Vulkano doesn't allow us to bind *nothing* in a descriptor, so we create a dummy
			// texture and a dummy sampler to use when a texture or a sampler is missing.
			let dummy_sampler = Sampler::simple_repeat_linear(queue.device().clone());
			let (dummy_texture, _) = ImmutableImage::from_iter(
				[0u8].iter().cloned(),
				Dimensions::Dim2d {
					width: 1,
					height: 1,
				},
				Format::R8Unorm,
				queue.clone(),
			).expect("Failed to create immutable image");

			let mut materials = Vec::new();
			for mat in gltf.materials() {
				// Create a buffer that stores some basic parameter values.
				// These fields are the same as the one found in the shader's source code.
				let pbr = mat.pbr_metallic_roughness();
				let material_params = params_buffer.next(fs::ty::MaterialParams {
					base_color_factor: pbr.base_color_factor(),
					base_color_texture_tex_coord: pbr
						.base_color_texture()
						.map(|t| t.tex_coord() as i32)
						.unwrap_or(-1),
					metallic_factor: pbr.metallic_factor(),
					roughness_factor: pbr.roughness_factor(),
					metallic_roughness_texture_tex_coord: pbr
						.metallic_roughness_texture()
						.map(|t| t.tex_coord() as i32)
						.unwrap_or(-1),
					normal_texture_scale: mat.normal_texture().map(|t| t.scale()).unwrap_or(0.0),
					normal_texture_tex_coord: mat
						.normal_texture()
						.map(|t| t.tex_coord() as i32)
						.unwrap_or(-1),
					occlusion_texture_tex_coord: mat
						.occlusion_texture()
						.map(|t| t.tex_coord() as i32)
						.unwrap_or(-1),
					occlusion_texture_strength: mat.occlusion_texture().map(|t| t.strength()).unwrap_or(0.0),
					emissive_texture_tex_coord: mat
						.emissive_texture()
						.map(|t| t.tex_coord() as i32)
						.unwrap_or(-1),
					emissive_factor: mat.emissive_factor(),
					_dummy0: [0; 12],
				});

				// Create the textures and samplers based on the glTF definition.
				let base_color = pbr
					.base_color_texture()
					.map(|t| gltf_textures[t.texture().index()].clone())
					.unwrap_or((dummy_texture.clone(), dummy_sampler.clone()));
				let metallic_roughness = pbr
					.metallic_roughness_texture()
					.map(|t| gltf_textures[t.texture().index()].clone())
					.unwrap_or((dummy_texture.clone(), dummy_sampler.clone()));
				let normal_texture = mat
					.normal_texture()
					.map(|t| gltf_textures[t.texture().index()].clone())
					.unwrap_or((dummy_texture.clone(), dummy_sampler.clone()));
				let occlusion_texture = mat
					.occlusion_texture()
					.map(|t| gltf_textures[t.texture().index()].clone())
					.unwrap_or((dummy_texture.clone(), dummy_sampler.clone()));
				let emissive_texture = mat
					.emissive_texture()
					.map(|t| gltf_textures[t.texture().index()].clone())
					.unwrap_or((dummy_texture.clone(), dummy_sampler.clone()));

				// Building the descriptor set with all the things we built above.
				let descriptor_set = Arc::new(
					PersistentDescriptorSet::start(pipeline_layout.clone(), 1)
						.add_buffer(material_params.unwrap())
						.unwrap()
						.add_sampled_image(base_color.0, base_color.1)
						.unwrap()
						.add_sampled_image(metallic_roughness.0, metallic_roughness.1)
						.unwrap()
						.add_sampled_image(normal_texture.0, normal_texture.1)
						.unwrap()
						.add_sampled_image(occlusion_texture.0, occlusion_texture.1)
						.unwrap()
						.add_sampled_image(emissive_texture.0, emissive_texture.1)
						.unwrap()
						.build()
						.unwrap(),
				);

				materials.push(descriptor_set as Arc<_>);
			}
			materials
		};

		// Each glTF mesh is made of one of more primitives.
		// In this loader, each primitive has its own graphics pipeline.
		let gltf_meshes = {
			let vs = vs::Shader::load(queue.device().clone()).expect("failed to create shader module");
			let fs = fs::Shader::load(queue.device().clone()).expect("failed to create shader module");

			let mut meshes = Vec::new();
			for mesh in gltf.meshes() {
				let mut mesh_prim_out = Vec::with_capacity(mesh.primitives().len());
				for primitive in mesh.primitives() {
					// We build a `RuntimeVertexDef` that analyzes the primitive definition and
					// builds the link between the vertex shader input and the glTF vertex buffers.
					let runtime_def = RuntimeVertexDef::from_primitive(primitive.clone());

					// This `runtime_def` generates the list of vertex buffers that must be bound
					// when drawing, as a list of glTF buffer ids and their offsets.
					// From this information we generate a `vertex_buffer` variable that we will
					// later be able to directly pass to the `draw()` function.
					let vertex_buffers = runtime_def
						.vertex_buffer_ids()
						.iter()
						.map(|&(buf_id, offset)| {
							let buf = gltf_buffers[buf_id].clone();
							let buf_len = buf.len();
							let slice = buf.into_buffer_slice().slice(offset..buf_len).unwrap();
							Arc::new(slice) as Arc<_> // TODO: meh for Arc'ing that
						})
						.collect();

					// Similarly, if the primitive indicates that it uses an index buffer we
					// immediately generate an `index_buffer` variable which we will later be
					// able to pass to `draw_indexed`.
					let index_buffer = if let Some(accessor) = primitive.indices() {
						let view = accessor.view();
						let total_offset = accessor.offset() + view.offset();
						let index_buffer = gltf_buffers[view.buffer().index()].clone();
						let index_buffer_len = index_buffer.len();
						let indices = index_buffer
							.into_buffer_slice()
							.slice(total_offset..index_buffer_len)
							.unwrap();
						// TODO: it is not guaranteed to be u16
						// TODO: add a function in vulkano that does that
						let indices: BufferSlice<[u16], Arc<ImmutableBuffer<[u8]>>> =
							unsafe { ::std::mem::transmute(indices) };
						let indices = indices.clone().slice(0..accessor.count() as usize).unwrap();
						Some(indices)
					} else {
						None
					};

					// Determine the kind of primitives based on the glTF definition.
					let primitive_topology = match primitive.mode() {
						gltf::mesh::Mode::Points => PrimitiveTopology::PointList,
						gltf::mesh::Mode::Lines => PrimitiveTopology::LineList,
						gltf::mesh::Mode::LineLoop => panic!("LineLoop not supported"),
						gltf::mesh::Mode::LineStrip => PrimitiveTopology::LineStrip,
						gltf::mesh::Mode::Triangles => PrimitiveTopology::TriangleList,
						gltf::mesh::Mode::TriangleStrip => PrimitiveTopology::TriangleStrip,
						gltf::mesh::Mode::TriangleFan => PrimitiveTopology::TriangleFan,
					};

					let material_id = primitive
						.material()
						.index()
						.expect("Default material not supported");

					// TODO: adjust some pipeline params based on material
					// TODO: pass pipeline_layout to the builder

					// Now building the graphics pipeline of this primitive.
					let pipeline = Arc::new(
						GraphicsPipeline::start()
							.vertex_input(runtime_def)
							.vertex_shader(vs.main_entry_point(), ())
							.primitive_topology(primitive_topology)
							.viewports_dynamic_scissors_irrelevant(1)
							.fragment_shader(fs.main_entry_point(), ())
							.render_pass(subpass.clone())
							.build(queue.device().clone())
							.unwrap(),
					);

					mesh_prim_out.push(PrimitiveInfo {
						pipeline: pipeline as Arc<_>,
						vertex_buffers: vertex_buffers,
						index_buffer: index_buffer,
						material: gltf_materials[material_id].clone(),
					});
				}
				meshes.push(mesh_prim_out);
			}
			meshes
		};

		// Before returning, we start all the pending transfers and wait until they are finished.
		let _ = final_future
			.then_signal_fence_and_flush()
			.unwrap()
			.wait(None)
			.unwrap();

		GltfModel {
			gltf: gltf,
			gltf_meshes: gltf_meshes,
			instance_params_upload: CpuBufferPool::new(
				queue.device().clone(),
				BufferUsage::uniform_buffer(),
			),
			pipeline_layout: pipeline_layout,
		}
	}

	/// Draws the glTF scene by adding commands to `builder`.
	///
	/// `viewport_dimensions` should be the dimensions of the framebuffer we're drawing to.
	///
	/// The `builder` must be inside a subpass compatible with the one that was passed in `new`.
	pub fn draw_default_scene(
		&self,
		viewport_dimensions: [u32; 2],
		pos: Point3<f32>,
		dir: Vector3<f32>,
		up: Vector3<f32>,
		builder: AutoCommandBufferBuilder,
	) -> AutoCommandBufferBuilder {
		if let Some(scene) = self.gltf.default_scene() {
			self.draw_scene(scene.index(), viewport_dimensions, pos, dir, up, builder)
		} else {
			builder
		}
	}

	/// Draws a single scene.
	///
	/// # Panic
	///
	/// - Panics if the scene is out of range.
	///
	pub fn draw_scene(
		&self,
		scene_id: usize,
		viewport_dimensions: [u32; 2],
		pos: Point3<f32>,
		dir: Vector3<f32>,
		up: Vector3<f32>,
		mut builder: AutoCommandBufferBuilder,
	) -> AutoCommandBufferBuilder {
		let scene = self.gltf.scenes().nth(scene_id).unwrap();
		for node in scene.nodes() {
			let fovy = Rad(45.0f32.to_radians());
			let mut aspect = viewport_dimensions[0] as f32 / viewport_dimensions[1] as f32;
			let near = 0.1;
			let far = 100.0;
			let mut proj = perspective(fovy, aspect, near, far);
			let view = Matrix4::look_at(pos, pos + dir, up);

			builder = self.draw_node(node.index(), proj * view, viewport_dimensions, builder);
		}

		builder
	}

	// Draws a single node.
	//
	// # Panic
	//
	// - Panics if the node is out of range.
	//
	fn draw_node(
		&self,
		node_id: usize,
		world_to_framebuffer: Matrix4<f32>,
		viewport_dimensions: [u32; 2],
		mut builder: AutoCommandBufferBuilder,
	) -> AutoCommandBufferBuilder {
		let node = self.gltf.nodes().nth(node_id).unwrap();
		let local_matrix = world_to_framebuffer * Matrix4::from(node.transform().matrix());

		if let Some(mesh) = node.mesh() {
			builder = self.draw_mesh(mesh.index(), local_matrix, viewport_dimensions, builder);
		}

		for child in node.children() {
			builder = self.draw_node(child.index(), local_matrix, viewport_dimensions, builder);
		}

		builder
	}

	/// Draws a single mesh of the glTF document.
	///
	/// # Panic
	///
	/// - Panics if the mesh is out of range.
	///
	pub fn draw_mesh(
		&self,
		mesh_id: usize,
		world_to_framebuffer: Matrix4<f32>,
		viewport_dimensions: [u32; 2],
		mut builder: AutoCommandBufferBuilder,
	) -> AutoCommandBufferBuilder {
		let instance_params = {
			let buf = self.instance_params_upload.next(vs::ty::InstanceParams {
				world_to_framebuffer: world_to_framebuffer.into(),
			});

			Arc::new(
				PersistentDescriptorSet::start(self.pipeline_layout.clone(), 0)
					.add_buffer(buf.unwrap())
					.unwrap()
					.build()
					.unwrap(),
			)
		};

		for primitive in self.gltf_meshes[mesh_id].iter() {
			let dynamic_state = DynamicState {
				viewports: Some(vec![
					Viewport {
						origin: [0.0, 0.0],
						dimensions: [viewport_dimensions[0] as f32, viewport_dimensions[1] as f32],
						depth_range: 0.0..1.0,
					},
				]),
				..DynamicState::none()
			};

			if let Some(ref indices) = primitive.index_buffer {
				builder = builder
					.draw_indexed(
						primitive.pipeline.clone(),
						dynamic_state,
						primitive.vertex_buffers.clone(),
						indices.clone(),
						(instance_params.clone(), primitive.material.clone()),
						(),
					)
					.unwrap();
			} else {
				builder = builder
					.draw(
						primitive.pipeline.clone(),
						dynamic_state,
						primitive.vertex_buffers.clone(),
						(instance_params.clone(), primitive.material.clone()),
						(),
					)
					.unwrap();
			}
		}

		builder
	}
}

mod vs {
	#[derive(VulkanoShader)]
	#[allow(dead_code)]
	#[ty = "vertex"]
	#[path = "shaders/gltf.vs"]
	struct Dummy;
}

mod fs {
	#[derive(VulkanoShader)]
	#[allow(dead_code)]
	#[ty = "fragment"]
	#[path = "shaders/gltf.fs"]
	struct Dummy;
}

pub struct RuntimeVertexDef {
	buffers: Vec<(u32, usize, InputRate)>,
	vertex_buffer_ids: Vec<(usize, usize)>,
	attributes: Vec<(String, u32, AttributeInfo)>,
	num_vertices: u32,
}

impl RuntimeVertexDef {
	pub fn from_primitive(primitive: gltf::Primitive) -> RuntimeVertexDef {
		use gltf::mesh::Semantic;
		use gltf::accessor::{DataType, Dimensions};

		let mut buffers = Vec::new();
		let mut vertex_buffer_ids = Vec::new();
		let mut attributes = Vec::new();

		let mut num_vertices = u32::max_value();

		for (attribute_id, attribute) in primitive.attributes().enumerate() {
			let (name, accessor) = match attribute.clone() {
				(Semantic::Positions, accessor) => ("i_position".to_owned(), accessor),
				(Semantic::Normals, accessor) => ("i_normal".to_owned(), accessor),
				(Semantic::Tangents, accessor) => ("i_tangent".to_owned(), accessor),
				(Semantic::Colors(0), accessor) => ("i_color_0".to_owned(), accessor),
				(Semantic::TexCoords(0), accessor) => ("i_texcoord_0".to_owned(), accessor),
				(Semantic::TexCoords(1), accessor) => ("i_texcoord_1".to_owned(), accessor),
				(Semantic::Joints(0), accessor) => ("i_joints_0".to_owned(), accessor),
				(Semantic::Weights(0), accessor) => ("i_weights_0".to_owned(), accessor),
				_ => unimplemented!(),
			};

			if (accessor.count() as u32) < num_vertices {
				num_vertices = accessor.count() as u32;
			}

			let infos = AttributeInfo {
				offset: 0,
				format: match (accessor.data_type(), accessor.dimensions()) {
					(DataType::I8, Dimensions::Scalar) => Format::R8Snorm,
					(DataType::U8, Dimensions::Scalar) => Format::R8Unorm,
					(DataType::F32, Dimensions::Vec2) => Format::R32G32Sfloat,
					(DataType::F32, Dimensions::Vec3) => Format::R32G32B32Sfloat,
					(DataType::F32, Dimensions::Vec4) => Format::R32G32B32A32Sfloat,
					_ => unimplemented!(),
				},
			};

			let view = accessor.view();
			buffers.push((
				attribute_id as u32,
				view.stride().unwrap_or(accessor.size()),
				InputRate::Vertex,
			));
			attributes.push((name, attribute_id as u32, infos));
			vertex_buffer_ids.push((view.buffer().index(), view.offset() + accessor.offset()));
		}

		println!("number of vertices: {}", num_vertices);

		RuntimeVertexDef {
			buffers: buffers,
			vertex_buffer_ids: vertex_buffer_ids,
			num_vertices: num_vertices,
			attributes: attributes,
		}
	}

	/// Returns the indices of the buffers to bind as vertex buffers and the byte offset, when
	/// drawing the primitive.
	pub fn vertex_buffer_ids(&self) -> &[(usize, usize)] {
		&self.vertex_buffer_ids
	}
}

unsafe impl<I> VertexDefinition<I> for RuntimeVertexDef
where
	I: ShaderInterfaceDef,
{
	type BuffersIter = VecIntoIter<(u32, usize, InputRate)>;
	type AttribsIter = VecIntoIter<(u32, u32, AttributeInfo)>;

	fn definition(
		&self,
		interface: &I,
	) -> Result<(Self::BuffersIter, Self::AttribsIter), IncompatibleVertexDefinitionError> {
		let buffers_iter = self.buffers.clone().into_iter();

		let mut attribs_iter = self
			.attributes
			.iter()
			.map(|&(ref name, buffer_id, ref infos)| {
				let attrib_loc = interface
					.elements()
					.find(|e| e.name.as_ref().map(|n| &n[..]) == Some(&name[..]))
					.unwrap()
					.location
					.start;
				(
					attrib_loc as u32,
					buffer_id,
					AttributeInfo {
						offset: infos.offset,
						format: infos.format,
					},
				)
			})
			.collect::<Vec<_>>();

		// Add dummy attributes.
		for binding in interface.elements() {
			if attribs_iter.iter().any(|a| a.0 == binding.location.start) {
				continue;
			}

			attribs_iter.push((
				binding.location.start,
				0,
				AttributeInfo {
					offset: 0,
					format: binding.format,
				},
			));
		}

		Ok((buffers_iter, attribs_iter.into_iter()))
	}
}

unsafe impl VertexSource<Vec<Arc<BufferAccess + Send + Sync>>> for RuntimeVertexDef {
	fn decode(
		&self,
		bufs: Vec<Arc<BufferAccess + Send + Sync>>,
	) -> (Vec<Box<BufferAccess + Send + Sync>>, usize, usize) {
		(
			bufs.into_iter().map(|b| Box::new(b) as Box<_>).collect(),
			self.num_vertices as usize,
			1,
		)
	}
}
