#include <cstdint>
#include <climits>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#define _USE_MATH_DEFINES
#include <cmath>

#include <veekay/veekay.hpp>
#include <veekay/input.hpp> 

#include <vulkan/vulkan_core.h>
#include <imgui.h>
#include <lodepng.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

namespace {

	constexpr uint32_t max_models = 1024;

	struct Vertex {
		veekay::vec3 position;
		veekay::vec3 normal;
		veekay::vec2 uv;
		// NOTE: You can add more attributes
	};

	struct PointLight {
		veekay::vec3 position;
		float intensity = 1.0f;
		veekay::vec3 color = { 1.0f, 1.0f, 1.0f };
		float _pad0;
	};

	struct SpotLightStruct {
		veekay::vec3 position;
		float _pad0;
		veekay::vec3 direction;
		float _pad1;
		veekay::vec3 color;
		float _pad2;
		float inner_cutOff_cos;
		float outer_cutOff_cos;
		float _pad3; // padding
		float _pad4; // padding
	};

	struct LightData {
		veekay::vec3 camera_position;
		float _p0; // padding
		veekay::vec3 ambient_color;
		float _p1; // padding
		veekay::vec3 directional_dir;
		float _p2; // padding
		veekay::vec3 directional_color;
		float _p3; // padding

		SpotLightStruct spot_lights[10];
		uint32_t spot_light_count;
		float _pad1;
		float _pad2;
		float _pad3;

		PointLight point_lights[10];
		uint32_t point_light_count;
		float _pad4; // padding
		float _pad5; // padding
		float _pad6; // padding
	};

	struct SceneUniforms {
		veekay::mat4 view_projection;
		veekay::vec3 view_position;
		float _pad0;
		veekay::vec3 ambient_light_intensity = { 0.1f, 0.1f, 0.1f };
		float _pad1;
		veekay::vec3 sun_light_direction = { 0.0f, -1.0f, 0.0f };
		float _pad2;
		veekay::vec3 sun_light_color = { 1.0f, 1.0f, 1.0f };
		float _pad3;
		uint32_t point_light_count = 0;
		float time = 0.0f;
		float _pad4;
		float _pad5;
		float _pad6;
	};

	struct ModelUniforms {
		veekay::mat4 model;
		// veekay::vec3 albedo_color;
		// float _pad0;
		// veekay::vec3 specular_color;
		// float shininess;
	};

	struct Mesh {
		veekay::graphics::Buffer* vertex_buffer;
		veekay::graphics::Buffer* index_buffer;
		uint32_t indices;
	};

	struct Transform {
		veekay::vec3 position = {};
		veekay::vec3 scale = { 1.0f, 1.0f, 1.0f };
		veekay::vec3 rotation = {};

		// NOTE: Model matrix (translation, rotation and scaling)
		veekay::mat4 matrix() const;
	};

	struct Material {
		veekay::graphics::Texture* texture_diffuse;   
		veekay::graphics::Texture* texture_specular;  
		veekay::graphics::Texture* texture_emissive;  
		VkSampler sampler; 
		VkDescriptorSet descriptor_set; 
	};

	struct Model {
		Mesh mesh;
		Transform transform;
		Material* material; 
	};

	struct Camera {
		constexpr static float default_fov = 60.0f;
		constexpr static float default_near_plane = 0.01f;
		constexpr static float default_far_plane = 100.0f;

		veekay::vec3 position = {};
		veekay::vec3 rotation = {};

		float fov = default_fov;
		float near_plane = default_near_plane;
		float far_plane = default_far_plane;

		// NOTE: View matrix of camera (inverse of a transform)
		veekay::mat4 view() const;

		// NOTE: View and projection composition
		veekay::mat4 view_projection(float aspect_ratio) const;
	};

	struct AmbientLight {
		veekay::vec3 color = { 0.1f, 0.1f, 0.1f };
	};

	struct DirectionalLight {
		veekay::vec3 direction = { 0.0f, -1.0f, 0.0f };
		veekay::vec3 color = { 1.0f, 1.0f, 1.0f };
	};

	struct SpotLight {
		veekay::vec3 position = { 0.0f, 2.0f, 0.0f };
		veekay::vec3 direction = { 0.0f, 1.0f, 0.0f };
		veekay::vec3 color = { 1.0f, 1.0f, 1.0f };
		float inner_cutOff = 12.5f;
		float outer_cutOff = 17.5f;
	};

	// NOTE: Scene objects
	inline namespace {
		Camera camera{
			.position = {0.0f, -0.5f, -3.0f}
		};

		std::vector<Model> models;

		AmbientLight ambient_light;
		DirectionalLight directional_light;

		std::vector<SpotLight> spot_lights;

		std::vector<PointLight> point_lights;
	}

	// NOTE: Vulkan objects
	inline namespace {
		VkShaderModule vertex_shader_module;
		VkShaderModule fragment_shader_module;

		VkDescriptorPool descriptor_pool;
		VkDescriptorSetLayout descriptor_set_layout; 
		VkDescriptorSetLayout material_set_layout;
		VkDescriptorSet descriptor_set;
		VkPipelineLayout pipeline_layout;
		VkPipeline pipeline;

		veekay::graphics::Buffer* scene_uniforms_buffer;
		veekay::graphics::Buffer* model_uniforms_buffer;
		veekay::graphics::Buffer* light_data_buffer;

		Mesh plane_mesh;
		Mesh cube_mesh;

		Material* missing_material; 
		Material* first_material; 
		Material* second_material; 
		Material* third_material;
		Material* floor_material; 

		uint32_t min_uniform_buffer_offset_alignment;
		uint32_t aligned_model_uniforms_size;
	}

	float toRadians(float degrees) {
		return degrees * float(M_PI) / 180.0f;
	}

	veekay::mat4 Transform::matrix() const {
		// TODO: Scaling and rotation
		auto t = veekay::mat4::translation(position);
		auto s = veekay::mat4::scaling(scale);
		auto rx = veekay::mat4::rotation({ 1,0,0 }, toRadians(rotation.x));
		auto ry = veekay::mat4::rotation({ 0,1,0 }, toRadians(rotation.y));
		auto rz = veekay::mat4::rotation({ 0,0,1 }, toRadians(rotation.z));
		return t * rz * ry * rx * s; 
	}

	veekay::mat4 Camera::view() const {
		// TODO: Rotation
		auto t = veekay::mat4::translation(-position);
		auto rx = veekay::mat4::rotation({ 1,0,0 }, toRadians(-rotation.x));
		auto ry = veekay::mat4::rotation({ 0,1,0 }, toRadians(-rotation.y));
		auto rz = veekay::mat4::rotation({ 0,0,1 }, toRadians(-rotation.z)); 
		return rz * ry * rx * t; 
	}

	veekay::mat4 Camera::view_projection(float aspect_ratio) const {
		auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);
		return view() * projection;
	}

	// NOTE: Loads shader byte code from file
	// NOTE: Your shaders are compiled via CMake with this code too, look it up
	VkShaderModule loadShaderModule(const char* path) {
		std::ifstream file(path, std::ios::binary | std::ios::ate);
		size_t size = file.tellg();
		std::vector<uint32_t> buffer(size / sizeof(uint32_t));
		file.seekg(0);
		file.read(reinterpret_cast<char*>(buffer.data()), size);
		file.close();

		VkShaderModuleCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = size,
			.pCode = buffer.data(),
		};

		VkShaderModule result;
		if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result) != VK_SUCCESS) {
			return nullptr;
		}

		return result;
	}

	auto createMaterial = [&](veekay::graphics::Texture* tex_diffuse, veekay::graphics::Texture* tex_specular, veekay::graphics::Texture* tex_emissive, VkFilter filter,
		VkSamplerAddressMode address_mode) -> Material* {
			Material* mat = new Material();
			mat->texture_diffuse = tex_diffuse;
			mat->texture_specular = tex_specular;
			mat->texture_emissive = tex_emissive;
			mat->sampler = VK_NULL_HANDLE;
			mat->descriptor_set = VK_NULL_HANDLE;

			VkDevice& device = veekay::app.vk_device;

			VkSamplerCreateInfo sampler_info{
				.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
				.pNext = nullptr,
				.flags = 0,
				.magFilter = filter, // Фильтр при увеличении
				.minFilter = filter, // Фильтр при уменьшении
				.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR, // Линейная интерполяция между mip-уровнями
				.addressModeU = address_mode, // Режим адресации по U
				.addressModeV = address_mode, // Режим адресации по V
				.addressModeW = address_mode,
				.mipLodBias = 0.0f,
				.anisotropyEnable = VK_TRUE, // Анизотропная фильтрация
				.maxAnisotropy = 16.0f, // Макс. уровень анизотропии
				.compareEnable = VK_FALSE,
				.compareOp = VK_COMPARE_OP_ALWAYS,
				.minLod = 0.0f,
				.maxLod = VK_LOD_CLAMP_NONE, // Использовать все mip-уровни
				.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
				.unnormalizedCoordinates = VK_FALSE,
			};

			if (vkCreateSampler(device, &sampler_info, nullptr, &mat->sampler) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan texture sampler\n";
				veekay::app.running = false;
				delete mat; 
				return nullptr;
			}

			// Выделение descriptor set для этого материала из пула
			VkDescriptorSetAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &material_set_layout,
			};
			if (vkAllocateDescriptorSets(device, &alloc_info, &mat->descriptor_set) != VK_SUCCESS) {
				std::cerr << "Failed to allocate material descriptor set\n";
				vkDestroySampler(device, mat->sampler, nullptr);
				delete mat;
				return nullptr;
			}

			// Запись в descriptor set: binding 0, 1, 2 связываются с текстурами и сэмплером
			VkDescriptorImageInfo image_info_diffuse{
				.sampler = mat->sampler,
				.imageView = mat->texture_diffuse->view,
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			};
			VkDescriptorImageInfo image_info_specular{
				.sampler = mat->sampler,
				.imageView = mat->texture_specular->view,
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			};
			VkDescriptorImageInfo image_info_emissive{
				.sampler = mat->sampler,
				.imageView = mat->texture_emissive->view,
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			};
			VkWriteDescriptorSet write_infos[3] = {
				VkWriteDescriptorSet{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = mat->descriptor_set,
					.dstBinding = 0, // layout(set=1, binding=0) в шейдере (diffuse)
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.pImageInfo = &image_info_diffuse,
				},
				VkWriteDescriptorSet{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = mat->descriptor_set,
					.dstBinding = 1, // layout(set=1, binding=1) в шейдере (specular)
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.pImageInfo = &image_info_specular,
				},
				VkWriteDescriptorSet{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = mat->descriptor_set,
					.dstBinding = 2, // layout(set=1, binding=2) в шейдере (emissive)
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.pImageInfo = &image_info_emissive,
				}
			};
			vkUpdateDescriptorSets(device, 3, write_infos, 0, nullptr);

			return mat;
		};


	void initialize(VkCommandBuffer cmd) {
		VkDevice& device = veekay::app.vk_device;
		VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

		{ // NOTE: Build graphics pipeline
			vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
			if (!vertex_shader_module) {
				std::cerr << "Failed to load Vulkan vertex shader from file\n";
				veekay::app.running = false;
				return;
			}

			fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
			if (!fragment_shader_module) {
				std::cerr << "Failed to load Vulkan fragment shader from file\n";
				veekay::app.running = false;
				return;
			}

			VkPipelineShaderStageCreateInfo stage_infos[2];

			// NOTE: Vertex shader stage
			stage_infos[0] = VkPipelineShaderStageCreateInfo{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_VERTEX_BIT,
				.module = vertex_shader_module,
				.pName = "main",
			};

			// NOTE: Fragment shader stage
			stage_infos[1] = VkPipelineShaderStageCreateInfo{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
				.module = fragment_shader_module,
				.pName = "main",
			};

			// NOTE: How many bytes does a vertex take?
			VkVertexInputBindingDescription buffer_binding{
				.binding = 0,
				.stride = sizeof(Vertex),
				.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
			};

			// NOTE: Declare vertex attributes
			VkVertexInputAttributeDescription attributes[] = {
				{
					.location = 0, // NOTE: First attribute
					.binding = 0, // NOTE: First vertex buffer
					.format = VK_FORMAT_R32G32B32_SFLOAT, // NOTE: 3-component vector of floats
					.offset = offsetof(Vertex, position), // NOTE: Offset of "position" field in a Vertex struct
				},
				{
					.location = 1,
					.binding = 0,
					.format = VK_FORMAT_R32G32B32_SFLOAT,
					.offset = offsetof(Vertex, normal),
				},
				{
					.location = 2,
					.binding = 0,
					.format = VK_FORMAT_R32G32_SFLOAT,
					.offset = offsetof(Vertex, uv),
				},
			};

			// NOTE: Describe inputs
			VkPipelineVertexInputStateCreateInfo input_state_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
				.vertexBindingDescriptionCount = 1,
				.pVertexBindingDescriptions = &buffer_binding,
				.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
				.pVertexAttributeDescriptions = attributes,
			};

			// NOTE: Every three vertices make up a triangle,
				//       so our vertex buffer contains a "list of triangles"
			VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
				.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
			};

			// NOTE: Declare clockwise triangle order as front-facing
				//       Discard triangles that are facing away
				//       Fill triangles, don't draw lines instaed
			VkPipelineRasterizationStateCreateInfo raster_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
				.polygonMode = VK_POLYGON_MODE_FILL,
				.cullMode = VK_CULL_MODE_BACK_BIT,
				.frontFace = VK_FRONT_FACE_CLOCKWISE,
				.lineWidth = 1.0f,
			};

			// NOTE: Use 1 sample per pixel
			VkPipelineMultisampleStateCreateInfo sample_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
				.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
				.sampleShadingEnable = false,
				.minSampleShading = 1.0f,
			};

			VkViewport viewport{
				.x = 0.0f,
				.y = 0.0f,
				.width = static_cast<float>(veekay::app.window_width),
				.height = static_cast<float>(veekay::app.window_height),
				.minDepth = 0.0f,
				.maxDepth = 1.0f,
			};

			VkRect2D scissor{
					.offset = {0, 0},
					.extent = {veekay::app.window_width, veekay::app.window_height},
			};

			// NOTE: Let rasterizer draw on the entire window
			VkPipelineViewportStateCreateInfo viewport_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,

				.viewportCount = 1,
				.pViewports = &viewport,

				.scissorCount = 1,
				.pScissors = &scissor,
			};

			// NOTE: Let rasterizer perform depth-testing and overwrite depth values on condition pass
			VkPipelineDepthStencilStateCreateInfo depth_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
				.depthTestEnable = true,
				.depthWriteEnable = true,
				.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
			};

			// NOTE: Let fragment shader write all the color channels
			VkPipelineColorBlendAttachmentState attachment_info{
				.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
								  VK_COLOR_COMPONENT_G_BIT |
								  VK_COLOR_COMPONENT_B_BIT |
								  VK_COLOR_COMPONENT_A_BIT,
			};

			// NOTE: Let rasterizer just copy resulting pixels onto a buffer, don't blend
			VkPipelineColorBlendStateCreateInfo blend_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,

				.logicOpEnable = false,
				.logicOp = VK_LOGIC_OP_COPY,

				.attachmentCount = 1,
				.pAttachments = &attachment_info
			};

			{
				VkDescriptorPoolSize pools[] = {
					{
						.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
						.descriptorCount = 8,
					},
					{
						.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
						.descriptorCount = 8,
					},
					{
						.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
						.descriptorCount = 48,
					},
					{
						.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
						.descriptorCount = 8, 
					}
				};

				VkDescriptorPoolCreateInfo info{
					.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
					.maxSets = 17,
					.poolSizeCount = sizeof(pools) / sizeof(pools[0]),
					.pPoolSizes = pools,
				};

				if (vkCreateDescriptorPool(device, &info, nullptr, &descriptor_pool) != VK_SUCCESS) {
					std::cerr << "Failed to create Vulkan descriptor pool\n";
					veekay::app.running = false;
					return;
				}
			}

			{
				VkDescriptorSetLayoutBinding bindings[] = {
					{
						.binding = 0,
						.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
					},
					{
						.binding = 1,
						.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
					},
					{
						.binding = 2,
						.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
					}
				};

				VkDescriptorSetLayoutCreateInfo info{
				   .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				   .bindingCount = sizeof(bindings) / sizeof(bindings[0]),
				   .pBindings = bindings,
				};

				if (vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
					std::cerr << "Failed to create Vulkan descriptor set layout\n";
					veekay::app.running = false;
					return;
				}
			}

			{
				VkDescriptorSetLayoutBinding bindings[3] = {
					VkDescriptorSetLayoutBinding{
						.binding = 0, // diffuse
						.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, 
					},
					VkDescriptorSetLayoutBinding{
						.binding = 1, // specular
						.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
					},
					VkDescriptorSetLayoutBinding{
						.binding = 2, // emissive
						.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
					}
				};

				VkDescriptorSetLayoutCreateInfo info{
				   .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				   .bindingCount = 3,
				   .pBindings = bindings,
				};

				if (vkCreateDescriptorSetLayout(device, &info, nullptr, &material_set_layout) != VK_SUCCESS) {
					std::cerr << "Failed to create Vulkan material descriptor set layout\n";
					veekay::app.running = false;
					return;
				}
			}

			{
				VkDescriptorSetAllocateInfo info{
					.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
					.descriptorPool = descriptor_pool,
					.descriptorSetCount = 1,
					.pSetLayouts = &descriptor_set_layout,
				};

				if (vkAllocateDescriptorSets(device, &info, &descriptor_set) != VK_SUCCESS) {
					std::cerr << "Failed to create Vulkan descriptor set\n";
					veekay::app.running = false;
					return;
				}
			}

			VkDescriptorSetLayout set_layouts[] = {
				descriptor_set_layout,
				material_set_layout,
			};

			VkPipelineLayoutCreateInfo layout_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
				.setLayoutCount = 2, // Два набора дескрипторов
				.pSetLayouts = set_layouts,
			};

			// NOTE: Create pipeline layout
			if (vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan pipeline layout\n";
				veekay::app.running = false;
				return;
			}

			VkGraphicsPipelineCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
				.stageCount = 2,
				.pStages = stage_infos,
				.pVertexInputState = &input_state_info,
				.pInputAssemblyState = &assembly_state_info,
				.pViewportState = &viewport_info,
				.pRasterizationState = &raster_info,
				.pMultisampleState = &sample_info,
				.pDepthStencilState = &depth_info,
				.pColorBlendState = &blend_info,
				.layout = pipeline_layout,
				.renderPass = veekay::app.vk_render_pass,
			};

			// NOTE: Create graphics pipeline
			if (vkCreateGraphicsPipelines(device, nullptr, 1, &info, nullptr, &pipeline) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan pipeline\n";
				veekay::app.running = false;
				return;
			}
		}

		VkPhysicalDeviceProperties physical_device_properties;
		vkGetPhysicalDeviceProperties(physical_device, &physical_device_properties);
		min_uniform_buffer_offset_alignment = static_cast<uint32_t>(
			physical_device_properties.limits.minUniformBufferOffsetAlignment);

		uint32_t model_uniforms_size = sizeof(ModelUniforms);
		aligned_model_uniforms_size = (model_uniforms_size + min_uniform_buffer_offset_alignment - 1)
			& ~(min_uniform_buffer_offset_alignment - 1);

		scene_uniforms_buffer = new veekay::graphics::Buffer(
			sizeof(SceneUniforms),
			nullptr,
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

		model_uniforms_buffer = new veekay::graphics::Buffer(
			max_models * aligned_model_uniforms_size,
			nullptr,
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

		light_data_buffer = new veekay::graphics::Buffer(
			sizeof(LightData),
			nullptr,
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);


		{
			veekay::vec4 white = { 0.0f, 0.0f, 0.0f, 0.0f };
			auto* tex = new veekay::graphics::Texture(
				cmd, 1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, &white);

			missing_material = createMaterial(tex, tex, tex, VK_FILTER_NEAREST, 
				VK_SAMPLER_ADDRESS_MODE_REPEAT);
		}

		{
			std::vector<unsigned char> image_data_d, image_data_s, image_data_e;
			unsigned width_d, height_d, width_s, height_s, width_e, height_e;
			unsigned error_d = lodepng::decode(image_data_d, width_d, height_d, "./assets/wood.png");
			unsigned error_s = lodepng::decode(image_data_s, width_s, height_s, "./assets/star.png"); 
			unsigned error_e = lodepng::decode(image_data_e, width_e, height_e, "./assets/emissive.png"); 

			if (error_d) {
				std::cerr << "Failed to load first diffuse texture. Using fallback.\n";
				first_material = missing_material;
			}
			else { 
				veekay::graphics::Texture* tex_d = new veekay::graphics::Texture(cmd, width_d, height_d, VK_FORMAT_B8G8R8A8_UNORM, image_data_d.data());

				veekay::graphics::Texture* tex_s;
				if (error_s) { 
					std::cerr << "Failed to load first specular texture. Using fallback (checkerboard).\n";
					tex_s = missing_material->texture_specular; 
				}
				else { 
					tex_s = new veekay::graphics::Texture(cmd, width_s, height_s, VK_FORMAT_B8G8R8A8_UNORM, image_data_s.data()); 
				}

				veekay::graphics::Texture* tex_e;
				if (error_e) { 
					std::cerr << "Failed to load first emissive texture. Using fallback (checkerboard).\n";
					tex_e = missing_material->texture_emissive; 
				}
				else { 
					tex_e = new veekay::graphics::Texture(cmd, width_e, height_e, VK_FORMAT_B8G8R8A8_UNORM, image_data_e.data()); 
				}

				first_material = createMaterial(tex_d, tex_s, tex_e, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT);
			}
		}

		{
			std::vector<unsigned char> image_data_d, image_data_s, image_data_e;
			unsigned width_d, height_d, width_s, height_s, width_e, height_e;

			unsigned error_d = lodepng::decode(image_data_d, width_d, height_d, "./assets/stone.png");
			if (error_d) {
				std::cerr << "Failed to load floor texture(s). Using fallback.\n";
				floor_material = missing_material; 
			}
			else {
				veekay::graphics::Texture* tex_d = new veekay::graphics::Texture(cmd, width_d, height_d, VK_FORMAT_B8G8R8A8_UNORM, image_data_d.data()); 
				veekay::graphics::Texture* tex_s = missing_material->texture_specular; 
				veekay::graphics::Texture* tex_e = missing_material->texture_emissive; 

				floor_material = createMaterial(tex_d, tex_s, tex_e, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT);
			}
		}

		{
			std::vector<unsigned char> image_data_d, image_data_s, image_data_e;
			unsigned width_d, height_d, width_s, height_s, width_e, height_e;

			unsigned error_d = lodepng::decode(image_data_d, width_d, height_d, "./assets/bow.png");
			unsigned error_s = lodepng::decode(image_data_s, width_s, height_s, "./assets/star.png");
			if (error_d) {
				std::cerr << "Failed to load second texture(s). Using fallback.\n";
				second_material = missing_material; 
			}
			else {
				veekay::graphics::Texture* tex_d = new veekay::graphics::Texture(cmd, width_d, height_d, VK_FORMAT_B8G8R8A8_UNORM, image_data_d.data());
				veekay::graphics::Texture* tex_s;
				if (error_s) {
					std::cerr << "Failed to load first specular texture. Using fallback (checkerboard).\n";
					tex_s = missing_material->texture_specular;
				}
				else {
					tex_s = new veekay::graphics::Texture(cmd, width_s, height_s, VK_FORMAT_B8G8R8A8_UNORM, image_data_s.data());
				}
				veekay::graphics::Texture* tex_e = missing_material->texture_emissive; 

				second_material = createMaterial(tex_d, tex_s, tex_e, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
			}
		}

		{
			std::vector<unsigned char> image_data_d, image_data_s, image_data_e;
			unsigned width_d, height_d, width_s, height_s, width_e, height_e;
			unsigned error_d = lodepng::decode(image_data_d, width_d, height_d, "./assets/lenna.png"); 
			unsigned error_s = lodepng::decode(image_data_s, width_s, height_s, "./assets/star.png");

			if (error_d) {
				std::cerr << "Failed to load third texture(s). Using fallback.\n";
				third_material = missing_material;
			}
			else {
				veekay::graphics::Texture* tex_d = new veekay::graphics::Texture(cmd, width_d, height_d, VK_FORMAT_B8G8R8A8_UNORM, image_data_d.data());
				veekay::graphics::Texture* tex_s;
				if (error_s) {
					std::cerr << "Failed to load first specular texture. Using fallback (checkerboard).\n";
					tex_s = missing_material->texture_specular;
				}
				else {
					tex_s = new veekay::graphics::Texture(cmd, width_s, height_s, VK_FORMAT_B8G8R8A8_UNORM, image_data_s.data());
				}
				veekay::graphics::Texture* tex_e = missing_material->texture_emissive;

				third_material = createMaterial(tex_d, tex_s, tex_e, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
			}
		}


		{
			VkDescriptorBufferInfo buffer_infos[] = {
				{
					.buffer = scene_uniforms_buffer->buffer,
					.offset = 0,
					.range = sizeof(SceneUniforms),
				},
				{
					.buffer = model_uniforms_buffer->buffer,
					.offset = 0,
					.range = aligned_model_uniforms_size, 
				},
				{
					.buffer = light_data_buffer->buffer,
					.offset = 0,
					.range = sizeof(LightData),
				}
			};

			VkWriteDescriptorSet write_infos[] = {
				{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = descriptor_set,
					.dstBinding = 0,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.pBufferInfo = &buffer_infos[0],
				},
				{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = descriptor_set,
					.dstBinding = 1,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.pBufferInfo = &buffer_infos[1],
				},
				{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = descriptor_set,
					.dstBinding = 2,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.pBufferInfo = &buffer_infos[2],
				}
			};

			vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
				write_infos, 0, nullptr);
		}

		// NOTE: Plane mesh initialization
		{
			// (v0)------(v1)
			//  |  \       |
			//  |   `--,   |
			//  |       \  |
			// (v3)------(v2)
			std::vector<Vertex> vertices = {
				{{-5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
				{{5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {10.0f, 0.0f}}, 
				{{5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {10.0f, 10.0f}},
				{{-5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 10.0f}},
			};

			std::vector<uint32_t> indices = {
				0, 1, 2, 2, 3, 0
			};

			plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
				vertices.size() * sizeof(Vertex), vertices.data(),
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

			plane_mesh.index_buffer = new veekay::graphics::Buffer(
				indices.size() * sizeof(uint32_t), indices.data(),
				VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

			plane_mesh.indices = uint32_t(indices.size());
		}

		// NOTE: Cube mesh initialization
		{
			std::vector<Vertex> vertices = {
				{{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
				{{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
				{{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
				{{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},

				{{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
				{{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
				{{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
				{{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

				{{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
				{{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
				{{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
				{{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},

				{{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
				{{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
				{{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
				{{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

				{{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
				{{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
				{{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
				{{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},

				{{-0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
				{{+0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
				{{+0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
				{{-0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
			};

			std::vector<uint32_t> indices = {
				0, 1, 2, 2, 3, 0,
				4, 5, 6, 6, 7, 4,
				8, 9, 10, 10, 11, 8,
				12, 13, 14, 14, 15, 12,
				16, 17, 18, 18, 19, 16,
				20, 21, 22, 22, 23, 20,
			};

			cube_mesh.vertex_buffer = new veekay::graphics::Buffer(
				vertices.size() * sizeof(Vertex), vertices.data(),
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

			cube_mesh.index_buffer = new veekay::graphics::Buffer(
				indices.size() * sizeof(uint32_t), indices.data(),
				VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

			cube_mesh.indices = uint32_t(indices.size());
		}

		// NOTE: Add models to scene
		models.emplace_back(Model{
			.mesh = plane_mesh,
			.transform = Transform{},
			.material = floor_material, 
			});

		models.emplace_back(Model{
			.mesh = cube_mesh,
			.transform = Transform{.position = {-2.0f, -0.5f, -1.5f}, },
			.material = first_material, 
			});

		models.emplace_back(Model{
			.mesh = cube_mesh,
			.transform = Transform{.position = {1.5f, -0.5f, -0.5f}, },
			.material = second_material,
			});

		models.emplace_back(Model{
			.mesh = cube_mesh,
			.transform = Transform{.position = {0.0f, -0.5f, 1.0f}, },
			.material = third_material, 
			});

		point_lights.push_back({
			.position = { -2.0f, 1.0f, -1.5f },
			.intensity = 5.0f, 
			.color = { 0.8f, 0.6f, 0.6f } 
			});
		point_lights.push_back({
			.position = { 1.5f, 1.0f, -0.5f },
			.intensity = 5.0f, 
			.color = { 0.6f, 0.8f, 0.6f } 
			});
		point_lights.push_back({
			.position = { 0.0f, 1.0f, 1.0f },
			.intensity = 5.0f, 
			.color = { 0.6f, 0.6f, 0.8f } 
			});

		spot_lights.emplace_back(SpotLight{
			.position = { 1.5f, 2.0f, 0.5f },
			.direction = { 0.0f, -1.0f, 0.0f },
			.color = { 1.0f, 1.0f, 1.0f },
			.inner_cutOff = 10.0f,
			.outer_cutOff = 15.0f
			});
	}

	// NOTE: Destroy resources here, do not cause leaks in your program!
	void shutdown() {
		VkDevice& device = veekay::app.vk_device;

		models.clear();

		delete cube_mesh.index_buffer;
		delete cube_mesh.vertex_buffer;
		delete plane_mesh.index_buffer;
		delete plane_mesh.vertex_buffer;

		delete light_data_buffer;
		delete model_uniforms_buffer;
		delete scene_uniforms_buffer;

		std::vector<veekay::graphics::Texture*> unique_textures;
		auto isTextureUnique = [&](veekay::graphics::Texture* tex) {
			if (!tex) return false;
			return std::find(unique_textures.begin(), unique_textures.end(), tex) == unique_textures.end();
			};

		std::vector<Material*> all_materials;
		all_materials.push_back(first_material);
		all_materials.push_back(second_material);
		all_materials.push_back(third_material);
		all_materials.push_back(floor_material);
		all_materials.push_back(missing_material);

		for (auto* mat : all_materials) {
			if (!mat) continue;
			if (mat->texture_diffuse && isTextureUnique(mat->texture_diffuse)) {
				unique_textures.push_back(mat->texture_diffuse);
			}
			if (mat->texture_specular && isTextureUnique(mat->texture_specular)) {
				unique_textures.push_back(mat->texture_specular);
			}
			if (mat->texture_emissive && isTextureUnique(mat->texture_emissive)) {
				unique_textures.push_back(mat->texture_emissive);
			}
		}

		for (auto* mat : all_materials) {
			if (!mat) continue;

			if (mat->sampler != VK_NULL_HANDLE) {
				vkDestroySampler(device, mat->sampler, nullptr);
				mat->sampler = VK_NULL_HANDLE;
			}

			// Don't delete textures here, they'll be deleted separately
			delete mat; // Delete the Material struct itself
		}

		first_material = nullptr;
		second_material = nullptr;
		third_material = nullptr;
		floor_material = nullptr;
		missing_material = nullptr;

		for (auto* tex : unique_textures) {
			delete tex;
		}

		vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
		descriptor_pool = VK_NULL_HANDLE;

		vkDestroyDescriptorSetLayout(device, material_set_layout, nullptr);
		material_set_layout = VK_NULL_HANDLE;

		vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
		descriptor_set_layout = VK_NULL_HANDLE;

		vkDestroyPipeline(device, pipeline, nullptr);
		pipeline = VK_NULL_HANDLE;

		vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
		pipeline_layout = VK_NULL_HANDLE;

		vkDestroyShaderModule(device, fragment_shader_module, nullptr);
		fragment_shader_module = VK_NULL_HANDLE;

		vkDestroyShaderModule(device, vertex_shader_module, nullptr);
		vertex_shader_module = VK_NULL_HANDLE;
	}

	void update(double time) {
		ImGui::Begin("Controls:");

		ImGui::Text("Ambient");
		ImGui::ColorEdit3("Color##Amb", &ambient_light.color.x);

		ImGui::Separator();
		ImGui::Text("Directional");
		ImGui::SliderFloat3("Dir", &directional_light.direction.x, -1.0f, 1.0f);
		directional_light.direction = veekay::vec3::normalized(directional_light.direction);
		ImGui::ColorEdit3("Color##Dir", &directional_light.color.x);

		ImGui::Separator();
		ImGui::Text("Spot Lights");
		for (size_t i = 0; i < spot_lights.size(); ++i) {
			char label[64];
			snprintf(label, sizeof(label), "Spot Light %zu", i);
			ImGui::Separator();
			ImGui::Text("%s", label);
			ImGui::SliderFloat3(("Pos##Spot" + std::to_string(i)).c_str(), &spot_lights[i].position.x, -10.0f, 10.0f);
			ImGui::SliderFloat3(("Dir##Spot" + std::to_string(i)).c_str(), &spot_lights[i].direction.x, -1.0f, 1.0f);
			spot_lights[i].direction = veekay::vec3::normalized(spot_lights[i].direction);
			ImGui::ColorEdit3(("Color##Spot" + std::to_string(i)).c_str(), &spot_lights[i].color.x);
			ImGui::SliderFloat(("Inner CutOff##Spot" + std::to_string(i)).c_str(), &spot_lights[i].inner_cutOff, 0.0f, 45.0f);
			ImGui::SliderFloat(("Outer CutOff##Spot" + std::to_string(i)).c_str(), &spot_lights[i].outer_cutOff, 0.0f, 45.0f);
		}

		ImGui::Separator();
		ImGui::Text("Point Lights");
		for (size_t i = 0; i < point_lights.size(); ++i) {
			char label[64];
			snprintf(label, sizeof(label), "Point Light %zu", i);
			ImGui::Separator();
			ImGui::Text("%s", label);
			ImGui::SliderFloat3(("Pos##Point" + std::to_string(i)).c_str(), &point_lights[i].position.x, -10.0f, 10.0f);
			ImGui::SliderFloat(("Intensity##Point" + std::to_string(i)).c_str(), &point_lights[i].intensity, 0.1f, 50.0f);
			ImGui::ColorEdit3(("Color##Point" + std::to_string(i)).c_str(), &point_lights[i].color.x);
		}

		ImGui::End();

		if (ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {
			return;
		}

		if (veekay::input::mouse::isButtonDown(veekay::input::mouse::Button::left)) {
			auto delta = veekay::input::mouse::cursorDelta();
			camera.rotation.y -= delta.x * 0.15f;
			camera.rotation.x -= delta.y * 0.15f;
			camera.rotation.x = std::clamp(camera.rotation.x, -89.0f, 89.0f);
		}

		veekay::mat4 view = camera.view();
		veekay::vec3 right = veekay::vec3::normalized({ view[0][0], view[1][0], view[2][0] });
		veekay::vec3 front = veekay::vec3::normalized({ -view[0][2], -view[1][2], -view[2][2] });

		float speed = 0.05f;

		if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::w)) camera.position += front * speed;
		if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::s)) camera.position -= front * speed;
		if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::d)) camera.position += right * speed;
		if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::a)) camera.position -= right * speed;
		if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::q)) camera.position.y += speed;
		if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::z)) camera.position.y -= speed;

		float aspect = float(veekay::app.window_width) / float(veekay::app.window_height);

		SceneUniforms scene_data;
		scene_data.view_projection = camera.view_projection(aspect);
		scene_data.view_position = camera.rotation; 
		scene_data.ambient_light_intensity = ambient_light.color;
		scene_data.sun_light_direction = directional_light.direction;
		scene_data.sun_light_color = directional_light.color;
		scene_data.point_light_count = static_cast<uint32_t>(point_lights.size());
		scene_data.time = static_cast<float>(time);
		*(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_data;

		std::vector<ModelUniforms> model_uniforms_data(models.size());
		for (size_t i = 0, n = models.size(); i < n; ++i) {
			ModelUniforms& uniforms = model_uniforms_data[i];
			uniforms.model = models[i].transform.matrix();
			// uniforms.albedo_color = models[i].albedo_color; 
			// uniforms.specular_color = models[i].material->specular; 
			// uniforms.shininess = models[i].material->shininess; 
		}

		uint8_t* buffer_ptr = static_cast<uint8_t*>(model_uniforms_buffer->mapped_region);
		for (size_t i = 0; i < model_uniforms_data.size(); ++i) {
			memcpy(buffer_ptr + i * aligned_model_uniforms_size,
				&model_uniforms_data[i], sizeof(ModelUniforms));
		}

		LightData light_data;
		light_data.camera_position = camera.rotation; 
		light_data._p0 = 0.0f; // padding
		light_data.ambient_color = ambient_light.color;
		light_data._p1 = 0.0f; // padding
		light_data.directional_dir = directional_light.direction;
		light_data._p2 = 0.0f; // padding
		light_data.directional_color = directional_light.color;
		light_data._p3 = 0.0f; // padding

		for (size_t i = 0; i < spot_lights.size() && i < 10; ++i) {
			light_data.spot_lights[i].position = spot_lights[i].position;
			light_data.spot_lights[i].direction = spot_lights[i].direction;
			light_data.spot_lights[i].color = spot_lights[i].color;
			light_data.spot_lights[i].inner_cutOff_cos = (float)cos(toRadians(spot_lights[i].inner_cutOff));
			light_data.spot_lights[i].outer_cutOff_cos = (float)cos(toRadians(spot_lights[i].outer_cutOff));
		}
		light_data.spot_light_count = static_cast<uint32_t>(spot_lights.size());
		light_data._pad1 = 0.0f; // padding
		light_data._pad2 = 0.0f; // padding
		light_data._pad3 = 0.0f; // padding


		for (size_t i = 0; i < point_lights.size() && i < 10; ++i) {
			light_data.point_lights[i].position = point_lights[i].position;
			light_data.point_lights[i].color = point_lights[i].color;
			light_data.point_lights[i].intensity = point_lights[i].intensity;
		}

		light_data.point_light_count = static_cast<uint32_t>(point_lights.size());
		light_data._pad4 = 0.0f; // padding
		light_data._pad5 = 0.0f; // padding
		light_data._pad6 = 0.0f; // padding

		memcpy(light_data_buffer->mapped_region, &light_data, sizeof(LightData));
	}

	void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
		vkResetCommandBuffer(cmd, 0);

		{ 
			VkCommandBufferBeginInfo info{
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
				.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
			};
			vkBeginCommandBuffer(cmd, &info);
		}

		{ 
			VkClearValue clear_color{ .color = {{0.1f, 0.1f, 0.1f, 1.0f}} };
			VkClearValue clear_depth{ .depthStencil = {1.0f, 0} };
			VkClearValue clear_values[] = { clear_color, clear_depth };

			VkRenderPassBeginInfo info{
				.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
				.renderPass = veekay::app.vk_render_pass,
				.framebuffer = framebuffer,
				.renderArea = {.extent = {veekay::app.window_width, veekay::app.window_height}, },
				.clearValueCount = 2,
				.pClearValues = clear_values,
			};

			vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

			VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
			VkBuffer current_index_buffer = VK_NULL_HANDLE;
			VkDescriptorSet current_material_set = VK_NULL_HANDLE;

			for (size_t i = 0, n = models.size(); i < n; ++i) {
				const Model& model = models[i];
				const Mesh& mesh = model.mesh;

				if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
					current_vertex_buffer = mesh.vertex_buffer->buffer;
					VkDeviceSize zero_offset = 0;
					vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
				}

				if (current_index_buffer != mesh.index_buffer->buffer) {
					current_index_buffer = mesh.index_buffer->buffer;
					vkCmdBindIndexBuffer(cmd, current_index_buffer, 0, VK_INDEX_TYPE_UINT32);
				}

				uint32_t dyn_offset = i * aligned_model_uniforms_size;
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
					0, 1, &descriptor_set, 1, &dyn_offset);

				VkDescriptorSet material_set = model.material ? model.material->descriptor_set
					: missing_material->descriptor_set;
				if (current_material_set != material_set) {
					current_material_set = material_set;
					vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
						1, 1, &current_material_set, 0, nullptr);
				}

				vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
			}

			vkCmdEndRenderPass(cmd);
		}

		vkEndCommandBuffer(cmd);
	}

} // namespace

int main() {
	return veekay::run({
		.init = initialize,
		.shutdown = shutdown,
		.update = update,
		.render = render,
		});
}
