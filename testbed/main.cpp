#include <cstdint>
#include <climits>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

#include <veekay/veekay.hpp>

#include <vulkan/vulkan_core.h>
#include <imgui.h>
#include <lodepng.h>

namespace {

	constexpr uint32_t max_models = 1024;
	constexpr uint32_t max_point_lights = 16;
	constexpr uint32_t max_spot_lights = 16;
	constexpr uint32_t materials_count = 4;

	struct Vertex {
		veekay::vec3 position;
		veekay::vec3 normal;
		veekay::vec2 uv;
	};

	struct PointLight {
		veekay::vec3 position;
		float intensity = 1.0f;
		veekay::vec3 color = { 1.0f, 1.0f, 1.0f };
		float _pad0;
	};

	struct SpotLight {
		veekay::vec3 position;
		float _pad0;
		veekay::vec3 direction;
		float _pad1;
		veekay::vec3 color;
		float _pad2;
		float inner_cutOff_cos;
		float outer_cutOff_cos;
		float _pad3;
		float _pad4;
	};

	struct LightData {
		veekay::vec3 camera_position;
		float _p0;
		veekay::vec3 ambient_color;
		float _p1;
		veekay::vec3 directional_dir;
		float _p2;
		veekay::vec3 directional_color;
		float _p3;

		SpotLight spot_lights[max_spot_lights]; 
		uint32_t spot_light_count;
		float _pad1;
		float _pad2;
		float _pad3;

		PointLight point_lights[max_point_lights];
		uint32_t point_light_count;
		float _pad4;
		float _pad5;
		float _pad6;
	};

	struct SceneUniforms {
		veekay::mat4 view_projection;
		veekay::mat4 shadow_projection; 
		veekay::vec3 view_position;
		float _pad0;
		float time; 
		float _pad5[3];
	};

	struct ModelUniforms {
		veekay::mat4 model;
		veekay::vec3 albedo_color; float _pad0;
		veekay::vec3 specular_color; float _pad1;
		float shininess; float _pad2[3];
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

		veekay::mat4 matrix() const;
	};

	struct Model {
		Mesh mesh;
		Transform transform;
		veekay::vec3 albedo_color;
		veekay::vec3 specular_color;
		float shininess;
		uint32_t material_index;
	};

	struct Camera {
		constexpr static float default_fov = 60.0f;
		constexpr static float default_near_plane = 0.01f;
		constexpr static float default_far_plane = 100.0f;

		veekay::vec3 position = {};
		veekay::vec3 rotation = {};
		veekay::vec3 forward = {};

		float fov = default_fov;
		float near_plane = default_near_plane;
		float far_plane = default_far_plane;

		veekay::mat4 view() const;
		veekay::mat4 lookat() const;
		veekay::mat4 view_projection(float aspect_ratio) const;
	};

	// NOTE: Scene objects
	inline namespace {
		bool g_use_lookat = false;
		float g_mouse_sensitivity = 0.12f * float(M_PI) / 180.0f;

		veekay::vec3 g_ambient = { 0.15f, 0.15f, 0.15f };
		veekay::vec3 g_sun_dir = veekay::vec3::normalized({ -0.3f, 0.5f, -0.2f });
		veekay::vec3 g_sun_color = { 1.0f, 1.0f, 1.0f };

		Camera camera{
			.position = {0.0f, -0.5f, -3.0f}
		};

		std::vector<PointLight> point_lights;

		std::vector<SpotLight> spot_lights;

		std::vector<Model> models;
	}

	// NOTE: Vulkan objects
	inline namespace {
		VkShaderModule vertex_shader_module;
		VkShaderModule fragment_shader_module;

		VkDescriptorPool descriptor_pool;
		VkDescriptorSetLayout descriptor_set_layout;

		VkDescriptorSet material_descriptor_sets[materials_count];

		VkPipelineLayout pipeline_layout;
		VkPipeline pipeline;

		veekay::graphics::Buffer* scene_uniforms_buffer;
		veekay::graphics::Buffer* model_uniforms_buffer;
		veekay::graphics::Buffer* light_data_buffer;

		Mesh plane_mesh;
		Mesh cube_mesh;

		veekay::graphics::Texture* fallback_texture = nullptr;
		VkSampler fallback_sampler = VK_NULL_HANDLE;

		std::vector<veekay::graphics::Texture*> textures;
		std::vector<VkSampler> texture_samplers;

		struct ShadowPass {
			VkFormat depth_image_format;
			VkImage depth_image;
			VkDeviceMemory depth_image_memory;
			VkImageView depth_image_view;

			VkShaderModule vertex_shader;

			VkDescriptorSetLayout descriptor_set_layout;
			VkDescriptorSet descriptor_set;
			VkPipelineLayout pipeline_layout;
			VkPipeline pipeline;

			veekay::graphics::Buffer* uniform_buffer;
			VkSampler sampler;

			veekay::mat4 matrix;
		};

		ShadowPass shadow;

		PFN_vkCmdBeginRenderingKHR vkCmdBeginRenderingKHR;
		PFN_vkCmdEndRenderingKHR   vkCmdEndRenderingKHR;

		constexpr uint32_t shadow_map_size = 4096;
	}

	float toRadians(float degrees) {
		return degrees * float(M_PI) / 180.0f;
	}

	static constexpr veekay::vec3 WORLD_UP{ 0.0f, -1.0f, 0.0f };

	static void cameraBasis(float yaw, float pitch,
		veekay::vec3& right,
		veekay::vec3& up,
		veekay::vec3& front)
	{
		front = veekay::vec3::normalized({
			std::cosf(pitch) * std::sinf(yaw), 
			std::sinf(pitch),                  
			-std::cosf(pitch) * std::cosf(yaw) 
			});
		right = veekay::vec3::normalized(veekay::vec3::cross(front, WORLD_UP));
		up = veekay::vec3::cross(right, front);
	}

	veekay::mat4 Camera::lookat() const {
		using veekay::vec3;
		using veekay::mat4;

		vec3 target = forward;
		vec3 eye = position;
		vec3 center = { eye.x + target.x,
						eye.y + target.y,
						eye.z + target.z };

		vec3 f = forward;
		vec3 r = vec3::normalized(vec3::cross(f, WORLD_UP));
		vec3 u = vec3::cross(r, f);

		mat4 m{};
		m[0][0] = -r.x; m[0][1] = -u.x; m[0][2] = -f.x; m[0][3] = 0.0f;
		m[1][0] = -r.y; m[1][1] = -u.y; m[1][2] = -f.y; m[1][3] = 0.0f;
		m[2][0] = -r.z; m[2][1] = -u.z; m[2][2] = -f.z; m[2][3] = 0.0f;
		m[3][0] = vec3::dot(r, eye);
		m[3][1] = vec3::dot(u, eye);
		m[3][2] = vec3::dot(f, eye);
		m[3][3] = 1.0f;

		return m;
	}

	veekay::mat4 lookat(veekay::vec3 forward, veekay::vec3 position) {
		using veekay::vec3;
		using veekay::mat4;

		vec3 target = forward;
		vec3 eye = position;
		vec3 center = { eye.x + target.x,
						eye.y + target.y,
						eye.z + target.z };

		vec3 f = forward;
		vec3 r = vec3::normalized(vec3::cross(f, WORLD_UP));
		vec3 u = vec3::cross(r, f);

		mat4 m{};
		m[0][0] = -r.x; m[0][1] = -u.x; m[0][2] = -f.x; m[0][3] = 0.0f;
		m[1][0] = -r.y; m[1][1] = -u.y; m[1][2] = -f.y; m[1][3] = 0.0f;
		m[2][0] = -r.z; m[2][1] = -u.z; m[2][2] = -f.z; m[2][3] = 0.0f;
		m[3][0] = vec3::dot(r, eye);
		m[3][1] = vec3::dot(u, eye);
		m[3][2] = vec3::dot(f, eye);
		m[3][3] = 1.0f;

		return m;
	}

	veekay::mat4 Transform::matrix() const {
		using veekay::vec3; using veekay::mat4;
		mat4 S = mat4::scaling(scale);

		mat4 Rz = mat4::rotation(vec3{ 0,0,1 }, rotation.z);
		mat4 Rx = mat4::rotation(vec3{ 1,0,0 }, rotation.x);
		mat4 Ry = mat4::rotation(vec3{ 0,1,0 }, rotation.y);
		mat4 R = Rz * Rx * Ry;

		mat4 T = mat4::translation(position);

		return S * R * T;
	}

	veekay::mat4 Camera::view() const {
		veekay::vec3 right, up, front;
		cameraBasis(rotation.y, rotation.x, right, up, front);

		veekay::mat4 m{};
		m[0][0] = -right.x; m[0][1] = -up.x; m[0][2] = -front.x; m[0][3] = 0.0f;
		m[1][0] = -right.y; m[1][1] = -up.y; m[1][2] = -front.y; m[1][3] = 0.0f;
		m[2][0] = -right.z; m[2][1] = -up.z; m[2][2] = -front.z; m[2][3] = 0.0f;
		m[3][0] = veekay::vec3::dot(right, position);
		m[3][1] = veekay::vec3::dot(up, position);
		m[3][2] = veekay::vec3::dot(front, position);
		m[3][3] = 1.0f;
		return m;
	}

	veekay::mat4 Camera::view_projection(float aspect_ratio) const {
		auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);

		veekay::mat4 view_matrix = g_use_lookat ? lookat() : view();

		return view_matrix * projection;
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
		if (vkCreateShaderModule(veekay::app.vk_device, &
			info, nullptr, &result) != VK_SUCCESS) {
			return nullptr;
		}

		return result;
	}

	void initialize(VkCommandBuffer cmd) {
		VkDevice& device = veekay::app.vk_device;
		VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;
		vkCmdBeginRenderingKHR = reinterpret_cast<PFN_vkCmdBeginRenderingKHR>(
			vkGetDeviceProcAddr(device, "vkCmdBeginRenderingKHR"));
		vkCmdEndRenderingKHR = reinterpret_cast<PFN_vkCmdEndRenderingKHR>(
			vkGetDeviceProcAddr(device, "vkCmdEndRenderingKHR"));

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

			shadow.vertex_shader = loadShaderModule("./shaders/shadow.vert.spv");
			if (!shadow.vertex_shader) {
				std::cerr << "Failed to load shadow vertex shader\n";
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
						.descriptorCount = 32,
					},
					{
						.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
						.descriptorCount = 8,
					},
				};

				VkDescriptorPoolCreateInfo info{
					.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
					.maxSets = materials_count + 1,
					.poolSizeCount = sizeof(pools) / sizeof(pools[0]),
					.pPoolSizes = pools,
				};

				if (vkCreateDescriptorPool(device, &info, nullptr,
					&descriptor_pool) != VK_SUCCESS) {
					std::cerr << "Failed to create Vulkan descriptor pool\n";
					veekay::app.running = false;
					return;
				}
			}

			// NOTE: Descriptor set layout specification
			{
				VkDescriptorSetLayoutBinding bindings[] = {
					{ // SceneUniforms
						.binding = 0,
						.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
					},
					{ // ModelUniforms
						.binding = 1,
						.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
					},
					{ // LightData
						.binding = 2,
						.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
					},
					{ // albedo_texture
						.binding = 4,
						.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
					},
					{ // specular_texture
						.binding = 5,
						.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
					},
					{ // emissive_texture
						.binding = 6,
						.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
					},
					{ // shadow map
						.binding = 7,
						.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
					},
				};

				VkDescriptorSetLayoutCreateInfo info{
					.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
					.bindingCount = sizeof(bindings) / sizeof(bindings[0]),
					.pBindings = bindings,
				};

				if (vkCreateDescriptorSetLayout(device, &info, nullptr,
					&descriptor_set_layout) != VK_SUCCESS) {
					std::cerr << "Failed to create Vulkan descriptor set layout\n";
					veekay::app.running = false;
					return;
				}
			}

			{
				VkDescriptorSetLayout layouts[materials_count];
				for (int i = 0; i < materials_count; i++)
					layouts[i] = descriptor_set_layout;

				VkDescriptorSetAllocateInfo info{
					.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
					.descriptorPool = descriptor_pool,
					.descriptorSetCount = materials_count,
					.pSetLayouts = layouts,
				};

				if (vkAllocateDescriptorSets(device, &info, material_descriptor_sets) != VK_SUCCESS) {
					std::cerr << "Failed to create Vulkan descriptor sets\n";
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
						.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
					},
					{
						.binding = 1,
						.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
					},
				};

				VkDescriptorSetLayoutCreateInfo info{
					.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
					.bindingCount = uint32_t(std::size(bindings)),
					.pBindings = bindings,
				};

				vkCreateDescriptorSetLayout(device, &info, nullptr,
					&shadow.descriptor_set_layout);
			}

			{
				VkDescriptorSetAllocateInfo info{
					.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
					.descriptorPool = descriptor_pool,
					.descriptorSetCount = 1,
					.pSetLayouts = &shadow.descriptor_set_layout,
				};
				vkAllocateDescriptorSets(device, &info, &shadow.descriptor_set);
			}

			// NOTE: Declare external data sources, only push constants this time
			VkPipelineLayoutCreateInfo layout_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
				.setLayoutCount = 1,
				.pSetLayouts = &descriptor_set_layout,
			};

			// NOTE: Create pipeline layout
			if (vkCreatePipelineLayout(device, &layout_info,
				nullptr, &pipeline_layout) != VK_SUCCESS) {
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
			if (vkCreateGraphicsPipelines(device, nullptr,
				1, &info, nullptr, &pipeline) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan pipeline\n";
				veekay::app.running = false;
				return;
			}

			{
				VkPipelineLayoutCreateInfo layout_info{
					.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
					.setLayoutCount = 1,
					.pSetLayouts = &shadow.descriptor_set_layout,
				};
				vkCreatePipelineLayout(device, &layout_info, nullptr,
					&shadow.pipeline_layout);
			}

			{
				if (shadow.depth_image_format == VK_FORMAT_UNDEFINED) {
					VkFormat candidates[] = {
						VK_FORMAT_D32_SFLOAT,
						VK_FORMAT_D32_SFLOAT_S8_UINT,
						VK_FORMAT_D24_UNORM_S8_UINT,
					};

					for (auto f : candidates) {
						VkFormatProperties props;
						vkGetPhysicalDeviceFormatProperties(physical_device, f, &props);
						if (props.optimalTilingFeatures &
							VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
							shadow.depth_image_format = f;
							break;
						}
					}
				}

				VkVertexInputBindingDescription shadow_binding{
					.binding = 0,
					.stride = sizeof(Vertex),
					.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
				};

				VkVertexInputAttributeDescription shadow_attribute{
					.location = 0,
					.binding = 0,
					.format = VK_FORMAT_R32G32B32_SFLOAT,
					.offset = offsetof(Vertex, position),
				};

				VkPipelineVertexInputStateCreateInfo shadow_input_state{
					.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
					.vertexBindingDescriptionCount = 1,
					.pVertexBindingDescriptions = &shadow_binding,
					.vertexAttributeDescriptionCount = 1,
					.pVertexAttributeDescriptions = &shadow_attribute,
				};

				VkPipelineShaderStageCreateInfo stage{
					.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
					.stage = VK_SHADER_STAGE_VERTEX_BIT,
					.module = shadow.vertex_shader,
					.pName = "main",
				};

				VkPipelineRasterizationStateCreateInfo raster_info{
					.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
					.polygonMode = VK_POLYGON_MODE_FILL,
					.cullMode = VK_CULL_MODE_FRONT_BIT,
					.frontFace = VK_FRONT_FACE_CLOCKWISE,
					.depthBiasEnable = VK_TRUE,
					.lineWidth = 1.0f,
				};

				VkPipelineColorBlendStateCreateInfo blend_info{
					.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
					.attachmentCount = 0,
				};

				VkDynamicState dyn_states[] = {
					VK_DYNAMIC_STATE_VIEWPORT,
					VK_DYNAMIC_STATE_SCISSOR,
					VK_DYNAMIC_STATE_DEPTH_BIAS,
				};
				VkPipelineDynamicStateCreateInfo dyn_state_info{
					.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
					.dynamicStateCount = uint32_t(std::size(dyn_states)),
					.pDynamicStates = dyn_states,
				};

				VkPipelineViewportStateCreateInfo viewport_info{
					.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
					.viewportCount = 1,
					.scissorCount = 1,
				};

				VkPipelineRenderingCreateInfoKHR format_info{
					.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
					.depthAttachmentFormat = shadow.depth_image_format,
				};

				VkGraphicsPipelineCreateInfo info{
					.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
					.pNext = &format_info,
					.stageCount = 1,
					.pStages = &stage,
					.pVertexInputState = &shadow_input_state,
					.pInputAssemblyState = &assembly_state_info,
					.pViewportState = &viewport_info,
					.pRasterizationState = &raster_info,
					.pMultisampleState = &sample_info,
					.pDepthStencilState = &depth_info,
					.pColorBlendState = &blend_info,
					.pDynamicState = &dyn_state_info,
					.layout = shadow.pipeline_layout,
				};

				vkCreateGraphicsPipelines(device, nullptr, 1, &info, nullptr,
					&shadow.pipeline);
			}

			{
				VkImageCreateInfo info{
					.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
					.imageType = VK_IMAGE_TYPE_2D,
					.format = shadow.depth_image_format,
					.extent = { shadow_map_size, shadow_map_size, 1 },
					.mipLevels = 1,
					.arrayLayers = 1,
					.samples = VK_SAMPLE_COUNT_1_BIT,
					.tiling = VK_IMAGE_TILING_OPTIMAL,
					.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
							VK_IMAGE_USAGE_SAMPLED_BIT,
				};

				vkCreateImage(device, &info, nullptr, &shadow.depth_image);

				VkMemoryRequirements req;
				vkGetImageMemoryRequirements(device, shadow.depth_image, &req);

				VkPhysicalDeviceMemoryProperties mem_props;
				vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);

				uint32_t index = UINT32_MAX;
				for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
					if ((req.memoryTypeBits & (1 << i)) &&
						(mem_props.memoryTypes[i].propertyFlags &
							VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
						index = i;
						break;
					}
				}

				VkMemoryAllocateInfo alloc{
					.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
					.allocationSize = req.size,
					.memoryTypeIndex = index,
				};
				vkAllocateMemory(device, &alloc, nullptr, &shadow.depth_image_memory);
				vkBindImageMemory(device, shadow.depth_image, shadow.depth_image_memory, 0);

				VkImageViewCreateInfo view_info{
					.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
					.image = shadow.depth_image,
					.viewType = VK_IMAGE_VIEW_TYPE_2D,
					.format = shadow.depth_image_format,
					.subresourceRange = {
						.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
						.baseMipLevel = 0,
						.levelCount = 1,
						.baseArrayLayer = 0,
						.layerCount = 1,
					},
				};
				vkCreateImageView(device, &view_info, nullptr, &shadow.depth_image_view);
			}
		}

		scene_uniforms_buffer = new veekay::graphics::Buffer(
			sizeof(SceneUniforms),
			nullptr,
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

		model_uniforms_buffer = new veekay::graphics::Buffer(
			max_models * veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms)),
			nullptr,
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

		light_data_buffer = new veekay::graphics::Buffer(
			sizeof(LightData),
			nullptr,
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
		);

		shadow.uniform_buffer = new veekay::graphics::Buffer(
			sizeof(veekay::mat4), nullptr,
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

		{
			VkDescriptorBufferInfo buffer_infos[] = {
				{
					.buffer = shadow.uniform_buffer->buffer,
					.range = sizeof(veekay::mat4),
				},
				{
					.buffer = model_uniforms_buffer->buffer,
					.range = sizeof(ModelUniforms),
				},
			};

			VkWriteDescriptorSet writes[] = {
				{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = shadow.descriptor_set,
					.dstBinding = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.pBufferInfo = &buffer_infos[0],
				},
				{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = shadow.descriptor_set,
					.dstBinding = 1,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.pBufferInfo = &buffer_infos[1],
				},
			};

			vkUpdateDescriptorSets(device,
				uint32_t(std::size(writes)), writes,
				0, nullptr);
		}

		// NOTE: This texture and sampler is used when texture could not be loaded
		// Создаём ОДНУ текстуру-заглушку (чёрную)
		{
			uint32_t black_pixels[4] = { 0xff000000, 0xff000000, 0xff000000, 0xff000000 };

			fallback_texture = new veekay::graphics::Texture(
				cmd, 2, 2,
				VK_FORMAT_B8G8R8A8_UNORM,
				black_pixels
			);

			VkSamplerCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
				.magFilter = VK_FILTER_NEAREST,
				.minFilter = VK_FILTER_NEAREST,
				.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
				.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
				.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
				.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
				.minLod = 0.0f,
				.maxLod = VK_LOD_CLAMP_NONE,
			};

			if (vkCreateSampler(device, &info, nullptr, &fallback_sampler) != VK_SUCCESS) {
				std::cerr << "Failed to create fallback sampler\n";
				veekay::app.running = false;
				return;
			}
		}

		// 1. Текстура для пола
		{
			veekay::graphics::Texture* texture = fallback_texture;
			VkSampler texture_sampler = fallback_sampler;

			uint32_t width, height;
			std::vector<uint8_t> pixels;
			if (lodepng::decode(pixels, width, height, "./assets/stone.png")) {
				std::cerr << "Failed to load floor texture\n";
			}
			else {
				texture = new veekay::graphics::Texture(
					cmd, width, height,
					VK_FORMAT_R8G8B8A8_UNORM,
					pixels.data());

				VkSamplerCreateInfo info{
					.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
					.magFilter = VK_FILTER_LINEAR,
					.minFilter = VK_FILTER_LINEAR,
					.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
					.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
					.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
					.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
					.anisotropyEnable = true,
					.maxAnisotropy = 16.0f,
					.minLod = 0.0f,
					.maxLod = VK_LOD_CLAMP_NONE,
				};

				if (vkCreateSampler(device, &info, nullptr, &texture_sampler) != VK_SUCCESS) {
					std::cerr << "Failed to create floor texture sampler\n";
					texture = fallback_texture;
					texture_sampler = fallback_sampler;
				}
			}

			textures.push_back(texture);
			texture_samplers.push_back(texture_sampler);
		}

		// 2. Specular текстура для пола
		{
			veekay::graphics::Texture* texture = fallback_texture;
			VkSampler texture_sampler = fallback_sampler;

			uint32_t width, height;
			std::vector<uint8_t> pixels;
			if (lodepng::decode(pixels, width, height, "./assets/star.png")) {
				std::cerr << "Failed to load floor specular texture\n";
			}
			else {
				texture = new veekay::graphics::Texture(
					cmd, width, height,
					VK_FORMAT_R8G8B8A8_UNORM,
					pixels.data());

				VkSamplerCreateInfo info{
					.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
					.magFilter = VK_FILTER_LINEAR,
					.minFilter = VK_FILTER_LINEAR,
					.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
					.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
					.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
					.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
					.anisotropyEnable = true,
					.maxAnisotropy = 16.0f,
					.minLod = 0.0f,
					.maxLod = VK_LOD_CLAMP_NONE,
				};

				if (vkCreateSampler(device, &info, nullptr, &texture_sampler) != VK_SUCCESS) {
					std::cerr << "Failed to create floor specular sampler\n";
					texture = fallback_texture;
					texture_sampler = fallback_sampler;
				}
			}

			textures.push_back(texture);
			texture_samplers.push_back(texture_sampler);
		}

		// 3. Текстура для первого куба
		{
			veekay::graphics::Texture* texture = fallback_texture;
			VkSampler texture_sampler = fallback_sampler;

			uint32_t width, height;
			std::vector<uint8_t> pixels;
			if (lodepng::decode(pixels, width, height, "./assets/wood.png")) {
				std::cerr << "Failed to load cube1 texture\n";
			}
			else {
				texture = new veekay::graphics::Texture(
					cmd, width, height,
					VK_FORMAT_R8G8B8A8_UNORM,
					pixels.data());

				VkSamplerCreateInfo info{
					.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
					.magFilter = VK_FILTER_LINEAR,
					.minFilter = VK_FILTER_LINEAR,
					.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
					.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
					.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
					.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
					.anisotropyEnable = true,
					.maxAnisotropy = 16.0f,
					.minLod = 0.0f,
					.maxLod = VK_LOD_CLAMP_NONE,
				};

				if (vkCreateSampler(device, &info, nullptr, &texture_sampler) != VK_SUCCESS) {
					std::cerr << "Failed to create cube1 texture sampler\n";
					texture = fallback_texture;
					texture_sampler = fallback_sampler;
				}
			}

			textures.push_back(texture);
			texture_samplers.push_back(texture_sampler);
		}

		// 4. Emissive текстура для первого куба
		{
			veekay::graphics::Texture* texture = fallback_texture;
			VkSampler texture_sampler = fallback_sampler;

			uint32_t width, height;
			std::vector<uint8_t> pixels;
			if (lodepng::decode(pixels, width, height, "./assets/emissive.png")) {
				std::cerr << "Failed to load cube3 emissive texture\n";
			}
			else {
				texture = new veekay::graphics::Texture(
					cmd, width, height,
					VK_FORMAT_R8G8B8A8_UNORM,
					pixels.data());

				VkSamplerCreateInfo info{
					.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
					.magFilter = VK_FILTER_LINEAR,
					.minFilter = VK_FILTER_LINEAR,
					.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
					.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
					.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
					.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
					.anisotropyEnable = true,
					.maxAnisotropy = 16.0f,
					.minLod = 0.0f,
					.maxLod = VK_LOD_CLAMP_NONE,
				};

				if (vkCreateSampler(device, &info, nullptr, &texture_sampler) != VK_SUCCESS) {
					std::cerr << "Failed to create cube3 emissive sampler\n";
					texture = fallback_texture;
					texture_sampler = fallback_sampler;
				}
			}

			textures.push_back(texture);
			texture_samplers.push_back(texture_sampler);
		}

		// 5. Текстура для второго куба
		{
			veekay::graphics::Texture* texture = fallback_texture;
			VkSampler texture_sampler = fallback_sampler;

			uint32_t width, height;
			std::vector<uint8_t> pixels;
			if (lodepng::decode(pixels, width, height, "./assets/bow.png")) {
				std::cerr << "Failed to load cube2 texture\n";
			}
			else {
				texture = new veekay::graphics::Texture(
					cmd, width, height,
					VK_FORMAT_R8G8B8A8_UNORM,
					pixels.data());

				VkSamplerCreateInfo info{
					.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
					.magFilter = VK_FILTER_LINEAR,
					.minFilter = VK_FILTER_LINEAR,
					.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
					.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
					.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
					.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
					.anisotropyEnable = true,
					.maxAnisotropy = 16.0f,
					.minLod = 0.0f,
					.maxLod = VK_LOD_CLAMP_NONE,
				};

				if (vkCreateSampler(device, &info, nullptr, &texture_sampler) != VK_SUCCESS) {
					std::cerr << "Failed to create cube2 texture sampler\n";
					texture = fallback_texture;
					texture_sampler = fallback_sampler;
				}
			}

			textures.push_back(texture);
			texture_samplers.push_back(texture_sampler);
		}


		// 6. Текстура для третьего куба
		{
			veekay::graphics::Texture* texture = fallback_texture;
			VkSampler texture_sampler = fallback_sampler;

			uint32_t width, height;
			std::vector<uint8_t> pixels;
			if (lodepng::decode(pixels, width, height, "./assets/lenna.png")) {
				std::cerr << "Failed to load cube3 texture\n";
			}
			else {
				texture = new veekay::graphics::Texture(
					cmd, width, height,
					VK_FORMAT_R8G8B8A8_UNORM,
					pixels.data());

				VkSamplerCreateInfo info{
					.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
					.magFilter = VK_FILTER_LINEAR,
					.minFilter = VK_FILTER_LINEAR,
					.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
					.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
					.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
					.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
					.anisotropyEnable = true,
					.maxAnisotropy = 16.0f,
					.minLod = 0.0f,
					.maxLod = VK_LOD_CLAMP_NONE,
				};

				if (vkCreateSampler(device, &info, nullptr, &texture_sampler) != VK_SUCCESS) {
					std::cerr << "Failed to create cube3 texture sampler\n";
					texture = fallback_texture;
					texture_sampler = fallback_sampler;
				}
			}

			textures.push_back(texture);
			texture_samplers.push_back(texture_sampler);
		}

		{
			VkSamplerCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
				.magFilter = VK_FILTER_LINEAR,
				.minFilter = VK_FILTER_LINEAR,
				.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
				.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
				.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
				.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
				.compareEnable = VK_TRUE,
				.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
				.minLod = 0.0f,
				.maxLod = VK_LOD_CLAMP_NONE,
				.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
			};

			vkCreateSampler(device, &info, nullptr, &shadow.sampler);
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
					.range = sizeof(ModelUniforms),
				},
				{
					.buffer = light_data_buffer->buffer,
					.offset = 0,
					.range = sizeof(LightData),
				},
			};

			// Материал для пола
			VkDescriptorImageInfo floor_image_infos[] = {
				{
					.sampler = texture_samplers[0],
					.imageView = textures[0]->view,
					.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				},
				{
					.sampler = texture_samplers[1],
					.imageView = textures[1]->view,
					.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				},
				{
					.sampler = fallback_sampler,
					.imageView = fallback_texture->view,
					.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				},
			};

			// Материал для первого куба
			VkDescriptorImageInfo cube1_image_infos[] = {
				{
					.sampler = texture_samplers[2],
					.imageView = textures[2]->view,
					.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				},
				{
					.sampler = texture_samplers[1],
					.imageView = textures[1]->view,
					.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				},
				{
					.sampler = texture_samplers[3],
					.imageView = textures[3]->view,
					.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				},
			};

			// Материал для второго куба
			VkDescriptorImageInfo cube2_image_infos[] = {
				{
					.sampler = texture_samplers[4],
					.imageView = textures[4]->view,
					.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				},
				{
					.sampler = texture_samplers[1],
					.imageView = textures[1]->view,
					.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				},
				{
					.sampler = fallback_sampler,
					.imageView = fallback_texture->view,
					.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				},
			};

			// Материал для третьего куба
			VkDescriptorImageInfo cube3_image_infos[] = {
				{
					.sampler = texture_samplers[5],
					.imageView = textures[5]->view,
					.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				},
				{
					.sampler = texture_samplers[1],
					.imageView = textures[1]->view,
					.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				},
				{
					.sampler = fallback_sampler,
					.imageView = fallback_texture->view,
					.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				},
			};

			VkDescriptorImageInfo shadow_image_info{
				.sampler = shadow.sampler,
				.imageView = shadow.depth_image_view,
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			};

			auto write_material_set = [&](VkDescriptorSet descriptorset,
				VkDescriptorImageInfo* image_infos) {
					VkWriteDescriptorSet writes[] = {
						{   // SceneUniforms
							.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
							.dstSet = descriptorset,
							.dstBinding = 0,
							.dstArrayElement = 0,
							.descriptorCount = 1,
							.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
							.pBufferInfo = &buffer_infos[0],
						},
						{   // ModelUniforms
							.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
							.dstSet = descriptorset,
							.dstBinding = 1,
							.dstArrayElement = 0,
							.descriptorCount = 1,
							.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
							.pBufferInfo = &buffer_infos[1],
						},
						{   // LightData 
							.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
							.dstSet = descriptorset,
							.dstBinding = 2,
							.dstArrayElement = 0,
							.descriptorCount = 1,
							.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
							.pBufferInfo = &buffer_infos[2],
						},
						{   // diffuse texture
							.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
							.dstSet = descriptorset,
							.dstBinding = 4,
							.dstArrayElement = 0,
							.descriptorCount = 1,
							.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
							.pImageInfo = &image_infos[0],
						},
						{   // specular texture
							.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
							.dstSet = descriptorset,
							.dstBinding = 5,
							.dstArrayElement = 0,
							.descriptorCount = 1,
							.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
							.pImageInfo = &image_infos[1],
						},
						{   // emissive texture
							.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
							.dstSet = descriptorset,
							.dstBinding = 6,
							.dstArrayElement = 0,
							.descriptorCount = 1,
							.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
							.pImageInfo = &image_infos[2],
						},
						{   // shadow map
							.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
							.dstSet = descriptorset,
							.dstBinding = 7,
							.descriptorCount = 1,
							.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
							.pImageInfo = &shadow_image_info,
						}
					};

					vkUpdateDescriptorSets(device,
						uint32_t(std::size(writes)), writes, 0, nullptr);
				};

			// floor material (индекс 0)
			write_material_set(material_descriptor_sets[0], floor_image_infos);
			// cube1 material (индекс 1)
			write_material_set(material_descriptor_sets[1], cube1_image_infos);
			// cube2 material (индекс 2)
			write_material_set(material_descriptor_sets[2], cube2_image_infos);
			// cube3 material (индекс 3) с emissive текстурой
			write_material_set(material_descriptor_sets[3], cube3_image_infos);
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
			.albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
			.specular_color = { 0.05f, 0.05f, 0.05f },
			.shininess = 8.0f,
			.material_index = 0, 
			});

		models.emplace_back(Model{
			.mesh = cube_mesh,
			.transform = Transform{
				.position = {-2.0f, -0.5f, -1.5f},
			},
			.albedo_color = { 1.0f, 1.0f, 1.0f },
			.specular_color = { 0.5f, 0.5f, 0.5f },
			.shininess = 32.0f,
			.material_index = 1,
			});

		models.emplace_back(Model{
			.mesh = cube_mesh,
			.transform = Transform{
				.position = {1.5f, -0.5f, -0.5f},
			},
			.albedo_color = { 1.0f, 1.0f, 1.0f },
			.specular_color = { 0.6f, 0.6f, 0.6f },
			.shininess = 64.0f,
			.material_index = 2,
			});

		models.emplace_back(Model{
			.mesh = cube_mesh,
			.transform = Transform{
				.position = {0.0f, -0.5f, 1.0f},
			},
			.albedo_color = { 1.0f, 1.0f, 1.0f },
			.specular_color = { 1.0f, 1.0f, 1.0f },
			.shininess = 128.0f,
			.material_index = 3, 
			});

		point_lights.push_back({
			.position = { -2.0f, 1.0f, -1.5f },
			.intensity = 5.0f,
			.color = { 0.8f, 0.4f, 0.2f }
			});
		point_lights.push_back({
			.position = { 1.5f, 1.0f, -0.5f },
			.intensity = 5.0f,
			.color = { 0.8f, 0.2f, 0.2f }
			});
		point_lights.push_back({
			.position = { 0.0f, 1.0f, 1.0f },
			.intensity = 5.0f,
			.color = { 0.6f, 0.6f, 0.8f }
			});

		spot_lights.push_back({
			.position = { -1.5f, 2.0f, 0.5f },
			.direction = veekay::vec3::normalized({ 1.0f, -0.6f, 0.2f }),
			.color = { 1.0f, 1.0f, 1.0f },
			.inner_cutOff_cos = std::cosf(12.5f * 3.14159265358979323846f / 180.0f),
			.outer_cutOff_cos = std::cosf(17.5f * 3.14159265358979323846f / 180.0f)
			});
	}

	// NOTE: Destroy resources here, do not cause leaks in your program!
	void shutdown() {
		VkDevice& device = veekay::app.vk_device;

		vkDestroySampler(device, fallback_sampler, nullptr);
		delete fallback_texture;

		const size_t textures_count = textures.size();
		for (size_t i = 0; i < textures_count; ++i) {
			delete textures[i];
		}

		for (VkSampler sampler : texture_samplers) {
			vkDestroySampler(device, sampler, nullptr);
		}

		delete cube_mesh.index_buffer;
		delete cube_mesh.vertex_buffer;

		delete plane_mesh.index_buffer;
		delete plane_mesh.vertex_buffer;

		delete model_uniforms_buffer;
		delete scene_uniforms_buffer;

		delete light_data_buffer; 

		vkDestroySampler(device, shadow.sampler, nullptr);
		delete shadow.uniform_buffer;
		vkDestroyPipeline(device, shadow.pipeline, nullptr);
		vkDestroyPipelineLayout(device, shadow.pipeline_layout, nullptr);
		vkDestroyDescriptorSetLayout(device, shadow.descriptor_set_layout, nullptr);
		vkDestroyShaderModule(device, shadow.vertex_shader, nullptr);
		vkDestroyImageView(device, shadow.depth_image_view, nullptr);
		vkFreeMemory(device, shadow.depth_image_memory, nullptr);
		vkDestroyImage(device, shadow.depth_image, nullptr);

		vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
		vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

		vkDestroyPipeline(device, pipeline, nullptr);
		vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
		vkDestroyShaderModule(device, fragment_shader_module, nullptr);
		vkDestroyShaderModule(device, vertex_shader_module, nullptr);
	}

	void update(double time) {
		ImGui::Begin("Controls:");

		ImGui::Text("Ambient");
		ImGui::ColorEdit3("Color##Amb", &g_ambient.x);

		ImGui::Separator();
		ImGui::Text("Directional");
		ImGui::SliderFloat3("Dir", &g_sun_dir.x, -1.0f, 1.0f);
		g_sun_dir = veekay::vec3::normalized(g_sun_dir);
		ImGui::ColorEdit3("Color##Dir", &g_sun_color.x);

		ImGui::Separator();
		ImGui::Text("Spot Lights");
		for (size_t i = 0; i < spot_lights.size(); ++i) {
			char label[64];
			snprintf(label, sizeof(label), "Spot Light %zu", i);
			ImGui::Separator();
			ImGui::Text("%s", label);
			ImGui::SliderFloat3(("Pos##Spot" + std::to_string(i)).c_str(),
				&spot_lights[i].position.x, -10.0f, 10.0f);

			veekay::vec3 dir_display = spot_lights[i].direction;
			ImGui::SliderFloat3(("Dir##Spot" + std::to_string(i)).c_str(),
				&dir_display.x, -1.0f, 1.0f);
			spot_lights[i].direction = veekay::vec3::normalized(dir_display);

			ImGui::ColorEdit3(("Color##Spot" + std::to_string(i)).c_str(),
				&spot_lights[i].color.x);

			float inner_deg = std::acos(spot_lights[i].inner_cutOff_cos) * 180.0f / M_PI;
			float outer_deg = std::acos(spot_lights[i].outer_cutOff_cos) * 180.0f / M_PI;

			if (ImGui::SliderFloat(("Inner##Spot" + std::to_string(i)).c_str(),
				&inner_deg, 0.0f, 45.0f)) {
				spot_lights[i].inner_cutOff_cos = std::cos(inner_deg * M_PI / 180.0f);
			}

			if (ImGui::SliderFloat(("Outer##Spot" + std::to_string(i)).c_str(),
				&outer_deg, 0.0f, 45.0f)) {
				spot_lights[i].outer_cutOff_cos = std::cos(outer_deg * M_PI / 180.0f);
			}
		}

		ImGui::Separator();
		ImGui::Text("Point Lights");
		for (size_t i = 0; i < point_lights.size(); ++i) {
			char label[64];
			snprintf(label, sizeof(label), "Point Light %zu", i);
			ImGui::Separator();
			ImGui::Text("%s", label);
			ImGui::SliderFloat3(("Pos##Point" + std::to_string(i)).c_str(),
				&point_lights[i].position.x, -10.0f, 10.0f);
			ImGui::SliderFloat(("Intensity##Point" + std::to_string(i)).c_str(),
				&point_lights[i].intensity, 0.1f, 50.0f);
			ImGui::ColorEdit3(("Color##Point" + std::to_string(i)).c_str(),
				&point_lights[i].color.x);
		}

		ImGui::End();

		if (!ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {
			using namespace veekay::input;

			if (mouse::isButtonDown(mouse::Button::left)) {
				auto delta = mouse::cursorDelta();
				camera.rotation.y += delta.x * g_mouse_sensitivity;
				camera.rotation.x += delta.y * g_mouse_sensitivity;

				const float pitch_limit = toRadians(89.0f);
				const float yaw_limit = float(M_PI);

				if (camera.rotation.x > pitch_limit) camera.rotation.x = pitch_limit;
				if (camera.rotation.x < -pitch_limit) camera.rotation.x = -pitch_limit;

				if (camera.rotation.y > yaw_limit)   camera.rotation.y -= 2.0f * yaw_limit;
				if (camera.rotation.y < -yaw_limit)   camera.rotation.y += 2.0f * yaw_limit;
			}

			static constexpr veekay::vec3 WORLD_UP{ 0.0f, -1.0f, 0.0f };
			veekay::vec3 right, up, front;

			front = veekay::vec3::normalized({
				std::cosf(camera.rotation.x) * std::sinf(camera.rotation.y),
				std::sinf(camera.rotation.x),
				-std::cosf(camera.rotation.x) * std::cosf(camera.rotation.y)
				});
			right = veekay::vec3::normalized(veekay::vec3::cross(front, WORLD_UP));

			veekay::vec3 front_xz = veekay::vec3::normalized({ front.x, 0.0f, front.z });
			veekay::vec3 right_xz = veekay::vec3::normalized({ right.x, 0.0f, right.z });
			const float move = 0.1f;

			if (keyboard::isKeyDown(keyboard::Key::w)) camera.position -= front_xz * move;
			if (keyboard::isKeyDown(keyboard::Key::s)) camera.position += front_xz * move;
			if (keyboard::isKeyDown(keyboard::Key::d)) camera.position -= right_xz * move;
			if (keyboard::isKeyDown(keyboard::Key::a)) camera.position += right_xz * move;
			if (keyboard::isKeyDown(keyboard::Key::q)) camera.position.y -= move;
			if (keyboard::isKeyDown(keyboard::Key::z)) camera.position.y += move;
		}

		float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
		veekay::mat4 camera_vp = camera.view_projection(aspect_ratio);

		veekay::vec3 sun_dir = -g_sun_dir;
		veekay::vec3 scene_center{ 0.0f, 0.0f, 0.0f };
		float light_distance = 15.0f;
		veekay::vec3 light_position = scene_center + sun_dir * light_distance;
		veekay::vec3 light_forward = sun_dir;
		veekay::mat4 light_view = lookat(light_forward, light_position);

		float ortho_size = 10.0f;
		float z_near = 1.0f;
		float z_far = 30.0f;
		veekay::mat4 light_ortho = veekay::mat4::orthographic(-ortho_size, ortho_size,
			-ortho_size, ortho_size,
			z_near, z_far);
		veekay::mat4 light_vp = light_view * light_ortho;

		shadow.matrix = light_vp;
		*static_cast<veekay::mat4*>(shadow.uniform_buffer->mapped_region) = shadow.matrix;

		SceneUniforms scene_uniforms{
			.view_projection = camera_vp,
			.shadow_projection = light_vp,
			.view_position = camera.position,
			.time = static_cast<float>(time),
		};

		*(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;

		LightData light_data{};
		light_data.camera_position = camera.position;
		light_data.ambient_color = g_ambient;
		light_data.directional_dir = g_sun_dir;
		light_data.directional_color = g_sun_color;

		light_data.spot_light_count = std::min((uint32_t)spot_lights.size(), max_spot_lights);
		for (size_t i = 0; i < light_data.spot_light_count; ++i) {
			light_data.spot_lights[i] = spot_lights[i];
		}

		light_data.point_light_count = std::min((uint32_t)point_lights.size(), max_point_lights);
		for (size_t i = 0; i < light_data.point_light_count; ++i) {
			light_data.point_lights[i] = point_lights[i];
		}

		memcpy(light_data_buffer->mapped_region, &light_data, sizeof(LightData));

		std::vector<ModelUniforms> model_uniforms(models.size());
		for (size_t i = 0, n = models.size(); i < n; ++i) {
			const Model& model = models[i];
			ModelUniforms& uniforms = model_uniforms[i];

			uniforms.model = model.transform.matrix();
			uniforms.albedo_color = model.albedo_color;
			uniforms.specular_color = model.specular_color;
			uniforms.shininess = model.shininess;
		}

		const size_t model_stride = veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));
		for (size_t i = 0, n = model_uniforms.size(); i < n; ++i) {
			const ModelUniforms& uniforms = model_uniforms[i];
			char* const pointer = static_cast<char*>(model_uniforms_buffer->mapped_region) + i * model_stride;
			*reinterpret_cast<ModelUniforms*>(pointer) = uniforms;
		}
	}

	void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
		vkResetCommandBuffer(cmd, 0);

		{ // NOTE: Start recording rendering commands
			VkCommandBufferBeginInfo info{
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
				.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
			};

			vkBeginCommandBuffer(cmd, &info);
		}

		{
			VkImageMemoryBarrier barrier{
				.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
				.srcAccessMask = 0,
				.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
				.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
				.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.image = shadow.depth_image,
				.subresourceRange = {
					.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
			};

			vkCmdPipelineBarrier(cmd,
				VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
				VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
				VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
				VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
				0, 0, nullptr, 0, nullptr, 1, &barrier);
		}

		{
			VkRenderingAttachmentInfoKHR depth_attachment{
				.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR,
				.imageView = shadow.depth_image_view,
				.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
				.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
				.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
				.clearValue = {.depthStencil = {1.0f, 0} },
			};

			VkRenderingInfoKHR rendering_info{
				.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR,
				.renderArea = { {0, 0}, {shadow_map_size, shadow_map_size} },
				.layerCount = 1,
				.pDepthAttachment = &depth_attachment,
			};

			vkCmdBeginRenderingKHR(cmd, &rendering_info);

			VkViewport viewport{
				.x = 0.0f, .y = 0.0f,
				.width = float(shadow_map_size),
				.height = float(shadow_map_size),
				.minDepth = 0.0f, .maxDepth = 1.0f,
			};
			vkCmdSetViewport(cmd, 0, 1, &viewport);

			VkRect2D scissor{ {0, 0}, {shadow_map_size, shadow_map_size} };
			vkCmdSetScissor(cmd, 0, 1, &scissor);

			vkCmdSetDepthBias(cmd, 1.25f, 0.0f, 1.0f);

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
				shadow.pipeline);

			VkDeviceSize zero_offset = 0;
			VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
			VkBuffer current_index_buffer = VK_NULL_HANDLE;

			const size_t model_uniforms_alignment =
				veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

			for (size_t i = 0, n = models.size(); i < n; ++i) {
				const Model& model = models[i];
				const Mesh& mesh = model.mesh;

				if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
					current_vertex_buffer = mesh.vertex_buffer->buffer;
					vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer,
						&zero_offset);
				}
				if (current_index_buffer != mesh.index_buffer->buffer) {
					current_index_buffer = mesh.index_buffer->buffer;
					vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset,
						VK_INDEX_TYPE_UINT32);
				}

				uint32_t offset = uint32_t(i * model_uniforms_alignment);
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
					shadow.pipeline_layout,
					0, 1, &shadow.descriptor_set,
					1, &offset);

				vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
			}

			vkCmdEndRenderingKHR(cmd);
		}

		{
			VkImageMemoryBarrier barrier{
				.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
				.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
				.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
				.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
				.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.image = shadow.depth_image,
				.subresourceRange = {
					.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
			};

			vkCmdPipelineBarrier(cmd,
				VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
				VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
				VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
				0, 0, nullptr, 0, nullptr, 1, &barrier);
		}

		{ // NOTE: Use current swapchain framebuffer and clear it
			VkClearValue clear_color{ .color = {{0.1f, 0.1f, 0.1f, 1.0f}} };
			VkClearValue clear_depth{ .depthStencil = {1.0f, 0} };

			VkClearValue clear_values[] = { clear_color, clear_depth };

			VkRenderPassBeginInfo info{
				.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
				.renderPass = veekay::app.vk_render_pass,
				.framebuffer = framebuffer,
				.renderArea = {
					.extent = {
						veekay::app.window_width,
						veekay::app.window_height
					},
				},
				.clearValueCount = 2,
				.pClearValues = clear_values,
			};

			vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
		}

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
		VkDeviceSize zero_offset = 0;

		VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
		VkBuffer current_index_buffer = VK_NULL_HANDLE;

		const size_t model_uniorms_alignment =
			veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

		for (size_t i = 0, n = models.size(); i < n; ++i) {
			const Model& model = models[i];
			const Mesh& mesh = model.mesh;

			if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
				current_vertex_buffer = mesh.vertex_buffer->buffer;
				vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
			}

			if (current_index_buffer != mesh.index_buffer->buffer) {
				current_index_buffer = mesh.index_buffer->buffer;
				vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
			}

			uint32_t offset = i * model_uniorms_alignment;
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
				0, 1, &material_descriptor_sets[model.material_index], 1, &offset);

			vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
		}

		vkCmdEndRenderPass(cmd);
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