use std::collections::HashSet;
use std::ffi::CStr;
use std::{u64};
use std::os::raw::c_void;

use vulkanalia::bytecode::Bytecode;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::vk::{ExtDebugUtilsExtension, Framebuffer, ImageView, KhrSurfaceExtension, KhrSwapchainExtension, Pipeline, PipelineLayout, RenderPass, Semaphore};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

/// Whether the validation layers should be enabled.
const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
/// The name of the validation layers.
const VALIDATION_LAYER: vk::ExtensionName = vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];

const MAX_FRAMES_IN_FLIGHT: usize = 2;

fn main() {
    match run_app() {
        Ok(()) => println!("Vulkan initialization and device query completed successfully!"),
        Err(e) => eprintln!("Error: {}", e),
    }
}

fn run_app() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    
    let mut app = BusyDeckApp::new();
    
    // Run the event loop
    event_loop.run_app(&mut app)?;

    Ok(())
}

fn device_name_from_properties(properties: &vk::PhysicalDeviceProperties) -> String {
    let raw_name = &properties.device_name;
    let end = raw_name.iter().position(|&b| b == 0).unwrap_or(raw_name.len());
    // Convert from i8 array to u8 array for from_utf8
    let name_bytes: Vec<u8> = raw_name[..end].iter().map(|&b| b as u8).collect();
    std::str::from_utf8(&name_bytes)
        .unwrap_or("Unknown Device")
        .to_string()
}

struct BusyDeckApp {
    window: Option<Window>,
    vulkan_app: Option<VulkanApp>,
    minimized: bool,
}

impl BusyDeckApp {
    fn new() -> Self {
        BusyDeckApp { window: None, vulkan_app: None, minimized: false }
    }
}

#[derive(Default)]
struct VulkanState {
    // Debug
    messenger: vk::DebugUtilsMessengerEXT,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    pipeline_layout: PipelineLayout,
    render_pass: RenderPass,
    pipeline: Pipeline,
    framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    images_in_flight: Vec<vk::Fence>,

    frame: usize,

    resized: bool,
}

struct VulkanApp {
    entry: Entry,
    instance: Instance,
    device: Device,
    state: VulkanState,
}

impl VulkanApp {
    fn new(window: &Window) -> Result<Self, Box<dyn std::error::Error>> {
        unsafe { VulkanApp::new_unsafe(window) }
    }

    unsafe fn new_unsafe(window: &Window) -> Result<Self, Box<dyn std::error::Error>> {
        let mut state = VulkanState::default();

        let (entry, instance) = VulkanApp::init_vulkan(window, &mut state)?;
        let surface = vulkanalia::window::create_surface(&instance, &window, &window)?;
        state.surface = surface;
        VulkanApp::create_physical_device(&instance, &mut state)?;
        let device = VulkanApp::create_logical_device(&instance, &mut state)?;
        VulkanApp::create_swapchain(window, &instance, &device, &mut state)?;
        VulkanApp::create_swapchain_image_views(&device, &mut state)?;
        VulkanApp::create_render_pass(&device, &mut state)?;
        VulkanApp::create_pipeline(&device, &mut state)?;
        VulkanApp::create_framebuffers(&device, &mut state)?;
        state.command_pool = VulkanApp::create_command_pool(&instance, &device, &surface, &state.physical_device)?;
        VulkanApp::create_command_buffers(&device, &mut state)?;
        VulkanApp::create_sync_objects(&device, &mut state)?;
        
        println!("Created all Vulkan objects.");

        Ok(Self {
            entry,
            instance,
            device,
            state
        })
    }

    fn init_vulkan(window: &Window, state: &mut VulkanState) -> Result<(Entry, Instance), Box<dyn std::error::Error>> {
        println!("Initializing Vulkan API...");
        
        // Create Vulkan entry point with libloading loader
        let loader = unsafe { LibloadingLoader::new(LIBRARY)? };
        let entry = match unsafe { vulkanalia::Entry::new(loader) } {
            Ok(entry) => entry,
            Err(e) => {
                eprintln!("Failed to create Vulkan entry: {:?}", e);
                return Err("Failed to initialize Vulkan".into());
            }
        };
        
        // Create Vulkan instance
        let instance = unsafe { VulkanApp::create_instance(&entry, window, state) }?;

        println!("Initialized Vulkan API");

        Ok((entry, instance))
    }

    unsafe fn create_instance(entry: &Entry, window: &Window, state: &mut VulkanState) -> Result<Instance, Box<dyn std::error::Error>> {
        let app_info = vk::ApplicationInfo::builder()
            .application_name(b"BusyDeck\0")
            .application_version(vk::make_version(1, 0, 0))
            .engine_name(b"No Engine\0")
            .engine_version(vk::make_version(1, 0, 0))
            .api_version(vk::make_version(1, 0, 0));
        
        let available_layers = entry
            .enumerate_instance_layer_properties()?
            .iter()
            .map(|l| l.layer_name)
            .collect::<HashSet<_>>();

        if VALIDATION_ENABLED && !available_layers.contains(&VALIDATION_LAYER) {
            return Err("Validation layer requested but not supported.".into());
        }

        let layers = if VALIDATION_ENABLED {
            vec![VALIDATION_LAYER.as_ptr()]
        } else {
            Vec::new()
        };

        let mut extensions = vulkanalia::window::get_required_instance_extensions(window)
            .iter()
            .map(|e| e.as_ptr())
            .collect::<Vec<_>>();

        if VALIDATION_ENABLED {
            extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
        }

        let mut info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions);

        let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .user_callback(Some(debug_callback));

        if VALIDATION_ENABLED {
            info = info.push_next(&mut debug_info);
        }
        
        let instance = entry.create_instance(&info, None)?;

        if VALIDATION_ENABLED {
            state.messenger = instance.create_debug_utils_messenger_ext(&debug_info, None)?;
        }
        
        println!("Vulkan instance created successfully!");
        Ok(instance)
    }

    fn create_physical_device(instance: &Instance, state: &mut VulkanState) -> Result<(), Box<dyn std::error::Error>> {
        match VulkanApp::query_and_print_devices(instance, &state.surface)? {
            Some(physical_device) => {
                println!("Selected and initialized physical device.");
                state.physical_device = physical_device;
                Ok(())
            }
            None => {
                Err("No device found".into())
            }
        }
    }
    
    fn query_and_print_devices(instance: &Instance, surface: &vk::SurfaceKHR) -> Result<Option<vk::PhysicalDevice>, Box<dyn std::error::Error>> {
        // Get all physical devices
        let physical_devices = unsafe { instance.enumerate_physical_devices() }?;
        
        println!("\nFound {} physical device(s):", physical_devices.len());
        println!("{}", "=".repeat(50));
        
        let mut graphics_device = None;
        
        for (index, device) in physical_devices.iter().enumerate() {
            // Get device properties
            let properties = unsafe { instance.get_physical_device_properties(*device) };
            
            // Get device features
            let features = unsafe { instance.get_physical_device_features(*device) };
            
            // Get memory properties
            let memory_props = unsafe { instance.get_physical_device_memory_properties(*device) };
            
            // Get queue family properties
            let queue_families = unsafe { instance.get_physical_device_queue_family_properties(*device) };
            
            println!("\nDevice #{}: {}", index, device_name_from_properties(&properties));
            println!("  Device Type: {:?}", properties.device_type);
            println!("  API Version: {}.{}.{}", 
                vk::version_major(properties.api_version),
                vk::version_minor(properties.api_version),
                vk::version_patch(properties.api_version)
            );
            println!("  Driver Version: {}", properties.driver_version);
            println!("  Vendor ID: 0x{:X}", properties.vendor_id);
            println!("  Device ID: 0x{:X}", properties.device_id);
            
            // Print memory information
            println!("  Memory Heaps: {}", memory_props.memory_heap_count);
            for i in 0..memory_props.memory_heap_count {
                let heap = &memory_props.memory_heaps[i as usize];
                println!("    Heap {}: {} MB (flags: {:?})", 
                    i, 
                    heap.size / (1024 * 1024),
                    heap.flags
                );
            }
            
            // Print queue families
            println!("  Queue Families: {}", queue_families.len());
            let mut has_graphics = false;
            for (qf_index, queue_family) in queue_families.iter().enumerate() {
                println!("    Queue Family {}: {} queues (flags: {:?})", 
                    qf_index,
                    queue_family.queue_count,
                    queue_family.queue_flags
                );
                
                if let Ok(_) = unsafe { check_physical_device(instance, surface, device) } {
                    has_graphics = true;
                }
            }
            
            // If this device supports graphics and we haven't found one yet, save it
            if has_graphics && graphics_device.is_none() {
                graphics_device = Some(*device);
                println!("  → Selected as graphics device");
            }
            
            // Print some key features
            println!("  Key Features:");
            println!("    Geometry Shader: {}", features.geometry_shader != 0);
            println!("    Tessellation Shader: {}", features.tessellation_shader != 0);
            println!("    Multi Viewport: {}", features.multi_viewport != 0);
            println!("    Anisotropic Filtering: {}", features.sampler_anisotropy != 0);
        }
        
        if let Some(device) = graphics_device {
            println!("\nSelected graphics device: {}", device_name_from_properties(
                &unsafe { instance.get_physical_device_properties(device) }
            ));
        } else {
            println!("\nNo graphics-capable device found!");
        }
        
        Ok(graphics_device)
    }

    fn create_logical_device(instance: &Instance, state: &mut VulkanState) -> Result<Device, Box<dyn std::error::Error>> {
        let indices = unsafe { QueueFamilyIndices::get(instance, &state.surface, &state.physical_device)? };
        
        let mut unique_indices = HashSet::new();
        unique_indices.insert(indices.graphics);
        unique_indices.insert(indices.present);
        
        let queue_priorities = &[1.0];
        let queue_infos = unique_indices
            .iter()
            .map(|i| {
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(*i)
                    .queue_priorities(queue_priorities)
            })
            .collect::<Vec<_>>();
        
        // Use default features
        let device_features = vk::PhysicalDeviceFeatures::default();
        
        // Declare empty vectors for layers and extensions
        let layers: Vec<*const i8> = vec![];
        let extensions = DEVICE_EXTENSIONS
            .iter()
            .map(|v| v.as_ptr())
            .collect::<Vec<_>>();
        
        // Create logical device
        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_infos)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions)
            .enabled_features(&device_features);
        
        let device = unsafe { instance.create_device(state.physical_device, &device_create_info, None) }?;
        
        // Get queue handle from the device
        let graphics_queue = unsafe { device.get_device_queue(indices.graphics, 0) };
        let present_queue = unsafe { device.get_device_queue(indices.present, 0) };
        
        state.graphics_queue = graphics_queue;
        state.present_queue = present_queue;

        println!("Logical device created successfully");
        
        // Return both device and queue
        Ok(device)
    }

    unsafe fn create_swapchain(
        window: &Window,
        instance: &Instance,
        device: &Device,
        state: &mut VulkanState,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let indices = QueueFamilyIndices::get(instance, &state.surface, &state.physical_device)?;
        let support = SwapchainSupport::get(instance, &state.surface, &state.physical_device)?;

        let surface_format = get_swapchain_surface_format(&support.formats);
        let present_mode = get_swapchain_present_mode(&support.present_modes);
        let extent = get_swapchain_extent(window, support.capabilities);

        let mut image_count = support.capabilities.min_image_count + 1;
        if support.capabilities.max_image_count != 0 && image_count > support.capabilities.max_image_count
        {
            image_count = support.capabilities.max_image_count;
        }

        let mut queue_family_indices = vec![];
        let image_sharing_mode = if indices.graphics != indices.present {
            queue_family_indices.push(indices.graphics);
            queue_family_indices.push(indices.present);
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        };

        let info: vk::SwapchainCreateInfoKHRBuilder<'_> = vk::SwapchainCreateInfoKHR::builder()
            .surface(state.surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(image_sharing_mode)
            .queue_family_indices(&queue_family_indices)
            .pre_transform(support.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null());

        let swapchain = device.create_swapchain_khr(&info, None)?;
        let images = device.get_swapchain_images_khr(swapchain)?;

        state.swapchain_format = surface_format.format;
        state.swapchain_extent = extent;
        state.swapchain_images = images;
        state.swapchain = swapchain;

        println!("Created swapchain.");

        Ok(())
    }

    unsafe fn create_swapchain_image_views(device: &Device, state: &mut VulkanState) -> Result<(), Box<dyn std::error::Error>> {
        let views = state.swapchain_images
            .iter()
            .map(|i| {
                let components = vk::ComponentMapping::builder()
                    .r(vk::ComponentSwizzle::IDENTITY)
                    .g(vk::ComponentSwizzle::IDENTITY)
                    .b(vk::ComponentSwizzle::IDENTITY)
                    .a(vk::ComponentSwizzle::IDENTITY);

                let subresource_range = vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1);

                let info = vk::ImageViewCreateInfo::builder()
                    .image(*i)
                    .view_type(vk::ImageViewType::_2D)
                    .format(state.swapchain_format)
                    .components(components)
                    .subresource_range(subresource_range);

                device.create_image_view(&info, None)
            })
            .collect::<Result<Vec<_>, _>>()?;

        state.swapchain_image_views = views;

        println!("Created swapchain image views.");
        Ok(())
    }

    unsafe fn create_pipeline(device: &Device, state: &mut VulkanState) -> Result<(), Box<dyn std::error::Error>> {
        let vert = include_bytes!("../shaders/vert.spv");
        let frag = include_bytes!("../shaders/frag.spv");

        let vert_shader_module = VulkanApp::create_shader_module(device, &vert[..])?;
        let frag_shader_module = VulkanApp::create_shader_module(device, &frag[..])?;

        let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_shader_module)
            .name(b"main\0");

        let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_shader_module)
            .name(b"main\0");

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder();

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport = vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(state.swapchain_extent.width as f32)
            .height(state.swapchain_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        let scissor = vk::Rect2D::builder()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(state.swapchain_extent);

        let viewports = &[viewport];
        let scissors = &[scissor];
        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(viewports)
            .scissors(scissors);

        // Rasterization State

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false);

        // Multisample State

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::_1);

        // Color Blend State

        let attachment = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::all())
            .blend_enable(false);

        let attachments = &[attachment];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(attachments)
            .blend_constants([0.0, 0.0, 0.0, 0.0]);

        // Layout

        let layout_info = vk::PipelineLayoutCreateInfo::builder();

        let pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;

        println!("Created pipeline layout.");

        let stages = &[vert_stage, frag_stage];
        let info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&color_blend_state)
            .layout(pipeline_layout)
            .render_pass(state.render_pass)
            .subpass(0);

        let pipeline = device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)?.0[0];
        
        device.destroy_shader_module(vert_shader_module, None);
        device.destroy_shader_module(frag_shader_module, None);

        state.pipeline_layout = pipeline_layout;
        state.pipeline = pipeline;

        println!("Created pipeline.");

        Ok(())
    }

    unsafe fn create_render_pass(device: &Device, state: &mut VulkanState) -> Result<(), Box<dyn std::error::Error>> {
        let dependency = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
        
        let color_attachment = vk::AttachmentDescription::builder()
            .format(state.swapchain_format)
            .samples(vk::SampleCountFlags::_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let color_attachment_ref = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        
        let color_attachments = &[color_attachment_ref];
        let subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(color_attachments);
        let attachments = &[color_attachment];
        let subpasses = &[subpass];
        let dependencies = &[dependency];
        
        let info = vk::RenderPassCreateInfo::builder()
            .attachments(attachments)
            .subpasses(subpasses)
            .dependencies(dependencies);
        let render_pass = device.create_render_pass(&info, None)?;

        state.render_pass = render_pass;
        Ok(())
    }

    unsafe fn create_framebuffers(device: &Device, state: &mut VulkanState) -> Result<(), Box<dyn std::error::Error>> {
        let framebuffers = state.swapchain_image_views
            .iter()
            .map(|i| {
                let attachments = &[*i];
                let create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(state.render_pass)
                    .attachments(attachments)
                    .width(state.swapchain_extent.width)
                    .height(state.swapchain_extent.height)
                    .layers(1);
                device.create_framebuffer(&create_info, None)
            })
            .collect::<Result<Vec<_>, _>>()?;
        
        state.framebuffers = framebuffers;

        println!("Created framebuffers.");

        Ok(())
    }

    unsafe fn create_command_pool(
        instance: &Instance,
        device: &Device,
        surface: &vk::SurfaceKHR,
        physical_device: &vk::PhysicalDevice
    ) -> Result<vk::CommandPool, Box<dyn std::error::Error>> {
        let indices = QueueFamilyIndices::get(instance, surface, physical_device)?;

        let info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::empty())
            .queue_family_index(indices.graphics);

        let command_pool = device.create_command_pool(&info, None)?;

        println!("Created command pool.");

        Ok(command_pool)
    }

    unsafe fn create_command_buffers(device: &Device, state: &mut VulkanState) -> Result<(), Box<dyn std::error::Error>> {
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(state.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(state.framebuffers.len() as u32);

        let command_buffers =  device.allocate_command_buffers(&allocate_info)?;

        for (i, command_buffer) in command_buffers.iter().enumerate() {
            let info = vk::CommandBufferBeginInfo::builder();

            device.begin_command_buffer(*command_buffer, &info)?;

            let render_area = vk::Rect2D::builder()
                .offset(vk::Offset2D::default())
                .extent(state.swapchain_extent);

            let color_clear_value = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            };

            let clear_values = &[color_clear_value];
            let info = vk::RenderPassBeginInfo::builder()
                .render_pass(state.render_pass)
                .framebuffer(state.framebuffers[i])
                .render_area(render_area)
                .clear_values(clear_values);

            device.cmd_begin_render_pass(*command_buffer, &info, vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(*command_buffer, vk::PipelineBindPoint::GRAPHICS, state.pipeline);
            device.cmd_draw(*command_buffer, 3, 1, 0, 0);
            device.cmd_end_render_pass(*command_buffer);
            device.end_command_buffer(*command_buffer)?;
        }

        state.command_buffers = command_buffers;

        println!("Created command buffers.");

        Ok(())
    }

    unsafe fn create_sync_objects(device: &Device, state: &mut VulkanState) -> Result<(), Box<dyn std::error::Error>> {
        let semaphore_info = vk::SemaphoreCreateInfo::builder();
        let fence_info = vk::FenceCreateInfo::builder()
            .flags(vk::FenceCreateFlags::SIGNALED);

        let mut image_available_semaphores = vec![];
        let mut render_finished_semaphores = vec![];
        let mut fences = vec![];
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            image_available_semaphores.push(device.create_semaphore(&semaphore_info, None)?);
            render_finished_semaphores.push(device.create_semaphore(&semaphore_info, None)?);
            fences.push(device.create_fence(&fence_info, None)?)
        }

        let images_in_flight = state.swapchain_images
            .iter()
            .map(|_| vk::Fence::null())
            .collect();
        
        state.image_available_semaphores = image_available_semaphores;
        state.render_finished_semaphores = render_finished_semaphores;
        state.in_flight_fences = fences;
        state.images_in_flight = images_in_flight;

        Ok(())
    }

    unsafe fn create_shader_module(device: &Device, bytecode: &[u8]) -> Result<vk::ShaderModule, Box<dyn std::error::Error>> {
        let bytecode = Bytecode::new(bytecode)?;
        let info = vk::ShaderModuleCreateInfo::builder()
            .code_size(bytecode.code_size())
            .code(bytecode.code());

        Ok(device.create_shader_module(&info, None)?)
    }

        fn cleanup_vulkan(&mut self) {
        unsafe { self.cleanup_vulkan_unsafe() };
    }

    unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<(), Box<dyn std::error::Error>> {
        println!("Recreating swapchain...");

        self.device.device_wait_idle()?;
        self.destroy_swapchain();

        VulkanApp::create_swapchain(window, &self.instance, &self.device, &mut self.state)?;
        VulkanApp::create_swapchain_image_views(&self.device, &mut self.state)?;
        VulkanApp::create_render_pass(&self.device, &mut self.state)?;
        VulkanApp::create_pipeline(&self.device, &mut self.state)?;
        VulkanApp::create_framebuffers(&self.device, &mut self.state)?;
        VulkanApp::create_command_buffers(&self.device, &mut self.state)?;
        self.state
            .images_in_flight
            .resize(self.state.swapchain_images.len(), vk::Fence::null());
        Ok(())
    }

    unsafe fn destroy_swapchain(&mut self) {
        let st = &self.state;
        let device = &self.device;

        device.free_command_buffers(st.command_pool, &st.command_buffers);
        println!("Freed command buffers.");

        st.framebuffers
            .iter()
            .for_each(|f| device.destroy_framebuffer(*f, None));
        println!("Framebuffers destroyed.");

        device.destroy_pipeline(st.pipeline, None);
        println!("Pipeline destroyed");

        device.destroy_pipeline_layout(st.pipeline_layout, None);
        println!("Pipeline layout destroyed.");

        device.destroy_render_pass(st.render_pass, None);
        println!("Render pass destroyed.");

        st.swapchain_image_views
            .iter()
            .for_each(|v| device.destroy_image_view(*v, None));
        println!("Swapchain image views destroyed.");

        device.destroy_swapchain_khr(st.swapchain, None);
        println!("Swapchain destroyed");
    }

    unsafe fn cleanup_vulkan_unsafe(&mut self) {
        self.destroy_swapchain();

        let st = &self.state;
        let device = &self.device;
        let instance = &self.instance;

        st.image_available_semaphores.iter()
            .for_each(|s|device.destroy_semaphore(*s, None));
        st.render_finished_semaphores.iter()
            .for_each(|s|device.destroy_semaphore(*s, None));
        
        println!("Semaphores destroyed");

        st.in_flight_fences.iter()
            .for_each(|v| device.destroy_fence(*v, None));

        println!("Fences destroyed.");
        
        device.destroy_command_pool(st.command_pool, None);
        println!("Command pool destroyed.");
        
        device.destroy_device(None);
        println!("Vulkan device destroyed.");
        instance.destroy_surface_khr(st.surface, None);
        println!("Surface destoyed.");

        if VALIDATION_ENABLED {
            instance.destroy_debug_utils_messenger_ext(st.messenger, None);
        }

        instance.destroy_instance(None);
        println!("Vulkan instance destroyed.");
    }

    fn render(&mut self, window: &Window) -> Result<(), Box<dyn std::error::Error>> {
        unsafe { self.render_unsafe(window) }
    }

    unsafe fn render_unsafe(&mut self, window: &Window) -> Result<(), Box<dyn std::error::Error>> {
        self.device.wait_for_fences(
            &[self.state.in_flight_fences[self.state.frame]],
            true,
        u64::MAX)?;

        let result = self
            .device
            .acquire_next_image_khr(
                self.state.swapchain,
                u64::MAX,
                self.state.image_available_semaphores[self.state.frame],
                vk::Fence::null(),
            );
        
        let image_index = match result {
            Ok((image_index, _)) => image_index as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
            Err(e) => return Err(e.into()),
        };

        if !self.state.images_in_flight[image_index as usize].is_null() {
            self.device.wait_for_fences(
                &[self.state.images_in_flight[image_index as usize]],
                true,
                u64::MAX,
            )?;
        }

        self.state.images_in_flight[image_index as usize] = self.state.in_flight_fences[self.state.frame];

        let wait_semaphores = &[self.state.image_available_semaphores[self.state.frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.state.command_buffers[image_index as usize]];
        let signal_semaphores = &[self.state.render_finished_semaphores[self.state.frame]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        self.device.reset_fences(&[self.state.in_flight_fences[self.state.frame]])?;

        self.device.queue_submit(
            self.state.graphics_queue, 
            &[submit_info],
            self.state.in_flight_fences[self.state.frame]
        )?;

        let swapchains = &[self.state.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let result = self.device.queue_present_khr(self.state.present_queue, &present_info);

        let changed = result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR) || result == Err(vk::ErrorCode::OUT_OF_DATE_KHR);

        if self.state.resized || changed {
            self.state.resized = false;
            self.recreate_swapchain(window)?;
        } else if let Err(e) = result {
            return Err(e.into());
        }

        self.state.frame = (self.state.frame + 1) % MAX_FRAMES_IN_FLIGHT;

        println!("Frame rendered.");

        Ok(())
    }

    fn wait_idle(&self) -> Result<(), Box<dyn std::error::Error>> {
        unsafe { self.device.device_wait_idle() }?;
        Ok(())
    }
}




extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    type_: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    let data = unsafe { *data };
    let message = unsafe { CStr::from_ptr(data.message) }.to_string_lossy();

    if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        println!("({:?}) {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        println!("({:?}) {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        println!("({:?}) {}", type_, message);
    } else {
        println!("({:?}) {}", type_, message);
    }

    vk::FALSE
}


impl ApplicationHandler for BusyDeckApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window_attributes = Window::default_attributes()
                .with_title("BusyDeck - Vulkan Window")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 800))
                .with_resizable(true);
            
            match event_loop.create_window(window_attributes) {
                Ok(window) => {
                    println!("Window created successfully: {}x{}", 1280, 800);

                    let vulkan_app = VulkanApp::new(&window);
                    match vulkan_app {
                        Ok(app) => {
                            self.vulkan_app = Some(app);
                        },
                        Err(e) => {
                            eprintln!("Failed to initialize Vulkan: {}", e);
                            event_loop.exit();
                        }
                    }

                    self.window = Some(window);
                }
                Err(e) => {
                    eprintln!("Failed to create window: {}", e);
                    event_loop.exit();
                    return;
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                println!("Window close requested, exiting...");
                if let Some(app) = self.vulkan_app.as_mut() {
                    app.wait_idle().unwrap();
                    app.cleanup_vulkan();
                }
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                // Handle redraw if needed
                if let Some((window, app)) = self.window.as_ref().zip(self.vulkan_app.as_mut()) {
                    if !event_loop.exiting() && !self.minimized {
                        app.render(&window).unwrap();
                    }
                    window.request_redraw();
                }
            }
            WindowEvent::Resized(size) => {
                if size.width == 0 || size.height == 0 {
                    self.minimized = true;
                } else {
                    self.minimized = false;
                    if let Some(app) = self.vulkan_app.as_mut() {
                        app.state.resized = true;
                    }
                }
                println!("Minimized: {}", self.minimized);
                
            },
            _ => {}
        }
    }
}

unsafe fn check_physical_device(
    instance: &Instance,
    surface: &vk::SurfaceKHR,
    physical_device: &vk::PhysicalDevice,
) -> Result<(), Box<dyn std::error::Error>> {
    QueueFamilyIndices::get(instance, surface, physical_device)?;
    check_physical_device_extensions(instance, physical_device)?;

    let support = SwapchainSupport::get(instance, surface, physical_device)?;
    if support.formats.is_empty() || support.present_modes.is_empty() {
        return Err("Insufficient swapchain support.".into());
    }
    Ok(())
}

unsafe fn check_physical_device_extensions(
    instance: &Instance,
    physical_device: &vk::PhysicalDevice,
) -> Result<(), Box<dyn std::error::Error>> {
    let extensions = instance
        .enumerate_device_extension_properties(*physical_device, None)?
        .iter()
        .map(|e| e.extension_name)
        .collect::<HashSet<_>>();
    if DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(e)) {
        Ok(())
    } else {
        Err("Missing required device extensions.".into())
    }
}

fn get_swapchain_surface_format(
    formats: &[vk::SurfaceFormatKHR],
) -> vk::SurfaceFormatKHR {
    formats
        .iter()
        .cloned()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or_else(|| formats[0])
}

fn get_swapchain_present_mode(
    present_modes: &[vk::PresentModeKHR],
) -> vk::PresentModeKHR {
    present_modes
        .iter()
        .cloned()
        .find(|m| *m == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO)
}

fn get_swapchain_extent(
    window: &Window,
    capabilities: vk::SurfaceCapabilitiesKHR,
) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        vk::Extent2D::builder()
            .width(window.inner_size().width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ))
            .height(window.inner_size().height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ))
            .build()
    }
}

#[derive(Copy, Clone, Debug)]
struct QueueFamilyIndices {
    graphics: u32,
    present: u32,
}

impl QueueFamilyIndices {
    unsafe fn get(instance: &Instance, surface: &vk::SurfaceKHR, physical_device: &vk::PhysicalDevice) -> Result<Self, Box<dyn std::error::Error>> {
        let properties = instance.get_physical_device_queue_family_properties(*physical_device);

        let graphics = properties
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|i| i as u32);

        let mut present = None;
        for (index, properties) in properties.iter().enumerate() {
            if instance.get_physical_device_surface_support_khr(*physical_device, index as u32, *surface)? {
                present = Some(index as u32);
                break;
            }
        }

        if let (Some(graphics), Some(present)) = (graphics, present) {
            Ok(Self { graphics, present })
        } else {
            Err("Missing required queue families.".into())
        }
    }
}

#[derive(Clone, Debug)]
struct SwapchainSupport {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    unsafe fn get(
        instance: &Instance,
        surface: &vk::SurfaceKHR,
        physical_device: &vk::PhysicalDevice,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            capabilities: instance
                .get_physical_device_surface_capabilities_khr(
                    *physical_device, *surface)?,
            formats: instance
                .get_physical_device_surface_formats_khr(
                    *physical_device, *surface)?,
            present_modes: instance
                .get_physical_device_surface_present_modes_khr(
                    *physical_device, *surface)?,
        })
    }
}
