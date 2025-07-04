use std::collections::HashSet;
use std::ffi::CStr;
use std::mem::size_of;
use std::os::raw::c_void;
use std::{u64};
use std::ptr::copy_nonoverlapping as memcpy;
use std::time::Instant;

use cgmath::{vec2, vec3};

use vulkanalia::bytecode::Bytecode;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::vk::{ExtDebugUtilsExtension, KhrSurfaceExtension, KhrSwapchainExtension, Pipeline, PipelineLayout, RenderPass};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use crate::source::{DisplayDataSource, LocalStatsDisplayDataSource, WebDisplayDataSource};

mod font;
mod source;

/// Whether the validation layers should be enabled.
const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
/// The name of the validation layers.
const VALIDATION_LAYER: vk::ExtensionName = vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];

const MAX_FRAMES_IN_FLIGHT: usize = 2;

const DISPLAY_MATRIX_SIZE: (u16, u16) = (72, 26);
const DISPLAY_FILL_FACTOR: f32 = 0.95;
const DISPLAY_MARGIN: f32 = 0.01;

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
    
    // FPS tracking
    fps_counter: u32,
    fps_start_time: Instant,
    
    // Display data source
    display_data_source: Box<dyn DisplayDataSource>,
}

impl BusyDeckApp {
    fn new() -> Self {
        // Choose display data source:
        // 1. LocalStatsDisplayDataSource - shows CPU/Memory stats
        // 2. WebDisplayDataSource - displays lines received via HTTP API
        // let display_data_source = Box::new(LocalStatsDisplayDataSource::new());
        let display_data_source = Box::new(WebDisplayDataSource::new());
        
        BusyDeckApp { 
            window: None, 
            vulkan_app: None, 
            minimized: false,
            fps_counter: 0,
            fps_start_time: Instant::now(),
            display_data_source,
        }
    }
}

type Vec2 = cgmath::Vector2<f32>;
type Vec3 = cgmath::Vector3<f32>;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    pos: Vec2,
    color: Vec3,
}

impl Vertex {
    const fn new(pos: Vec2, color: Vec3) -> Self {
        Self { pos, color }
    }

    fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }

    fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        let pos = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(0)
            .build();

        let color = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(size_of::<Vec2>() as u32)
            .build();

        [pos, color]
    }
}

fn generate_matrix(
    matrix_size: (u16, u16), // (columns, rows)
    margin: f32,
    fill_factor: f32,
    state: &VulkanState,
) -> (Vec<Vertex>, Vec<u16>) {
    let (cols, rows) = matrix_size;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    
    // Calculate square size based on fill factor
    // Total width should be fill_factor * 2.0 (since position ranges from -1.0 to 1.0)
    let total_width = fill_factor * 2.0;
    let square_size = (total_width - (cols - 1) as f32 * margin) / cols as f32;
    
    // Calculate total height maintaining square aspect ratio
    let total_height = square_size * rows as f32 + (rows - 1) as f32 * margin;
    
    // Starting positions to center the matrix
    let start_x = -total_width / 2.0;
    let start_y = -total_height / 2.0;
    
    let red_color = vec3(1.0, 0.0, 0.0);
    let black_color = vec3(0.0, 0.0, 0.0);
    
    // Get font data and display text
    let font_data = font::build_font_2();
    let line1 = &state.display_line1;
    let line2 = &state.display_line2;
    let char_width = 5;
    let char_height = 5;
    let char_spacing = 1; // Space between characters
    let line_spacing = 1; // Space between lines
    
    // Generate vertices for each square in the matrix
    for row in 0..rows {
        for col in 0..cols {
            let x = start_x + col as f32 * (square_size + margin);
            let y = start_y + row as f32 * (square_size + margin);
            
            // Determine if this pixel should be red (text) or black (background)
            let mut pixel_color = black_color;
            
            // Check if this position is within the text area
            let char_index = col as usize / (char_width + char_spacing);
            let char_x = col as usize % (char_width + char_spacing);
            let char_y = row as usize;
            
            // Determine which line we're on and adjust for line spacing
            let line_height = char_height + line_spacing;
            let current_line = char_y / line_height;
            let char_y_in_line = char_y % line_height;
            
            let text_to_use = if current_line == 0 {
                line1
            } else if current_line == 1 {
                line2
            } else {
                ""
            };
            
            if char_index < text_to_use.len() && char_x < char_width && char_y_in_line < char_height {
                if let Some(current_char) = text_to_use.chars().nth(char_index) {
                    if let Some(char_bitmap) = font_data.get(&current_char) {
                        if char_bitmap[char_y_in_line][char_x] == 1 {
                            pixel_color = red_color;
                        }
                    }
                }
            }
            
            let base_vertex = vertices.len() as u16;
            
            // Create 4 vertices for each square (bottom-left, bottom-right, top-right, top-left)
            vertices.push(Vertex::new(vec2(x, y - square_size), pixel_color)); // bottom-left
            vertices.push(Vertex::new(vec2(x + square_size, y - square_size), pixel_color)); // bottom-right
            vertices.push(Vertex::new(vec2(x + square_size, y), pixel_color)); // top-right
            vertices.push(Vertex::new(vec2(x, y), pixel_color)); // top-left
            
            // Create indices for two triangles that form the square
            indices.extend_from_slice(&[
                base_vertex, base_vertex + 1, base_vertex + 2, // first triangle
                base_vertex + 2, base_vertex + 3, base_vertex   // second triangle
            ]);
        }
    }
    
    (vertices, indices)
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

    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,

    frame: usize,
    resized: bool,
    
    // Display text
    display_line1: String,
    display_line2: String,
}

struct VulkanApp {
    #[allow(dead_code)]
    entry: Entry,
    instance: Instance,
    device: Device,
    state: VulkanState,

    display_matrix_vertices: Vec<Vertex>,
    display_matrix_indices: Vec<u16>,
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

        let (vertices, indices) = generate_matrix(
            DISPLAY_MATRIX_SIZE,
            DISPLAY_MARGIN,
            DISPLAY_FILL_FACTOR,
            &state
        );

        VulkanApp::create_vertex_buffer(&instance, &device, &mut state, &vertices)?;
        VulkanApp::create_index_buffer(&instance, &device, &mut state, &indices)?;
        VulkanApp::create_command_buffers(&device, &mut state, indices.len() as u32)?;
        VulkanApp::create_sync_objects(&device, &mut state)?;
        
        println!("Created all Vulkan objects.");

        Ok(Self {
            entry,
            instance,
            device,
            state,
            display_matrix_vertices: vertices,
            display_matrix_indices: indices,
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

        let binding_descriptions = &[Vertex::binding_description()];
        let attribute_descriptions = Vertex::attribute_descriptions();
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

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

    unsafe fn create_vertex_buffer(
        instance: &Instance,
        device: &Device,
        state: &mut VulkanState,
        vertex_data: &Vec<Vertex>,
    ) -> Result<(), Box<dyn std::error::Error>> {

        println!("Creating vertex buffer...");

        let size = (size_of::<Vertex>() * vertex_data.len()) as u64;

        let (vertex_buffer, vertex_buffer_memory) = VulkanApp::create_buffer(
            instance,
            device,
            state,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        )?;

        state.vertex_buffer = vertex_buffer;
        state.vertex_buffer_memory = vertex_buffer_memory;

        VulkanApp::upload_to_device_buffer(instance, device, state, vertex_data, vertex_buffer)?;

        Ok(())
    }

    unsafe fn create_index_buffer(
        instance: &Instance,
        device: &Device,
        state: &mut VulkanState,
        data: &Vec<u16>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let size = (size_of::<u16>() * data.len()) as u64;

        let (index_buffer, index_buffer_memory) = VulkanApp::create_buffer(
            instance,
            device,
            state,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        state.index_buffer = index_buffer;
        state.index_buffer_memory = index_buffer_memory;

        VulkanApp::upload_to_device_buffer(instance, device, state, data, index_buffer)?;

        Ok(())
    }

    unsafe fn create_command_buffers(device: &Device, state: &mut VulkanState, index_count: u32) -> Result<(), Box<dyn std::error::Error>> {
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
            device.cmd_bind_vertex_buffers(*command_buffer, 0, &[state.vertex_buffer], &[0]);
            device.cmd_bind_index_buffer(*command_buffer, state.index_buffer, 0, vk::IndexType::UINT16);
            device.cmd_draw_indexed(*command_buffer, index_count, 1, 0, 0, 0);
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
        VulkanApp::create_command_buffers(&self.device, &mut self.state, self.display_matrix_indices.len() as u32)?;
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

        device.destroy_buffer(st.index_buffer, None);
        device.free_memory(st.index_buffer_memory, None);

        device.destroy_buffer(st.vertex_buffer, None);
        device.free_memory(st.vertex_buffer_memory, None);

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

        let (vertices, indices) = generate_matrix(DISPLAY_MATRIX_SIZE, DISPLAY_MARGIN, DISPLAY_FILL_FACTOR, &self.state);
        self.display_matrix_vertices = vertices;
        self.display_matrix_indices = indices;

        VulkanApp::upload_to_device_buffer(&self.instance, &self.device, &self.state, &self.display_matrix_vertices, self.state.vertex_buffer)?;
        VulkanApp::upload_to_device_buffer(&self.instance, &self.device, &self.state, &self.display_matrix_indices, self.state.index_buffer)?;

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

        Ok(())
    }

    fn wait_idle(&self) -> Result<(), Box<dyn std::error::Error>> {
        unsafe { self.device.device_wait_idle() }?;
        Ok(())
    }

    unsafe fn get_memory_type_index(
        instance: &Instance,
        state: &VulkanState,
        properties: vk::MemoryPropertyFlags,
        requirements: vk::MemoryRequirements,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        let memory = instance.get_physical_device_memory_properties(state.physical_device);
        (0..memory.memory_type_count)
            .find(|i| {
                let suitable = (requirements.memory_type_bits & (1 << i)) != 0;
                let memory_type = memory.memory_types[*i as usize];
                suitable && memory_type.property_flags.contains(properties)
            })
            .ok_or_else(|| "Failed to find suitable memory type.".into())
    }

    unsafe fn create_buffer(
        instance: &Instance,
        device: &Device,
        state: &VulkanState,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), Box<dyn std::error::Error>> {
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = device.create_buffer(&buffer_info, None)?;

        let requirements = device.get_buffer_memory_requirements(buffer);

        let memory_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(VulkanApp::get_memory_type_index(
                instance,
                state,
                properties,
                requirements,
            )?);

        let buffer_memory = device.allocate_memory(&memory_info, None)?;

        device.bind_buffer_memory(buffer, buffer_memory, 0)?;

        Ok((buffer, buffer_memory))
    }

    unsafe fn copy_buffer(
        device: &Device,
        state: &VulkanState,
        source: vk::Buffer,
        destination: vk::Buffer,
        size: vk::DeviceSize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(state.command_pool)
        .command_buffer_count(1);

        let command_buffer = device.allocate_command_buffers(&info)?[0];

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device.begin_command_buffer(command_buffer, &info)?;

        let regions = vk::BufferCopy::builder().size(size);
        device.cmd_copy_buffer(command_buffer, source, destination, &[regions]);

        device.end_command_buffer(command_buffer)?;

        let command_buffers = &[command_buffer];
        let info = vk::SubmitInfo::builder()
            .command_buffers(command_buffers);

        device.queue_submit(state.graphics_queue, &[info], vk::Fence::null())?;
        device.queue_wait_idle(state.graphics_queue)?;

        device.free_command_buffers(state.command_pool, &[command_buffer]);

        Ok(())
    }

    unsafe fn upload_to_device_buffer<T>(instance: &Instance, device: &Device, state: &VulkanState, data: &Vec<T>, device_buffer: vk::Buffer) -> Result<(), Box<dyn std::error::Error>> {
        let size = (size_of::<T>() * data.len()) as u64;
        let (staging_buffer, staging_buffer_memory) = VulkanApp::create_buffer(
            instance,
            device,
            state,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;
        let memory = device.map_memory(
            staging_buffer_memory,
            0,
            size,
            vk::MemoryMapFlags::empty(),
        )?;
        memcpy(data.as_ptr(), memory.cast(), data.len());
        device.unmap_memory(staging_buffer_memory);
        VulkanApp::copy_buffer(device, state, staging_buffer, device_buffer, size)?;
        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);
        Ok(())
    }

    fn set_display_text(&mut self, line1: &str, line2: &str) {
        self.state.display_line1 = line1.to_string();
        self.state.display_line2 = line2.to_string();
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
                // Increment FPS counter
                self.fps_counter += 1;

                // Check if 1 second has elapsed and calculate FPS
                let elapsed = self.fps_start_time.elapsed();
                if elapsed.as_secs_f32() >= 1.0 {
                    let fps = self.fps_counter as f32 / elapsed.as_secs_f32();
                    println!("FPS: {:.2}", fps);
                    
                    // Reset counter and start time
                    self.fps_counter = 0;
                    self.fps_start_time = Instant::now();
                }

                // Handle redraw if needed
                if let Some((window, app)) = self.window.as_ref().zip(self.vulkan_app.as_mut()) {
                    if !event_loop.exiting() && !self.minimized {
                        let lines = self.display_data_source.get_lines();
                        let line1 = lines.get(0).map(|s| s.as_str()).unwrap_or("");
                        let line2 = lines.get(1).map(|s| s.as_str()).unwrap_or("");
                        app.set_display_text(line1, line2);
                        
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
        for (index, _) in properties.iter().enumerate() {
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
