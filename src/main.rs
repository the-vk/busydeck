use std::collections::HashSet;

use vulkanalia::prelude::v1_0::*;
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::vk::{ImageView, KhrSurfaceExtension, KhrSwapchainExtension};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];


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
    vulkan_state: Option<VulkanState>,
}

impl BusyDeckApp {
    fn new() -> Self {
        Self { 
            window: None,
            vulkan_state: None,
        }
    }
}

struct VulkanState {
    entry: Entry,
    instance: Instance,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    device: Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
}

impl BusyDeckApp {
    fn init_vulkan(&mut self, window: &Window) -> Result<(Entry, Instance), Box<dyn std::error::Error>> {
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
        let instance = self.create_instance(&entry, window)?;

        Ok((entry, instance))
    }
    
    fn create_instance(&self, entry: &Entry, window: &Window) -> Result<Instance, Box<dyn std::error::Error>> {
        let app_info = vk::ApplicationInfo::builder()
            .application_name(b"BusyDeck\0")
            .application_version(vk::make_version(1, 0, 0))
            .engine_name(b"No Engine\0")
            .engine_version(vk::make_version(1, 0, 0))
            .api_version(vk::make_version(1, 0, 0));
        
        let extensions = vulkanalia::window::get_required_instance_extensions(window)
            .iter()
            .map(|e| e.as_ptr())
            .collect::<Vec<_>>();

        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extensions);
        
        let instance = unsafe { entry.create_instance(&create_info, None) }?;
        
        println!("Vulkan instance created successfully!");
        Ok(instance)
    }

    fn create_physical_device(&self, instance: &Instance, surface: &vk::SurfaceKHR) -> Result<vk::PhysicalDevice, Box<dyn std::error::Error>> {
        match self.query_and_print_devices(instance, surface)? {
            Some(physical_device) => {
                Ok(physical_device)
            }
            None => {
                Err("No device found".into())
            }
        }
    }
    
    fn query_and_print_devices(&self, instance: &Instance, surface: &vk::SurfaceKHR) -> Result<Option<vk::PhysicalDevice>, Box<dyn std::error::Error>> {
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

    fn create_logical_device(&self, instance: &Instance, surface: &vk::SurfaceKHR, physical_device: &vk::PhysicalDevice) -> Result<(Device, vk::Queue, vk::Queue), Box<dyn std::error::Error>> {
        let indices = unsafe { QueueFamilyIndices::get(instance, surface, physical_device)? };
        
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
        
        let device = unsafe { instance.create_device(*physical_device, &device_create_info, None) }?;
        
        // Get queue handle from the device
        let graphics_queue = unsafe { device.get_device_queue(indices.graphics, 0) };
        let present_queue = unsafe { device.get_device_queue(indices.present, 0) };
        
        println!("Logical device created successfully");
        
        // Return both device and queue
        Ok((device, graphics_queue, present_queue))
    }

    unsafe fn create_swapchain(
        &self,
        window: &Window,
        instance: &Instance,
        physical_device: &vk::PhysicalDevice,
        device: &Device,
        surface: &vk::SurfaceKHR,
    ) -> Result<(
            vk::Format,
            vk::Extent2D,
            vk::SwapchainKHR,
            Vec<vk::Image>
        ), Box<dyn std::error::Error>> {
        let indices = QueueFamilyIndices::get(instance, surface, physical_device)?;
        let support = SwapchainSupport::get(instance, surface, physical_device)?;

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
            .surface(*surface)
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

        Ok((surface_format.format, extent, swapchain, images))
    }

    unsafe fn create_swapchain_image_views(&self, device: &Device, swapchain_format: &vk::Format, swapchain_images: &Vec<vk::Image>) -> Result<Vec<ImageView>, Box<dyn std::error::Error>> {
        let views = swapchain_images
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
                    .format(*swapchain_format)
                    .components(components)
                    .subresource_range(subresource_range);

                device.create_image_view(&info, None)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(views)
    }
    
    fn cleanup_vulkan(&mut self) {
        if let Some(vulkan_state) = &self.vulkan_state {
            unsafe {
                let views = &vulkan_state.swapchain_image_views;
                views
                    .iter()
                    .for_each(|v| vulkan_state.device.destroy_image_view(*v, None));
                println!("Swapchain image views destroyed.");
                vulkan_state.device.destroy_swapchain_khr(vulkan_state.swapchain, None);
                println!("Swapchain destroyed");
                vulkan_state.device.destroy_device(None);
                println!("Vulkan device destroyed.");
                vulkan_state.instance.destroy_surface_khr(vulkan_state.surface, None);
                println!("Surface destoyed.");
                vulkan_state.instance.destroy_instance(None);
                println!("Vulkan instance destroyed.");
            }
        }
        self.vulkan_state = None;
    }

    unsafe fn init_vulkan_pipeline(&mut self, window: &Window) -> Result<(), Box<dyn std::error::Error>> {
        let (entry, instance) = self.init_vulkan(window)?;
        let surface = vulkanalia::window::create_surface(&instance, &window, &window)?;
        let physical_device = self.create_physical_device(&instance, &surface)?;
        let (device, graphics_queue, present_queue) = self.create_logical_device(&instance, &surface, &physical_device)?;
        let (format, extent, swapchain, swapchain_images) = self.create_swapchain(window, &instance, &physical_device, &device, &surface)?;
        let swapchain_image_views = self.create_swapchain_image_views(&device, &format, &swapchain_images)?;

        self.vulkan_state = Some(VulkanState {
            entry,
            instance,
            surface,
            physical_device,
            device,
            graphics_queue,
            present_queue,
            swapchain_format: format,
            swapchain_extent: extent,
            swapchain,
            swapchain_images,
            swapchain_image_views,
        });

        Ok(())
    }
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

                if let Err(e) = unsafe { self.init_vulkan_pipeline(&window) } {
                    eprintln!("Failed to initialize Vulkan: {}", e);
                    event_loop.exit();
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
                self.cleanup_vulkan();
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                // Handle redraw if needed
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
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
