use vulkanalia::prelude::v1_0::*;
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

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
    entry: Option<Entry>,
    instance: Option<Instance>,
    physical_device: Option<vk::PhysicalDevice>

}

impl BusyDeckApp {
    fn new() -> Self {
        Self { window: None, entry: None, instance: None, physical_device: None }
    }
}

impl BusyDeckApp {
    fn init_vulkan(&mut self) -> Result<(Entry, Instance), Box<dyn std::error::Error>> {
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
        let instance = self.create_instance(&entry)?;

        Ok((entry, instance))
    }
    
    fn create_instance(&self, entry: &Entry) -> Result<Instance, Box<dyn std::error::Error>> {
        let app_info = vk::ApplicationInfo::builder()
            .application_name(b"BusyDeck\0")
            .application_version(vk::make_version(1, 0, 0))
            .engine_name(b"No Engine\0")
            .engine_version(vk::make_version(1, 0, 0))
            .api_version(vk::make_version(1, 0, 0));
        
        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info);
        
        let instance = unsafe { entry.create_instance(&create_info, None) }?;
        
        println!("Vulkan instance created successfully!");
        Ok(instance)
    }

    fn create_physical_device(&self, instance: &Instance) -> Result<vk::PhysicalDevice, Box<dyn std::error::Error>> {
        match self.query_and_print_devices(instance)? {
            Some(physical_device) => {
                Ok(physical_device)
            }
            None => {
                Err("No device found".into())
            }
        }
    }
    
    fn query_and_print_devices(&self, instance: &Instance) -> Result<Option<vk::PhysicalDevice>, Box<dyn std::error::Error>> {
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
                
                // Check if this queue family supports graphics
                if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
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
    
    fn cleanup_vulkan(&mut self) {
        if let Some(device) = &self.device {
            unsafe { device.destroy_device(None); }
            println!("Vulkan device destroyed.")
        }
        if let Some(instance) = &self.instance {
            unsafe { instance.destroy_instance(None) };
            println!("Vulkan instance destroyed.");
        }

        self.device = None;
        self.physical_device = None;
        self.instance = None;
        self.entry = None;
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
                    self.window = Some(window);
                }
                Err(e) => {
                    eprintln!("Failed to create window: {}", e);
                    event_loop.exit();
                }
            }

            match self.init_vulkan() {
                Ok((entry, instance)) => {
                    self.entry = Some(entry);
                    self.instance = Some(instance);
                }
                Err(e) => {
                    eprintln!("Failed to init Vulkan: {}", e);
                    event_loop.exit();
                }
            }

            match self.create_physical_device(self.instance.as_ref().unwrap()) {
                Ok(physical_device) => {
                    self.physical_device = Some(physical_device);
                }
                Err(e) => {
                    eprintln!("Failed to select physical device: {}", e);
                    event_loop.exit();
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
