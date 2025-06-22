use vulkanalia::prelude::v1_0::*;
use vulkanalia::loader::{LibloadingLoader, LIBRARY};

fn main() {
    match run_app() {
        Ok(()) => println!("Vulkan initialization and device query completed successfully!"),
        Err(e) => eprintln!("Error: {}", e),
    }
}

fn run_app() -> Result<(), Box<dyn std::error::Error>> {
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
    let instance = create_instance(&entry)?;
    
    // Query and print physical devices
    query_and_print_devices(&instance)?;
    
    // Clean up
    unsafe { instance.destroy_instance(None) };
    
    Ok(())
}

fn create_instance(entry: &Entry) -> Result<Instance, Box<dyn std::error::Error>> {
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

fn query_and_print_devices(instance: &Instance) -> Result<(), Box<dyn std::error::Error>> {
    // Get all physical devices
    let physical_devices = unsafe { instance.enumerate_physical_devices() }?;
    
    println!("\nFound {} physical device(s):", physical_devices.len());
    println!("{}", "=".repeat(50));
    
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
        for (qf_index, queue_family) in queue_families.iter().enumerate() {
            println!("    Queue Family {}: {} queues (flags: {:?})", 
                qf_index,
                queue_family.queue_count,
                queue_family.queue_flags
            );
        }
        
        // Print some key features
        println!("  Key Features:");
        println!("    Geometry Shader: {}", features.geometry_shader != 0);
        println!("    Tessellation Shader: {}", features.tessellation_shader != 0);
        println!("    Multi Viewport: {}", features.multi_viewport != 0);
        println!("    Anisotropic Filtering: {}", features.sampler_anisotropy != 0);
    }
    
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
