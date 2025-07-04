use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

pub trait DisplayDataSource {
    fn get_lines(&mut self) -> Vec<String>;
}

pub struct LocalStatsDisplayDataSource {
    system: System,
}

impl LocalStatsDisplayDataSource {
    pub fn new() -> Self {
        let system = System::new_with_specifics(
            RefreshKind::new()
                .with_memory(MemoryRefreshKind::everything())
                .with_cpu(CpuRefreshKind::everything())
        );
        
        LocalStatsDisplayDataSource { system }
    }
}

impl DisplayDataSource for LocalStatsDisplayDataSource {
    fn get_lines(&mut self) -> Vec<String> {
        // Update system information
        self.system.refresh_cpu_all();
        self.system.refresh_memory();
        
        // Get CPU usage (average across all cores)
        let cpu_usage = self.system.global_cpu_usage();
        
        // Get used memory in MB
        let used_memory = (self.system.used_memory() / 1024 / 1024) as u32;
        
        // Prepare display text
        let line1 = format!("CPU {}%", cpu_usage as u32);
        let line2 = format!("MEM {}MB", used_memory);
        
        vec![line1, line2]
    }
}
