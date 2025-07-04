use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use serde::{Deserialize, Serialize};
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Request, Response, Server, Method, StatusCode};
use std::convert::Infallible;
use std::net::SocketAddr;
use tokio::sync::oneshot;

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

#[derive(Deserialize, Serialize)]
struct DisplayLinesRequest {
    line1: String,
    line2: String,
}

pub struct WebDisplayDataSource {
    lines: Arc<Mutex<(String, String)>>,
    thread_handle: Option<JoinHandle<()>>,
    shutdown_sender: Option<oneshot::Sender<()>>,
}

impl WebDisplayDataSource {
    pub fn new() -> Self {
        let lines = Arc::new(Mutex::new(("".to_string(), "".to_string())));
        let lines_clone = lines.clone();
        let (shutdown_sender, shutdown_receiver) = oneshot::channel();
        
        // Start the web server in a background thread
        let thread_handle = thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let addr = SocketAddr::from(([127, 0, 0, 1], 8000));
                
                let make_svc = make_service_fn(move |_conn| {
                    let lines = lines_clone.clone();
                    async move {
                        Ok::<_, Infallible>(service_fn(move |req| {
                            handle_request(req, lines.clone())
                        }))
                    }
                });
                
                let server = Server::bind(&addr).serve(make_svc);
                println!("Starting web server on 127.0.0.1:8000");
                
                // Run server with graceful shutdown
                let graceful = server.with_graceful_shutdown(async {
                    shutdown_receiver.await.ok();
                });
                
                if let Err(e) = graceful.await {
                    eprintln!("Server error: {}", e);
                }
                
                println!("Web server stopped");
            });
        });
        
        WebDisplayDataSource { 
            lines,
            thread_handle: Some(thread_handle),
            shutdown_sender: Some(shutdown_sender),
        }
    }
}

impl DisplayDataSource for WebDisplayDataSource {
    fn get_lines(&mut self) -> Vec<String> {
        let guard = self.lines.lock().unwrap();
        vec![guard.0.clone(), guard.1.clone()]
    }
}

impl Drop for WebDisplayDataSource {
    fn drop(&mut self) {
        println!("Shutting down WebDisplayDataSource...");
        
        // Send shutdown signal to the server
        if let Some(sender) = self.shutdown_sender.take() {
            let _ = sender.send(());
        }
        
        // Wait for the thread to finish, with a 5-second timeout
        if let Some(handle) = self.thread_handle.take() {
            // Create a channel to communicate the join result
            let (tx, rx) = std::sync::mpsc::channel();
            
            // Spawn a thread to handle the join operation
            let join_thread = std::thread::spawn(move || {
                let result = handle.join();
                let _ = tx.send(result);
            });
            
            // Wait for 5 seconds for the thread to finish
            match rx.recv_timeout(Duration::from_secs(5)) {
                Ok(Ok(())) => {
                    println!("Web server thread shut down gracefully");
                    let _ = join_thread.join();
                }
                Ok(Err(_)) => {
                    println!("Web server thread panicked during shutdown");
                    let _ = join_thread.join();
                }
                Err(_) => {
                    // Timeout occurred - the thread is still running
                    println!("Web server thread was aborted after 5 seconds timeout");
                    // The join_thread and the original thread will be forcefully terminated
                    // when they go out of scope
                }
            }
        }
        
        println!("WebDisplayDataSource shutdown complete");
    }
}

async fn handle_request(
    req: Request<Body>,
    lines: Arc<Mutex<(String, String)>>,
) -> Result<Response<Body>, Infallible> {
    match (req.method(), req.uri().path()) {
        (&Method::POST, "/display/lines-2") => {
            let body_bytes = match hyper::body::to_bytes(req.into_body()).await {
                Ok(bytes) => bytes,
                Err(_) => {
                    return Ok(Response::builder()
                        .status(StatusCode::BAD_REQUEST)
                        .body(Body::from("Failed to read body"))
                        .unwrap());
                }
            };
            
            let request: DisplayLinesRequest = match serde_json::from_slice(&body_bytes) {
                Ok(req) => req,
                Err(_) => {
                    return Ok(Response::builder()
                        .status(StatusCode::BAD_REQUEST)
                        .body(Body::from("Invalid JSON format"))
                        .unwrap());
                }
            };
            
            let mut guard = lines.lock().unwrap();
            *guard = (request.line1, request.line2);
            drop(guard);
            
            Ok(Response::builder()
                .status(StatusCode::OK)
                .body(Body::from("OK"))
                .unwrap())
        }
        _ => {
            Ok(Response::builder()
                .status(StatusCode::NOT_FOUND)
                .body(Body::from("Not Found"))
                .unwrap())
        }
    }
}

// Test the WebDisplayDataSource with:
// curl -X POST http://127.0.0.1:8000/display/lines-2 -H "Content-Type: application/json" -d '{"line1": "Hello", "line2": "World"}'

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_web_display_data_source_creation_and_drop() {
        // Create the WebDisplayDataSource
        let mut source = WebDisplayDataSource::new();
        
        // Give the server a moment to start
        std::thread::sleep(Duration::from_millis(100));
        
        // Test getting initial lines (should be empty)
        let lines = source.get_lines();
        assert_eq!(lines, vec!["", ""]);
        
        // The Drop implementation will be called automatically when source goes out of scope
        println!("Test completed, source will be dropped now");
    }
}
