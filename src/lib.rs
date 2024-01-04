use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use gyroflow_core::{ StabilizationManager, filesystem, stabilization::{ RGBA8, RGBA16, RGBAf } };
use gyroflow_core::gpu::{ BufferDescription, Buffers, BufferSource };

use after_effects as ae;
use after_effects_sys as ae_sys;

use lru::LruCache;
use parking_lot::Mutex;

// We should cache managers globally because it's common to have the effect applied to the same clip and cut the clip into multiple pieces
// We don't want to create a new manager for each piece of the same clip
// Cache key is specific enough

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
enum Params {
    BrowseButton,
    Smoothness,
    StabilizationOverview,
}

struct Plugin {
    manager_cache: Mutex<LruCache<String, Arc<StabilizationManager>>>
}
impl Default for Plugin {
    fn default() -> Self {
        Self {
            manager_cache: Mutex::new(LruCache::new(std::num::NonZeroUsize::new(8).unwrap())),
        }
    }
}
impl Drop for Plugin {
    fn drop(&mut self) {
        log::info!("dropping plugin: {:?}", self as *const _);
        {
            let mut lock = self.manager_cache.lock();
            for (_, v) in lock.iter() {
                log::info!("arc count: {}", Arc::strong_count(v));
            }
            lock.clear();
        }
        log::info!("dropped plugin: {:?}", self as *const _);
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Instance {
    width: usize,
    height: usize,
    project_path: String,
	smoothness: f32,
	stab_overview: bool,

    instance_id: String,
    embedded_lens: String,
    embedded_preset: String,

    input_rotation: f64,
    video_rotation: f64,

    include_project_data: bool,
    project_data: String,

    original_video_size: (usize, usize),
    original_output_size: (usize, usize),
    num_frames: usize,
    fps: f64,
    ever_changed: bool,

    #[serde(skip)]
    gyrodata: Option<LruCache<String, Arc<StabilizationManager>>>,
}
impl Default for Instance {
    fn default() -> Self {
        log::info!("new instance");
        Self {
            width: 0,
            height: 0,
            project_path: String::new(),
            smoothness: 50.0,
            stab_overview: false,

            instance_id: format!("{}", fastrand::u64(..)),
            embedded_lens: String::new(),
            embedded_preset: String::new(),

            input_rotation: 0.0,
            video_rotation: 0.0,

            include_project_data: false,
            project_data: String::new(),

            original_video_size: (0, 0),
            original_output_size: (0, 0),
            num_frames: 0,
            fps: 0.0,
            ever_changed: true,

            gyrodata: Some(LruCache::new(std::num::NonZeroUsize::new(8).unwrap())),
        }
    }
}

ae::register_plugin!(Plugin, Instance, Params);

impl Drop for Instance {
    fn drop(&mut self) {
        log::info!("dropping instance: {:?}", self as *const _);
        self.gyrodata.as_mut().unwrap().clear();
        //gyroflow_core::stabilization::clear_thread_local_cache();
        log::info!("dropped instance: {:?}", self as *const _);
    }
}
impl Instance {
    fn open_gyroflow(&self) {
        // TODO
    }
    fn update_loaded_state(&mut self, _enabled: bool) {
        // TODO
    }
    fn gyrodata(&mut self, global: &Plugin, bit_depth: usize, input_rect: ae::Rect, output_rect: ae::Rect) -> Option<Arc<StabilizationManager>> {
        let disable_stretch = false; // TODO

        let in_size = (input_rect.width() as usize, input_rect.height() as usize);
        let out_size = (output_rect.width() as usize, output_rect.height() as usize);

        let instance_id = &self.instance_id;
        let path = &self.project_path;
        if path.is_empty() {
            log::info!("no path!");
            self.update_loaded_state(false);
            return None;
        }
        let key = format!("{path}{bit_depth:?}{in_size:?}{out_size:?}{disable_stretch}{instance_id}");
        let cloned = global.manager_cache.lock().get(&key).map(Arc::clone);
        log::info!("key: {key}");
        let stab = if let Some(stab) = cloned {
            // Cache it in this instance as well
            /*if !self.gyrodata.as_ref()?.contains(&key) {
                self.gyrodata.as_mut()?.put(key.to_owned(), stab.clone());
            }*/
            stab
        } else {
            let mut stab = StabilizationManager::default();
            {
                // Find first lens profile database with loaded profiles
                let lock = global.manager_cache.lock();
                for (_, v) in lock.iter() {
                    if v.lens_profile_db.read().loaded {
                        stab.lens_profile_db = v.lens_profile_db.clone();
                        break;
                    }
                }
            }

            if !path.ends_with(".gyroflow") {
                match stab.load_video_file(&filesystem::path_to_url(&path), None) {
                    Ok(md) => {
                        if !self.embedded_lens.is_empty() {
                            if let Err(e) = stab.load_lens_profile(&self.embedded_lens) {
                                rfd::MessageDialog::new()
                                    .set_description(&format!("Failed to load lens profile: {e:?}"))
                                    .show();
                            }
                        }
                        if !self.embedded_preset.is_empty() {
                            let mut is_preset = false;
                            if let Err(e) = stab.import_gyroflow_data(self.embedded_preset.as_bytes(), true, None, |_|(), Arc::new(AtomicBool::new(false)), &mut is_preset) {
                                rfd::MessageDialog::new()
                                    .set_description(&format!("Failed to load preset: {e:?}"))
                                    .show();
                            }
                        }
                        if self.include_project_data {
                            if let Ok(data) = stab.export_gyroflow_data(gyroflow_core::GyroflowProjectType::WithGyroData, "{}", None) {
                                self.project_data = data;
                            }
                        }
                        if md.rotation != 0 {
                            let r = ((360 - md.rotation) % 360) as f64;
                            self.input_rotation = r;
                            stab.params.write().video_rotation = r;
                        }
                    },
                    Err(e) => {
                        let embedded_data = &self.project_data;
                        if !embedded_data.is_empty() {
                            let mut is_preset = false;
                            match stab.import_gyroflow_data(embedded_data.as_bytes(), true, None, |_|(), Arc::new(AtomicBool::new(false)), &mut is_preset) {
                                Ok(_) => { },
                                Err(e) => {
                                    log::error!("load_gyro_data error: {}", &e);
                                    self.update_loaded_state(false);
                                }
                            }
                        } else {
                            log::error!("An error occured: {e:?}");
                            self.update_loaded_state(false);
                            // self.param_status.set_label("Failed to load file info!")?;
                            // self.param_status.set_hint(&format!("Error loading {path}: {e:?}."))?;
                            return None;
                        }
                    }
                }
            } else {
                let project_data = {
                    if self.include_project_data && !self.project_data.is_empty() {
                        self.project_data.clone()
                    } else if let Ok(data) = std::fs::read_to_string(&path) {
                        if self.include_project_data {
                            self.project_data = data.clone();
                        } else {
                            self.project_data.clear();
                        }
                        data
                    } else {
                        String::new()
                    }
                };
                let mut is_preset = false;
                if let Err(e) = stab.import_gyroflow_data(project_data.as_bytes(), true, Some(&filesystem::path_to_url(&path)), |_|(), Arc::new(AtomicBool::new(false)), &mut is_preset) {
                    log::error!("load_gyro_data error: {}", &e);
                    self.update_loaded_state(false);
                }
            }

            let loaded = {
                stab.params.write().calculate_ramped_timestamps(&stab.keyframes.read(), false, true);
                let params = stab.params.read();
                self.original_video_size = params.video_size;
                self.original_output_size = params.video_output_size;
                self.num_frames = params.frame_count;
                self.fps = params.fps;
                let loaded = params.duration_ms > 0.0;
                //if loaded && self.reload_values_from_project {
                //    self.reload_values_from_project = false;
                //    let smooth = stab.smoothing.read();
                //    let smoothness = smooth.current().get_parameter("smoothness");
                //}
                loaded
            };

            self.update_loaded_state(loaded);

            if disable_stretch {
                stab.disable_lens_stretch();
            }

            stab.set_fov_overview(self.stab_overview);

            let video_size = {
                let mut params = stab.params.write();
                params.framebuffer_inverted = false;
                params.video_size
            };

            let org_ratio = video_size.0 as f64 / video_size.1 as f64;

            let src_rect = get_center_rect(in_size.0, in_size.1, org_ratio);
            stab.set_size(src_rect.2, src_rect.3);
            stab.set_output_size(out_size.0, out_size.1);

            {
                let mut stab = stab.stabilization.write();
                stab.share_wgpu_instances = true;
                stab.interpolation = gyroflow_core::stabilization::Interpolation::Lanczos4;
            }

            stab.invalidate_smoothing();
            stab.recompute_blocking();
            //let inverse = !(self.keyframable_params.read().use_gyroflows_keyframes.get_value()? && stab.keyframes.read().is_keyframed_internally(&KeyframeType::VideoSpeed));
            //stab.params.write().calculate_ramped_timestamps(&stab.keyframes.read(), inverse, inverse);

            let stab = Arc::new(stab);
            // Insert to static global cache
            global.manager_cache.lock().put(key.to_owned(), stab.clone());
            // Cache it in this instance as well
            //self.gyrodata.as_mut()?.put(key.to_owned(), stab.clone());

            stab
        };

        Some(stab)
    }
}

impl Instance {
    fn smart_render(&mut self, global: &Plugin, in_data: ae::pf::InData, extra: SmartRenderExtra, is_gpu: bool) -> Result<(), ae::Error> {
        let cb = extra.callbacks();
        if let Ok(input_world) = cb.checkout_layer_pixels(in_data.effect_ref(), 0) {
            if let Ok(output_world) = cb.checkout_output(in_data.effect_ref()) {
                if let Ok(world_suite) = ae::WorldSuite2::new() {
                    let pixel_format = world_suite.get_pixel_format(input_world).unwrap();
                    if is_gpu && pixel_format != ae::PixelFormat::GpuBgra128 {
                        log::info!("GPU render requested but pixel format is not GpuBgra128");
                        return Err(Error::UnrecogizedParameterType);
                    }
                    if let Some(stab) = extra.pre_render_data::<Arc<StabilizationManager>>() {
                        log::info!("pixel_format: {pixel_format:?}, is_gpu: {is_gpu}, arc count: {}", Arc::strong_count(&stab));
                        log::info!("smart_render: {}, size: {:?}", in_data.current_timestamp(), stab.params.read().size);

                        let timestamp_us = (in_data.current_timestamp() * 1_000_000.0).round() as i64;

                        let org_ratio = {
                            let params = stab.params.read();
                            params.video_size.0 as f64 / params.video_size.1 as f64
                        };

                        let src_size = (input_world.width(), input_world.height(), input_world.row_bytes());
                        let dest_size = (output_world.width(), output_world.height(), output_world.row_bytes());
                        let src_rect = get_center_rect(input_world.width(),  input_world.height(), org_ratio);

                        let what_gpu = extra.what_gpu();
                        log::info!("Render API: {what_gpu:?}, src_size: {src_size:?}, src_rect: {src_rect:?}, dest_size: {dest_size:?}");

                        if is_gpu && !ae::pf::GPUDeviceSuite1::new().is_err() {
                            if let Ok(gpu_suite) = ae::pf::GPUDeviceSuite1::new() {
                                let device_info = gpu_suite.get_device_info(in_data, extra.device_index())?;

                                let in_ptr = gpu_suite.get_gpu_world_data(in_data, input_world)?;
                                let out_ptr = gpu_suite.get_gpu_world_data(in_data, output_world)?;

                                let mut buffers = Buffers {
                                    input: BufferDescription {
                                        size: src_size,
                                        rect: Some(src_rect),
                                        data: match what_gpu {
                                            #[cfg(any(target_os = "macos", target_os = "ios"))]
                                            ae::GpuFramework::Metal  => BufferSource::MetalBuffer { buffer: in_ptr, command_queue: device_info.command_queuePV },
                                            ae::GpuFramework::OpenCl => BufferSource::OpenCL      { texture: in_ptr, queue: device_info.command_queuePV },
                                            ae::GpuFramework::Cuda   => BufferSource::CUDABuffer  { buffer: in_ptr },
                                            _ => panic!("Invalid GPU framework")
                                        },
                                        rotation: None,
                                        texture_copy: true
                                    },
                                    output: BufferDescription {
                                        size: dest_size,
                                        rect: None,
                                        data: match what_gpu {
                                            #[cfg(any(target_os = "macos", target_os = "ios"))]
                                            ae::GpuFramework::Metal  => BufferSource::MetalBuffer { buffer: out_ptr, command_queue: device_info.command_queuePV },
                                            ae::GpuFramework::OpenCl => BufferSource::OpenCL      { texture: out_ptr, queue: std::ptr::null_mut() },
                                            ae::GpuFramework::Cuda   => BufferSource::CUDABuffer  { buffer: out_ptr },
                                            _ => panic!("Invalid GPU framework")
                                        },
                                        rotation: None,
                                        texture_copy: true
                                    }
                                };

                                log::info!("Render GPU: {in_ptr:?} -> {out_ptr:?}. API: {what_gpu:?}, pixel_format: {pixel_format:?}");
                                match pixel_format {
                                    ae::PixelFormat::Argb32 => {
                                        match stab.process_pixels::<RGBA8>(timestamp_us, &mut buffers) {
                                            Ok(i)  => { log::info!("process_pixels ok: {i:?}"); },
                                            Err(e) => { log::info!("process_pixels error: {e:?}"); }
                                        }
                                    },
                                    ae::PixelFormat::GpuBgra128 => {
                                        match stab.process_pixels::<RGBAf>(timestamp_us, &mut buffers) {
                                            Ok(i)  => { log::info!("process_pixels ok: {i:?}"); },
                                            Err(e) => { log::info!("process_pixels error: {e:?}"); }
                                        }
                                    },
                                    _ => {
                                        log::info!("Unhandled pixel format: {pixel_format:?}");
                                    }
                                }

                            } else {
                                log::error!("Missing gpu suite");
                            }
                        } else {
                            let src = input_world.data_as_ptr_mut();
                            let dest = output_world.data_as_ptr_mut();

                            let inframe  = unsafe { std::slice::from_raw_parts_mut(src, src_size.1 * src_size.2) };
                            let outframe = unsafe { std::slice::from_raw_parts_mut(dest, dest_size.1 * dest_size.2) };

                            let mut buffers = Buffers {
                                input: BufferDescription {
                                    size: src_size,
                                    rect: Some(src_rect),
                                    data: BufferSource::Cpu { buffer: inframe },
                                    rotation: None,
                                    texture_copy: false
                                },
                                output: BufferDescription {
                                    size: dest_size,
                                    rect: None,
                                    data: BufferSource::Cpu { buffer: outframe },
                                    rotation: None,
                                    texture_copy: false
                                }
                            };

                            log::info!("pixel_format: {pixel_format:?}");
                            match pixel_format {
                                ae::PixelFormat::Argb128 => {
                                    if let Err(e) = stab.process_pixels::<RGBAf>(timestamp_us, &mut buffers) {
                                        log::info!("process_pixels error: {e:?}");
                                    }
                                },
                                ae::PixelFormat::Argb64 => {
                                    if let Err(e) = stab.process_pixels::<RGBA16>(timestamp_us, &mut buffers) {
                                        log::info!("process_pixels error: {e:?}");
                                    }
                                },
                                ae::PixelFormat::Argb32 => {
                                    if let Err(e) = stab.process_pixels::<RGBA8>(timestamp_us, &mut buffers) {
                                        log::info!("process_pixels error: {e:?}");
                                    }
                                },
                                _ => {
                                    log::info!("Unhandled pixel format: {pixel_format:?}");
                                }
                            }
                        }
                    }
                }
            }
            log::info!("checkin_layer_pixels");
            cb.checkin_layer_pixels(in_data.effect_ref(), 0).unwrap();
        }
        Ok(())
    }
}

impl AdobePluginGlobal for Plugin {
    fn can_load(_host_name: &str, _host_version: &str) -> bool {
        true
    }

    fn params_setup(&self, params: &mut ae::Parameters<Params>) -> Result<(), Error> {
        // Project file
        params.add_param_with_flags(Params::BrowseButton, "Project file or video", ButtonDef::new().label("Browse"), ParamFlag::SUPERVISE | ParamFlag::CANNOT_TIME_VARY);

        // Smoothness
        params.add_param_with_flags(Params::Smoothness, "Smoothness", ae::FloatSliderDef::new()
            .set_valid_min(0.0)
            .set_slider_min(0.0)
            .set_valid_max(100.0)
            .set_slider_max(100.0)
            .set_value(50.0)
            .set_default(50.0)
            .precision(1)
            .display_flags(ValueDisplayFlag::NONE)
        , ParamFlag::SUPERVISE);

        // Stabilization overview
        params.add_param_with_flags(Params::StabilizationOverview, "Stabilization overview", CheckBoxDef::new().label("ON"), ParamFlag::SUPERVISE);

        Ok(())
    }

    fn handle_command(&self, cmd: ae::Command, in_data: ae::InData, mut out_data: ae::OutData) -> Result<(), ae::Error> {
        let _ = log::set_logger(&win_dbg_logger::DEBUGGER_LOGGER);
        log::set_max_level(log::LevelFilter::Debug);
        log_panics::init();

        log::info!("handle_command: {:?}, thread: {:?}, ptr: {:?}, effect_ref: {:?}", cmd, std::thread::current().id(), self as *const _, in_data.effect_ref().as_ptr());

        match cmd {
            ae::Command::About => {
                out_data.set_return_msg("Gyroflow, v1.0\nCopyright 2023 AdrianEddy\rGyroflow plugin.");
            }
            ae::Command::GlobalSetup => {
                if &in_data.application_id() == b"PrMr" {
                    use ae::pr::PixelFormat::*;
                    let pixel_format = ae::pf::PixelFormatSuite::new()?;
                    pixel_format.clear_supported_pixel_formats(in_data.effect_ref())?;
                    let supported_formats = [
                        Bgra4444_8u,  Vuya4444_8u,  Vuya4444_8u709,  Argb4444_8u,  Bgrx4444_8u,  Vuyx4444_8u,  Vuyx4444_8u709,  Xrgb4444_8u,  Bgrp4444_8u,  Vuyp4444_8u,  Vuyp4444_8u709,  Prgb4444_8u,
                        Bgra4444_16u, Vuya4444_16u,                  Argb4444_16u, Bgrx4444_16u,                                Xrgb4444_16u, Bgrp4444_16u,                                Prgb4444_16u,
                        Bgra4444_32f, Vuya4444_32f, Vuya4444_32f709, Argb4444_32f, Bgrx4444_32f, Vuyx4444_32f, Vuyx4444_32f709, Xrgb4444_32f, Bgrp4444_32f, Vuyp4444_32f, Vuyp4444_32f709, Prgb4444_32f,
                        Bgra4444_32fLinear, Bgrp4444_32fLinear, Bgrx4444_32fLinear, Argb4444_32fLinear, Prgb4444_32fLinear, Xrgb4444_32fLinear
                    ];
                    for x in supported_formats {
                        pixel_format.add_pr_supported_pixel_format(in_data.effect_ref(), x)?;
                    }
                } else {
                    out_data.add_out_flag2(ae_sys::PF_OutFlag2_SUPPORTS_GPU_RENDER_F32);
                }
                log::info!("added all flag");
            }
            ae::Command::GlobalSetdown => {
                //self.manager_cache.lock().clear();
            }
            ae::Command::GpuDeviceSetup { extra } => {
                let device_info = ae::pf::GPUDeviceSuite1::new().unwrap().get_device_info(in_data, extra.device_index())?;

                let what_gpu = extra.what_gpu();

                log::info!("Device info: {device_info:?}. GPU: {what_gpu:?}");

                if what_gpu != ae::GpuFramework::None {
                    out_data.add_out_flag2(ae_sys::PF_OutFlag2_SUPPORTS_GPU_RENDER_F32);
                }
            }
            ae::Command::GpuDeviceSetdown { extra } => {
                log::info!("gpu_device_setdown: {:?}", extra.what_gpu());
            }

            _ => {}
        }
        Ok(())
    }
}

impl AdobePluginInstance for Instance {
    fn flatten(&self) -> Result<Vec<u8>, Error> {
        log::info!("flatten path: {}, ptr: {:?}", self.project_path, self as *const _);
        Ok(bincode::serialize(self).unwrap())
    }
    fn unflatten(bytes: &[u8]) -> Result<Self, Error> {
        let mut inst = bincode::deserialize::<Self>(bytes).unwrap();
        inst.gyrodata = Some(LruCache::new(std::num::NonZeroUsize::new(8).unwrap()));
        log::info!("unflatten path: {}, ptr: {:?}, bytes: {}", inst.project_path, &inst as *const _, pretty_hex::pretty_hex(&bytes));
        Ok(inst)
    }

    fn user_changed_param(&mut self, global: &mut Plugin, param: Params, params: &mut ae::Parameters<Params>) -> Result<(), ae::Error> {
        match param {
            Params::BrowseButton => {
                let mut d = rfd::FileDialog::new()
                    .add_filter("Gyroflow project files", &["gyroflow"])
                    .add_filter("Video files", &["mp4", "mov", "mxf", "braw", "r3d", "insv"]);
                let current_path = &self.project_path;
                if !current_path.is_empty() {
                    if let Some(path) = std::path::Path::new(current_path).parent() {
                        d = d.set_directory(path);
                    }
                }
                if let Some(d) = d.pick_file() {
                    self.project_path = d.display().to_string();
                    log::info!("path: {}", self.project_path);
                    params.get_param_def(Params::BrowseButton).set_value_has_changed();
                }
            }
            Params::Smoothness => {
                let val = params.get_float_slider(Params::Smoothness).value();
                self.smoothness = val as f32;
            }
            Params::StabilizationOverview => {
                let val = params.get_checkbox(Params::StabilizationOverview).value();
                self.stab_overview = val;
            }
        }
        log::info!("PF_Cmd_USER_CHANGED_PARAM: {} {}", self.smoothness, self.stab_overview);

        Ok(())
    }

    fn render(&self, global: &Plugin, in_data: ae::InData, src: &Layer, dst: &mut Layer, _params: &ae::Parameters<Params>) -> Result<(), ae::Error> {
        log::info!("render: {}", in_data.current_timestamp());

        if let Some(stab) = in_data.frame_data::<Arc<StabilizationManager>>() {
            let timestamp_us = (in_data.current_timestamp() * 1_000_000.0).round() as i64;

            let org_ratio = {
                let params = stab.params.read();
                params.video_size.0 as f64 / params.video_size.1 as f64
            };

            let src_size  = (src.width() as usize, src.height() as usize, src.stride().abs() as usize);
            let dest_size = (dst.width() as usize, dst.height() as usize, dst.stride().abs() as usize);
            let src_rect = get_center_rect(src_size.0, src_size.1, org_ratio);

            log::info!("org_ratio: {org_ratio:?}, src_size: {src_size:?}, src_rect: {src_rect:?}, dest_size: {dest_size:?}, src.stride: {}", src.stride());

            let inframe  = unsafe { std::slice::from_raw_parts_mut(src.buffer().as_ptr() as *mut u8, src.buffer().len()) };
            let outframe = unsafe { std::slice::from_raw_parts_mut(dst.buffer().as_ptr() as *mut u8, dst.buffer().len()) };

            let mut buffers = Buffers {
                input: BufferDescription {
                    size: src_size,
                    rect: Some(src_rect),
                    data: BufferSource::Cpu { buffer: inframe },
                    rotation: None,
                    texture_copy: false
                },
                output: BufferDescription {
                    size: dest_size,
                    rect: None,
                    data: BufferSource::Cpu { buffer: outframe },
                    rotation: None,
                    texture_copy: false
                }
            };

            if let Err(e) = stab.process_pixels::<RGBA8>(timestamp_us, &mut buffers) {
                log::info!("process_pixels error: {e:?}");
            }
        }

        Ok(())
    }

    fn handle_command(&mut self, global: &mut Plugin, cmd: ae::Command, mut in_data: ae::InData, mut out_data: ae::OutData) -> Result<(), ae::Error> {
        log::info!("sequence command: {:?}, thread: {:?}, ptr: {:?}", cmd, std::thread::current().id(), self as *const _);

        match cmd {
            ae::Command::UserChangedParam { .. } => {
                out_data.set_force_rerender();
            }
            ae::Command::SequenceSetup | ae::Command::SequenceResetup => {
                if let Ok(interface_suite) = ae::aegp::PFInterfaceSuite::new() {
                    if let Ok(layer_suite) = ae::aegp::LayerSuite::new() {
                        if let Ok(footage_suite) = ae::aegp::FootageSuite::new() {
                            let layer = interface_suite.effect_layer(in_data.effect_ref())?;
                            let item = layer_suite.layer_source_item(layer)?;
                            let footage = footage_suite.main_footage_from_item(item)?;
                            log::info!("footage path: {:?}", footage_suite.footage_path(footage, 0, 0));
                        }
                    }
                }
            },
            ae::Command::SmartPreRender { mut extra } => {
                let what_gpu = extra.what_gpu();
                let req = extra.output_request();

                if what_gpu != ae::GpuFramework::None {
                    extra.set_gpu_render_possible(true);
                }

                let cb = extra.callbacks();
                if let Ok(in_result) = cb.checkout_layer(in_data.effect_ref(), 0, 0, &req, in_data.current_time(), in_data.time_step(), in_data.time_scale()) {
                    let      result_rect = extra.union_result_rect(in_result.result_rect.into());
                    let _max_result_rect = extra.union_max_result_rect(in_result.max_result_rect.into());

                    if let Some(stab) = self.gyrodata(global, 8, result_rect, result_rect) {
                        stab.set_fov_overview(self.stab_overview);
                        stab.recompute_undistortion();
                        log::info!("setting pre-render extra: {result_rect:?}, in: {:?}, stab_overview: {}", in_data.extent_hint(), self.stab_overview);
                        extra.set_pre_render_data::<Arc<StabilizationManager>>(stab);
                    }
                }
            }
            ae::Command::FrameSetup { out_layer, .. } => {
                if let Some(stab) = self.gyrodata(global, 8, in_data.extent_hint(), out_layer.extent_hint()) {
                    out_data.set_frame_data::<Arc<StabilizationManager>>(stab);
                }
            }
            ae::Command::FrameSetdown => {
                in_data.destroy_frame_data::<Arc<StabilizationManager>>();
            }
            ae::Command::SmartRender { extra } => {
                self.smart_render(global, in_data, extra, false)?;
            }
            ae::Command::SmartRenderGpu { extra } => {
                self.smart_render(global, in_data, extra, true)?;
            }
            _ => { }
        }

        Ok(())
    }
}

fn get_center_rect(width: usize, height: usize, org_ratio: f64) -> (usize, usize, usize, usize) {
    // If aspect ratio is different
    let new_ratio = width as f64 / height as f64;
    if (new_ratio - org_ratio).abs() > 0.1 {
        // Get center rect of original aspect ratio
        let rect = if new_ratio > org_ratio {
            ((height as f64 * org_ratio).round() as usize, height)
        } else {
            (width, (width as f64 / org_ratio).round() as usize)
        };
        (
            (width - rect.0) / 2, // x
            (height - rect.1) / 2, // y
            rect.0, // width
            rect.1 // height
        )
    } else {
        (0, 0, width, height)
    }
}
