use after_effects_sys as ae_sys;
use after_effects as ae;
use cstr_literal::cstr;

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use gyroflow_core::{ StabilizationManager, filesystem, stabilization::{ RGBA8, RGBA16, RGBAf } };
use gyroflow_core::gpu::{ BufferDescription, Buffers, BufferSource };

#[repr(i32)]
#[derive(Debug, PartialEq)]
enum PluginParams {
    InputLayer = 0,
    // ArbitraryParam,
    BrowseButton,
    Smoothness,
    StabilizationOverview,
    NumParams
}

#[derive(Default)]
struct Instance {
    width: usize,
    height: usize,
    stab: StabilizationManager,
    project_path: String,
	smoothness: f32,
	stab_overview: bool,
}
/*impl Default for Instance {
    fn default() -> Self {
        let mut stab = StabilizationManager::default();
        {
            let mut stab = stab.stabilization.write();
            stab.share_wgpu_instances = true;
            stab.interpolation = gyroflow_core::stabilization::Interpolation::Lanczos4;
        }
        Self {
            stab,
            ..Default::default()
        }
    }
}*/

impl Instance {
    pub fn load_path(&mut self, path: &str) {
        log::debug!("loading: {path}");
        {
            let mut stab = self.stab.stabilization.write();
            stab.share_wgpu_instances = true;
            stab.interpolation = gyroflow_core::stabilization::Interpolation::Lanczos4;
        }

        if !path.ends_with(".gyroflow") {
            if let Err(e) = self.stab.load_video_file(&filesystem::path_to_url(&path), None) {
                log::error!("An error occured: {e:?}");
            }
        } else {
            if let Err(e) = self.stab.import_gyroflow_file(&filesystem::path_to_url(&path), true, |_|(), Arc::new(AtomicBool::new(false))) {
                log::error!("import_gyroflow_file error: {e:?}");
            }
        }

        let video_size = self.stab.params.read().video_size;

        let org_ratio = video_size.0 as f64 / video_size.1 as f64;

        let src_rect = get_center_rect(self.width, self.height, org_ratio);
        self.stab.set_size(src_rect.2, src_rect.3);
        self.stab.set_output_size(self.width, self.height);

        self.stab.invalidate_smoothing();
        self.stab.recompute_blocking();

        self.project_path = path.to_owned();
    }
}

#[no_mangle]
pub unsafe extern "C" fn PluginDataEntryFunction2(
    in_ptr: ae_sys::PF_PluginDataPtr,
    in_plugin_data_callback_ptr: ae_sys::PF_PluginDataCB2,
    _in_sp_basic_suite_ptr: *const ae_sys::SPBasicSuite,
    in_host_name: *const std::ffi::c_char,
    in_host_version: *const std::ffi::c_char) -> ae_sys::PF_Err
{
    log::set_max_level(log::LevelFilter::Debug);
    log::info!("PluginDataEntryFunction2: {:?}, {:?}", std::ffi::CStr::from_ptr(in_host_name), std::ffi::CStr::from_ptr(in_host_version));

    if let Some(cb_ptr) = in_plugin_data_callback_ptr {
        cb_ptr(in_ptr,
            cstr!(env!("PIPL_NAME"))       .as_ptr() as *const u8, // Name
            cstr!(env!("PIPL_MATCH_NAME")) .as_ptr() as *const u8, // Match Name
            cstr!(env!("PIPL_CATEGORY"))   .as_ptr() as *const u8, // Category
            cstr!(env!("PIPL_ENTRYPOINT")) .as_ptr() as *const u8, // Entry point
            env!("PIPL_KIND")              .parse().unwrap(),
            env!("PIPL_AE_SPEC_VER_MAJOR") .parse().unwrap(),
            env!("PIPL_AE_SPEC_VER_MINOR") .parse().unwrap(),
            env!("PIPL_AE_RESERVED")       .parse().unwrap(),
            cstr!(env!("PIPL_SUPPORT_URL")).as_ptr() as *const u8, // Support url
        )
    } else {
        ae_sys::PF_Err_INVALID_CALLBACK as ae_sys::PF_Err
    }
}

fn write_str(ae_buffer: &mut [ae_sys::A_char], s: String) {
    let buf = std::ffi::CString::new(s).unwrap().into_bytes_with_nul();
    ae_buffer[0..buf.len()].copy_from_slice(unsafe { std::mem::transmute(buf.as_slice()) });
}

fn union_rect(src: &ae_sys::PF_LRect, dst: &mut ae_sys::PF_LRect) {
    fn is_empty_rect(r: &ae_sys::PF_LRect) -> bool{
        (r.left >= r.right) || (r.top >= r.bottom)
    }
	if is_empty_rect(dst) {
		*dst = *src;
	} else if !is_empty_rect(src) {
		dst.left 	= dst.left.min(src.left);
		dst.top  	= dst.top.min(src.top);
		dst.right 	= dst.right.max(src.right);
		dst.bottom  = dst.bottom.max(src.bottom);
	}
}
unsafe fn pre_render(in_data: ae::pf::InDataHandle, extra: *mut ae_sys::PF_PreRenderExtra) -> ae_sys::PF_Err {
	let req = (*(*extra).input).output_request;

	(*(*extra).output).flags |= ae_sys::PF_RenderOutputFlag_GPU_RENDER_POSSIBLE as i16;

    let pre_render_cb = ae::pf::PreRenderCallbacks::from_raw((*extra).cb);
    if let Ok(in_result) = pre_render_cb.checkout_layer(in_data.effect_ref(), PluginParams::InputLayer as i32, 0, &req, in_data.current_time(), in_data.time_step(), in_data.time_scale()) {
        union_rect(&in_result.result_rect,     &mut (*(*extra).output).result_rect);
        union_rect(&in_result.max_result_rect, &mut (*(*extra).output).max_result_rect);
    }

	ae_sys::PF_Err_NONE as ae_sys::PF_Err
}

unsafe fn smart_render(in_data: ae::pf::InDataHandle, extra: *mut ae_sys::PF_SmartRenderExtra, is_gpu: bool) -> ae_sys::PF_Err {
    let mut err = ae_sys::PF_Err_NONE as ae_sys::PF_Err;

    let cb = ae::pf::SmartRenderCallbacks::from_raw((*extra).cb);
    if let Ok(input_world) = cb.checkout_layer_pixels(in_data.effect_ref(), PluginParams::InputLayer as u32) {
        if let Ok(output_world) = cb.checkout_output(in_data.effect_ref()) {
            if let Ok(world_suite) = ae::WorldSuite2::new() {
                let pixel_format = world_suite.get_pixel_format(input_world).unwrap();
                log::info!("pixel_format: {pixel_format:?}, is_gpu: {is_gpu}");

                let seq_ptr = ae::pf::EffectSequenceDataSuite1::new()
                    .and_then(|x| x.get_const_sequence_data(in_data))
                    .unwrap_or((*in_data.as_ptr()).sequence_data as *const _);

                log::info!("smart_render: {}, seq_data: {:?}", in_data.current_timestamp(), seq_ptr);

                if !seq_ptr.is_null() {
                    let mut instance_handle = ae::pf::Handle::<Instance>::from_raw(seq_ptr as ae_sys::PF_Handle).unwrap();
                    {
                        let instance = instance_handle.lock().unwrap();
                        let instance = instance.as_ref_mut().unwrap();
                        let timestamp_us = (in_data.current_timestamp() * 1_000_000.0).round() as i64;

                        let org_ratio = {
                            let params = instance.stab.params.read();
                            params.video_size.0 as f64 / params.video_size.1 as f64
                        };

                        let src_size = (input_world.width(), input_world.height(), input_world.row_bytes());
                        let dest_size = (output_world.width(), output_world.height(), output_world.row_bytes());
                        let src_rect = get_center_rect(input_world.width(),  input_world.height(), org_ratio);

                        if is_gpu {
                            let what_gpu = (*(*extra).input).what_gpu;
                            log::info!("Render API: {what_gpu}");

                            if let Ok(gpu_suite) = ae::pf::GPUDeviceSuite1::new() {
                                let device_info = gpu_suite.get_device_info(in_data, (*(*extra).input).device_index).unwrap();

                                let what_gpu = (*(*extra).input).what_gpu;
                                let in_ptr = gpu_suite.get_gpu_world_data(in_data, input_world).unwrap();
                                let out_ptr = gpu_suite.get_gpu_world_data(in_data, output_world).unwrap();

                                log::info!("Render GPU: {in_ptr:?} -> {out_ptr:?}. API: {what_gpu}");

                                match what_gpu {
                                    ae_sys::PF_GPU_Framework_CUDA => {
                                    },
                                    ae_sys::PF_GPU_Framework_OPENCL => {
                                    },
                                    ae_sys::PF_GPU_Framework_METAL => {
                                    },
                                    _ => { }
                                }
                            } else {
                                log::error!("Missing gpu suite");
                            }

                            // err = smart_render_gpu(in_data, out_data, pixel_format, input_worldP, output_worldP, extraP, infoP);
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

                            match pixel_format {
                                ae_sys::PF_PixelFormat_ARGB128 => {
                                    log::debug!("PF_PixelFormat_ARGB128");
                                    if let Err(e) = instance.stab.process_pixels::<RGBAf>(timestamp_us, &mut buffers) {
                                        log::debug!("process_pixels error: {e:?}");
                                    }
                                },
                                ae_sys::PF_PixelFormat_ARGB64 => {
                                    log::debug!("PF_PixelFormat_ARGB64");
                                    if let Err(e) = instance.stab.process_pixels::<RGBA16>(timestamp_us, &mut buffers) {
                                        log::debug!("process_pixels error: {e:?}");
                                    }
                                },
                                ae_sys::PF_PixelFormat_ARGB32 => {
                                    log::debug!("PF_PixelFormat_ARGB32");
                                    if let Err(e) = instance.stab.process_pixels::<RGBA8>(timestamp_us, &mut buffers) {
                                        log::debug!("process_pixels error: {e:?}");
                                    }
                                },
                                _ => {
                                    log::info!("Unhandled pixel format: {pixel_format}");
                                }
                            }
                        }
                    }
                    std::mem::forget(instance_handle);
                }
            }
        }
        cb.checkin_layer_pixels(in_data.effect_ref(), PluginParams::InputLayer as u32).unwrap();
    }

    err
}

unsafe fn render(in_data: ae::pf::InDataHandle, params: *mut *mut ae_sys::PF_ParamDef, dest: *mut ae_sys::PF_LayerDef) -> ae_sys::PF_Err {
    if &in_data.application_id() == b"PrMr" && !(*in_data.as_ptr()).sequence_data.is_null() {
        let src = unsafe { &mut (*(*params.add(PluginParams::InputLayer as usize))).u.ld };

        log::info!("pr render");

        let mut instance_handle = ae::pf::Handle::<Instance>::from_raw((*in_data.as_ptr()).sequence_data as ae_sys::PF_Handle).unwrap();
        {
            let instance = instance_handle.lock().unwrap();
            let instance = instance.as_ref_mut().unwrap();
            log::info!("render: {}", in_data.current_timestamp());
            let timestamp_us = (in_data.current_timestamp() * 1_000_000.0).round() as i64;

            let org_ratio = {
                let params = instance.stab.params.read();
                params.video_size.0 as f64 / params.video_size.1 as f64
            };

            let src_size = ((*src).width as usize, (*src).height as usize, (*src).rowbytes as usize);
            let dest_size = ((*dest).width as usize, (*dest).height as usize, (*dest).rowbytes as usize);
            let src_rect = get_center_rect((*src).width as usize, (*src).height as usize, org_ratio);

            let inframe  = unsafe { std::slice::from_raw_parts_mut((*src).data as *mut u8, src_size.1 * src_size.2) };
            let outframe = unsafe { std::slice::from_raw_parts_mut((*dest).data as *mut u8, dest_size.1 * dest_size.2) };

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

            if let Err(e) = instance.stab.process_pixels::<RGBAf>(timestamp_us, &mut buffers) {
                log::debug!("process_pixels error: {e:?}");
            }
        }
        std::mem::forget(instance_handle);
	}
    ae_sys::PF_Err_NONE as ae_sys::PF_Err
}

/////////////////////////////////////////////////// GPU ///////////////////////////////////////////////////

unsafe fn gpu_device_setup(in_data: ae::pf::InDataHandle, out_data: *mut ae_sys::PF_OutData, extra: *mut ae_sys::PF_GPUDeviceSetupExtra) -> ae_sys::PF_Err {
    let device_info = ae::pf::GPUDeviceSuite1::new().unwrap().get_device_info(in_data, (*(*extra).input).device_index).unwrap();

    let what_gpu = (*(*extra).input).what_gpu;

    log::info!("Device info: {device_info:?}. GPU: {what_gpu}");

    match what_gpu {
        ae_sys::PF_GPU_Framework_CUDA => {
            //(*out_data).out_flags2 |= ae_sys::PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
        },
        ae_sys::PF_GPU_Framework_OPENCL => {
            //(*out_data).out_flags2 |= ae_sys::PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
        },
        ae_sys::PF_GPU_Framework_METAL => {
            //(*out_data).out_flags2 |= ae_sys::PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
        },
        _ => { }
    }

    ae_sys::PF_Err_NONE as ae_sys::PF_Err
}

unsafe fn gpu_device_setdown(in_data: ae::pf::InDataHandle, out_data: *mut ae_sys::PF_OutData, extra: *mut ae_sys::PF_GPUDeviceSetdownExtra) -> ae_sys::PF_Err {
    let what_gpu = (*(*extra).input).what_gpu;

    log::info!("gpu_device_setdown: {what_gpu}");

    ae_sys::PF_Err_NONE as ae_sys::PF_Err
}

/////////////////////////////////////////////////// GPU ///////////////////////////////////////////////////

#[no_mangle]
pub unsafe extern "C" fn EffectMain(
    cmd: ae_sys::PF_Cmd,
    in_data_ptr: *const ae_sys::PF_InData,
    out_data: *mut ae_sys::PF_OutData,
    params: *mut *mut ae_sys::PF_ParamDef,
    output: *mut ae_sys::PF_LayerDef,
    extra: *mut std::ffi::c_void) -> ae_sys::PF_Err
{
    let _pica = ae::PicaBasicSuite::from_pf_in_data_raw(in_data_ptr);

    let _ = log::set_logger(&win_dbg_logger::DEBUGGER_LOGGER);
    log::set_max_level(log::LevelFilter::Debug);


    let in_data = ae::pf::InDataHandle::from_raw(in_data_ptr);
    log::info!("[{:?}]: {:?} {}x{}", std::thread::current().id(), ae::pf::Command::try_from(cmd).unwrap(), in_data.width(), in_data.height());

    let mut err = ae_sys::PF_Err_NONE as ae_sys::PF_Err;

    log::info!("cmd: {cmd}, in seq: {:?}, out seq: {:?}", (*in_data_ptr).sequence_data, (*out_data).sequence_data);
    match cmd as ae::EnumIntType {
        ae_sys::PF_Cmd_ABOUT => {
            write_str(&mut (*out_data).return_msg,
                format!("Gyroflow, v1.0\nCopyright 2023 AdrianEddy\rGyroflow plugin.")
            );
        },
        ae_sys::PF_Cmd_USER_CHANGED_PARAM => {
            let extra = extra as *const ae_sys::PF_UserChangedParamExtra;
            if !(*in_data_ptr).sequence_data.is_null() {
                if (*extra).param_index > 0 && (*extra).param_index < PluginParams::NumParams as i32 {
                    let mut instance_handle = ae::pf::Handle::<Instance>::from_raw((*in_data_ptr).sequence_data as ae_sys::PF_Handle).unwrap();
                    {
                        let instance = instance_handle.lock().unwrap();
                        let instance = instance.as_ref_mut().unwrap();
                        let param = std::mem::transmute((*extra).param_index);
                        if param == PluginParams::BrowseButton {
                            let mut d = rfd::FileDialog::new()
                                .add_filter("Gyroflow project files", &["gyroflow"])
                                .add_filter("Video files", &["mp4", "mov", "mxf", "braw", "r3d", "insv"]);
                            let current_path = &instance.project_path;
                            if !current_path.is_empty() {
                                if let Some(path) = std::path::Path::new(current_path).parent() {
                                    d = d.set_directory(path);
                                }
                            }
                            if let Some(d) = d.pick_file() {
                                instance.load_path(&d.display().to_string());
                            }
                        } else {
                            match (param, ae::pf::ParamDef::from_raw(in_data_ptr, unsafe { *params.add((*extra).param_index as usize) }).to_param()) {
                                (PluginParams::Smoothness, ae::Param::FloatSlider(p))  => {
                                    instance.smoothness = p.value() as f32;
                                    instance.stab.smoothing.write().current_mut().set_parameter("smoothness", instance.smoothness as f64);
                                    instance.stab.invalidate_smoothing();
                                    instance.stab.recompute_blocking();
                                },
                                (PluginParams::StabilizationOverview, ae::Param::CheckBox(p))  => {
                                    instance.stab_overview = p.value();
                                    instance.stab.set_fov_overview(instance.stab_overview);
                                    instance.stab.recompute_undistortion();
                                },
                                _ => { }
                            }
                        }
                        log::info!("PF_Cmd_USER_CHANGED_PARAM: {} {}", instance.smoothness, instance.stab_overview);
                    }

                    std::mem::forget(instance_handle);
                }
            }

            log::info!("PF_Cmd_USER_CHANGED_PARAM: {}", (*extra).param_index);
        },
        ae_sys::PF_Cmd_SEQUENCE_SETUP => {
            log::info!("setup");
            let mut instance = Instance::default();
            instance.width = in_data.width() as usize;
            instance.height = in_data.height() as usize;

            (*out_data).sequence_data = ae::pf::Handle::into_raw(ae::pf::Handle::new(instance).unwrap()) as *mut _;
            log::debug!("setting sequence_data: {:?}", (*out_data).sequence_data);
        },
        ae_sys::PF_Cmd_SEQUENCE_RESETUP => {
            log::info!("RESETUP: {:?} | {:?}", (*in_data_ptr).sequence_data, (*out_data).sequence_data);
            if !(*in_data_ptr).sequence_data.is_null() {
                let instance = ae::pf::FlatHandle::from_raw((*in_data_ptr).sequence_data as ae_sys::PF_Handle).unwrap();
                let _lock = instance.lock().unwrap();
                let bytes = instance.as_slice().unwrap();
                if let Ok(path) = String::from_utf8(bytes.to_vec()) {

                    log::info!("resetup serialized: {path}");
                    let path = path.strip_prefix("path:").unwrap_or_default();

                    let mut instance = Instance::default();
                    instance.width = in_data.width() as usize;
                    instance.height = in_data.height() as usize;
                    instance.load_path(&path);

                    (*out_data).sequence_data = ae::pf::Handle::into_raw(ae::pf::Handle::new(instance).unwrap()) as *mut _;
                    log::debug!("setting sequence_data: {:?}", (*out_data).sequence_data);
                }
            } else {
                (*out_data).sequence_data = std::ptr::null_mut();
            }
        },
        ae_sys::PF_Cmd_SEQUENCE_FLATTEN => {
            log::info!("FLATTEN: {:?}", (*in_data_ptr).sequence_data);
            if !(*in_data_ptr).sequence_data.is_null() {
                let mut instance_handle = ae::pf::Handle::<Instance>::from_raw((*in_data_ptr).sequence_data as ae_sys::PF_Handle).unwrap();
                {
                    let instance = instance_handle.lock().unwrap();
                    let instance = instance.as_ref().unwrap();

                    let serialized = format!("path:{}", instance.project_path).as_bytes().to_vec();

                    (*out_data).sequence_data = ae::pf::FlatHandle::into_raw(ae::pf::FlatHandle::new(serialized).unwrap()) as *mut _;
                    log::info!("FLATTEN set: {:?}", (*out_data).sequence_data);
                }
                std::mem::forget(instance_handle);
            } else {
                (*out_data).sequence_data = std::ptr::null_mut();
            }
        },
        ae_sys::PF_Cmd_SEQUENCE_SETDOWN => {
            log::info!("SETDOWN: {:?}", (*in_data_ptr).sequence_data);
            if !(*in_data_ptr).sequence_data.is_null() {
                ae::pf::Handle::<Instance>::from_raw((*in_data_ptr).sequence_data as ae_sys::PF_Handle).unwrap();
            }
        },
        ae_sys::PF_Cmd_GLOBAL_SETUP => {
            log_panics::init();

            (*out_data).my_version = env!("PIPL_VERSION").parse().unwrap();
            (*out_data).out_flags  = env!("PIPL_OUTFLAGS").parse().unwrap();
            (*out_data).out_flags2 = env!("PIPL_OUTFLAGS2").parse().unwrap();

            if &in_data.application_id() == b"PrMr" {
                let pixel_format = ae::pf::PixelFormatSuite::new().unwrap();
                pixel_format.clear_supported_pixel_formats(in_data.effect_ref()).unwrap();
                pixel_format.add_supported_pixel_format(in_data.effect_ref(), ae_sys::PrPixelFormat_PrPixelFormat_VUYA_4444_32f).unwrap();
            } else {
                //(*out_data).out_flags2 |= ae_sys::PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
            }
        },
        ae_sys::PF_Cmd_PARAMS_SETUP => {
            use ae::*;
            //let mut arb = ArbitraryDef::new();
            //ParamDef::new(in_data).name("data").param(Param::Arbitrary(arb)).add(-1);

	        // Project file
            let mut btn = ButtonDef::new();
            btn.label("Browse");
            ParamDef::new(in_data).name("Project file or video").flags(ParamFlag::SUPERVISE).param(Param::Button(btn)).add(-1);

	        // Smoothness
            ParamDef::new(in_data).name("Smoothness").flags(ParamFlag::SUPERVISE).param(Param::FloatSlider(*FloatSliderDef::new()
                .set_valid_min(0.0)
                .set_slider_min(0.0)
                .set_valid_max(100.0)
                .set_slider_max(100.0)
                .set_value(50.0)
                .set_default(50.0)
                .precision(1)
                .display_flags(ValueDisplayFlag::NONE)
            )).add(-1);

	        // Stabilization overview
            let mut cb = CheckBoxDef::new();
            cb.label("ON");
            ParamDef::new(in_data).name("Stabilization overview").flags(ParamFlag::SUPERVISE).param(Param::CheckBox(cb)).add(-1);

            (*out_data).num_params = PluginParams::NumParams as i32;
        },
        ae_sys::PF_Cmd_GPU_DEVICE_SETUP => {
            err = gpu_device_setup(in_data, out_data, extra as *mut ae_sys::PF_GPUDeviceSetupExtra);
        },
        ae_sys::PF_Cmd_GPU_DEVICE_SETDOWN => {
            err = gpu_device_setdown(in_data, out_data, extra as *mut ae_sys::PF_GPUDeviceSetdownExtra);
        },
        ae_sys::PF_Cmd_RENDER => {
            err = render(in_data, params, output);
        },
        ae_sys::PF_Cmd_SMART_PRE_RENDER => {
            err = pre_render(in_data, extra as *mut ae_sys::PF_PreRenderExtra);
        },
        ae_sys::PF_Cmd_SMART_RENDER => {
            err = smart_render(in_data, extra as *mut ae_sys::PF_SmartRenderExtra, false);
        },
        ae_sys::PF_Cmd_SMART_RENDER_GPU => {
            err = smart_render(in_data, extra as *mut ae_sys::PF_SmartRenderExtra, true);
        },
        _ => {
            log::debug!("Unknown cmd: {cmd:?}");
        }
    }

    err
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
