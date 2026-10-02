/* Copyright 2021 Aristocratos (jakob@qvantnet.com)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

indent = tab
tab-size = 4
*/

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <numeric>
#include <optional>
#include <ranges>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <arpa/inet.h> // for inet_ntop()
#include <dlfcn.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <fmt/format.h>
#include <fmt/std.h>

#if defined(RSMI_STATIC)
	#include <rocm_smi/rocm_smi.h>
#endif

#if !(defined(STATIC_BUILD) && defined(__GLIBC__))
	#include <pwd.h>
#endif

#include "../btop_config.hpp"
#include "../btop_log.hpp"
#include "../btop_shared.hpp"
#include "../btop_tools.hpp"

#if defined(GPU_SUPPORT)
	// Redefining C++ keywords fortunately has a warning in clang, however it's unavoidable here
	// since the C library uses "class" as a struct member and keywords are not allowed to be used
	// as identifiers in C++.
	#if defined(__clang__)
		#pragma clang diagnostic push
		#pragma clang diagnostic ignored "-Wkeyword-macro"
	#endif // __clang__

	#define class class_
extern "C" {
	#include "intel_gpu_top/intel_gpu_top.h"
}
	#undef class

	#if defined(__clang__)
		#pragma clang diagnostic pop
	#endif // __clang__

	//? Level Zero Sysman headers, for struct/enum definitions only — the actual library is
	//? dlopen'd at runtime (see Gpu::Intel::LevelZero::init()), so this is a build-time-only
	//? dependency, same trade-off Rsmi already accepts for <rocm_smi/rocm_smi.h>. See
	//? INTEL_XPU.md for the package needed on each distro.
	#include <level_zero/zes_api.h>
#endif

using std::abs;
using std::clamp;
using std::cmp_greater;
using std::cmp_less;
using std::ifstream;
using std::max;
using std::min;
using std::numeric_limits;
using std::round;
using std::streamsize;
using std::vector;
using std::future;
using std::async;
using std::pair;


namespace fs = std::filesystem;
namespace rng = std::ranges;

using namespace Tools;
using namespace std::literals; // for operator""s
using namespace std::chrono_literals;
//? --------------------------------------------------- FUNCTIONS -----------------------------------------------------

namespace
{

long long get_monotonicTimeUSec()
{
	struct timespec time;
	clock_gettime(CLOCK_MONOTONIC, &time);
	return time.tv_sec * 1000000 + time.tv_nsec / 1000;
}

// Wrapper with join behavior of std::jthread to avoid -fexperimental-library
// flag usage for Clang 19, ideally replace once flag is no longer needed.
struct join_thread
{
	std::thread t;

	join_thread() noexcept = default;

	template<typename... Args>
	explicit join_thread(Args&&... args) : t(std::forward<Args>(args)...) {}

	~join_thread() { if (t.joinable()) t.join(); }

	join_thread(join_thread&&) noexcept = default;

	join_thread& operator=(join_thread&& other) noexcept {
		if (t.joinable()) t.join();
		t = std::move(other.t);
		return *this;
	}

	void join() {
		t.join();
	}

};

}

namespace Cpu {
	vector<long long> core_old_totals;
	vector<long long> core_old_idles;
	vector<fs::path> core_freq;
	vector<string> available_fields = {"Auto", "total"};
	vector<string> available_sensors = {"Auto"};
	cpu_info current_cpu;
	bool got_sensors{};
	bool cpu_temp_only{};
	bool supports_watts = true;

	//* Populate found_sensors map
	bool get_sensors();

	//* Get current cpu clock speed
	string get_cpuHz();

	//* Search /proc/cpuinfo for a cpu name
	string get_cpuName();

	struct Sensor {
		fs::path path;
		int64_t temp{};
		int64_t crit{};
	};

	std::unordered_map<string, Sensor> found_sensors;
	string cpu_sensor;
	vector<string> core_sensors;
	std::unordered_map<int, int> core_mapping;
}

#if defined(GPU_SUPPORT)

namespace Gpu {
	vector<gpu_info> gpus;
	//? NVIDIA data collection
	namespace Nvml {
		//? NVML defines, structs & typedefs
		#define NVML_DEVICE_NAME_BUFFER_SIZE        64
		#define NVML_SUCCESS                         0
		#define NVML_TEMPERATURE_THRESHOLD_SHUTDOWN  0
		#define NVML_CLOCK_GRAPHICS                  0
		#define NVML_CLOCK_MEM                       2
		#define NVML_TEMPERATURE_GPU                 0
		#define NVML_PCIE_UTIL_TX_BYTES              0
		#define NVML_PCIE_UTIL_RX_BYTES              1

		typedef void* nvmlDevice_t; // we won't be accessing any of the underlying struct's properties, so this is fine
		typedef int nvmlReturn_t, // enums are basically ints
					nvmlTemperatureThresholds_t,
					nvmlClockType_t,
					nvmlPstates_t,
					nvmlTemperatureSensors_t,
					nvmlPcieUtilCounter_t;

		struct nvmlUtilization_t {unsigned int gpu, memory;};
		struct nvmlMemory_t {unsigned long long total, free, used;};

		//? Function pointers
		const char* (*nvmlErrorString)(nvmlReturn_t);
		nvmlReturn_t (*nvmlInit)();
		nvmlReturn_t (*nvmlShutdown)();
		nvmlReturn_t (*nvmlDeviceGetCount)(unsigned int*);
		nvmlReturn_t (*nvmlDeviceGetHandleByIndex)(unsigned int, nvmlDevice_t*);
		nvmlReturn_t (*nvmlDeviceGetName)(nvmlDevice_t, char*, unsigned int);
		nvmlReturn_t (*nvmlDeviceGetPowerManagementLimit)(nvmlDevice_t, unsigned int*);
		nvmlReturn_t (*nvmlDeviceGetTemperatureThreshold)(nvmlDevice_t, nvmlTemperatureThresholds_t, unsigned int*);
		nvmlReturn_t (*nvmlDeviceGetUtilizationRates)(nvmlDevice_t, nvmlUtilization_t*);
		nvmlReturn_t (*nvmlDeviceGetClockInfo)(nvmlDevice_t, nvmlClockType_t, unsigned int*);
		nvmlReturn_t (*nvmlDeviceGetPowerUsage)(nvmlDevice_t, unsigned int*);
		nvmlReturn_t (*nvmlDeviceGetPowerState)(nvmlDevice_t, nvmlPstates_t*);
		nvmlReturn_t (*nvmlDeviceGetTemperature)(nvmlDevice_t, nvmlTemperatureSensors_t, unsigned int*);
		nvmlReturn_t (*nvmlDeviceGetMemoryInfo)(nvmlDevice_t, nvmlMemory_t*);
		nvmlReturn_t (*nvmlDeviceGetPcieThroughput)(nvmlDevice_t, nvmlPcieUtilCounter_t, unsigned int*);
		nvmlReturn_t (*nvmlDeviceGetEncoderUtilization)(nvmlDevice_t, unsigned int*, unsigned int*);
		nvmlReturn_t (*nvmlDeviceGetDecoderUtilization)(nvmlDevice_t, unsigned int*, unsigned int*);

		//? Data
		void* nvml_dl_handle;
		bool initialized = false;
		bool init();
		bool shutdown();
		template <bool is_init> bool collect(gpu_info* gpus_slice);
		vector<nvmlDevice_t> devices;
		unsigned int device_count = 0;
	}

	//? AMD data collection
	namespace Rsmi {

	//? RSMI defines, structs & typedefs
	#define RSMI_DEVICE_NAME_BUFFER_SIZE 128

	#if !defined(RSMI_STATIC)
		#define RSMI_MAX_NUM_FREQUENCIES_V5  32
		#define RSMI_MAX_NUM_FREQUENCIES_V6  33
		#define RSMI_STATUS_SUCCESS           0
		#define RSMI_MEM_TYPE_VRAM            0
		#define RSMI_TEMP_CURRENT             0
		#define RSMI_TEMP_TYPE_EDGE           0
		#define RSMI_CLK_TYPE_MEM             4
		#define RSMI_CLK_TYPE_SYS             0
		#define RSMI_TEMP_MAX                 1

		typedef int rsmi_status_t,
					rsmi_temperature_metric_t,
					rsmi_clk_type_t,
					rsmi_memory_type_t;

		struct rsmi_version_t {uint32_t major,  minor,  patch; const char* build;};
		struct rsmi_frequencies_t_v5 {uint32_t num_supported, current; uint64_t frequency[RSMI_MAX_NUM_FREQUENCIES_V5];};
		struct rsmi_frequencies_t_v6 {bool has_deep_sleep; uint32_t num_supported, current; uint64_t frequency[RSMI_MAX_NUM_FREQUENCIES_V6];};

		//? Function pointers
		rsmi_status_t (*rsmi_init)(uint64_t);
		rsmi_status_t (*rsmi_shut_down)();
		rsmi_status_t (*rsmi_version_get)(rsmi_version_t*);
		rsmi_status_t (*rsmi_num_monitor_devices)(uint32_t*);
		rsmi_status_t (*rsmi_dev_name_get)(uint32_t, char*, size_t);
		rsmi_status_t (*rsmi_dev_power_cap_get)(uint32_t, uint32_t, uint64_t*);
		rsmi_status_t (*rsmi_dev_temp_metric_get)(uint32_t, uint32_t, rsmi_temperature_metric_t, int64_t*);
		rsmi_status_t (*rsmi_dev_busy_percent_get)(uint32_t, uint32_t*);
		rsmi_status_t (*rsmi_dev_memory_busy_percent_get)(uint32_t, uint32_t*);
		rsmi_status_t (*rsmi_dev_gpu_clk_freq_get_v5)(uint32_t, rsmi_clk_type_t, rsmi_frequencies_t_v5*);
		rsmi_status_t (*rsmi_dev_gpu_clk_freq_get_v6)(uint32_t, rsmi_clk_type_t, rsmi_frequencies_t_v6*);
		rsmi_status_t (*rsmi_dev_power_ave_get)(uint32_t, uint32_t, uint64_t*);
		rsmi_status_t (*rsmi_dev_memory_total_get)(uint32_t, rsmi_memory_type_t, uint64_t*);
		rsmi_status_t (*rsmi_dev_memory_usage_get)(uint32_t, rsmi_memory_type_t, uint64_t*);
		rsmi_status_t (*rsmi_dev_pci_throughput_get)(uint32_t, uint64_t*, uint64_t*, uint64_t*);

		uint32_t version_major = 0;

		//? Data
		void* rsmi_dl_handle;
	#endif
		bool initialized = false;
		bool init();
		bool shutdown();
		template <bool is_init> bool collect(gpu_info* gpus_slice);
		uint32_t device_count = 0;
	}


	//? Intel data collection
	//? Three backends are tried in order at startup (Shared::init), richest/most-permission-
	//? free first: LevelZero -> Sysfs -> this legacy PMU backend. See INTEL_XPU.md for why.
	namespace Intel {
		const char* device = "i915";
		struct engines *engines = nullptr;

		bool initialized = false;
		bool init();
		bool shutdown();
		template <bool is_init> bool collect(gpu_info* gpus_slice);
		uint32_t device_count = 0;

		//? GT RC6/idle-residency based busy% — correct on Xe-architecture GPUs where the
		//? legacy PMU per-engine "busy" counters above undercount GPU occupancy when a single
		//? workload is split across multiple parallel engine instances. Pure sysfs, no
		//? perf_event_open/CAP_PERFMON permissions needed. See INTEL_XPU.md.
		namespace Sysfs {
			struct GtNode {
				string label;
				std::filesystem::path residency_path;
				std::filesystem::path freq_act_path;
				std::filesystem::path freq_cur_path;
				std::filesystem::path freq_max_path;
				long long prev_res_ms = -1;
				long long prev_t_us = 0;
			};

			struct device_paths {
				std::filesystem::path dev;    //? .../cardN/device
				std::filesystem::path hwmon;
				vector<GtNode> gts;
				uint32_t pci_device_id = 0;
				//? True only if an actual power source file was found under hwmon (not just
				//? the hwmon directory itself) — this, not hwmon presence, is what guarantees
				//? "gpu-pwr-totals" gets populated. See collect() below.
				bool has_power = false;
				long long prev_energy_uj = -1;
				long long prev_energy_t_us = 0;
			};

			bool initialized = false;
			bool init();
			bool shutdown();
			template <bool is_init> bool collect(gpu_info* gpus_slice);
			uint32_t device_count = 0;
			vector<device_paths> devices;
		}

		//? Level Zero Sysman — richest telemetry (accurate VRAM, true per-engine activity)
		//? when libze_loader is present on the system; dlopen'd at runtime, so this is
		//? gracefully skipped if it isn't installed. See INTEL_XPU.md.
		namespace LevelZero {
			struct engine_info {
				zes_engine_handle_t handle;
				zes_engine_group_t type;
				zes_engine_stats_t stats;
			};

			struct memory_info {
				zes_mem_handle_t handle;
				uint64_t physical_size;
				uint64_t size;
				uint64_t free;
			};

			struct XeDeviceState {
				zes_device_handle_t device = nullptr;
				vector<engine_info> engines;
				vector<memory_info> memory_modules;
				zes_pwr_handle_t power_handle = nullptr;
				zes_power_energy_counter_t power_stats{};
				long long power_limit = 0;
				zes_freq_handle_t frequency_handle = nullptr;
				bool initialized = false;
			};

			vector<XeDeviceState> device_states;
			bool initialized = false;
			bool init();
			bool shutdown();
			template <bool is_init> bool collect(gpu_info* gpus_slice);
			uint32_t device_count = 0;
			void* ze_dl_handle = nullptr;
		}
	}

	//? AMD sysfs (consumer GPU / iGPU) data collection — fallback when amd-smi/rocm-smi
	//? cannot enumerate (e.g. APUs without /dev/kfd, or systems without ROCm libs installed).
	//? Reads only standard amdgpu DRM sysfs nodes; no library dependency.
	namespace Asysfs {
		struct device_paths {
			std::filesystem::path device;
			std::filesystem::path hwmon;
			std::filesystem::path power;            //? empty if no power sensor
			uint32_t pci_device_id;
			bool has_temp;
			bool has_freq;
			bool has_busy;
			bool has_vram;
		};

		bool initialized = false;
		bool init();
		bool shutdown();
		template <bool is_init> bool collect(gpu_info* gpus_slice);
		uint32_t device_count = 0;
		vector<device_paths> devices;
	}
}

//? Intel NPU data collection (brief rows in the CPU panel only — no dedicated box).
//? Reuses Gpu::gpu_info/gpu_info_supported since the drawing widgets (Meter, Graph)
//? and most fields map directly; NPU just leaves mem_total/mem_used/pcie_txrx/
//? encoder_utilization/decoder_utilization at their default false (not applicable —
//? the NPU shares system RAM rather than having dedicated VRAM). See INTEL_XPU.md.
namespace Npu {
	struct device_paths {
		std::filesystem::path dev;          //? .../accelN/device
		std::filesystem::path accel;        //? .../accelN
		std::filesystem::path hwmon;
		std::filesystem::path busy_path;    //? npu_busy_time_us, or runtime_active_time fallback
		bool busy_is_us = true;             //? false when using the coarser ms-granularity fallback
		bool has_power = false;
		long long prev_busy = -1;
		long long prev_t_us = 0;
		long long prev_energy_uj = -1;
		long long prev_energy_t_us = 0;
		uint32_t pci_device_id = 0;
	};

	bool initialized = false;
	bool init();
	bool shutdown();
	void sample();
	auto collect(bool no_update) -> vector<Gpu::gpu_info>&;
	uint32_t device_count = 0;
	vector<device_paths> devices;
	vector<Gpu::gpu_info> npus;
	vector<string> npu_names;
}

#endif // GPU_SUPPORT

namespace Mem {
	double old_uptime;
}

namespace Shared {

	fs::path procPath, passwd_path;
	long pageSize, clkTck, coreCount;

	void init() {

		//? Shared global variables init
		procPath = (fs::is_directory(fs::path("/proc")) and access("/proc", R_OK) != -1) ? "/proc" : "";
		if (procPath.empty())
			throw std::runtime_error("Proc filesystem not found or no permission to read from it!");

		passwd_path = (fs::is_regular_file(fs::path("/etc/passwd")) and access("/etc/passwd", R_OK) != -1) ? "/etc/passwd" : "";
		if (passwd_path.empty())
			Logger::warning("Could not read /etc/passwd, will show UID instead of username.");

		coreCount = sysconf(_SC_NPROCESSORS_ONLN);
		if (coreCount < 1) {
			coreCount = sysconf(_SC_NPROCESSORS_CONF);
			if (coreCount < 1) {
				coreCount = 1;
				Logger::warning("Could not determine number of cores, defaulting to 1.");
			}
		}

		pageSize = sysconf(_SC_PAGE_SIZE);
		if (pageSize <= 0) {
			pageSize = 4096;
			Logger::warning("Could not get system page size. Defaulting to 4096, processes memory usage might be incorrect.");
		}

		clkTck = sysconf(_SC_CLK_TCK);
		if (clkTck <= 0) {
			clkTck = 100;
			Logger::warning("Could not get system clock ticks per second. Defaulting to 100, processes cpu usage might be incorrect.");
		}

		//? Init for namespace Cpu
		Cpu::current_cpu.core_percent.insert(Cpu::current_cpu.core_percent.begin(), Shared::coreCount, {});
		Cpu::current_cpu.temp.insert(Cpu::current_cpu.temp.begin(), Shared::coreCount + 1, {});
		Cpu::core_old_totals.insert(Cpu::core_old_totals.begin(), Shared::coreCount, 0);
		Cpu::core_old_idles.insert(Cpu::core_old_idles.begin(), Shared::coreCount, 0);

		for (int i = 0; i < Shared::coreCount; ++i) {
			Cpu::core_freq.push_back("/sys/devices/system/cpu/cpufreq/policy" + to_string(i) + "/scaling_cur_freq");
			if (not fs::exists(Cpu::core_freq.back()) or access(Cpu::core_freq.back().c_str(), R_OK) == -1) {
				Cpu::core_freq.pop_back();
			}
		}

		Cpu::collect();
		if (Runner::coreNum_reset) Runner::coreNum_reset = false;
		for (auto& [field, vec] : Cpu::current_cpu.cpu_percent) {
			if (not vec.empty() and not v_contains(Cpu::available_fields, field)) Cpu::available_fields.push_back(field);
		}
		Cpu::cpuName = Cpu::get_cpuName();
		Cpu::got_sensors = Cpu::get_sensors();
		for (const auto& [sensor, ignored] : Cpu::found_sensors) {
			Cpu::available_sensors.push_back(sensor);
		}
		Cpu::core_mapping = Cpu::get_core_mapping();

		Cpu::container_engine = detect_container();

		//? Init for namespace Gpu
	#ifdef GPU_SUPPORT
		auto shown_gpus = Config::getS("shown_gpus");
		if (shown_gpus.contains("nvidia")) {
		    Gpu::Nvml::init();
		}

		if (shown_gpus.contains("amd")) {
			Gpu::Rsmi::init();
			Gpu::Asysfs::init(); //? self-skips when rocm-smi already enumerated devices
		}

		if (shown_gpus.contains("intel")) {
			//? Prefer the richest/most-reliable backend available at runtime, falling back in
			//? order: Level Zero (richest telemetry, needs libze_loader) -> Sysfs (GT
			//? RC6-residency, always available on a modern i915/xe kernel, no special
			//? permissions) -> legacy PMU (last resort; undercounts busy% on Xe-architecture
			//? GPUs that split a workload across multiple engine instances). See INTEL_XPU.md.
			if (not Gpu::Intel::LevelZero::init())
				if (not Gpu::Intel::Sysfs::init())
					Gpu::Intel::init();
		}

		//? Init for namespace Npu (brief rows in the CPU panel only, see INTEL_XPU.md)
		if (Config::getS("shown_npus").contains("intel")) {
			Npu::init();
		}

		if (not Gpu::gpu_names.empty()) {
			for (auto const& [key, _] : Gpu::gpus[0].gpu_percent)
				Cpu::available_fields.push_back(key);
			for (auto const& [key, _] : Gpu::shared_gpu_percent)
				Cpu::available_fields.push_back(key);
			//? Pure UI alias for "gpu-totals" (same combined GPU+NPU split view,
			//? see INTEL_XPU.md) — not a real gpu_percent key, resolved in
			//? Cpu::draw(). A clearer name now that NPUs can join the split too.
			Cpu::available_fields.push_back("xpu-totals");

			using namespace Gpu;
			count = gpus.size();
			gpu_b_height_offsets.resize(gpus.size());
			for (size_t i = 0; i < gpu_b_height_offsets.size(); ++i)
				gpu_b_height_offsets[i] = gpus[i].supported_functions.gpu_utilization
					   + gpus[i].supported_functions.pwr_usage
					   + (gpus[i].supported_functions.encoder_utilization or gpus[i].supported_functions.decoder_utilization)
					   + (gpus[i].supported_functions.mem_total or gpus[i].supported_functions.mem_used)
						* (1 + 2*(gpus[i].supported_functions.mem_total and gpus[i].supported_functions.mem_used) + 2*gpus[i].supported_functions.mem_utilization);
		}
	#endif

		//? Init for namespace Mem
		Mem::old_uptime = system_uptime();
		Mem::collect();

		Logger::debug("Shared::init() : Initialized.");
	}
}

namespace Cpu {
	string cpuName;
	string cpuHz;
	bool has_battery = true;
	tuple<int, float, long, string> current_bat;

	const array time_names {
		"user"s, "nice"s, "system"s, "idle"s, "iowait"s,
		"irq"s, "softirq"s, "steal"s, "guest"s, "guest_nice"s
	};

	std::unordered_map<string, long long> cpu_old = {
			{"totals", 0},
			{"idles", 0},
			{"user", 0},
			{"nice", 0},
			{"system", 0},
			{"idle", 0},
			{"iowait", 0},
			{"irq", 0},
			{"softirq", 0},
			{"steal", 0},
			{"guest", 0},
			{"guest_nice", 0}
	};

	string get_cpuName() {
		string name;
		ifstream cpuinfo(Shared::procPath / "cpuinfo");
		if (cpuinfo.good()) {
			for (string instr; getline(cpuinfo, instr, ':') and not instr.starts_with("model name");)
				cpuinfo.ignore(SSmax, '\n');
			if (cpuinfo.bad()) return name;
			else if (not cpuinfo.eof()) {
				cpuinfo.ignore(1);
				getline(cpuinfo, name);
			}
			else if (fs::exists("/sys/devices")) {
				for (const auto& d : fs::directory_iterator("/sys/devices")) {
					if (string(d.path().filename()).starts_with("arm")) {
						name = d.path().filename();
						break;
					}
				}
				if (not name.empty()) {
					auto name_vec = ssplit(name, '_');
					if (name_vec.size() < 2) return capitalize(name);
					else return capitalize(name_vec.at(1)) + (name_vec.size() > 2 ? ' ' + capitalize(name_vec.at(2)) : "");
				}

			}

			name = trim_name(name);
		}

		return name;
	}

	bool get_sensors() {
		bool got_cpu = false, got_coretemp = false;
		vector<fs::path> search_paths;
		try {
			//? Setup up paths to search for sensors
			if (fs::exists(fs::path("/sys/class/hwmon")) and access("/sys/class/hwmon", R_OK) != -1) {
				for (const auto& dir : fs::directory_iterator(fs::path("/sys/class/hwmon"))) {
					fs::path add_path = fs::canonical(dir.path());
					if (v_contains(search_paths, add_path) or v_contains(search_paths, add_path / "device")) continue;

					if (std::string_view { add_path.c_str() }.contains("coretemp"))
						got_coretemp = true;

					for (const auto & file : fs::directory_iterator(add_path)) {
						if (file.path().filename() == "device") {
							for (const auto & dev_file : fs::directory_iterator(file.path())) {
								string dev_filename = dev_file.path().filename();
								if (dev_filename.starts_with("temp") and dev_filename.ends_with("_input")) {
									search_paths.push_back(file.path());
									break;
								}
							}
						}

						string filename = file.path().filename();
						if (filename.starts_with("temp") and filename.ends_with("_input")) {
							search_paths.push_back(add_path);
							break;
						}
					}
				}
			}
			if (not got_coretemp and fs::exists(fs::path("/sys/devices/platform/coretemp.0/hwmon"))) {
				for (auto& d : fs::directory_iterator(fs::path("/sys/devices/platform/coretemp.0/hwmon"))) {
					fs::path add_path = fs::canonical(d.path());

					for (const auto & file : fs::directory_iterator(add_path)) {
						string filename = file.path().filename();
						if (filename.starts_with("temp") and filename.ends_with("_input") and not v_contains(search_paths, add_path)) {
								search_paths.push_back(add_path);
								got_coretemp = true;
								break;
						}
					}
				}
			}
			//? Scan any found directories for temperature sensors
			if (not search_paths.empty()) {
				for (const auto& path : search_paths) {
					const string pname = readfile(path / "name", path.filename());
					for (const auto & file : fs::directory_iterator(path)) {
						const string file_suffix = "input";
						const int file_id = atoi(file.path().filename().c_str() + 4); // skip "temp" prefix
						string file_path = file.path();

						if (!file_path.contains(file_suffix) or file_path.contains("nvme")) {
							continue;
						}

						const string basepath = file_path.erase(file_path.find(file_suffix), file_suffix.length());
						const string label = readfile(fs::path(basepath + "label"), "temp" + to_string(file_id));
						const string sensor_name = pname + "/" + label;
						const int64_t temp = stol(readfile(fs::path(basepath + "input"), "0")) / 1000;
						const int64_t crit = stol(readfile(fs::path(basepath + "crit"), "95000")) / 1000;

						found_sensors[sensor_name] = Sensor { fs::path(basepath + "input"), temp, crit };

						if (not got_cpu and (label.starts_with("Package id") or label.starts_with("Tdie") or label.starts_with("SoC Temperature"))) {
							got_cpu = true;
							cpu_sensor = sensor_name;
						}
						else if (label.starts_with("Core") or label.starts_with("Tccd")) {
							got_coretemp = true;
							if (not v_contains(core_sensors, sensor_name)) core_sensors.push_back(sensor_name);
						}
					}
				}
			}
			//? If no good candidate for cpu temp has been found scan /sys/class/thermal
			if (not got_cpu and fs::exists(fs::path("/sys/class/thermal"))) {
				const string rootpath = fs::path("/sys/class/thermal/thermal_zone");
				for (int i = 0; fs::exists(fs::path(rootpath + to_string(i))); i++) {
					const fs::path basepath = rootpath + to_string(i);
					if (not fs::exists(basepath / "temp")) continue;
					const string label = readfile(basepath / "type", "temp" + to_string(i));
					const string sensor_name = "thermal" + to_string(i) + "/" + label;
					const int64_t temp = stol(readfile(basepath / "temp", "0")) / 1000;

					int64_t high = 0;
					int64_t crit = 0;
					for (int ii = 0; fs::exists(basepath / fmt::format("trip_point_{}_temp", ii)); ii++) {
						const string trip_type = readfile(basepath / fmt::format("trip_point_{}_type", ii));
						if (not is_in(trip_type, "high", "critical")) continue;
						auto& val = (trip_type == "high" ? high : crit);
						val = stol(readfile(basepath / fmt::format("trip_point_{}_temp", ii), "0")) / 1000;
					}
					if (high < 1) high = 80;
					if (crit < 1) crit = 95;

					found_sensors[sensor_name] = Sensor { basepath / "temp", temp, crit };
				}
			}

		}
		catch (...) {}

		if (not got_coretemp or core_sensors.empty()) {
			cpu_temp_only = true;
		}
		else {
			rng::sort(core_sensors, rng::less{});
			rng::stable_sort(core_sensors, [](const auto& a, const auto& b){
				return a.size() < b.size();
			});
		}

		if (cpu_sensor.empty() and not found_sensors.empty()) {
			for (const auto& [name, sensor] : found_sensors) {
				if (str_to_lower(name).contains("cpu") or str_to_lower(name).contains("k10temp")) {
					cpu_sensor = name;
					break;
				}
			}
			if (cpu_sensor.empty()) {
				cpu_sensor = found_sensors.begin()->first;
				Logger::warning("No good candidate for cpu sensor found, using random from all found sensors.");
			}
		}

		return not found_sensors.empty();
	}

	static void update_sensors() {
		if (cpu_sensor.empty()) return;

		const auto& cpu_sensor = (not Config::getS("cpu_sensor").empty() and found_sensors.contains(Config::getS("cpu_sensor")) ? Config::getS("cpu_sensor") : Cpu::cpu_sensor);

		found_sensors.at(cpu_sensor).temp = stol(readfile(found_sensors.at(cpu_sensor).path, "0")) / 1000;
		current_cpu.temp.at(0).push_back(found_sensors.at(cpu_sensor).temp);
		current_cpu.temp_max = found_sensors.at(cpu_sensor).crit;
		if (current_cpu.temp.at(0).size() > 20) current_cpu.temp.at(0).pop_front();

		if (Config::getB("show_coretemp") and not cpu_temp_only) {
			for (vector<string_view> done; const auto& sensor : core_sensors) {
				if (v_contains(done, sensor)) continue;
				found_sensors.at(sensor).temp = stol(readfile(found_sensors.at(sensor).path, "0")) / 1000;
				done.push_back(sensor);
			}
			for (const auto& [core, temp] : core_mapping) {
				if (cmp_less(core + 1, current_cpu.temp.size()) and cmp_less(temp, core_sensors.size())) {
					current_cpu.temp.at(core + 1).push_back(found_sensors.at(core_sensors.at(temp)).temp);
					if (current_cpu.temp.at(core + 1).size() > 20) current_cpu.temp.at(core + 1).pop_front();
				}
			}
		}
	}

	static string normalize_frequency(double hz) {
		string str;
		if (hz > 999999) {
			str = fmt::format("{:.1f}", hz / 1'000'000);
			str.resize(3);
			if (str.back() == '.') str.pop_back();
			str += " THz";
		}
		else if (hz > 999) {
			str = fmt::format("{:.1f}", hz / 1'000);
			str.resize(3);
			if (str.back() == '.') str.pop_back();
			str += " GHz";
		}
		else {
			str = fmt::format("{:.0f} MHz", hz);
		}
		return str;
	}

	string get_cpuHz() {
		static int failed{};

		if (failed > 4)
			return ""s;

		string cpuhz;

		const auto &freq_mode = Config::getS("freq_mode");

		try {
			double hz = 0.0;
			// Read frequencies from all CPU cores
			vector<double> frequencies;
			for (auto it = Cpu::core_freq.begin(); it != Cpu::core_freq.end(); ) {
    			if (it->empty()) {
        			it = Cpu::core_freq.erase(it);
        			continue;
    			}

    			double core_hz = stod(readfile(*it, "0.0")) / 1000;
    			if (core_hz <= 0.0 and ++failed >= 2) {
        			it = Cpu::core_freq.erase(it);
    			} else {
        			frequencies.push_back(core_hz);
        			if (freq_mode == "first") break;
        			++it;
    			}
 			}

			if (not frequencies.empty()) {
				if (freq_mode == "first") {
					hz = frequencies.front();
				}
				if (freq_mode == "average") {
					hz = std::accumulate(frequencies.begin(), frequencies.end(), 0.0) / static_cast<double>(frequencies.size());
				}
				else if (freq_mode == "highest") {
					hz = *std::max_element(frequencies.begin(), frequencies.end());
				}
				else if (freq_mode == "lowest") {
					hz = *std::min_element(frequencies.begin(), frequencies.end());
				}
				else if (freq_mode == "range") {
					auto [min_hz,max_hz] = std::minmax_element(frequencies.begin(), frequencies.end());

					// Format as range
					string min_str, max_str;
					min_str = normalize_frequency(*min_hz);
					max_str = normalize_frequency(*max_hz);

					return min_str + " - " + max_str;
				}
			}
			//? If freq from /sys failed or is missing try to use /proc/cpuinfo
			if (hz <= 0.0) {
				ifstream cpufreq(Shared::procPath / "cpuinfo");
				if (cpufreq.good()) {
					while (cpufreq.ignore(SSmax, '\n')) {
						// peek is caps sensitive so it was skipping 'CPU MHz'. This aims to fix it.
						if (cpufreq.peek() == 'c' || cpufreq.peek() == 'C') {
							cpufreq.ignore(SSmax, ' ');
							if (cpufreq.peek() == 'M') {
								cpufreq.ignore(SSmax, ':');
								cpufreq.ignore(1);
								cpufreq >> hz;
								break;
							}
						}
					}
				}
			}

			if (hz <= 1 or hz >= 999999999)
				throw std::runtime_error("Failed to read /sys/devices/system/cpu/cpufreq/policy and /proc/cpuinfo.");

			cpuhz = normalize_frequency(hz);

		}
		catch (const std::exception& e) {
			if (++failed < 5)
				return ""s;
			else {
				Logger::warning("get_cpuHZ() : {}", e.what());
				return ""s;
			}
		}

		return cpuhz;
	}

	auto get_core_mapping() -> std::unordered_map<int, int> {
		std::unordered_map<int, int> core_map;
		if (cpu_temp_only) return core_map;

		const int num_sensors = core_sensors.size();

		//? Try to get core mapping from /proc/cpuinfo
		ifstream cpuinfo(Shared::procPath / "cpuinfo");
		if (cpuinfo.good()) {
			int cpu{};
			int core{};
			int max_core{};
			std::unordered_map<int, int> cpu_to_core;
			for (string instr; cpuinfo >> instr;) {
				if (instr == "processor") {
					cpuinfo.ignore(SSmax, ':');
					cpuinfo >> cpu;
				}
				else if (instr.starts_with("core")) {
					cpuinfo.ignore(SSmax, ':');
					cpuinfo >> core;
					cpu_to_core[cpu] = core;
					max_core = std::max(max_core, core);
				}
				cpuinfo.ignore(SSmax, '\n');
			}
			if (not cpu_to_core.empty())
				//? Divide core ID space evenly across sensors (handles AMD multi-CCD, e.g. 5950x: IDs 0-7 -> Tccd1, 8-15 -> Tccd2)
				for (const auto& [cpu_id, core_id] : cpu_to_core) {
					core_map[cpu_id] = core_id * num_sensors / (max_core + 1);
				}
		}

		//? If core mapping from cpuinfo was incomplete try to guess remainder, if missing completely, map 0-0 1-1 2-2 etc.
		if (cmp_less(core_map.size(), Shared::coreCount)) {
			if (Shared::coreCount % 2 == 0 and (long)core_map.size() == Shared::coreCount / 2) {
				//? SMT siblings mirror their first half of cores.
				for (int i = 0; i < Shared::coreCount / 2; i++) {
					core_map[Shared::coreCount / 2 + i] = core_map.count(i) ? core_map.at(i) : (i % num_sensors);
				}
			}
			else {
				core_map.clear();
				for (int i = 0; i < Shared::coreCount; i++) {
					core_map[i] = (i * num_sensors) / Shared::coreCount;
				}
			}
		}

		//? Apply user set custom mapping if any
		const auto& custom_map = Config::getS("cpu_core_map");
		if (not custom_map.empty()) {
			try {
				for (const auto& split : ssplit(custom_map)) {
					const auto vals = ssplit(split, ':');
					if (vals.size() != 2) continue;
					int change_id = std::stoi(vals.at(0));
					int new_id = std::stoi(vals.at(1));
					if (not core_map.contains(change_id) or cmp_greater(new_id, num_sensors)) continue;
					core_map.at(change_id) = new_id;
				}
			}
			catch (...) {}
		}

		return core_map;
	}

	struct battery {
		fs::path base_dir, energy_now, charge_now, energy_full, charge_full, power_now, current_now, voltage_now, status, online;
		string device_type;
		bool use_energy_or_charge = true;
		bool use_power = true;
	};

	auto get_battery() -> tuple<int, float, long, string> {
		if (not has_battery) return {0, 0, 0, ""};
		static string auto_sel;
		static std::unordered_map<string, battery> batteries;

		//? Get paths to needed files and check for valid values on first run
		if (batteries.empty() and has_battery) {
			try {
				if (fs::exists("/sys/class/power_supply")) {
					for (const auto& d : fs::directory_iterator("/sys/class/power_supply")) {
						//? Only consider online power supplies of type Battery or UPS
						//? see kernel docs for details on the file structure and contents
						//? https://www.kernel.org/doc/Documentation/ABI/testing/sysfs-class-power
						battery new_bat;
						fs::path bat_dir;
						try {
							if (not d.is_directory()
								or not fs::exists(d.path() / "type")
								or not fs::exists(d.path() / "present")
								or stoi(readfile(d.path() / "present")) != 1)
								continue;
							string dev_type = readfile(d.path() / "type");
							if (is_in(dev_type, "Battery", "UPS")) {
								bat_dir = d.path();
								new_bat.base_dir = d.path();
								new_bat.device_type = dev_type;
							}
						} catch (...) {
							//? skip power supplies not conforming to the kernel standard
							continue;
						}

						if (fs::exists(bat_dir / "energy_now")) new_bat.energy_now = bat_dir / "energy_now";
						else if (fs::exists(bat_dir / "charge_now")) new_bat.charge_now = bat_dir / "charge_now";
						else new_bat.use_energy_or_charge = false;

						if (fs::exists(bat_dir / "energy_full")) new_bat.energy_full = bat_dir / "energy_full";
						else if (fs::exists(bat_dir / "charge_full")) new_bat.charge_full = bat_dir / "charge_full";
						else new_bat.use_energy_or_charge = false;

						if (not new_bat.use_energy_or_charge and not fs::exists(bat_dir / "capacity")) {
							continue;
						}

						if (fs::exists(bat_dir / "power_now")) {
							new_bat.power_now = bat_dir / "power_now";
						}
						else if ((fs::exists(bat_dir / "current_now")) and (fs::exists(bat_dir / "voltage_now"))) {
							 new_bat.current_now = bat_dir / "current_now";
							 new_bat.voltage_now = bat_dir / "voltage_now";
						}
						else {
							new_bat.use_power = false;
						}

						if (fs::exists(bat_dir / "AC0/online")) new_bat.online = bat_dir / "AC0/online";
						else if (fs::exists(bat_dir / "AC/online")) new_bat.online = bat_dir / "AC/online";

						batteries[bat_dir.filename()] = new_bat;
						Config::available_batteries.push_back(bat_dir.filename());
					}
				}
			}
			catch (...) {
				batteries.clear();
			}
			if (batteries.empty()) {
				has_battery = false;
				return {0, 0, 0, ""};
			}
		}

		auto& battery_sel = Config::getS("selected_battery");

		if (auto_sel.empty()) {
			for (auto& [name, bat] : batteries) {
				if (bat.device_type == "Battery") {
					auto_sel = name;
					break;
				}
			}
			if (auto_sel.empty()) auto_sel = batteries.begin()->first;
		}

		auto& b = (battery_sel != "Auto" and batteries.contains(battery_sel) ? batteries.at(battery_sel) : batteries.at(auto_sel));

		int percent = -1;
		long seconds = -1;
		float watts = -1;

		//? Try to get battery percentage
		if (percent < 0) {
			try {
				percent = stoi(readfile(b.base_dir / "capacity", "-1"));
			}
			catch (const std::invalid_argument&) { }
			catch (const std::out_of_range&) { }
		}
		if (b.use_energy_or_charge and percent < 0) {
			try {
				percent = round(100.0 * stod(readfile(b.energy_now, "-1")) / stod(readfile(b.energy_full, "1")));
			}
			catch (const std::invalid_argument&) { }
			catch (const std::out_of_range&) { }
		}
		if (b.use_energy_or_charge and percent < 0) {
			try {
				percent = round(100.0 * stod(readfile(b.charge_now, "-1")) / stod(readfile(b.charge_full, "1")));
			}
			catch (const std::invalid_argument&) { }
			catch (const std::out_of_range&) { }
		}
		if (percent < 0) {
			has_battery = false;
			return {0, 0, 0, ""};
		}

		//? Get charging/discharging status
		string status = str_to_lower(readfile(b.base_dir / "status", "unknown"));
		if (status == "unknown" and not b.online.empty()) {
			const auto online = readfile(b.online, "0");
			if (online == "1" and percent < 100) status = "charging";
			else if (online == "1") status = "full";
			else status = "discharging";
		}

		//? Get seconds to empty
		if (not is_in(status, "charging", "full")) {
			if (b.use_energy_or_charge ) {
				if (not b.power_now.empty()) {
					try {
						seconds = abs(round(stod(readfile(b.energy_now, "0")) / stod(readfile(b.power_now, "1")) * 3600));
					}
					catch (const std::invalid_argument&) { }
					catch (const std::out_of_range&) { }
				}
				else if (not b.current_now.empty()) {
					try {
						seconds = abs(round(stod(readfile(b.charge_now, "0")) / stod(readfile(b.current_now, "1")) * 3600));
					}
					catch (const std::invalid_argument&) { }
					catch (const std::out_of_range&) { }
				}
			}

			if (seconds < 0 and fs::exists(b.base_dir / "time_to_empty")) {
				try {
					seconds = stoll(readfile(b.base_dir / "time_to_empty", "0")) * 60;
				}
				catch (const std::invalid_argument&) { }
				catch (const std::out_of_range&) { }
			}
		}
		//? Or get seconds to full
		else if(is_in(status, "charging")) {
			if (b.use_energy_or_charge ) {
				if (not b.power_now.empty()) {
					try {
						seconds = (round(stod(readfile(b.energy_full , "0")) - round(stod(readfile(b.energy_now, "0"))))
									/ abs(stod(readfile(b.power_now, "1"))) * 3600);
					}
					catch (const std::invalid_argument&) { }
					catch (const std::out_of_range&) { }
				}
				else if (not b.current_now.empty()) {
					try {
						seconds = (round(stod(readfile(b.charge_full , "0")) - stod(readfile(b.charge_now, "0")))
									/ std::abs(stod(readfile(b.current_now, "1"))) * 3600);
					}
					catch (const std::invalid_argument&) { }
					catch (const std::out_of_range&) { }
				}
			}
		}

		//? Get power draw
		if (b.use_power) {
			if (not b.power_now.empty()) {
				try {
					watts = stof(readfile(b.power_now, "-1")) / 1000000.0F;
				}
				catch (const std::invalid_argument&) { }
				catch (const std::out_of_range&) { }
			}
			else if (not b.voltage_now.empty() and not b.current_now.empty()) {
				try {
					watts = stof(readfile(b.current_now, "-1")) / 1000000.0F * stof(readfile(b.voltage_now, "1")) / 1000000.0F;
				}
				catch (const std::invalid_argument&) { }
				catch (const std::out_of_range&) { }
			}

		}

		return {percent, watts, seconds, status};
	}

	long long get_cpuConsumptionUJoules()
	{
		long long consumption = -1;
		const auto rapl_power_usage_path = "/sys/class/powercap/intel-rapl:0/energy_uj";
		std::ifstream file(rapl_power_usage_path);
		if(file.good())
		{
			file >> consumption;
		}
		return consumption;
	}

	float get_cpuConsumptionWatts()
	{
		static long long previous_usage = 0;
		static long long previous_timestamp = 0;

		if (previous_usage == 0)
		{
			previous_usage = get_cpuConsumptionUJoules();
			previous_timestamp = get_monotonicTimeUSec();
			supports_watts = (previous_usage > 0);
			return 0;
		}

		if (!supports_watts)
		{
			return -1;
		}

		auto current_timestamp = get_monotonicTimeUSec();
		auto current_usage = get_cpuConsumptionUJoules();

		auto watts = (float)(current_usage - previous_usage) / (float)(current_timestamp - previous_timestamp);

		previous_timestamp = current_timestamp;
		previous_usage = current_usage;

		return watts;
	}

    static constexpr auto to_int(std::string_view view) {
        std::uint32_t value {};
        std::from_chars(view.data(), view.data() + view.size(), value);
        return value;
    }

    static constexpr auto detect_active_cpus() {
        auto stream = std::ifstream { "/sys/fs/cgroup/cpuset.cpus.effective" };
        auto buf = std::string { std::istreambuf_iterator<char> { stream }, {} };

        if (buf.empty()) {
            return std::views::iota(0, Shared::coreCount) | std::ranges::to<std::vector<std::int32_t>>();
        }

        return buf | std::views::split(',') | std::views::transform([](auto&& range) -> auto {
                   auto view = std::string_view { range };
                   auto dash = view.find('-');

                   if (dash == std::string_view::npos) {
                       // Single CPU, return iota of single element
                       auto value = to_int(view);
                       return std::views::iota(value, value + 1);
                   }

                   auto start = to_int(view.substr(0, dash));
                   auto end = to_int(view.substr(dash + 1));
                   return std::views::iota(start, end + 1);
               }) |
               std::views::join | std::ranges::to<std::vector<std::int32_t>>();
    }

	auto collect(bool no_update) -> cpu_info& {
		if (Runner::stopping or (no_update and not current_cpu.cpu_percent.at("total").empty())) return current_cpu;
		auto& cpu = current_cpu;

		if (Config::getB("show_cpu_freq"))
			cpuHz = get_cpuHz();

		if (getloadavg(cpu.load_avg.data(), cpu.load_avg.size()) < 0) {
			Logger::error("failed to get load averages");
		}

		ifstream cread;

		try {
			//? Get cpu total times for all cores from /proc/stat
			string cpu_name;
			cread.open(Shared::procPath / "stat");
			int i = 0;
			int target = Shared::coreCount;
			for (; i <= target or (cread.good() and cread.peek() == 'c'); i++) {
				//? Make sure to add zero value for missing core values if at end of file
				if ((not cread.good() or cread.peek() != 'c') and i <= target) {
					if (i == 0) throw std::runtime_error("Failed to parse /proc/stat");
					else {
						//? Fix container sizes if new cores are detected
						while (cmp_less(cpu.core_percent.size(), i)) {
							core_old_totals.push_back(0);
							core_old_idles.push_back(0);
							cpu.core_percent.emplace_back();
						}
						cpu.core_percent.at(i-1).push_back(0);
					}
				}
				else {
					if (i == 0) cread.ignore(SSmax, ' ');
					else {
						cread >> cpu_name;
						int cpuNum = std::stoi(cpu_name.substr(3));
						if (cpuNum >= target - 1) target = cpuNum + (cread.peek() == 'c' ? 2 : 1);

						//? Add zero value for core if core number is missing from /proc/stat
						while (i - 1 < cpuNum) {
							//? Fix container sizes if new cores are detected
							while (cmp_less(cpu.core_percent.size(), i)) {
								core_old_totals.push_back(0);
								core_old_idles.push_back(0);
								cpu.core_percent.emplace_back();
							}
							cpu.core_percent[i-1].push_back(0);
							if (cpu.core_percent.at(i-1).size() > 40) cpu.core_percent.at(i-1).pop_front();
							i++;
						}
					}

					//? Expected on kernel 2.6.3> : 0=user, 1=nice, 2=system, 3=idle, 4=iowait, 5=irq, 6=softirq, 7=steal, 8=guest, 9=guest_nice
					vector<long long> times;
					long long total_sum = 0;

					for (uint64_t val; cread >> val; total_sum += val) {
						times.push_back(val);
					}
					cread.clear();
					if (times.size() < 4) throw std::runtime_error("Malformed /proc/stat");

					//? Subtract fields 8-9 and any future unknown fields
					const long long totals = max(0ll, total_sum - (times.size() > 8 ? std::accumulate(times.begin() + 8, times.end(), 0ll) : 0));

					//? Add iowait field if present
					const long long idles = max(0ll, times.at(3) + (times.size() > 4 ? times.at(4) : 0));

					//? Calculate values for totals from first line of stat
					if (i == 0) {
						const long long calc_totals = max(1ll, totals - cpu_old.at("totals"));
						const long long calc_idles = max(0ll, idles - cpu_old.at("idles"));
						cpu_old.at("totals") = totals;
						cpu_old.at("idles") = idles;

						//? Total usage of cpu
						cpu.cpu_percent.at("total").push_back(clamp((long long)round((double)(calc_totals - calc_idles) * 100 / calc_totals), 0ll, 100ll));

						//? Reduce size if there are more values than needed for graph
						while (cmp_greater(cpu.cpu_percent.at("total").size(), width * 2)) cpu.cpu_percent.at("total").pop_front();

						//? Populate cpu.cpu_percent with all fields from stat
						for (int ii = 0; const auto& val : times) {
							cpu.cpu_percent.at(time_names.at(ii)).push_back(clamp((long long)round((double)(val - cpu_old.at(time_names.at(ii))) * 100 / calc_totals), 0ll, 100ll));
							cpu_old.at(time_names.at(ii)) = val;

							//? Reduce size if there are more values than needed for graph
							while (cmp_greater(cpu.cpu_percent.at(time_names.at(ii)).size(), width * 2)) cpu.cpu_percent.at(time_names.at(ii)).pop_front();

							if (++ii == 10) break;
						}
						continue;
					}
					//? Calculate cpu total for each core
					else {
						//? Fix container sizes if new cores are detected
						while (cmp_less(cpu.core_percent.size(), i)) {
							core_old_totals.push_back(0);
							core_old_idles.push_back(0);
							cpu.core_percent.emplace_back();
						}
						const long long calc_totals = max(1ll, totals - core_old_totals.at(i-1));
						const long long calc_idles = max(0ll, idles - core_old_idles.at(i-1));
						core_old_totals.at(i-1) = totals;
						core_old_idles.at(i-1) = idles;

						cpu.core_percent.at(i-1).push_back(clamp((long long)round((double)(calc_totals - calc_idles) * 100 / calc_totals), 0ll, 100ll));
					}
				}

				//? Reduce size if there are more values than needed for graph
				if (cpu.core_percent.at(i-1).size() > 40) cpu.core_percent.at(i-1).pop_front();
			}

			//? Notify main thread to redraw screen if we found more cores than previously detected
			if (cmp_greater(cpu.core_percent.size(), Shared::coreCount)) {
				Logger::debug("Changing CPU max corecount from {} to {}.", Shared::coreCount, cpu.core_percent.size());
				Runner::coreNum_reset = true;
				Shared::coreCount = cpu.core_percent.size();
				while (cmp_less(current_cpu.temp.size(), cpu.core_percent.size() + 1)) current_cpu.temp.push_back({0});
			}

		}
		catch (const std::exception& e) {
			Logger::debug("Cpu::collect() : {}", e.what());
			if (cread.bad()) throw std::runtime_error("Failed to read /proc/stat");
			else throw std::runtime_error(fmt::format("Cpu::collect() : {}", e.what()));
		}

		if (Config::getB("check_temp") and got_sensors)
			update_sensors();

		if (Config::getB("show_battery") and has_battery)
			current_bat = get_battery();

		if (Config::getB("show_cpu_watts") and supports_watts)
			current_cpu.usage_watts = get_cpuConsumptionWatts();

		cpu.active_cpus = std::make_optional(detect_active_cpus());

		return cpu;
	}
}

#ifdef GPU_SUPPORT
namespace Gpu {
    //? NVIDIA
    namespace Nvml {
		bool init() {
			if (initialized) return false;

			//? Dynamic loading & linking
			//? Try possible library names for libnvidia-ml.so
			const array libNvAlts = {
				"libnvidia-ml.so",
				"libnvidia-ml.so.1",
			};

			for (const auto& l : libNvAlts) {
				nvml_dl_handle = dlopen(l, RTLD_LAZY);
				if (nvml_dl_handle != nullptr) {
					break;
				}
			}
 			if (!nvml_dl_handle) {
				Logger::info("Failed to load libnvidia-ml.so, NVIDIA GPUs will not be detected: {}", dlerror());
 				return false;
 			}

			auto load_nvml_sym = [&](const char sym_name[]) {
				auto sym = dlsym(nvml_dl_handle, sym_name);
				auto err = dlerror();
				if (err != nullptr) {
					Logger::error("NVML: Couldn't find function {}: {}", sym_name, err);
					return (void*)nullptr;
				} else return sym;
			};

            #define LOAD_SYM(NAME)  if ((NAME = (decltype(NAME))load_nvml_sym(#NAME)) == nullptr) return false

		    LOAD_SYM(nvmlErrorString);
		    LOAD_SYM(nvmlInit);
		    LOAD_SYM(nvmlShutdown);
		    LOAD_SYM(nvmlDeviceGetCount);
		    LOAD_SYM(nvmlDeviceGetHandleByIndex);
		    LOAD_SYM(nvmlDeviceGetName);
		    LOAD_SYM(nvmlDeviceGetPowerManagementLimit);
		    LOAD_SYM(nvmlDeviceGetTemperatureThreshold);
		    LOAD_SYM(nvmlDeviceGetUtilizationRates);
		    LOAD_SYM(nvmlDeviceGetClockInfo);
		    LOAD_SYM(nvmlDeviceGetPowerUsage);
		    LOAD_SYM(nvmlDeviceGetPowerState);
		    LOAD_SYM(nvmlDeviceGetTemperature);
		    LOAD_SYM(nvmlDeviceGetMemoryInfo);
		    LOAD_SYM(nvmlDeviceGetPcieThroughput);
			LOAD_SYM(nvmlDeviceGetEncoderUtilization);
			LOAD_SYM(nvmlDeviceGetDecoderUtilization);

            #undef LOAD_SYM

			//? Function calls
			nvmlReturn_t result = nvmlInit();
    		if (result != NVML_SUCCESS) {
    			Logger::debug("Failed to initialize NVML, NVIDIA GPUs will not be detected: {}", nvmlErrorString(result));
    			return false;
    		}

			//? Device count
			result = nvmlDeviceGetCount(&device_count);
    		if (result != NVML_SUCCESS) {
    			Logger::warning("NVML: Failed to get device count: {}", nvmlErrorString(result));
    			return false;
    		}

			if (device_count > 0) {
				devices.resize(device_count);
				gpus.resize(device_count);
				gpu_names.resize(device_count);

				initialized = true;

				//? Check supported functions & get maximums
				Nvml::collect<1>(gpus.data());

				return true;
			} else {initialized = true; shutdown(); return false;}
		}

		bool shutdown() {
			if (!initialized) return false;
			nvmlReturn_t result = nvmlShutdown();
			if (NVML_SUCCESS == result) {
				initialized = false;
				dlclose(nvml_dl_handle);
			} else Logger::warning("Failed to shutdown NVML: {}", nvmlErrorString(result));

			return !initialized;
		}

		template <bool is_init> // collect<1> is called in Nvml::init(), and populates gpus.supported_functions
		bool collect(gpu_info* gpus_slice) { // raw pointer to vector data, size == device_count
			if (!initialized) return false;

			nvmlReturn_t result;
			join_thread pcie_tx_thread;
			join_thread pcie_rx_thread;
			// DebugTimer nvTotalTimer("Nvidia Total");
			for (unsigned int i = 0; i < device_count; ++i) {
				if constexpr(is_init) {
					//? Device Handle
    				result = nvmlDeviceGetHandleByIndex(i, devices.data() + i);
        			if (result != NVML_SUCCESS) {
    					Logger::warning("NVML: Failed to get device handle: {}", nvmlErrorString(result));
						gpus[i].supported_functions = {false, false, false, false, false, false, false, false, false, false};
    					continue;
        			}

					//? Device name
					char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    				result = nvmlDeviceGetName(devices[i], name, NVML_DEVICE_NAME_BUFFER_SIZE);
        			if (result != NVML_SUCCESS)
    					Logger::warning("NVML: Failed to get device name: {}", nvmlErrorString(result));
        			else {
        				gpu_names[i] = string(name);
        				for (const auto& brand : {"NVIDIA", "Nvidia", "(R)", "(TM)"}) {
							gpu_names[i] = s_replace(gpu_names[i], brand, "");
						}
						gpu_names[i] = trim(gpu_names[i]);
        			}

    				//? Power usage
    				unsigned int max_power;
    				result = nvmlDeviceGetPowerManagementLimit(devices[i], &max_power);
    				if (result != NVML_SUCCESS)
						Logger::warning("NVML: Failed to get maximum GPU power draw, defaulting to 225W: {}", nvmlErrorString(result));
					else {
						gpus[i].pwr_max_usage = max_power; // RSMI reports power in microWatts
						gpu_pwr_total_max += max_power;
					}

					//? Get temp_max
					unsigned int temp_max;
    				result = nvmlDeviceGetTemperatureThreshold(devices[i], NVML_TEMPERATURE_THRESHOLD_SHUTDOWN, &temp_max);
        			if (result != NVML_SUCCESS)
    					Logger::warning("NVML: Failed to get maximum GPU temperature, defaulting to 110°C: {}", nvmlErrorString(result));
    				else gpus[i].temp_max = (long long)temp_max;
				}

				//? PCIe link speeds, the data collection takes >=20ms each call so they run on separate threads
				if (gpus_slice[i].supported_functions.pcie_txrx and (Config::getB("nvml_measure_pcie_speeds") or is_init)) {
					pcie_tx_thread = join_thread([gpus_slice, i]() {
						unsigned int tx;
						nvmlReturn_t result = nvmlDeviceGetPcieThroughput(devices[i], NVML_PCIE_UTIL_TX_BYTES, &tx);
    					if (result != NVML_SUCCESS) {
							Logger::warning("NVML: Failed to get PCIe TX throughput: {}", nvmlErrorString(result));
							if constexpr(is_init) gpus_slice[i].supported_functions.pcie_txrx = false;
						} else gpus_slice[i].pcie_tx = (long long)tx;
					});

					pcie_rx_thread = join_thread([gpus_slice, i]() {
						unsigned int rx;
						nvmlReturn_t result = nvmlDeviceGetPcieThroughput(devices[i], NVML_PCIE_UTIL_RX_BYTES, &rx);
    					if (result != NVML_SUCCESS) {
							Logger::warning("NVML: Failed to get PCIe RX throughput: {}", nvmlErrorString(result));
						} else gpus_slice[i].pcie_rx = (long long)rx;
					});
				} else {
					gpus_slice[i].pcie_tx = -1;
					gpus_slice[i].pcie_rx = -1;
				}

				// DebugTimer nvTimer("Nv utilization");
				//? GPU & memory utilization
				if (gpus_slice[i].supported_functions.gpu_utilization) {
					nvmlUtilization_t utilization;
					result = nvmlDeviceGetUtilizationRates(devices[i], &utilization);
    				if (result != NVML_SUCCESS) {
						Logger::warning("NVML: Failed to get GPU utilization: {}", nvmlErrorString(result));
						if constexpr(is_init) gpus_slice[i].supported_functions.gpu_utilization = false;
						if constexpr(is_init) gpus_slice[i].supported_functions.mem_utilization = false;
    				} else {
						gpus_slice[i].gpu_percent.at("gpu-totals").push_back((long long)utilization.gpu);
						gpus_slice[i].mem_utilization_percent.push_back((long long)utilization.memory);
    				}
				}

				// nvTimer.stop_rename_reset("Nv clock");
				//? Clock speeds
				if (gpus_slice[i].supported_functions.gpu_clock) {
					unsigned int gpu_clock;
					result = nvmlDeviceGetClockInfo(devices[i], NVML_CLOCK_GRAPHICS, &gpu_clock);
    				if (result != NVML_SUCCESS) {
						Logger::warning("NVML: Failed to get GPU clock speed: {}", nvmlErrorString(result));
						if constexpr(is_init) gpus_slice[i].supported_functions.gpu_clock = false;
					} else gpus_slice[i].gpu_clock_speed = (long long)gpu_clock;
				}

				if (gpus_slice[i].supported_functions.mem_clock) {
					unsigned int mem_clock;
					result = nvmlDeviceGetClockInfo(devices[i], NVML_CLOCK_MEM, &mem_clock);
    				if (result != NVML_SUCCESS) {
						Logger::warning("NVML: Failed to get VRAM clock speed: {}", nvmlErrorString(result));
						if constexpr(is_init) gpus_slice[i].supported_functions.mem_clock = false;
					} else gpus_slice[i].mem_clock_speed = (long long)mem_clock;
				}

				// nvTimer.stop_rename_reset("Nv power");
    			//? Power usage & state
				if (gpus_slice[i].supported_functions.pwr_usage) {
    				unsigned int power;
    				result = nvmlDeviceGetPowerUsage(devices[i], &power);
    				if (result != NVML_SUCCESS) {
						Logger::warning("NVML: Failed to get GPU power usage: {}", nvmlErrorString(result));
						if constexpr(is_init) gpus_slice[i].supported_functions.pwr_usage = false;
    				} else {
    					gpus_slice[i].pwr_usage = (long long)power;
						if (gpus_slice[i].pwr_usage > gpus_slice[i].pwr_max_usage)
								gpus_slice[i].pwr_max_usage = gpus_slice[i].pwr_usage;
    					gpus_slice[i].gpu_percent.at("gpu-pwr-totals").push_back(clamp((long long)round((double)gpus_slice[i].pwr_usage * 100.0 / (double)gpus_slice[i].pwr_max_usage), 0ll, 100ll));
    				}
    			}

				if (gpus_slice[i].supported_functions.pwr_state) {
					nvmlPstates_t pState;
    				result = nvmlDeviceGetPowerState(devices[i], &pState);
    				if (result != NVML_SUCCESS) {
						Logger::warning("NVML: Failed to get GPU power state: {}", nvmlErrorString(result));
						if constexpr(is_init) gpus_slice[i].supported_functions.pwr_state = false;
    				} else gpus_slice[i].pwr_state = static_cast<int>(pState);
    			}

				// nvTimer.stop_rename_reset("Nv temp");
    			//? GPU temperature
				if (gpus_slice[i].supported_functions.temp_info) {
    				if (Config::getB("check_temp")) {
						unsigned int temp;
						nvmlReturn_t result = nvmlDeviceGetTemperature(devices[i], NVML_TEMPERATURE_GPU, &temp);
    					if (result != NVML_SUCCESS) {
							Logger::warning("NVML: Failed to get GPU temperature: {}", nvmlErrorString(result));
							if constexpr(is_init) gpus_slice[i].supported_functions.temp_info = false;
    					} else gpus_slice[i].temp.push_back((long long)temp);
					}
				}

				// nvTimer.stop_rename_reset("Nv mem");
				//? Memory info
				if (gpus_slice[i].supported_functions.mem_total) {
					nvmlMemory_t memory;
					result = nvmlDeviceGetMemoryInfo(devices[i], &memory);
    				if (result != NVML_SUCCESS) {
						Logger::warning("NVML: Failed to get VRAM info: {}", nvmlErrorString(result));
						if constexpr(is_init) gpus_slice[i].supported_functions.mem_total = false;
						if constexpr(is_init) gpus_slice[i].supported_functions.mem_used = false;
					} else {
						gpus_slice[i].mem_total = memory.total;
						gpus_slice[i].mem_used = memory.used;
						//gpu.mem_free = memory.free;

						auto used_percent = (long long)round((double)memory.used * 100.0 / (double)memory.total);
						gpus_slice[i].gpu_percent.at("gpu-vram-totals").push_back(used_percent);
					}
				}

				// nvTimer.stop_rename_reset("Nv enc");
				//? Encoder info
				if (gpus_slice[i].supported_functions.encoder_utilization) {
					unsigned int utilization;
					unsigned int samplingPeriodUs;
					result = nvmlDeviceGetEncoderUtilization(devices[i], &utilization, &samplingPeriodUs);
					if (result != NVML_SUCCESS) {
						Logger::warning("NVML: Failed to get encoder utilization: {}", nvmlErrorString(result));
						if constexpr(is_init) gpus_slice[i].supported_functions.encoder_utilization = false;
					} else gpus_slice[i].encoder_utilization = (long long)utilization;
				}

				// nvTimer.stop_rename_reset("Nv dec");
				//? Decoder info
				if (gpus_slice[i].supported_functions.decoder_utilization) {
					unsigned int utilization;
					unsigned int samplingPeriodUs;
					result = nvmlDeviceGetDecoderUtilization(devices[i], &utilization, &samplingPeriodUs);
					if (result != NVML_SUCCESS) {
						Logger::warning("NVML: Failed to get decoder utilization: {}", nvmlErrorString(result));
						if constexpr(is_init) gpus_slice[i].supported_functions.decoder_utilization = false;
					} else gpus_slice[i].decoder_utilization = (long long)utilization;
				}

    			//? TODO: Processes using GPU
    				/*unsigned int proc_info_len;
    				nvmlProcessInfo_t* proc_info = 0;
    				result = nvmlDeviceGetComputeRunningProcesses_v3(device, &proc_info_len, proc_info);
    				if (result != NVML_SUCCESS) {
						Logger::warning("NVML: Failed to get compute processes: {}", nvmlErrorString(result));
    				} else {
    					for (unsigned int i = 0; i < proc_info_len; ++i)
    						gpus_slice[i].graphics_processes.push_back({proc_info[i].pid, proc_info[i].usedGpuMemory});
    				}*/

				// nvTimer.stop_rename_reset("Nv pcie thread join");
				//? Join PCIE TX/RX threads
				if constexpr(is_init) { // there doesn't seem to be a better way to do this, but this should be fine considering it's just 2 lines
					pcie_tx_thread.join();
					pcie_rx_thread.join();
				} else if (gpus_slice[i].supported_functions.pcie_txrx and Config::getB("nvml_measure_pcie_speeds")) {
					pcie_tx_thread.join();
					pcie_rx_thread.join();
				}
    		}

			return true;
		}
    }

	//? AMD
	namespace Rsmi {
		bool init() {
			if (initialized) return false;

			//? Dynamic loading & linking
		#if !defined(RSMI_STATIC)

			//? Try possible library paths and names for librocm_smi64.so
			const array libRocAlts = {
				"/opt/rocm/lib/librocm_smi64.so",
				"librocm_smi64.so",
				"librocm_smi64.so.5", // fedora
				"librocm_smi64.so.1.0", // debian
				"librocm_smi64.so.6",
				"librocm_smi64.so.7" // rocm 7 support / Ubuntu 26.04
			};

			for (const auto& l : libRocAlts) {
				rsmi_dl_handle = dlopen(l, RTLD_LAZY);
				if (rsmi_dl_handle != nullptr) {
					break;
				}
 			}

			if (!rsmi_dl_handle) {
				Logger::info("Failed to load librocm_smi64.so, AMD GPUs will not be detected: {}", dlerror());
				return false;
			}

			auto load_rsmi_sym = [&](const char sym_name[]) {
				auto sym = dlsym(rsmi_dl_handle, sym_name);
				auto err = dlerror();
				if (err != nullptr) {
					Logger::error("ROCm SMI: Couldn't find function {}: {}", sym_name, err);
					return (void*)nullptr;
				} else return sym;
			};

            #define LOAD_SYM(NAME)  if ((NAME = (decltype(NAME))load_rsmi_sym(#NAME)) == nullptr) return false

		    LOAD_SYM(rsmi_init);
		    LOAD_SYM(rsmi_shut_down);
			LOAD_SYM(rsmi_version_get);
		    LOAD_SYM(rsmi_num_monitor_devices);
		    LOAD_SYM(rsmi_dev_name_get);
		    LOAD_SYM(rsmi_dev_power_cap_get);
		    LOAD_SYM(rsmi_dev_temp_metric_get);
		    LOAD_SYM(rsmi_dev_busy_percent_get);
		    LOAD_SYM(rsmi_dev_memory_busy_percent_get);
		    LOAD_SYM(rsmi_dev_power_ave_get);
		    LOAD_SYM(rsmi_dev_memory_total_get);
		    LOAD_SYM(rsmi_dev_memory_usage_get);
		    LOAD_SYM(rsmi_dev_pci_throughput_get);

            #undef LOAD_SYM
        #endif

			//? Function calls
			rsmi_status_t result = rsmi_init(0);
			if (result != RSMI_STATUS_SUCCESS) {
				Logger::debug("Failed to initialize ROCm SMI, AMD GPUs will not be detected");
				return false;
			}

		#if !defined(RSMI_STATIC)
			//? Check version
			rsmi_version_t version;
			result = rsmi_version_get(&version);
			if (result != RSMI_STATUS_SUCCESS) {
				Logger::warning("ROCm SMI: Failed to get version");
				return false;
			}

			// Two distinct real-world libraries report version.major == 1:
			//   - ROCm 7.2 ships the v6 ABI (see upstream PR #1566).
			//   - Debian/Ubuntu's librocm-smi64 is built from 5.x sources but
			//     rocm_smi64Config.h reports version 1.0.0, so the ABI is v5.
			// Probe a 6.x-only symbol to disambiguate instead of guessing.
			uint32_t effective_major = version.major;
			if (version.major == 1) {
				bool has_v6_symbol = (dlsym(rsmi_dl_handle, "rsmi_dev_activity_metric_get") != nullptr);
				(void)dlerror(); // clear error state from the probe
				effective_major = has_v6_symbol ? 6 : 5;
				Logger::warning("ROCm SMI: library reports version 1.x; assuming {}.x ABI based on symbol probe",
					has_v6_symbol ? "6" : "5");
			}

			if (effective_major == 5) {
				if ((rsmi_dev_gpu_clk_freq_get_v5 = (decltype(rsmi_dev_gpu_clk_freq_get_v5))load_rsmi_sym("rsmi_dev_gpu_clk_freq_get")) == nullptr)
					return false;
			// In the release tarballs of rocm 6.0.0 and 6.0.2 the version queried with rsmi_version_get is 7.0.0.0
			} else if (effective_major == 6 || effective_major == 7) {
				if ((rsmi_dev_gpu_clk_freq_get_v6 = (decltype(rsmi_dev_gpu_clk_freq_get_v6))load_rsmi_sym("rsmi_dev_gpu_clk_freq_get")) == nullptr)
					return false;
			} else {
				Logger::warning("ROCm SMI: Dynamic loading only supported for version 5 and 6");
				return false;
			}
			version_major = effective_major;
		#endif

			//? Device count
			result = rsmi_num_monitor_devices(&device_count);
			if (result != RSMI_STATUS_SUCCESS) {
				Logger::warning("ROCm SMI: Failed to fetch number of devices");
				return false;
			}

			if (device_count > 0) {
				gpus.resize(gpus.size() + device_count);
				gpu_names.resize(gpus.size() + device_count);

				initialized = true;

				//? Check supported functions & get maximums
				Rsmi::collect<1>(gpus.data() + Nvml::device_count);

				return true;
			} else {initialized = true; shutdown(); return false;}
		}

		bool shutdown() {
			if (!initialized) return false;
    		if (rsmi_shut_down() == RSMI_STATUS_SUCCESS) {
				initialized = false;
			#if !defined(RSMI_STATIC)
				dlclose(rsmi_dl_handle);
			#endif
			} else Logger::warning("Failed to shutdown ROCm SMI");

			return true;
		}

		template <bool is_init>
		bool collect(gpu_info* gpus_slice) { // raw pointer to vector data, size == device_count, offset by Nvml::device_count elements
			if (!initialized) return false;
			rsmi_status_t result;

			for (uint32_t i = 0; i < device_count; ++i) {
				if constexpr(is_init) {
					//? Device name
					char name[RSMI_DEVICE_NAME_BUFFER_SIZE];
    				result = rsmi_dev_name_get(i, name, RSMI_DEVICE_NAME_BUFFER_SIZE);
        			if (result != RSMI_STATUS_SUCCESS)
    					Logger::warning("ROCm SMI: Failed to get device name");
        			else gpu_names[Nvml::device_count + i] = string(name);

    				//? Power usage
    				uint64_t max_power;
    				result = rsmi_dev_power_cap_get(i, 0, &max_power);
    				if (result != RSMI_STATUS_SUCCESS)
						Logger::warning("ROCm SMI: Failed to get maximum GPU power draw, defaulting to 225W");
					else {
						gpus_slice[i].pwr_max_usage = (long long)(max_power/1000); // RSMI reports power in microWatts
						gpu_pwr_total_max += gpus_slice[i].pwr_max_usage;
					}

					//? Get temp_max
					int64_t temp_max;
    				result = rsmi_dev_temp_metric_get(i, RSMI_TEMP_TYPE_EDGE, RSMI_TEMP_MAX, &temp_max);
        			if (result != RSMI_STATUS_SUCCESS)
    					Logger::warning("ROCm SMI: Failed to get maximum GPU temperature, defaulting to 110°C");
    				else gpus_slice[i].temp_max = (long long)temp_max;

					//? Disable encoder and decoder utilisation on AMD
					gpus_slice[i].supported_functions.encoder_utilization = false;
					gpus_slice[i].supported_functions.decoder_utilization = false;
    			}

				//? GPU utilization
				if (gpus_slice[i].supported_functions.gpu_utilization) {
					uint32_t utilization;
					result = rsmi_dev_busy_percent_get(i, &utilization);
    				if (result != RSMI_STATUS_SUCCESS) {
						Logger::warning("ROCm SMI: Failed to get GPU utilization");
						if constexpr(is_init) gpus_slice[i].supported_functions.gpu_utilization = false;
    				} else gpus_slice[i].gpu_percent.at("gpu-totals").push_back((long long)utilization);
				}

				//? Memory utilization
				if (gpus_slice[i].supported_functions.mem_utilization) {
					uint32_t utilization;
					result = rsmi_dev_memory_busy_percent_get(i, &utilization);
    				if (result != RSMI_STATUS_SUCCESS) {
						Logger::warning("ROCm SMI: Failed to get VRAM utilization");
						if constexpr(is_init) gpus_slice[i].supported_functions.mem_utilization = false;
    				} else gpus_slice[i].mem_utilization_percent.push_back((long long)utilization);
				}
			#if !defined(RSMI_STATIC)
				//? Clock speeds
				// Some AMD devices (e.g. integrated GPUs with the Debian 5.7.0 rocm_smi package) return
				// RSMI_STATUS_SUCCESS from rsmi_dev_gpu_clk_freq_get but leave frequencies.current
				// uninitialized, which crashes when used as an array index. Bound-check before indexing.
				if (gpus_slice[i].supported_functions.gpu_clock) {
					if (version_major == 5) {
						rsmi_frequencies_t_v5 frequencies{};
						result = rsmi_dev_gpu_clk_freq_get_v5(i, RSMI_CLK_TYPE_SYS, &frequencies);
						if (result != RSMI_STATUS_SUCCESS) {
							Logger::warning("ROCm SMI: Failed to get GPU clock speed: ");
							if constexpr(is_init) gpus_slice[i].supported_functions.gpu_clock = false;
						} else if (frequencies.num_supported == 0 || frequencies.current >= frequencies.num_supported
								|| frequencies.num_supported > RSMI_MAX_NUM_FREQUENCIES_V5) {
							Logger::warning("ROCm SMI: GPU clock speed unsupported on this device");
							if constexpr(is_init) gpus_slice[i].supported_functions.gpu_clock = false;
						} else gpus_slice[i].gpu_clock_speed = (long long)frequencies.frequency[frequencies.current]/1000000; // Hz to MHz
					}
					else if (version_major == 6 || version_major == 7) {
						rsmi_frequencies_t_v6 frequencies{};
						result = rsmi_dev_gpu_clk_freq_get_v6(i, RSMI_CLK_TYPE_SYS, &frequencies);
						if (result != RSMI_STATUS_SUCCESS) {
							Logger::warning("ROCm SMI: Failed to get GPU clock speed: ");
							if constexpr(is_init) gpus_slice[i].supported_functions.gpu_clock = false;
						} else if (frequencies.num_supported == 0 || frequencies.current >= frequencies.num_supported
								|| frequencies.num_supported > RSMI_MAX_NUM_FREQUENCIES_V6) {
							Logger::warning("ROCm SMI: GPU clock speed unsupported on this device");
							if constexpr(is_init) gpus_slice[i].supported_functions.gpu_clock = false;
						} else gpus_slice[i].gpu_clock_speed = (long long)frequencies.frequency[frequencies.current]/1000000; // Hz to MHz
					}
				}

				if (gpus_slice[i].supported_functions.mem_clock) {
					if (version_major == 5) {
						rsmi_frequencies_t_v5 frequencies{};
						result = rsmi_dev_gpu_clk_freq_get_v5(i, RSMI_CLK_TYPE_MEM, &frequencies);
						if (result != RSMI_STATUS_SUCCESS) {
							Logger::warning("ROCm SMI: Failed to get VRAM clock speed: ");
							if constexpr(is_init) gpus_slice[i].supported_functions.mem_clock = false;
						} else if (frequencies.num_supported == 0 || frequencies.current >= frequencies.num_supported
								|| frequencies.num_supported > RSMI_MAX_NUM_FREQUENCIES_V5) {
							Logger::warning("ROCm SMI: VRAM clock speed unsupported on this device");
							if constexpr(is_init) gpus_slice[i].supported_functions.mem_clock = false;
						} else gpus_slice[i].mem_clock_speed = (long long)frequencies.frequency[frequencies.current]/1000000; // Hz to MHz
					}
					else if (version_major == 6 || version_major == 7) {
						rsmi_frequencies_t_v6 frequencies{};
						result = rsmi_dev_gpu_clk_freq_get_v6(i, RSMI_CLK_TYPE_MEM, &frequencies);
						if (result != RSMI_STATUS_SUCCESS) {
							Logger::warning("ROCm SMI: Failed to get VRAM clock speed: ");
							if constexpr(is_init) gpus_slice[i].supported_functions.mem_clock = false;
						} else if (frequencies.num_supported == 0 || frequencies.current >= frequencies.num_supported
								|| frequencies.num_supported > RSMI_MAX_NUM_FREQUENCIES_V6) {
							Logger::warning("ROCm SMI: VRAM clock speed unsupported on this device");
							if constexpr(is_init) gpus_slice[i].supported_functions.mem_clock = false;
						} else gpus_slice[i].mem_clock_speed = (long long)frequencies.frequency[frequencies.current]/1000000; // Hz to MHz
					}
				}
			#else
				//? Clock speeds
				// Same defensive pattern as the dynamic path above: bound-check the reported
				// frequencies before indexing, in case the driver returns SUCCESS without
				// populating the struct.
				if (gpus_slice[i].supported_functions.gpu_clock) {
					rsmi_frequencies_t frequencies{};
					result = rsmi_dev_gpu_clk_freq_get(i, RSMI_CLK_TYPE_SYS, &frequencies);
    				if (result != RSMI_STATUS_SUCCESS) {
						Logger::warning("ROCm SMI: Failed to get GPU clock speed: ");
						if constexpr(is_init) gpus_slice[i].supported_functions.gpu_clock = false;
					} else if (frequencies.num_supported == 0 || frequencies.current >= frequencies.num_supported
							|| frequencies.num_supported > RSMI_MAX_NUM_FREQUENCIES) {
						Logger::warning("ROCm SMI: GPU clock speed unsupported on this device");
						if constexpr(is_init) gpus_slice[i].supported_functions.gpu_clock = false;
    				} else gpus_slice[i].gpu_clock_speed = (long long)frequencies.frequency[frequencies.current]/1000000; // Hz to MHz
				}

				if (gpus_slice[i].supported_functions.mem_clock) {
					rsmi_frequencies_t frequencies{};
					result = rsmi_dev_gpu_clk_freq_get(i, RSMI_CLK_TYPE_MEM, &frequencies);
    				if (result != RSMI_STATUS_SUCCESS) {
						Logger::warning("ROCm SMI: Failed to get VRAM clock speed: ");
						if constexpr(is_init) gpus_slice[i].supported_functions.mem_clock = false;
					} else if (frequencies.num_supported == 0 || frequencies.current >= frequencies.num_supported
							|| frequencies.num_supported > RSMI_MAX_NUM_FREQUENCIES) {
						Logger::warning("ROCm SMI: VRAM clock speed unsupported on this device");
						if constexpr(is_init) gpus_slice[i].supported_functions.mem_clock = false;
    				} else gpus_slice[i].mem_clock_speed = (long long)frequencies.frequency[frequencies.current]/1000000; // Hz to MHz
				}
			#endif
    			//? Power usage & state
				if (gpus_slice[i].supported_functions.pwr_usage) {
    				uint64_t power;
    				result = rsmi_dev_power_ave_get(i, 0, &power);
    				if (result != RSMI_STATUS_SUCCESS) {
						Logger::warning("ROCm SMI: Failed to get GPU power usage");
						if constexpr(is_init) gpus_slice[i].supported_functions.pwr_usage = false;
    				} else {
							gpus_slice[i].pwr_usage = (long long)power / 1000;
							if (gpus_slice[i].pwr_usage > gpus_slice[i].pwr_max_usage)
								gpus_slice[i].pwr_max_usage = gpus_slice[i].pwr_usage;
							gpus_slice[i].gpu_percent.at("gpu-pwr-totals").push_back(clamp((long long)round((double)gpus_slice[i].pwr_usage * 100.0 / (double)gpus_slice[i].pwr_max_usage), 0ll, 100ll));
						}

					if constexpr(is_init) gpus_slice[i].supported_functions.pwr_state = false;
				}

    			//? GPU temperature
				if (gpus_slice[i].supported_functions.temp_info) {
    				if (Config::getB("check_temp") or is_init) {
						int64_t temp;
    					result = rsmi_dev_temp_metric_get(i, RSMI_TEMP_TYPE_EDGE, RSMI_TEMP_CURRENT, &temp);
        				if (result != RSMI_STATUS_SUCCESS) {
    						Logger::warning("ROCm SMI: Failed to get GPU temperature");
							if constexpr(is_init) gpus_slice[i].supported_functions.temp_info = false;
    					} else gpus_slice[i].temp.push_back((long long)temp/1000);
    				}
				}

				//? Memory info
				if (gpus_slice[i].supported_functions.mem_total) {
					uint64_t total;
					result = rsmi_dev_memory_total_get(i, RSMI_MEM_TYPE_VRAM, &total);
    				if (result != RSMI_STATUS_SUCCESS) {
						Logger::warning("ROCm SMI: Failed to get total VRAM");
						if constexpr(is_init) gpus_slice[i].supported_functions.mem_total = false;
					} else gpus_slice[i].mem_total = total;
				}

				if (gpus_slice[i].supported_functions.mem_used) {
					uint64_t used;
					result = rsmi_dev_memory_usage_get(i, RSMI_MEM_TYPE_VRAM, &used);
    				if (result != RSMI_STATUS_SUCCESS) {
						Logger::warning("ROCm SMI: Failed to get VRAM usage");
						if constexpr(is_init) gpus_slice[i].supported_functions.mem_used = false;
					} else {
						gpus_slice[i].mem_used = used;
						if (gpus_slice[i].supported_functions.mem_total)
							gpus_slice[i].gpu_percent.at("gpu-vram-totals").push_back((long long)round((double)used * 100.0 / (double)gpus_slice[i].mem_total));
					}
				}

				//? PCIe link speeds
				if ((gpus_slice[i].supported_functions.pcie_txrx and Config::getB("rsmi_measure_pcie_speeds")) or is_init) {
					uint64_t tx, rx;
					result = rsmi_dev_pci_throughput_get(i, &tx, &rx, nullptr);
    				if (result != RSMI_STATUS_SUCCESS) {
						Logger::warning("ROCm SMI: Failed to get PCIe throughput");
						if constexpr(is_init) gpus_slice[i].supported_functions.pcie_txrx = false;
					} else {
						gpus_slice[i].pcie_tx = (long long)tx;
						gpus_slice[i].pcie_rx = (long long)rx;
					}
				} else {
					gpus_slice[i].pcie_tx = -1;
					gpus_slice[i].pcie_rx = -1;
				}
    		}

			return true;
		}
	}

	namespace Intel {
		bool init() {
			if (initialized) return false;

			char *gpu_path = find_intel_gpu_dir();
			if (!gpu_path) {
				Logger::debug("Failed to find Intel GPU sysfs path, Intel GPUs will not be detected");
				return false;
			}

			char *gpu_device_id = get_intel_device_id(gpu_path);
			if (!gpu_device_id) {
				Logger::debug("Failed to find Intel GPU device ID, Intel GPUs will not be detected");
				return false;
			}

			char *gpu_device_name = get_intel_device_name(gpu_device_id);
			if (!gpu_device_name) {
				Logger::warning("Failed to find Intel GPU device name in internal database");
			}

			free(gpu_device_id);

			engines = discover_engines(device);
			if (!engines) {
				Logger::debug("Failed to find Intel GPU engines, Intel GPUs will not be detected");
				free(gpu_device_name);
				return false;
			}

			int ret = pmu_init(engines);
			if (ret) {
				Logger::warning("Intel GPU: Failed to initialize PMU");
				free(gpu_device_name);
				free_engines(engines);
				engines = nullptr;
				return false;
			}

			pmu_sample(engines);

			device_count = 1;

			gpus.resize(gpus.size() + device_count);
			gpu_names.resize(gpus.size() + device_count);

			if (gpu_device_name) {
				gpu_names[Nvml::device_count + Rsmi::device_count + Asysfs::device_count] = string(gpu_device_name);
			} else {
				gpu_names[Nvml::device_count + Rsmi::device_count + Asysfs::device_count] = "Intel GPU";
			}

			free(gpu_device_name);

			initialized = true;
			Intel::collect<1>(gpus.data() + Nvml::device_count + Rsmi::device_count + Asysfs::device_count);

			return true;
		}

		bool shutdown() {
			if (!initialized) return false;
			if (engines) {
				free_engines(engines);
				engines = nullptr;
			}
			initialized = false;
			return true;
		}

		template <bool is_init> bool collect(gpu_info* gpus_slice) {
			if (!initialized) return false;

			if constexpr(is_init) {
				gpus_slice->supported_functions = {
					.gpu_utilization = true,
					.mem_utilization = false,
					.gpu_clock = true,
					.mem_clock = false,
					.pwr_usage = true,
					.pwr_state = false,
					.temp_info = false,
					.mem_total = false,
					.mem_used = false,
					.pcie_txrx = false,
					.encoder_utilization = false,
					.decoder_utilization = false
				};

				gpus_slice->pwr_max_usage = 10'000; //? 10W
			}

			pmu_sample(engines);
			double t = (double)(engines->ts.cur - engines->ts.prev) / 1e9;

			double max_util = 0;
			for (unsigned int i = 0; i < engines->num_engines; i++) {
				struct engine *engine = &(&engines->engine)[i];
				double util = pmu_calc(&engine->busy.val, 1e9, t, 100);
				if (util > max_util) {
					max_util = util;
				}
			}
			gpus_slice->gpu_percent.at("gpu-totals").push_back((long long)round(max_util));

			double pwr = pmu_calc(&engines->r_gpu.val, 1, t, engines->r_gpu.scale); // in Watts
			gpus_slice->pwr_usage = (long long)round(pwr * 1000);
			if (gpus_slice->pwr_usage > gpus_slice->pwr_max_usage)
				gpus_slice->pwr_max_usage = gpus_slice->pwr_usage;

			gpus_slice->gpu_percent.at("gpu-pwr-totals").push_back(clamp((long long)round((double)gpus_slice->pwr_usage * 100.0 / (double)gpus_slice->pwr_max_usage), 0ll, 100ll));

			double freq = pmu_calc(&engines->freq_act.val, 1, t, 1); // in MHz
			gpus_slice->gpu_clock_speed = (unsigned int)round(freq);

			return true;
		}
	}

	namespace Intel::Sysfs {
		//? Read a sysfs node containing a single integer; return fallback on missing/parse error.
		static long long read_ll(const std::filesystem::path& path, long long fallback = 0) {
			try {
				return std::stoll(readfile(path, std::to_string(fallback)));
			} catch (const std::exception&) {
				return fallback;
			}
		}

		//? Match /sys/class/drm/cardN (no '-', all digits after "card"). Skips card1-DP-1, renderD*, etc.
		static bool is_card_node(const string& fname) {
			if (not fname.starts_with("card") or fname.size() <= 4) return false;
			return std::ranges::all_of(fname.begin() + 4, fname.end(),
				[](char c) { return c >= '0' and c <= '9'; });
		}

		//? Pick the first hwmon* subdirectory under <device>/hwmon, or empty path if none.
		static std::filesystem::path find_hwmon(const std::filesystem::path& device) {
			const auto hwmon_dir = device / "hwmon";
			std::error_code ec;
			if (not std::filesystem::is_directory(hwmon_dir, ec)) return {};
			for (const auto& h : std::filesystem::directory_iterator(hwmon_dir, ec)) {
				if (h.is_directory()) return h.path();
			}
			return {};
		}

		//? Discover GT residency nodes for a single card: try the xe layout first, then
		//? i915's multi-GT layout (Meteor Lake+), then i915's legacy single-GT layout.
		static void discover_gts(const std::filesystem::path& dev, const string& driver, vector<GtNode>& out) {
			std::error_code ec;

			if (driver == "xe") {
				for (const auto& tentry : std::filesystem::directory_iterator(dev, ec)) {
					if (ec) break;
					const string tname = tentry.path().filename().string();
					if (not tname.starts_with("tile")) continue;
					std::error_code gec;
					for (const auto& gentry : std::filesystem::directory_iterator(tentry.path(), gec)) {
						const string gname = gentry.path().filename().string();
						if (not gname.starts_with("gt")) continue;
						const auto gpath = gentry.path();
						const auto res = gpath / "gtidle" / "idle_residency_ms";
						if (not std::filesystem::exists(res)) continue;

						GtNode gt;
						gt.label = tname + "/" + gname;
						gt.residency_path = res;
						if (std::filesystem::exists(gpath / "freq0" / "act_freq")) gt.freq_act_path = gpath / "freq0" / "act_freq";
						if (std::filesystem::exists(gpath / "freq0" / "cur_freq")) gt.freq_cur_path = gpath / "freq0" / "cur_freq";
						if (std::filesystem::exists(gpath / "freq0" / "rp0_freq")) gt.freq_max_path = gpath / "freq0" / "rp0_freq";
						else if (std::filesystem::exists(gpath / "freq0" / "max_freq")) gt.freq_max_path = gpath / "freq0" / "max_freq";
						out.push_back(std::move(gt));
					}
				}
				return;
			}

			if (driver == "i915") {
				const auto card = dev.parent_path(); //? dev == <card>/device, so card == dev.parent_path()
				const auto gtdir = card / "gt";
				if (std::filesystem::is_directory(gtdir, ec)) {
					for (const auto& gentry : std::filesystem::directory_iterator(gtdir, ec)) {
						const string gname = gentry.path().filename().string();
						if (not gname.starts_with("gt")) continue;
						const auto gpath = gentry.path();
						const auto res = gpath / "rc6_residency_ms";
						if (not std::filesystem::exists(res)) continue;

						GtNode gt;
						gt.label = gname;
						gt.residency_path = res;
						if (std::filesystem::exists(gpath / "rps_act_freq_mhz")) gt.freq_act_path = gpath / "rps_act_freq_mhz";
						if (std::filesystem::exists(gpath / "rps_cur_freq_mhz")) gt.freq_cur_path = gpath / "rps_cur_freq_mhz";
						if (std::filesystem::exists(gpath / "rps_RP0_freq_mhz")) gt.freq_max_path = gpath / "rps_RP0_freq_mhz";
						else if (std::filesystem::exists(gpath / "rps_max_freq_mhz")) gt.freq_max_path = gpath / "rps_max_freq_mhz";
						out.push_back(std::move(gt));
					}
				}

				//? Single-GT / older i915: card-level legacy attributes, only if the per-GT
				//? directory above yielded nothing.
				if (out.empty()) {
					const auto res = card / "power" / "rc6_residency_ms";
					if (std::filesystem::exists(res)) {
						GtNode gt;
						gt.label = "gt0";
						gt.residency_path = res;
						if (std::filesystem::exists(card / "gt_act_freq_mhz")) gt.freq_act_path = card / "gt_act_freq_mhz";
						if (std::filesystem::exists(card / "gt_cur_freq_mhz")) gt.freq_cur_path = card / "gt_cur_freq_mhz";
						if (std::filesystem::exists(card / "gt_RP0_freq_mhz")) gt.freq_max_path = card / "gt_RP0_freq_mhz";
						else if (std::filesystem::exists(card / "gt_max_freq_mhz")) gt.freq_max_path = card / "gt_max_freq_mhz";
						out.push_back(std::move(gt));
					}
				}
			}
		}

		bool init() {
			if (initialized) return false;
			devices.clear();

			const std::filesystem::path drm_root("/sys/class/drm");
			std::error_code ec;
			if (not std::filesystem::is_directory(drm_root, ec)) {
				Logger::debug("Intel sysfs: /sys/class/drm not present");
				return false;
			}

			vector<string> device_names;

			for (const auto& entry : std::filesystem::directory_iterator(drm_root, ec)) {
				if (not is_card_node(entry.path().filename().string())) continue;

				const auto device_link = entry.path() / "device";
				if (not std::filesystem::exists(device_link)) continue;

				//? Vendor must be exactly 0x8086 (Intel). The sysfs node ends with a newline,
				//? so trim before comparing.
				string vendor = readfile(device_link / "vendor", "");
				while (not vendor.empty() and (vendor.back() == '\n' or vendor.back() == ' ')) vendor.pop_back();
				if (vendor != "0x8086") continue;

				//? Dynamically detect the bound driver (i915 or xe) instead of assuming one —
				//? this is the fix for GPUs the legacy PMU backend above can't see at all
				//? because it hardcodes "i915".
				std::error_code dec;
				const auto driver_link = std::filesystem::read_symlink(device_link / "driver", dec);
				if (dec) continue;
				const string driver = driver_link.filename().string();
				if (driver != "i915" and driver != "xe") continue;

				device_paths d{};
				d.dev = device_link;
				try {
					//? "device" file holds e.g. "0x56a0\n" — base 0 lets stoul autodetect the 0x prefix.
					d.pci_device_id = (uint32_t)std::stoul(readfile(device_link / "device", "0"), nullptr, 0);
				} catch (const std::exception&) {
					d.pci_device_id = 0;
				}
				d.hwmon = find_hwmon(device_link);
				if (not d.hwmon.empty()) {
					//? Only an actual power source file, not merely the hwmon directory,
					//? guarantees collect() below will have something to report every tick.
					d.has_power = std::filesystem::exists(d.hwmon / "power1_average")
						or std::filesystem::exists(d.hwmon / "power1_input")
						or std::filesystem::exists(d.hwmon / "energy1_input");
				}
				discover_gts(device_link, driver, d.gts);

				//? No residency counters on this card means there's nothing this backend can
				//? report for it — let the next tier (legacy PMU) have a shot instead.
				if (d.gts.empty()) {
					Logger::debug("Intel sysfs: skipping {} — no GT residency counters found", device_link.string());
					continue;
				}

				//? Reuse the vendored Intel device-name lookup (same database the legacy PMU
				//? backend uses below) so naming stays consistent across backends. Pass this
				//? card's own directory (not find_intel_gpu_dir(), which only ever returns the
				//? first Intel card found and would misname every GPU past the first).
				string name;
				char *gpu_device_id_c = get_intel_device_id(entry.path().c_str());
				if (gpu_device_id_c) {
					char *gpu_device_name_c = get_intel_device_name(gpu_device_id_c);
					if (gpu_device_name_c) {
						name = string(gpu_device_name_c);
						free(gpu_device_name_c);
					}
					free(gpu_device_id_c);
				}
				if (name.empty()) name = fmt::format("Intel GPU [{:04x}]", d.pci_device_id);

				device_names.push_back(std::move(name));
				devices.push_back(std::move(d));
			}

			device_count = (uint32_t)devices.size();
			if (device_count == 0) {
				Logger::debug("Intel sysfs: no Intel GPUs with GT residency counters found");
				return false;
			}

			gpus.resize(gpus.size() + device_count);
			gpu_names.resize(Nvml::device_count + Rsmi::device_count + Asysfs::device_count + device_count);
			for (uint32_t i = 0; i < device_count; ++i) {
				gpu_names[Nvml::device_count + Rsmi::device_count + Asysfs::device_count + i] = device_names[i];
			}

			initialized = true;
			Logger::info("Using Intel sysfs (GT residency) for {} Intel GPU(s)", device_count);
			Sysfs::collect<1>(gpus.data() + Nvml::device_count + Rsmi::device_count + Asysfs::device_count);
			return true;
		}

		bool shutdown() {
			if (not initialized) return false;
			devices.clear();
			device_count = 0;
			initialized = false;
			return true;
		}

		template <bool is_init> bool collect(gpu_info* gpus_slice) {
			if (not initialized) return false;

			const long long t_us = get_monotonicTimeUSec();

			for (uint32_t i = 0; i < device_count; ++i) {
				gpu_info& gpu = gpus_slice[i];
				device_paths& d = devices[i];

				bool have_freq_path = false;
				for (auto& gt : d.gts) {
					if (not gt.freq_act_path.empty() or not gt.freq_cur_path.empty()) { have_freq_path = true; break; }
				}

				if constexpr (is_init) {
					gpu.supported_functions = {
						.gpu_utilization = true,
						.mem_utilization = false,
						.gpu_clock = have_freq_path,
						.mem_clock = false,
						.pwr_usage = d.has_power,
						.pwr_state = false,
						.temp_info = not d.hwmon.empty() and std::filesystem::exists(d.hwmon / "temp1_input"),
						.mem_total = false, //? Intel has no simple universal sysfs VRAM node; deferred, see INTEL_XPU.md
						.mem_used = false,
						.pcie_txrx = false,
						.encoder_utilization = false,
						.decoder_utilization = false,
					};
					gpu.pwr_max_usage = 0;
				}

				//? Busy% = max(100 - idle%) across this card's GTs, derived from the delta in
				//? RC6/idle-residency over wall time. This is immune to the legacy PMU tier's
				//? "max over individual engine instances" undercount — see INTEL_XPU.md.
				double best_busy = -1.0;
				double act_sum = 0;
				int act_cnt = 0;
				for (auto& gt : d.gts) {
					const long long res = read_ll(gt.residency_path, -1);
					if (res >= 0) {
						if (gt.prev_res_ms >= 0 and t_us > gt.prev_t_us) {
							const double dwall_ms = (double)(t_us - gt.prev_t_us) / 1000.0;
							const double dres_ms = (double)(res - gt.prev_res_ms);
							if (dwall_ms > 0) {
								const double idle_pct = std::clamp(100.0 * dres_ms / dwall_ms, 0.0, 100.0);
								const double busy = 100.0 - idle_pct;
								if (busy > best_busy) best_busy = busy;
							}
						}
						gt.prev_res_ms = res;
						gt.prev_t_us = t_us;
					}

					//? act_freq reads 0 while the GT is parked in RC6; cur_freq (requested
					//? frequency) is the meaningful number then.
					if (not gt.freq_act_path.empty() or not gt.freq_cur_path.empty()) {
						long long f = gt.freq_act_path.empty() ? 0 : read_ll(gt.freq_act_path, 0);
						if (f <= 0 and not gt.freq_cur_path.empty()) f = read_ll(gt.freq_cur_path, 0);
						if (f > 0) { act_sum += (double)f; act_cnt++; }
					}
				}

				if (best_busy >= 0.0) {
					gpu.gpu_percent.at("gpu-totals").push_back((long long)std::round(best_busy));
				} else if constexpr (is_init) {
					//? Seed the graph for callers that expect a value immediately after init.
					gpu.gpu_percent.at("gpu-totals").push_back(0);
				}

				if (act_cnt > 0) {
					gpu.gpu_clock_speed = (unsigned int)std::round(act_sum / act_cnt);
				}

				if (d.has_power) {
					const auto avg_path = d.hwmon / "power1_average";
					const auto inst_path = d.hwmon / "power1_input";
					long long pw_uw = -1;
					if (std::filesystem::exists(avg_path)) pw_uw = read_ll(avg_path, -1);
					else if (std::filesystem::exists(inst_path)) pw_uw = read_ll(inst_path, -1);

					if (pw_uw < 0) {
						//? Neither instantaneous power file worked (or exists) — fall back to
						//? an energy-counter delta, the same technique xpu-top uses for cards
						//? exposing only energy1_input (cumulative microjoules, monotonic).
						const auto energy_path = d.hwmon / "energy1_input";
						if (std::filesystem::exists(energy_path)) {
							const long long energy_uj = read_ll(energy_path, -1);
							if (energy_uj >= 0) {
								if (d.prev_energy_uj >= 0 and t_us > d.prev_energy_t_us and energy_uj >= d.prev_energy_uj) {
									const double dt_s = (double)(t_us - d.prev_energy_t_us) / 1e6;
									if (dt_s > 0) {
										const double watts = (double)(energy_uj - d.prev_energy_uj) / 1e6 / dt_s; //? uJ/s -> W
										pw_uw = (long long)std::round(watts * 1e6); //? back to uW so the logic below stays uniform
									}
								}
								d.prev_energy_uj = energy_uj;
								d.prev_energy_t_us = t_us;
							}
						}
					}

					if (pw_uw >= 0) {
						gpu.pwr_usage = pw_uw / 1000; //? microwatts -> milliwatts

						if constexpr (is_init) {
							//? Prefer the driver-reported cap over letting the observed peak set
							//? the scale (power1_max -> power1_rated_max -> power1_cap).
							long long cap_uw = -1;
							if (std::filesystem::exists(d.hwmon / "power1_max")) cap_uw = read_ll(d.hwmon / "power1_max", -1);
							else if (std::filesystem::exists(d.hwmon / "power1_rated_max")) cap_uw = read_ll(d.hwmon / "power1_rated_max", -1);
							else if (std::filesystem::exists(d.hwmon / "power1_cap")) cap_uw = read_ll(d.hwmon / "power1_cap", -1);
							if (cap_uw > 0) gpu.pwr_max_usage = cap_uw / 1000;
						}
						gpu.pwr_max_usage = std::max(gpu.pwr_max_usage, gpu.pwr_usage);

						if (gpu.pwr_max_usage > 0) {
							gpu.gpu_percent.at("gpu-pwr-totals").push_back(
								std::clamp((long long)std::round((double)gpu.pwr_usage * 100.0 / (double)gpu.pwr_max_usage), 0ll, 100ll));
						} else if constexpr (is_init) {
							//? Seed the graph even when we can't yet establish a scale (e.g. the
							//? very first sample reads 0) — guarantees the deque is never left
							//? empty while supported_functions.pwr_usage claims otherwise. This
							//? is the fix for the crash in Cpu::draw()'s .back() on an empty
							//? "gpu-pwr-totals" deque — see INTEL_XPU.md.
							gpu.gpu_percent.at("gpu-pwr-totals").push_back(0);
						}
					} else if constexpr (is_init) {
						//? No usable reading at all yet (e.g. energy1_input needs a second
						//? sample to compute a delta) — seed anyway so the deque is never
						//? left empty.
						gpu.gpu_percent.at("gpu-pwr-totals").push_back(0);
					}
				}

				if (not d.hwmon.empty()) {
					const auto temp_path = d.hwmon / "temp1_input";
					if (std::filesystem::exists(temp_path)) {
						gpu.temp.push_back(read_ll(temp_path) / 1000); //? millidegrees -> degrees
					}
				}
			}
			return true;
		}

		//? Explicit template instantiations referenced from Shared::init and Gpu::collect.
		template bool collect<0>(gpu_info*);
		template bool collect<1>(gpu_info*);
	}

	namespace Intel::LevelZero {
		//? Function pointers, dlsym'd at runtime — Level Zero is optional and not everyone
		//? has libze_loader installed, so this mirrors how Nvml/Rsmi are loaded dynamically.
		//? decltype(&::symbol) pulls the exact prototype from the real header included above,
		//? so there's no risk of a hand-typed signature mismatching the real ABI.
		decltype(&::zesInit) zesInit = nullptr;
		decltype(&::zesDriverGet) zesDriverGet = nullptr;
		decltype(&::zesDeviceGet) zesDeviceGet = nullptr;
		decltype(&::zesDeviceGetProperties) zesDeviceGetProperties = nullptr;
		decltype(&::zesDeviceEnumEngineGroups) zesDeviceEnumEngineGroups = nullptr;
		decltype(&::zesEngineGetProperties) zesEngineGetProperties = nullptr;
		decltype(&::zesEngineGetActivity) zesEngineGetActivity = nullptr;
		decltype(&::zesDeviceEnumMemoryModules) zesDeviceEnumMemoryModules = nullptr;
		decltype(&::zesMemoryGetProperties) zesMemoryGetProperties = nullptr;
		decltype(&::zesMemoryGetState) zesMemoryGetState = nullptr;
		decltype(&::zesDeviceEnumPowerDomains) zesDeviceEnumPowerDomains = nullptr;
		decltype(&::zesPowerGetEnergyCounter) zesPowerGetEnergyCounter = nullptr;
		decltype(&::zesPowerGetProperties) zesPowerGetProperties = nullptr;
		decltype(&::zesPowerGetLimits) zesPowerGetLimits = nullptr;
		decltype(&::zesDeviceEnumFrequencyDomains) zesDeviceEnumFrequencyDomains = nullptr;
		decltype(&::zesFrequencyGetProperties) zesFrequencyGetProperties = nullptr;
		decltype(&::zesFrequencyGetState) zesFrequencyGetState = nullptr;

		bool init() {
			if (initialized) return false;

			//? Try possible library names for libze_loader.so
			const array libZeAlts = {
				"libze_loader.so.1",
				"libze_loader.so",
			};

			for (const auto& l : libZeAlts) {
				ze_dl_handle = dlopen(l, RTLD_LAZY);
				if (ze_dl_handle != nullptr) break;
			}
			if (!ze_dl_handle) {
				Logger::debug("Intel GPU: Failed to load libze_loader.so, Level Zero backend unavailable: {}", dlerror());
				return false;
			}

			auto load_ze_sym = [&](const char sym_name[]) {
				auto sym = dlsym(ze_dl_handle, sym_name);
				auto err = dlerror();
				if (err != nullptr) {
					Logger::debug("Intel GPU: Level Zero: couldn't find function {}: {}", sym_name, err);
					return (void*)nullptr;
				} else return sym;
			};

			#define LOAD_SYM(NAME) if ((NAME = (decltype(NAME))load_ze_sym(#NAME)) == nullptr) { \
				dlclose(ze_dl_handle); ze_dl_handle = nullptr; return false; }

			LOAD_SYM(zesInit);
			LOAD_SYM(zesDriverGet);
			LOAD_SYM(zesDeviceGet);
			LOAD_SYM(zesDeviceGetProperties);
			LOAD_SYM(zesDeviceEnumEngineGroups);
			LOAD_SYM(zesEngineGetProperties);
			LOAD_SYM(zesEngineGetActivity);
			LOAD_SYM(zesDeviceEnumMemoryModules);
			LOAD_SYM(zesMemoryGetProperties);
			LOAD_SYM(zesMemoryGetState);
			LOAD_SYM(zesDeviceEnumPowerDomains);
			LOAD_SYM(zesPowerGetEnergyCounter);
			LOAD_SYM(zesPowerGetProperties);
			LOAD_SYM(zesPowerGetLimits);
			LOAD_SYM(zesDeviceEnumFrequencyDomains);
			LOAD_SYM(zesFrequencyGetProperties);
			LOAD_SYM(zesFrequencyGetState);

			#undef LOAD_SYM

			//? Required by the Level Zero Sysman API before zesInit, or driver/device
			//? enumeration silently comes back empty. Don't clobber a user's own setting.
			setenv("ZES_ENABLE_SYSMAN", "1", 0);

			const auto init_result = zesInit(0);
			if (init_result != ZE_RESULT_SUCCESS) {
				Logger::debug("Intel GPU: zesInit failed ({})", static_cast<uint32_t>(init_result));
				dlclose(ze_dl_handle); ze_dl_handle = nullptr;
				return false;
			}

			const auto enumerate_handles = []<typename handle_t>(vector<handle_t>& handles, const auto& query) {
				handles.clear();
				uint32_t count{};
				if (query(&count, nullptr) != ZE_RESULT_SUCCESS or count == 0)
					return false;
				handles.resize(count);
				if (query(&count, handles.data()) != ZE_RESULT_SUCCESS) {
					handles.clear();
					return false;
				}
				return true;
			};

			vector<zes_driver_handle_t> drivers;
			if (not enumerate_handles(drivers, [](uint32_t* count, zes_driver_handle_t* handles) {
					return zesDriverGet(count, handles);
				})) {
				Logger::debug("Intel GPU: Level Zero did not report any drivers");
				dlclose(ze_dl_handle); ze_dl_handle = nullptr;
				return false;
			}

			const auto property_string = [](const auto& value) {
				return string(std::begin(value), std::find(std::begin(value), std::end(value), '\0'));
			};

			for (const auto driver : drivers) {
				vector<zes_device_handle_t> devices;
				enumerate_handles(devices, [driver](uint32_t* count, zes_device_handle_t* handles) {
					return zesDeviceGet(driver, count, handles);
				});

				for (const auto candidate : devices) {
					zes_device_properties_t properties{};
					properties.stype = ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES;
					properties.core.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
					if (zesDeviceGetProperties(candidate, &properties) != ZE_RESULT_SUCCESS) continue;

					if (properties.core.type == ZE_DEVICE_TYPE_GPU and properties.core.vendorId == 0x8086) {
						XeDeviceState state;
						state.device = candidate;

						//? Engines
						vector<zes_engine_handle_t> engine_handles;
						enumerate_handles(engine_handles, [candidate](uint32_t* count, zes_engine_handle_t* handles) {
							return zesDeviceEnumEngineGroups(candidate, count, handles);
						});
						for (const auto handle : engine_handles) {
							zes_engine_properties_t e_props{};
							e_props.stype = ZES_STRUCTURE_TYPE_ENGINE_PROPERTIES;
							zes_engine_stats_t e_stats{};
							if (zesEngineGetProperties(handle, &e_props) == ZE_RESULT_SUCCESS
								and zesEngineGetActivity(handle, &e_stats) == ZE_RESULT_SUCCESS) {
								state.engines.push_back({handle, e_props.type, e_stats});
							}
						}

						//? Memory
						vector<zes_mem_handle_t> memory_handles;
						enumerate_handles(memory_handles, [candidate](uint32_t* count, zes_mem_handle_t* handles) {
							return zesDeviceEnumMemoryModules(candidate, count, handles);
						});
						for (const auto handle : memory_handles) {
							zes_mem_properties_t m_props{};
							m_props.stype = ZES_STRUCTURE_TYPE_MEM_PROPERTIES;
							zesMemoryGetProperties(handle, &m_props);
							zes_mem_state_t m_state{};
							m_state.stype = ZES_STRUCTURE_TYPE_MEM_STATE;
							if (zesMemoryGetState(handle, &m_state) == ZE_RESULT_SUCCESS) {
								const uint64_t total = m_state.size > 0 ? m_state.size : m_props.physicalSize;
								if (total > 0) state.memory_modules.push_back({handle, m_props.physicalSize, total, min(m_state.free, total)});
							}
						}

						//? Power
						vector<zes_pwr_handle_t> power_handles;
						enumerate_handles(power_handles, [candidate](uint32_t* count, zes_pwr_handle_t* handles) {
							return zesDeviceEnumPowerDomains(candidate, count, handles);
						});
						for (const auto handle : power_handles) {
							zes_power_energy_counter_t p_stats{};
							if (zesPowerGetEnergyCounter(handle, &p_stats) == ZE_RESULT_SUCCESS) {
								zes_power_properties_t p_props{};
								p_props.stype = ZES_STRUCTURE_TYPE_POWER_PROPERTIES;
								zes_power_sustained_limit_t sustained{};
								if (zesPowerGetLimits(handle, &sustained, nullptr, nullptr) == ZE_RESULT_SUCCESS and sustained.power > 0)
									state.power_limit = sustained.power;
								else if (zesPowerGetProperties(handle, &p_props) == ZE_RESULT_SUCCESS and p_props.defaultLimit > 0)
									state.power_limit = p_props.defaultLimit;

								state.power_handle = handle;
								state.power_stats = p_stats;
								break;
							}
						}

						//? Frequency
						vector<zes_freq_handle_t> freq_handles;
						enumerate_handles(freq_handles, [candidate](uint32_t* count, zes_freq_handle_t* handles) {
							return zesDeviceEnumFrequencyDomains(candidate, count, handles);
						});
						for (const auto handle : freq_handles) {
							zes_freq_properties_t f_props{};
							f_props.stype = ZES_STRUCTURE_TYPE_FREQ_PROPERTIES;
							if (zesFrequencyGetProperties(handle, &f_props) == ZE_RESULT_SUCCESS
								and f_props.type == ZES_FREQ_DOMAIN_GPU) {
								state.frequency_handle = handle;
								break;
							}
						}

						if (state.engines.empty() and state.memory_modules.empty()
							and state.power_handle == nullptr and state.frequency_handle == nullptr) continue;

						state.initialized = true;
						device_states.push_back(state);
					}
				}
			}

			if (device_states.empty()) {
				Logger::debug("Intel GPU: Level Zero did not report any Intel GPUs with telemetry");
				dlclose(ze_dl_handle); ze_dl_handle = nullptr;
				return false;
			}

			device_count = (uint32_t)device_states.size();
			gpus.resize(gpus.size() + device_count);
			gpu_names.resize(gpus.size());
			const auto gpu_index = Nvml::device_count + Rsmi::device_count + Asysfs::device_count;

			for (uint32_t i = 0; i < device_count; ++i) {
				zes_device_properties_t props{};
				props.stype = ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES;
				props.core.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
				zesDeviceGetProperties(device_states[i].device, &props);

				const string core_name = property_string(props.core.name);
				const string model_name = property_string(props.modelName);
				auto usable_name = [](const string& name) {
					return not name.empty() and name != "unknown" and name != "Unknown" and name != "(unknown)";
				};

				if (usable_name(model_name)) gpu_names[gpu_index + i] = model_name;
				else if (usable_name(core_name)) gpu_names[gpu_index + i] = core_name;
				else gpu_names[gpu_index + i] = fmt::format("Intel GPU [0x{:04x}]", props.core.deviceId);
			}

			initialized = true;
			Intel::LevelZero::collect<1>(gpus.data() + gpu_index);

			return true;
		}

		bool shutdown() {
			if (!initialized) return false;
			device_states.clear();
			device_count = 0;
			initialized = false;
			if (ze_dl_handle) {
				dlclose(ze_dl_handle);
				ze_dl_handle = nullptr;
			}
			return true;
		}

		template <bool is_init> bool collect(gpu_info* gpus_slice) {
			if (!initialized) return false;

			if constexpr(is_init) {
				for (uint32_t i = 0; i < device_count; ++i) {
					auto& state = device_states[i];
					gpu_info& gpu = gpus_slice[i];
					gpu.supported_functions = {
						.gpu_utilization = not state.engines.empty(),
						.mem_utilization = false,
						.gpu_clock = state.frequency_handle != nullptr,
						.mem_clock = false,
						.pwr_usage = state.power_handle != nullptr,
						.pwr_state = false,
						.temp_info = false,
						.mem_total = not state.memory_modules.empty(),
						.mem_used = not state.memory_modules.empty(),
						.pcie_txrx = false,
						.encoder_utilization = false,
						.decoder_utilization = false
					};

					gpu.gpu_clock_speed = 0;
					gpu.pwr_usage = 0;
					if (state.power_limit > 0)
						gpu.pwr_max_usage = state.power_limit;
					if (state.power_handle != nullptr)
						gpu_pwr_total_max += gpu.pwr_max_usage;
				}
			}

			if (not device_states.empty()) {
				struct engine_sample {
					zes_engine_group_t type;
					double utilization;
				};
				vector<engine_sample> samples;
				samples.reserve(device_states[0].engines.size());

				for (uint32_t i = 0; i < device_count; ++i) {
					auto& state = device_states[i];
					gpu_info& gpu = gpus_slice[i];
					samples.clear();

					for (auto& engine : state.engines) {
						zes_engine_stats_t current{};
						if (zesEngineGetActivity(engine.handle, &current) != ZE_RESULT_SUCCESS) continue;

						if (current.timestamp > engine.stats.timestamp
							and current.activeTime >= engine.stats.activeTime) {
							const double active = current.activeTime - engine.stats.activeTime;
							const double elapsed = current.timestamp - engine.stats.timestamp;
							samples.push_back({
								engine.type,
								clamp(active * 100.0 / elapsed, 0.0, 100.0)
							});
							engine.stats = current;
						} else if (current.timestamp < engine.stats.timestamp
							or current.activeTime < engine.stats.activeTime) {
							//? Counter reset: establish a new baseline without emitting a false 0% sample.
							engine.stats = current;
						}
					}

					const auto aggregate_priority = [](const zes_engine_group_t type) {
						switch (type) {
							case ZES_ENGINE_GROUP_ALL: return 3;
							case ZES_ENGINE_GROUP_RENDER_ALL: return 2;
							case ZES_ENGINE_GROUP_COMPUTE_ALL: return 1;
							default: return 0;
						}
					};

					std::optional<double> selected_utilization;
					int selected_priority = 0;
					for (const auto& sample : samples) {
						const int priority = aggregate_priority(sample.type);
						if (priority > selected_priority) {
							selected_utilization = sample.utilization;
							selected_priority = priority;
						}
					}

					if (not selected_utilization) {
						double maximum = -1.0;
						for (const auto& sample : samples) {
							switch (sample.type) {
								case ZES_ENGINE_GROUP_COMPUTE_SINGLE:
								case ZES_ENGINE_GROUP_RENDER_SINGLE:
								case ZES_ENGINE_GROUP_COPY_SINGLE:
								case ZES_ENGINE_GROUP_MEDIA_DECODE_SINGLE:
								case ZES_ENGINE_GROUP_MEDIA_ENCODE_SINGLE:
								case ZES_ENGINE_GROUP_MEDIA_ENHANCEMENT_SINGLE:
								case ZES_ENGINE_GROUP_MEDIA_CODEC_SINGLE:
								case ZES_ENGINE_GROUP_3D_SINGLE:
									maximum = max(maximum, sample.utilization);
								break;
								default:
								break;
							}
						}
						if (maximum >= 0.0) selected_utilization = maximum;
					}

					if (selected_utilization) {
						gpu.gpu_percent.at("gpu-totals").push_back(
							(long long)round(*selected_utilization));
					} else if constexpr(is_init) {
						//? Seed the graph for callers that expect a value immediately after init.
						gpu.gpu_percent.at("gpu-totals").push_back(0);
					}
				}
			}

			if (not device_states.empty()) {
				for (uint32_t i = 0; i < device_count; ++i) {
					auto& state = device_states[i];
					gpu_info& gpu = gpus_slice[i];
					uint64_t total_memory = 0;
					uint64_t free_memory = 0;
					bool has_memory_sample = false;

					for (auto& memory : state.memory_modules) {
						zes_mem_state_t m_state{};
						m_state.stype = ZES_STRUCTURE_TYPE_MEM_STATE;
						if (zesMemoryGetState(memory.handle, &m_state) == ZE_RESULT_SUCCESS) {
							const uint64_t module_total = m_state.size > 0 ? m_state.size : memory.physical_size;
							if (module_total > 0) {
								memory.size = module_total;
								memory.free = min(m_state.free, module_total);
							}
						}

						total_memory += memory.size;
						free_memory += memory.free;
						has_memory_sample = true;
					}

					if (has_memory_sample and total_memory > 0) {
						gpu.mem_total = static_cast<long long>(total_memory);
						gpu.mem_used = static_cast<long long>(total_memory - free_memory);
						gpu.gpu_percent.at("gpu-vram-totals").push_back(
							clamp((long long)round((double)gpu.mem_used * 100.0
								/ (double)gpu.mem_total), 0ll, 100ll));
					}
				}
			}

			if (not device_states.empty()) {
				for (uint32_t i = 0; i < device_count; ++i) {
					auto& state = device_states[i];
					gpu_info& gpu = gpus_slice[i];
					zes_power_energy_counter_t current{};
					if (state.power_handle != nullptr and zesPowerGetEnergyCounter(state.power_handle, &current) == ZE_RESULT_SUCCESS) {
						if (current.timestamp > state.power_stats.timestamp and current.energy >= state.power_stats.energy) {
							//? microjoules / microseconds = watts
							const double power = (double)(current.energy - state.power_stats.energy)
								/ (double)(current.timestamp - state.power_stats.timestamp);
							gpu.pwr_usage = (long long)round(power * 1000);
						}
						state.power_stats = current;
					}
					if (state.power_handle != nullptr) {
						gpu.gpu_percent.at("gpu-pwr-totals").push_back(
							clamp((long long)round((double)gpu.pwr_usage * 100.0
								/ (double)gpu.pwr_max_usage), 0ll, 100ll));
					}
				}
			}

			if (not device_states.empty()) {
				for (uint32_t i = 0; i < device_count; ++i) {
					auto& state = device_states[i];
					gpu_info& gpu = gpus_slice[i];
					if (state.frequency_handle != nullptr) {
						zes_freq_state_t f_state{};
						f_state.stype = ZES_STRUCTURE_TYPE_FREQ_STATE;
						if (zesFrequencyGetState(state.frequency_handle, &f_state) == ZE_RESULT_SUCCESS and f_state.actual >= 0)
							gpu.gpu_clock_speed = (unsigned int)round(f_state.actual);
					}
				}
			}

			return true;
		}

		//? Explicit template instantiations referenced from Shared::init and Gpu::collect.
		template bool collect<0>(gpu_info*);
		template bool collect<1>(gpu_info*);
	}

	namespace Asysfs {
		//? Read a sysfs node containing a single integer; return fallback on missing/parse error.
		static long long read_ll(const std::filesystem::path& path, long long fallback = 0) {
			try {
				return std::stoll(readfile(path, std::to_string(fallback)));
			} catch (const std::exception&) {
				return fallback;
			}
		}

		//? Match /sys/class/drm/cardN (no '-', all digits after "card"). Skips card1-DP-1, renderD*, etc.
		static bool is_card_node(const string& fname) {
			if (not fname.starts_with("card") or fname.size() <= 4) return false;
			return std::ranges::all_of(fname.begin() + 4, fname.end(),
				[](char c) { return c >= '0' and c <= '9'; });
		}

		//? Pick the first hwmon* subdirectory under <device>/hwmon, or empty path if none.
		static std::filesystem::path find_hwmon(const std::filesystem::path& device) {
			const auto hwmon_dir = device / "hwmon";
			std::error_code ec;
			if (not std::filesystem::is_directory(hwmon_dir, ec)) return {};
			for (const auto& h : std::filesystem::directory_iterator(hwmon_dir, ec)) {
				if (h.is_directory()) return h.path();
			}
			return {};
		}

		bool init() {
			if (initialized) return false;
			//? Self-skip when rocm-smi already enumerated AMD devices — avoids double-counting on
			//? systems where both backends would succeed.
			if (Rsmi::device_count > 0) return false;
			devices.clear();

			const std::filesystem::path drm_root("/sys/class/drm");
			std::error_code ec;
			if (not std::filesystem::is_directory(drm_root, ec)) {
				Logger::debug("amdgpu sysfs: /sys/class/drm not present");
				return false;
			}

			for (const auto& entry : std::filesystem::directory_iterator(drm_root, ec)) {
				if (not is_card_node(entry.path().filename().string())) continue;

				const auto device_link = entry.path() / "device";
				if (not std::filesystem::exists(device_link)) continue;

				//? Vendor must be exactly 0x1002 (AMD). The sysfs node ends with a newline,
				//? so trim before comparing.
				string vendor = readfile(device_link / "vendor", "");
				while (not vendor.empty() and (vendor.back() == '\n' or vendor.back() == ' ')) vendor.pop_back();
				if (vendor != "0x1002") continue;

				//? Driver must be amdgpu (rules out radeon-driver-only legacy GPUs which use a
				//? different sysfs layout).
				const auto driver_link = std::filesystem::read_symlink(device_link / "driver", ec);
				if (ec or driver_link.filename().string() != "amdgpu") continue;

				device_paths d{};
				d.device = device_link;
				try {
					//? "device" file holds e.g. "0x150e\n" — base 0 lets stoul autodetect the 0x prefix.
					d.pci_device_id = (uint32_t)std::stoul(readfile(device_link / "device", "0"), nullptr, 0);
				} catch (const std::exception&) {
					d.pci_device_id = 0;
				}
				d.hwmon = find_hwmon(device_link);

				d.has_busy = std::filesystem::exists(device_link / "gpu_busy_percent");
				d.has_vram = std::filesystem::exists(device_link / "mem_info_vram_total");

				if (not d.hwmon.empty()) {
					d.has_temp = std::filesystem::exists(d.hwmon / "temp1_input");
					d.has_freq = std::filesystem::exists(d.hwmon / "freq1_input");
					//? Prefer power1_average (filtered, EMA) over power1_input (instantaneous).
					const auto avg = d.hwmon / "power1_average";
					const auto inst = d.hwmon / "power1_input";
					if (std::filesystem::exists(avg)) d.power = avg;
					else if (std::filesystem::exists(inst)) d.power = inst;
				}

				//? Skip cards with no readable signals — virtual GPUs, fresh-bound devices,
				//? or driver states where nothing useful is exposed yet. Otherwise btop would
				//? render an empty GPU entry every tick.
				if (not (d.has_busy or d.has_vram or d.has_temp or d.has_freq or not d.power.empty())) {
					Logger::debug("amdgpu sysfs: skipping {} — no readable metrics", device_link.string());
					continue;
				}

				devices.push_back(std::move(d));
			}

			device_count = (uint32_t)devices.size();
			if (device_count == 0) {
				Logger::debug("amdgpu sysfs: no AMD cards found in /sys/class/drm");
				return false;
			}

			gpus.resize(gpus.size() + device_count);
			gpu_names.resize(Nvml::device_count + Rsmi::device_count + device_count);
			for (uint32_t i = 0; i < device_count; ++i) {
				gpu_names[Nvml::device_count + Rsmi::device_count + i] =
					fmt::format("AMD GPU (1002:{:04x})", devices[i].pci_device_id);
			}

			initialized = true;
			Logger::info("Using amdgpu sysfs for {} AMD GPU(s)", device_count);
			Asysfs::collect<1>(gpus.data() + Nvml::device_count + Rsmi::device_count);
			return true;
		}

		bool shutdown() {
			if (not initialized) return false;
			devices.clear();
			device_count = 0;
			initialized = false;
			return true;
		}

		template <bool is_init> bool collect(gpu_info* gpus_slice) {
			if (not initialized) return false;

			for (uint32_t i = 0; i < device_count; ++i) {
				gpu_info& gpu = gpus_slice[i];
				const device_paths& d = devices[i];

				if constexpr (is_init) {
					gpu.supported_functions = {
						.gpu_utilization = d.has_busy,
						.mem_utilization = false,
						.gpu_clock = d.has_freq,
						.mem_clock = false, //? only DPM table is exposed via pp_dpm_mclk, not the current frequency
						.pwr_usage = not d.power.empty(),
						.pwr_state = false,
						.temp_info = d.has_temp,
						.mem_total = d.has_vram,
						.mem_used = d.has_vram,
						.pcie_txrx = false,
						.encoder_utilization = false,
						.decoder_utilization = false,
					};
					//? Start at zero and let the observed peak set the scale, like Intel does (line 1958-ish).
					gpu.pwr_max_usage = 0;
				}

				if (d.has_busy) {
					gpu.gpu_percent.at("gpu-totals").push_back(
						std::clamp(read_ll(d.device / "gpu_busy_percent"), 0LL, 100LL));
				}

				if (d.has_temp) {
					gpu.temp.push_back(read_ll(d.hwmon / "temp1_input") / 1000); //? millidegrees → degrees
				}

				if (not d.power.empty()) {
					gpu.pwr_usage = read_ll(d.power) / 1000; //? microwatts → milliwatts
					gpu.pwr_max_usage = std::max(gpu.pwr_max_usage, gpu.pwr_usage);
					if (gpu.pwr_max_usage > 0) {
						gpu.gpu_percent.at("gpu-pwr-totals").push_back(
							std::clamp((long long)std::round((double)gpu.pwr_usage * 100.0 / (double)gpu.pwr_max_usage), 0LL, 100LL));
					}
				}

				if (d.has_freq) {
					gpu.gpu_clock_speed = (unsigned int)(read_ll(d.hwmon / "freq1_input") / 1'000'000); //? Hz → MHz
				}

				if (d.has_vram) {
					gpu.mem_total = read_ll(d.device / "mem_info_vram_total");
					gpu.mem_used = read_ll(d.device / "mem_info_vram_used");
					if (gpu.mem_total > 0) {
						gpu.gpu_percent.at("gpu-vram-totals").push_back(
							std::clamp((long long)std::round((double)gpu.mem_used * 100.0 / (double)gpu.mem_total), 0LL, 100LL));
					}
				}
			}
			return true;
		}

		//? Explicit template instantiations referenced from Shared::init and Gpu::collect.
		template bool collect<0>(gpu_info*);
		template bool collect<1>(gpu_info*);
	}

	//? Collect data from GPU-specific libraries
	auto collect(bool no_update) -> vector<gpu_info>& {
		if (Runner::stopping or (no_update and not gpus.empty())) return gpus;

		// DebugTimer gpu_timer("GPU Total");

		//* Collect data
		Nvml::collect<0>(gpus.data()); // raw pointer to vector data, size == Nvml::device_count
		Rsmi::collect<0>(gpus.data() + Nvml::device_count); // size = Rsmi::device_count
		Asysfs::collect<0>(gpus.data() + Nvml::device_count + Rsmi::device_count); // size = Asysfs::device_count
		//? At most one of these three is ever initialized (see Shared::init); each is a no-op
		//? when its own device_count == 0, exactly like the backends above.
		Intel::LevelZero::collect<0>(gpus.data() + Nvml::device_count + Rsmi::device_count + Asysfs::device_count);
		Intel::Sysfs::collect<0>(gpus.data() + Nvml::device_count + Rsmi::device_count + Asysfs::device_count);
		Intel::collect<0>(gpus.data() + Nvml::device_count + Rsmi::device_count + Asysfs::device_count); // size = Intel::device_count

		//* Calculate average usage
		long long avg = 0;
		long long mem_usage_total = 0;
		long long mem_total = 0;
		long long pwr_total = 0;
		for (auto& gpu : gpus) {
			if (gpu.supported_functions.gpu_utilization)
				avg += gpu.gpu_percent.at("gpu-totals").back();
			if (gpu.supported_functions.mem_used)
				mem_usage_total += gpu.mem_used;
			if (gpu.supported_functions.mem_total)
				mem_total += gpu.mem_total;
			if (gpu.supported_functions.pwr_usage)
				pwr_total += gpu.pwr_usage;

			//* Trim vectors if there are more values than needed for graphs
			if (width != 0) {
				//? GPU & memory utilization
				while (cmp_greater(gpu.gpu_percent.at("gpu-totals").size(), width * 2)) gpu.gpu_percent.at("gpu-totals").pop_front();
				while (cmp_greater(gpu.mem_utilization_percent.size(), width)) gpu.mem_utilization_percent.pop_front();
				//? Power usage
				while (cmp_greater(gpu.gpu_percent.at("gpu-pwr-totals").size(), width)) gpu.gpu_percent.at("gpu-pwr-totals").pop_front();
				//? Temperature
				while (cmp_greater(gpu.temp.size(), 18)) gpu.temp.pop_front();
				//? Memory usage
				while (cmp_greater(gpu.gpu_percent.at("gpu-vram-totals").size(), width/2)) gpu.gpu_percent.at("gpu-vram-totals").pop_front();
			}
		}

		shared_gpu_percent.at("gpu-average").push_back(avg / gpus.size());
		if (mem_total != 0)
			shared_gpu_percent.at("gpu-vram-total").push_back(static_cast<long long>(round(mem_usage_total * 100.0 / mem_total)));
		if (gpu_pwr_total_max != 0)
			shared_gpu_percent.at("gpu-pwr-total").push_back(clamp(static_cast<long long>(round(pwr_total * 100.0 / gpu_pwr_total_max)), 0ll, 100ll));

		if (width != 0) {
			while (cmp_greater(shared_gpu_percent.at("gpu-average").size(), width * 2)) shared_gpu_percent.at("gpu-average").pop_front();
			while (cmp_greater(shared_gpu_percent.at("gpu-pwr-total").size(), width * 2)) shared_gpu_percent.at("gpu-pwr-total").pop_front();
			while (cmp_greater(shared_gpu_percent.at("gpu-vram-total").size(), width * 2)) shared_gpu_percent.at("gpu-vram-total").pop_front();
		}

		count = gpus.size();

		return gpus;
	}
}

namespace Npu {
	int count = 0;

	//? Match /sys/class/accel/accelN (no '-', all digits after "accel").
	static bool is_accel_node(const string& fname) {
		if (not fname.starts_with("accel") or fname.size() <= 5) return false;
		return std::ranges::all_of(fname.begin() + 5, fname.end(),
			[](char c) { return c >= '0' and c <= '9'; });
	}

	//? Read a sysfs node containing a single integer; return fallback on missing/parse error.
	static long long read_ll(const std::filesystem::path& path, long long fallback = 0) {
		try {
			return std::stoll(readfile(path, std::to_string(fallback)));
		} catch (const std::exception&) {
			return fallback;
		}
	}

	//? Pick the first hwmon* subdirectory under <device>/hwmon, or empty path if none.
	static std::filesystem::path find_hwmon(const std::filesystem::path& device) {
		const auto hwmon_dir = device / "hwmon";
		std::error_code ec;
		if (not std::filesystem::is_directory(hwmon_dir, ec)) return {};
		for (const auto& h : std::filesystem::directory_iterator(hwmon_dir, ec)) {
			if (h.is_directory()) return h.path();
		}
		return {};
	}

	bool init() {
		if (initialized) return false;
		devices.clear();
		npus.clear();
		npu_names.clear();

		const std::filesystem::path accel_root("/sys/class/accel");
		std::error_code ec;
		if (not std::filesystem::is_directory(accel_root, ec)) {
			Logger::debug("Intel NPU: /sys/class/accel not present");
			return false;
		}

		for (const auto& entry : std::filesystem::directory_iterator(accel_root, ec)) {
			if (not is_accel_node(entry.path().filename().string())) continue;

			const auto dev = entry.path() / "device";
			if (not std::filesystem::exists(dev)) continue;

			//? Only Intel's ivpu driver (intel_vpu on older kernels, ivpu on 6.10+) is
			//? supported for now — AMD's amdxdna exposes utilization via a DRM ioctl
			//? rather than a plain sysfs busy-time file, which is separate work.
			std::error_code dec;
			const auto driver_link = std::filesystem::read_symlink(dev / "driver", dec);
			if (dec) continue;
			const string driver = driver_link.filename().string();
			if (driver != "intel_vpu" and driver != "ivpu") continue;

			device_paths d{};
			d.dev = dev;
			d.accel = entry.path();
			d.hwmon = find_hwmon(dev);

			for (const char* c : { "npu_busy_time_us", "npu_busy_time" }) {
				if (std::filesystem::exists(d.accel / c)) { d.busy_path = d.accel / c; d.busy_is_us = true; break; }
				if (std::filesystem::exists(d.dev / c)) { d.busy_path = d.dev / c; d.busy_is_us = true; break; }
			}
			if (d.busy_path.empty()) {
				//? Runtime-PM fallback for kernels older than 6.10 (coarser: "device
				//? resumed" time in milliseconds, not "jobs executing").
				if (std::filesystem::exists(d.dev / "power" / "runtime_active_time")) {
					d.busy_path = d.dev / "power" / "runtime_active_time";
					d.busy_is_us = false;
				} else if (std::filesystem::exists(d.accel / "power" / "runtime_active_time")) {
					d.busy_path = d.accel / "power" / "runtime_active_time";
					d.busy_is_us = false;
				}
			}

			//? No busy-time counter at all means there's nothing this backend can report.
			if (d.busy_path.empty()) {
				Logger::debug("Intel NPU: skipping {} — no busy-time counter found", dev.string());
				continue;
			}

			if (not d.hwmon.empty()) {
				d.has_power = std::filesystem::exists(d.hwmon / "power1_average")
					or std::filesystem::exists(d.hwmon / "power1_input")
					or std::filesystem::exists(d.hwmon / "energy1_input");
			}

			try {
				d.pci_device_id = (uint32_t)std::stoul(readfile(dev / "device", "0"), nullptr, 0);
			} catch (const std::exception&) {
				d.pci_device_id = 0;
			}

			//? Reuse the vendored Intel device-name lookup (same database the GPU
			//? backends use) — pass the accel node's own directory, not
			//? find_intel_gpu_dir() (which only ever returns the first Intel *GPU*
			//? card and would misname/miss every NPU).
			string name;
			char *npu_device_id_c = get_intel_device_id(entry.path().c_str());
			if (npu_device_id_c) {
				char *npu_device_name_c = get_intel_device_name(npu_device_id_c);
				if (npu_device_name_c) {
					name = string(npu_device_name_c);
					free(npu_device_name_c);
				}
				free(npu_device_id_c);
			}
			if (name.empty()) name = fmt::format("Intel NPU [{:04x}]", d.pci_device_id);

			npu_names.push_back(std::move(name));
			devices.push_back(std::move(d));
		}

		device_count = (uint32_t)devices.size();
		if (device_count == 0) {
			Logger::debug("Intel NPU: no Intel NPUs found");
			return false;
		}

		npus.resize(device_count);
		for (uint32_t i = 0; i < device_count; ++i) {
			npus[i].supported_functions = {
				.gpu_utilization = true,
				.mem_utilization = false,
				.gpu_clock = false,
				.mem_clock = false,
				.pwr_usage = devices[i].has_power,
				.pwr_state = false,
				.temp_info = not devices[i].hwmon.empty() and std::filesystem::exists(devices[i].hwmon / "temp1_input"),
				.mem_total = false,
				.mem_used = false,
				.pcie_txrx = false,
				.encoder_utilization = false,
				.decoder_utilization = false,
			};
			npus[i].pwr_max_usage = 0;
		}

		initialized = true;
		Logger::info("Using Intel NPU sysfs for {} NPU(s)", device_count);
		sample(); //? seed first values so callers never see an empty deque
		return true;
	}

	bool shutdown() {
		if (not initialized) return false;
		devices.clear();
		npus.clear();
		npu_names.clear();
		device_count = 0;
		initialized = false;
		return true;
	}

	void sample() {
		const long long t_us = get_monotonicTimeUSec();

		for (uint32_t i = 0; i < device_count; ++i) {
			auto& d = devices[i];
			auto& npu = npus[i];

			const long long raw = read_ll(d.busy_path, -1);
			if (raw >= 0) {
				if (d.prev_busy >= 0 and t_us > d.prev_t_us and raw >= d.prev_busy) {
					const double dwall_us = (double)(t_us - d.prev_t_us);
					const double dbusy_us = d.busy_is_us
						? (double)(raw - d.prev_busy)
						: (double)(raw - d.prev_busy) * 1000.0; //? ms -> us
					if (dwall_us > 0) {
						const double p = std::clamp(100.0 * dbusy_us / dwall_us, 0.0, 100.0);
						npu.gpu_percent.at("gpu-totals").push_back((long long)std::round(p));
					}
				}
				d.prev_busy = raw;
				d.prev_t_us = t_us;
			}
			//? Seed so .back() is never called on an empty deque, whatever path above
			//? did or didn't run this tick — same safeguard the GPU Sysfs tier needed.
			if (npu.gpu_percent.at("gpu-totals").empty())
				npu.gpu_percent.at("gpu-totals").push_back(0);

			if (d.has_power) {
				const auto avg_path = d.hwmon / "power1_average";
				const auto inst_path = d.hwmon / "power1_input";
				long long pw_uw = -1;
				if (std::filesystem::exists(avg_path)) pw_uw = read_ll(avg_path, -1);
				else if (std::filesystem::exists(inst_path)) pw_uw = read_ll(inst_path, -1);

				if (pw_uw < 0) {
					const auto energy_path = d.hwmon / "energy1_input";
					if (std::filesystem::exists(energy_path)) {
						const long long energy_uj = read_ll(energy_path, -1);
						if (energy_uj >= 0) {
							if (d.prev_energy_uj >= 0 and t_us > d.prev_energy_t_us and energy_uj >= d.prev_energy_uj) {
								const double dt_s = (double)(t_us - d.prev_energy_t_us) / 1e6;
								if (dt_s > 0) {
									const double watts = (double)(energy_uj - d.prev_energy_uj) / 1e6 / dt_s;
									pw_uw = (long long)std::round(watts * 1e6);
								}
							}
							d.prev_energy_uj = energy_uj;
							d.prev_energy_t_us = t_us;
						}
					}
				}

				if (pw_uw >= 0) {
					npu.pwr_usage = pw_uw / 1000;
					npu.pwr_max_usage = std::max(npu.pwr_max_usage, npu.pwr_usage);
					if (npu.pwr_max_usage > 0) {
						npu.gpu_percent.at("gpu-pwr-totals").push_back(
							std::clamp((long long)std::round((double)npu.pwr_usage * 100.0 / (double)npu.pwr_max_usage), 0ll, 100ll));
					}
				}
				if (npu.gpu_percent.at("gpu-pwr-totals").empty())
					npu.gpu_percent.at("gpu-pwr-totals").push_back(0);
			}

			if (not d.hwmon.empty()) {
				const auto temp_path = d.hwmon / "temp1_input";
				if (std::filesystem::exists(temp_path)) {
					npu.temp.push_back(read_ll(temp_path) / 1000); //? millidegrees -> degrees
				}
			}
		}
	}

	auto collect(bool no_update) -> vector<Gpu::gpu_info>& {
		if (Runner::stopping or (no_update and not npus.empty())) return npus;

		sample();

		//? No dedicated box exists for NPUs (brief rows only, see INTEL_XPU.md), so
		//? there's no "width" to size against like Gpu does — cap growth at a
		//? generous fixed size instead.
		for (auto& npu : npus) {
			while (cmp_greater(npu.gpu_percent.at("gpu-totals").size(), 500)) npu.gpu_percent.at("gpu-totals").pop_front();
			while (cmp_greater(npu.gpu_percent.at("gpu-pwr-totals").size(), 500)) npu.gpu_percent.at("gpu-pwr-totals").pop_front();
			while (cmp_greater(npu.temp.size(), 18)) npu.temp.pop_front();
		}

		count = npus.size();

		return npus;
	}
}
#endif

/// Convert ascii escapes like \040 into chars.
static auto convert_ascii_escapes(const std::string& input) -> std::string {
    std::string out;
    out.reserve(input.size());

    for (std::size_t i = 0; i < input.size(); ++i) {

        if (input[i] == '\\' &&
	    	// Peek the next three characters.
            i + 3 < input.size() &&
            std::isdigit(input[i + 1]) &&
            std::isdigit(input[i + 2]) &&
            std::isdigit(input[i + 3])) {

			// Convert octal chars to decimal int.
			//   '0' - '0' -> 0, '4' - '0' -> 4, '0' - '0' -> 0.
			//   0 * 64 (0)
			//   + 4 * 8 (32)
			//   + 0
			//   = 32 (ascii space)
            int value = ((input[i + 1] - '0') * 64) + ((input[i + 2] - '0') * 8) + (input[i + 3] - '0');
            out.push_back(static_cast<char>(value));
            // Consume the three digits.
            i += 3;
        } else {
            out.push_back(input[i]);
        }
    }
    return out;
}

namespace Mem {
	bool has_swap{};
	vector<string> fstab;
	fs::file_time_type fstab_time;
	int disk_ios{};
	vector<string> last_found;

	//?* Find the filepath to the specified ZFS object's stat file
	fs::path get_zfs_stat_file(const string& device_name, size_t dataset_name_start, bool zfs_hide_datasets);

	//?* Collect total ZFS pool io stats
	bool zfs_collect_pool_total_stats(struct disk_info &disk);

	mem_info current_mem {};

	uint64_t get_totalMem() {
		ifstream meminfo(Shared::procPath / "meminfo");
		int64_t totalMem = 0;
		if (meminfo.good()) {
			meminfo.ignore(SSmax, ':');
			meminfo >> totalMem;
			totalMem <<= 10;
		}
		if (not meminfo.good() or totalMem == 0)
			throw std::runtime_error("Could not get total memory size from /proc/meminfo");

		return totalMem;
	}

	auto collect(bool no_update) -> mem_info& {
		if (Runner::stopping or (no_update and not current_mem.percent.at("used").empty())) return current_mem;
		auto show_swap = Config::getB("show_swap");
		auto swap_disk = Config::getB("swap_disk");
		auto show_disks = Config::getB("show_disks");
		auto zfs_arc_cached = Config::getB("zfs_arc_cached");
		auto totalMem = get_totalMem();
		auto& mem = current_mem;

		mem.stats.at("swap_total") = 0;

		//? Read ZFS ARC info from /proc/spl/kstat/zfs/arcstats
		uint64_t arc_size = 0, arc_min_size = 0;
		if (zfs_arc_cached) {
			ifstream arcstats(Shared::procPath / "spl/kstat/zfs/arcstats");
			if (arcstats.good()) {
				for (string label; arcstats >> label;) {
					if (label == "c_min") {
						arcstats >> arc_min_size >> arc_min_size; // double read skips type column
					}
					else if (label == "size") {
						arcstats >> arc_size >> arc_size;
						break;
					}
				}
			}
			arcstats.close();
		}

		//? Read memory info from /proc/meminfo
		ifstream meminfo(Shared::procPath / "meminfo");
		if (meminfo.good()) {
			bool got_avail = false;
			for (string label; meminfo.peek() != 'D' and meminfo >> label;) {
				if (label == "MemFree:") {
					meminfo >> mem.stats.at("free");
					mem.stats.at("free") <<= 10;
				}
				else if (label == "MemAvailable:") {
					meminfo >> mem.stats.at("available");
					mem.stats.at("available") <<= 10;
					got_avail = true;
				}
				else if (label == "Cached:") {
					meminfo >> mem.stats.at("cached");
					mem.stats.at("cached") <<= 10;
					if (not show_swap and not swap_disk) break;
				}
				else if (label == "SwapTotal:") {
					meminfo >> mem.stats.at("swap_total");
					mem.stats.at("swap_total") <<= 10;
				}
				else if (label == "SwapFree:") {
					meminfo >> mem.stats.at("swap_free");
					mem.stats.at("swap_free") <<= 10;
					break;
				}
				meminfo.ignore(SSmax, '\n');
			}
			if (not got_avail) mem.stats.at("available") = mem.stats.at("free") + mem.stats.at("cached");
			if (zfs_arc_cached) {
				mem.stats.at("cached") += arc_size;
				// The ARC will not shrink below arc_min_size, so that memory is not available
				if (arc_size > arc_min_size)
					mem.stats.at("available") += arc_size - arc_min_size;
			}
			mem.stats.at("used") = totalMem - (mem.stats.at("available") <= totalMem ? mem.stats.at("available") : mem.stats.at("free"));

			if (mem.stats.at("swap_total") > 0) mem.stats.at("swap_used") = mem.stats.at("swap_total") - mem.stats.at("swap_free");
		}
		else
			throw std::runtime_error("Failed to read /proc/meminfo");

		meminfo.close();

		//? Calculate percentages
		for (const auto& name : mem_names) {
			mem.percent.at(name).push_back(round((double)mem.stats.at(name) * 100 / totalMem));
			while (cmp_greater(mem.percent.at(name).size(), width * 2)) mem.percent.at(name).pop_front();
		}

		if (show_swap and mem.stats.at("swap_total") > 0) {
			for (const auto& name : swap_names) {
				mem.percent.at(name).push_back(round((double)mem.stats.at(name) * 100 / mem.stats.at("swap_total")));
				while (cmp_greater(mem.percent.at(name).size(), width * 2)) mem.percent.at(name).pop_front();
			}
			has_swap = true;
		}
		else
			has_swap = false;

		//? Get disks stats
		if (show_disks) {
			static vector<string> ignore_list;
			double uptime = system_uptime();
			auto free_priv = Config::getB("disk_free_priv");
			try {
				auto& disks_filter = Config::getS("disks_filter");
				bool filter_exclude = false;
				auto use_fstab = Config::getB("use_fstab");
				auto only_physical = Config::getB("only_physical");
				auto zfs_hide_datasets = Config::getB("zfs_hide_datasets");
				auto& disks = mem.disks;
				static std::unordered_map<string, future<pair<disk_info, int>>> disks_stats_promises;
				ifstream diskread;

				vector<string> filter;
				if (not disks_filter.empty()) {
					filter = ssplit(disks_filter);
					if (filter.at(0).starts_with("exclude=")) {
						filter_exclude = true;
						filter.at(0) = filter.at(0).substr(8);
					}
				}

				//? Get list of "real" filesystems from /proc/filesystems
				vector<string> fstypes;
				if (only_physical and not use_fstab) {
					fstypes = {"zfs", "wslfs", "drvfs"};
					diskread.open(Shared::procPath / "filesystems");
					if (diskread.good()) {
						for (string fstype; diskread >> fstype;) {
							if (not is_in(fstype, "nodev", "squashfs", "nullfs"))
								fstypes.push_back(fstype);
							diskread.ignore(SSmax, '\n');
						}
					}
					else
						throw std::runtime_error("Failed to read /proc/filesystems");
					diskread.close();
				}

				//? Get disk list to use from fstab if enabled
				if (use_fstab and fs::last_write_time("/etc/fstab") != fstab_time) {
					fstab.clear();
					fstab_time = fs::last_write_time("/etc/fstab");
					diskread.open("/etc/fstab");
					if (diskread.good()) {
						for (string instr; diskread >> instr;) {
							if (not instr.starts_with('#')) {
								diskread >> instr;
								#ifdef SNAPPED
									if (instr == "/") fstab.push_back("/mnt");
									else if (not is_in(instr, "none", "swap")) fstab.push_back(instr);
								#else
									if (not is_in(instr, "none", "swap")) fstab.push_back(instr);
								#endif
							}
							diskread.ignore(SSmax, '\n');
						}
					}
					else
						throw std::runtime_error("Failed to read /etc/fstab");
					diskread.close();
				}

				//? Get mounts from /etc/mtab or /proc/self/mounts
				diskread.open((fs::exists("/etc/mtab") ? fs::path("/etc/mtab") : Shared::procPath / "self/mounts"));
				if (diskread.good()) {
					vector<string> found;
					found.reserve(last_found.size());
					string dev, mountpoint, fstype;
					while (not diskread.eof()) {
						std::error_code ec;
						diskread >> dev >> mountpoint >> fstype;
						diskread.ignore(SSmax, '\n');

						// A mountpoint can ascii escape codes, which will not work with `statvfs`.
						mountpoint = convert_ascii_escapes(mountpoint);

						if (v_contains(ignore_list, mountpoint) or v_contains(found, mountpoint)) continue;

						//? Match filter if not empty
						if (not filter.empty()) {
							bool match = v_contains(filter, mountpoint);
							if ((filter_exclude and match) or (not filter_exclude and not match))
								continue;
						}

						//? Skip ZFS datasets if zfs_hide_datasets option is enabled
						size_t zfs_dataset_name_start = 0;
						if (fstype == "zfs" && (zfs_dataset_name_start = dev.find('/')) != std::string::npos && zfs_hide_datasets) continue;

						if ((not use_fstab and not only_physical)
						or (use_fstab and v_contains(fstab, mountpoint))
						or (not use_fstab and only_physical and v_contains(fstypes, fstype))) {
							found.push_back(mountpoint);
							if (not v_contains(last_found, mountpoint)) redraw = true;

							//? Save mountpoint, name, fstype, dev path and path to /sys/block stat file
							if (not disks.contains(mountpoint)) {
								disks[mountpoint] = disk_info{fs::canonical(dev, ec), fs::path(mountpoint).filename(), fstype};
								if (disks.at(mountpoint).dev.empty()) disks.at(mountpoint).dev = dev;
								#ifdef SNAPPED
									if (mountpoint == "/mnt") disks.at(mountpoint).name = "root";
								#endif
								if (disks.at(mountpoint).name.empty()) disks.at(mountpoint).name = (mountpoint == "/" ? "root" : mountpoint);
								string devname = disks.at(mountpoint).dev.filename();
								int c = 0;
								while (devname.size() >= 2) {
									const auto stat = fmt::format("/sys/block/{}/stat", devname);
									if (fs::exists(stat, ec) and access(stat.c_str(), R_OK) == 0) {
										const auto mount_stat = fmt::format("/sys/block/{}/{}/stat", devname, disks.at(mountpoint).dev.filename());
										if (c > 0 and fs::exists(mount_stat, ec))
											disks.at(mountpoint).stat = std::move(mount_stat);
										else
											disks.at(mountpoint).stat = std::move(stat);
										break;
									//? Set ZFS stat filepath
									} else if (fstype == "zfs") {
										disks.at(mountpoint).stat = get_zfs_stat_file(dev, zfs_dataset_name_start, zfs_hide_datasets);
										if (disks.at(mountpoint).stat.empty()) {
											Logger::debug("Failed to get ZFS stat file for device {}", dev);
										}
										break;
									}
									devname.resize(devname.size() - 1);
									c++;
								}
							}

							//? If zfs_hide_datasets option was switched, refresh stat filepath
							if (fstype == "zfs" && ((zfs_hide_datasets && !is_directory(disks.at(mountpoint).stat))
								|| (!zfs_hide_datasets && is_directory(disks.at(mountpoint).stat)))) {
								disks.at(mountpoint).stat = get_zfs_stat_file(dev, zfs_dataset_name_start, zfs_hide_datasets);
								if (disks.at(mountpoint).stat.empty()) {
									Logger::debug("Failed to get ZFS stat file for device {}", dev);
								}
							}
						}
					}

					//? Remove disks no longer mounted or filtered out
					if (swap_disk and has_swap) found.push_back("swap");
					for (auto it = disks.begin(); it != disks.end();) {
						if (not v_contains(found, it->first))
							it = disks.erase(it);
						else
							it++;
					}
					if (found.size() != last_found.size()) redraw = true;
					last_found = std::move(found);
				}
				else
					throw std::runtime_error("Failed to get mounts from /etc/mtab and /proc/self/mounts");
				diskread.close();

				//? Get disk/partition stats
				for (auto it = disks.begin(); it != disks.end(); ) {
					auto &[mountpoint, disk] = *it;
					if (v_contains(ignore_list, mountpoint) or disk.name == "swap") {
						it = disks.erase(it);
						continue;
					}
					if(auto promises_it = disks_stats_promises.find(mountpoint); promises_it != disks_stats_promises.end()){
						auto& promise = promises_it->second;
						if(promise.valid() &&
						   promise.wait_for(0s) == std::future_status::timeout) {
							++it;
							continue;
						}
						auto promise_res = promises_it->second.get();
						if(promise_res.second != -1){
							ignore_list.push_back(mountpoint);
							Logger::warning("Failed to get disk/partition stats for mount \"{}\" with statvfs error code: {}. Ignoring...", mountpoint, promise_res.second);
							it = disks.erase(it);
							continue;
						}
						auto &updated_stats = promise_res.first;
						disk.total = updated_stats.total;
						disk.free = updated_stats.free;
						disk.used = updated_stats.used;
						disk.used_percent = updated_stats.used_percent;
						disk.free_percent = updated_stats.free_percent;
					}
					disks_stats_promises[mountpoint] = async(std::launch::async, [mountpoint, free_priv]() -> pair<disk_info, int> {
						struct statvfs vfs;
						disk_info disk;
						if (statvfs(mountpoint.c_str(), &vfs) < 0) {
							return pair{disk, errno};
						}
						disk.total = vfs.f_blocks * vfs.f_frsize;
						disk.free = (free_priv ? vfs.f_bfree : vfs.f_bavail) * vfs.f_frsize;
						disk.used = disk.total - disk.free;
						if (disk.total != 0) {
							disk.used_percent = round((double)disk.used * 100 / disk.total);
							disk.free_percent = 100 - disk.used_percent;
						} else {
							disk.used_percent = 0;
							disk.free_percent = 0;
						}
						return pair{disk, -1};
					});
					++it;
				}

				//? Setup disks order in UI and add swap if enabled
				mem.disks_order.clear();
				#ifdef SNAPPED
					if (disks.contains("/mnt")) mem.disks_order.push_back("/mnt");
				#else
					if (disks.contains("/")) mem.disks_order.push_back("/");
				#endif
				if (swap_disk and has_swap) {
					mem.disks_order.push_back("swap");
					if (not disks.contains("swap")) disks["swap"] = {"", "swap", "swap"};
					disks.at("swap").total = mem.stats.at("swap_total");
					disks.at("swap").used = mem.stats.at("swap_used");
					disks.at("swap").free = mem.stats.at("swap_free");
					disks.at("swap").used_percent = mem.percent.at("swap_used").back();
					disks.at("swap").free_percent = mem.percent.at("swap_free").back();
				}
				for (const auto& name : last_found)
					#ifdef SNAPPED
						if (not is_in(name, "/mnt", "swap")) mem.disks_order.push_back(name);
					#else
						if (not is_in(name, "/", "swap")) mem.disks_order.push_back(name);
					#endif

				//? Get disks IO
				int64_t sectors_read, sectors_write, io_ticks, io_ticks_temp;
				disk_ios = 0;
				for (auto& [ignored, disk] : disks) {
					if (disk.stat.empty() or access(disk.stat.c_str(), R_OK) != 0) continue;
					if (disk.fstype == "zfs" && zfs_hide_datasets && zfs_collect_pool_total_stats(disk)) {
						disk_ios++;
						continue;
					}
					diskread.open(disk.stat);
					if (diskread.good()) {
						disk_ios++;
						//? ZFS Pool Support
						if (disk.fstype == "zfs") {
							// skip first three lines
							for (int i = 0; i < 3; i++) diskread.ignore(numeric_limits<streamsize>::max(), '\n');
							// skip characters until '4' is reached, indicating data type 4, next value will be out target
							diskread.ignore(numeric_limits<streamsize>::max(), '4');
							diskread >> io_ticks;

							// skip characters until '4' is reached, indicating data type 4, next value will be out target
							diskread.ignore(numeric_limits<streamsize>::max(), '4');
							diskread >> sectors_write; // nbytes written
							if (disk.io_write.empty())
								disk.io_write.push_back(0);
							else
								disk.io_write.push_back(max((int64_t)0, (sectors_write - disk.old_io.at(1))));
							disk.old_io.at(1) = sectors_write;
							while (cmp_greater(disk.io_write.size(), width * 2)) disk.io_write.pop_front();

							// skip characters until '4' is reached, indicating data type 4, next value will be out target
							diskread.ignore(numeric_limits<streamsize>::max(), '4');
							diskread >> io_ticks_temp;
							io_ticks += io_ticks_temp;

							// skip characters until '4' is reached, indicating data type 4, next value will be out target
							diskread.ignore(numeric_limits<streamsize>::max(), '4');
							diskread >> sectors_read; // nbytes read
							if (disk.io_read.empty())
								disk.io_read.push_back(0);
							else
								disk.io_read.push_back(max((int64_t)0, (sectors_read - disk.old_io.at(0))));
							disk.old_io.at(0) = sectors_read;
							while (cmp_greater(disk.io_read.size(), width * 2)) disk.io_read.pop_front();

							if (disk.io_activity.empty())
								disk.io_activity.push_back(0);
							else
								disk.io_activity.push_back(max((int64_t)0, (io_ticks - disk.old_io.at(2))));
							disk.old_io.at(2) = io_ticks;
							while (cmp_greater(disk.io_activity.size(), width * 2)) disk.io_activity.pop_front();
						} else {
							for (int i = 0; i < 2; i++) { diskread >> std::ws; diskread.ignore(SSmax, ' '); }
							diskread >> sectors_read;
							if (disk.io_read.empty())
								disk.io_read.push_back(0);
							else
								disk.io_read.push_back(max((int64_t)0, (sectors_read - disk.old_io.at(0)) * 512));
							disk.old_io.at(0) = sectors_read;
							while (cmp_greater(disk.io_read.size(), width * 2)) disk.io_read.pop_front();

							for (int i = 0; i < 3; i++) { diskread >> std::ws; diskread.ignore(SSmax, ' '); }
							diskread >> sectors_write;
							if (disk.io_write.empty())
								disk.io_write.push_back(0);
							else
								disk.io_write.push_back(max((int64_t)0, (sectors_write - disk.old_io.at(1)) * 512));
							disk.old_io.at(1) = sectors_write;
							while (cmp_greater(disk.io_write.size(), width * 2)) disk.io_write.pop_front();

							for (int i = 0; i < 2; i++) { diskread >> std::ws; diskread.ignore(SSmax, ' '); }
							diskread >> io_ticks;
							if (uptime == old_uptime || disk.io_activity.empty())
								disk.io_activity.push_back(0);
							else
								disk.io_activity.push_back(clamp((long)round((double)(io_ticks - disk.old_io.at(2)) / (uptime - old_uptime) / 10), 0l, 100l));
							disk.old_io.at(2) = io_ticks;
							while (cmp_greater(disk.io_activity.size(), width * 2)) disk.io_activity.pop_front();
						}
					} else {
						Logger::debug("Error in Mem::collect() : when opening {}", disk.stat);
					}
					diskread.close();
				}
				old_uptime = uptime;
			}
			catch (const std::exception& e) {
				Logger::warning("Error in Mem::collect() : {}", e.what());
			}
		}

		return mem;
	}

	fs::path get_zfs_stat_file(const string& device_name, size_t dataset_name_start, bool zfs_hide_datasets) {
		fs::path zfs_pool_stat_path;
		if (zfs_hide_datasets) {
			zfs_pool_stat_path = Shared::procPath / "spl/kstat/zfs" / device_name;
			if (access(zfs_pool_stat_path.c_str(), R_OK) == 0) {
				return zfs_pool_stat_path;
			} else {
				Logger::debug("Can't access folder: {}", zfs_pool_stat_path);
				return "";
			}
		}

		ifstream filestream;
		string filename;
		string name_compare;

		if (dataset_name_start != std::string::npos) { // device is a dataset
			zfs_pool_stat_path = Shared::procPath / "spl/kstat/zfs" / device_name.substr(0, dataset_name_start);
		} else { // device is a pool
			zfs_pool_stat_path = Shared::procPath / "spl/kstat/zfs" / device_name;
		}

		// looking through all files that start with 'objset' to find the one containing `device_name` object stats
		try {
			for (const auto& file: fs::directory_iterator(zfs_pool_stat_path)) {
				filename = file.path().filename();
				if (filename.starts_with("objset")) {
					filestream.open(file.path());
					if (filestream.good()) {
						// skip first two lines
						for (int i = 0; i < 2; i++) filestream.ignore(numeric_limits<streamsize>::max(), '\n');
						// skip characters until '7' is reached, indicating data type 7, next value will be object name
						filestream.ignore(numeric_limits<streamsize>::max(), '7');
						filestream >> name_compare;
						if (name_compare == device_name) {
							filestream.close();
							if (access(file.path().c_str(), R_OK) == 0) {
								return file.path();
							} else {
								Logger::debug("Can't access file: {}", file.path());
								return "";
							}
						}
					}
					filestream.close();
				}
			}
		}
		catch (fs::filesystem_error& e) {}

		Logger::debug("Could not read directory: {}", zfs_pool_stat_path);
		return "";
	}

	bool zfs_collect_pool_total_stats(struct disk_info &disk) {
		ifstream diskread;

		int64_t bytes_read;
		int64_t bytes_write;
		int64_t io_ticks;
		int64_t bytes_read_total{};
		int64_t bytes_write_total{};
		int64_t io_ticks_total{};
		int64_t objects_read{};

		// looking through all files that start with 'objset'
		for (const auto& file: fs::directory_iterator(disk.stat)) {
			if ((file.path().filename()).string().starts_with("objset")) {
				diskread.open(file.path());
				if (diskread.good()) {
					try {
						// skip first three lines
						for (int i = 0; i < 3; i++) diskread.ignore(numeric_limits<streamsize>::max(), '\n');
						// skip characters until '4' is reached, indicating data type 4, next value will be out target
						diskread.ignore(numeric_limits<streamsize>::max(), '4');
						diskread >> io_ticks;
						io_ticks_total += io_ticks;

						// skip characters until '4' is reached, indicating data type 4, next value will be out target
						diskread.ignore(numeric_limits<streamsize>::max(), '4');
						diskread >> bytes_write;
						bytes_write_total += bytes_write;

						// skip characters until '4' is reached, indicating data type 4, next value will be out target
						diskread.ignore(numeric_limits<streamsize>::max(), '4');
						diskread >> io_ticks;
						io_ticks_total += io_ticks;

						// skip characters until '4' is reached, indicating data type 4, next value will be out target
						diskread.ignore(numeric_limits<streamsize>::max(), '4');
						diskread >> bytes_read;
						bytes_read_total += bytes_read;
					} catch (const std::exception& e) {
						continue;
					}

					// increment read objects counter if no errors were encountered
					objects_read++;
				} else {
					Logger::debug("Could not read file: {}", file.path());
				}
				diskread.close();
			}
		}

		// if for some reason no objects were read
		if (objects_read == 0) return false;

		if (disk.io_write.empty())
			disk.io_write.push_back(0);
		else
			disk.io_write.push_back(max((int64_t)0, (bytes_write_total - disk.old_io.at(1))));
		disk.old_io.at(1) = bytes_write_total;
		while (cmp_greater(disk.io_write.size(), width * 2)) disk.io_write.pop_front();

		if (disk.io_read.empty())
			disk.io_read.push_back(0);
		else
			disk.io_read.push_back(max((int64_t)0, (bytes_read_total - disk.old_io.at(0))));
		disk.old_io.at(0) = bytes_read_total;
		while (cmp_greater(disk.io_read.size(), width * 2)) disk.io_read.pop_front();

		if (disk.io_activity.empty())
			disk.io_activity.push_back(0);
		else
			disk.io_activity.push_back(max((int64_t)0, (io_ticks_total - disk.old_io.at(2))));
		disk.old_io.at(2) = io_ticks_total;
		while (cmp_greater(disk.io_activity.size(), width * 2)) disk.io_activity.pop_front();

		return true;
	}

}

namespace Net {
	std::unordered_map<string, net_info> current_net;
	net_info empty_net = {};
	vector<string> interfaces;
	string selected_iface;
	int errors{};
	std::unordered_map<string, uint64_t> graph_max = { {"download", {}}, {"upload", {}} };
	std::unordered_map<string, array<int, 2>> max_count = { {"download", {}}, {"upload", {}} };
	bool rescale{true};
	uint64_t timestamp{};

	auto collect(bool no_update) -> net_info& {
		if (Runner::stopping) return empty_net;
		auto& net = current_net;
		auto& config_iface = Config::getS("net_iface");
		auto net_sync = Config::getB("net_sync");
		auto net_auto = Config::getB("net_auto");
		auto new_timestamp = time_ms();

		if (not no_update and errors < 3) {
			//? Get interface list using getifaddrs() wrapper
			IfAddrsPtr if_addrs {};
			if (if_addrs.get_status() != 0) {
				errors++;
				Logger::error("Net::collect() -> getifaddrs() failed with id {}", if_addrs.get_status());
				redraw = true;
				return empty_net;
			}
			int family = 0;
			static_assert(INET6_ADDRSTRLEN >= INET_ADDRSTRLEN); // 46 >= 16, compile-time assurance.
			enum { IPBUFFER_MAXSIZE = INET6_ADDRSTRLEN }; // manually using the known biggest value, guarded by the above static_assert
			char ip[IPBUFFER_MAXSIZE];
			interfaces.clear();
			string ipv4, ipv6;

			//? Iteration over all items in getifaddrs() list
			for (auto* ifa = if_addrs.get(); ifa != nullptr; ifa = ifa->ifa_next) {
				if (ifa->ifa_addr == nullptr) continue;
				family = ifa->ifa_addr->sa_family;
				const auto& iface = ifa->ifa_name;

				//? Update available interfaces vector and get status of interface
				if (not v_contains(interfaces, iface)) {
					interfaces.push_back(iface);
					net[iface].connected = (ifa->ifa_flags & IFF_RUNNING);

					// An interface can have more than one IP of the same family associated with it,
					// but we pick only the first one to show in the NET box.
					// Note: Interfaces without any IPv4 and IPv6 set are still valid and monitorable!
					net[iface].ipv4.clear();
					net[iface].ipv6.clear();
				}


				//? Get IPv4 address
				if (family == AF_INET) {
					if (net[iface].ipv4.empty()) {
						if (nullptr != inet_ntop(family, &(reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr)->sin_addr), ip, IPBUFFER_MAXSIZE)) {
							net[iface].ipv4 = ip;
						} else {
							int errsv = errno;
							Logger::error("Net::collect() -> Failed to convert IPv4 to string for iface {}, errno: {}", iface, strerror(errsv));
						}
					}
				}
				//? Get IPv6 address
				else if (family == AF_INET6) {
					if (net[iface].ipv6.empty()) {
						if (nullptr != inet_ntop(family, &(reinterpret_cast<struct sockaddr_in6*>(ifa->ifa_addr)->sin6_addr), ip, IPBUFFER_MAXSIZE)) {
							net[iface].ipv6 = ip;
						} else {
							int errsv = errno;
							Logger::error("Net::collect() -> Failed to convert IPv6 to string for iface {}, errno: {}", iface, strerror(errsv));
						}
					}
				} //else, ignoring family==AF_PACKET (see man 3 getifaddrs) which is the first one in the `for` loop.
			}

			//? Get total received and transmitted bytes + device address if no ip was found
			for (const auto& iface : interfaces) {
				auto& netif = net.at(iface);
				if (netif.ipv4.empty() and netif.ipv6.empty())
					netif.ipv4 = readfile("/sys/class/net/" + iface + "/address");

				for (const string dir : {"download", "upload"}) {
					const fs::path sys_file = "/sys/class/net/" + iface + "/statistics/" + (dir == "download" ? "rx_bytes" : "tx_bytes");
					auto& saved_stat = netif.stat.at(dir);
					auto& bandwidth = netif.bandwidth.at(dir);

					uint64_t val{};
					try { val = stoull(readfile(sys_file, "0")); }
					catch (const std::invalid_argument&) {}
					catch (const std::out_of_range&) {}

					//? Update speed, total and top values
					if (val < saved_stat.last) {
						saved_stat.rollover += saved_stat.last;
						saved_stat.last = 0;
					}
					if (cmp_greater((unsigned long long)saved_stat.rollover + (unsigned long long)val, numeric_limits<uint64_t>::max())) {
						saved_stat.rollover = 0;
						saved_stat.last = 0;
					}
					saved_stat.speed = round((double)(val - saved_stat.last) / ((double)(new_timestamp - timestamp) / 1000));
					if (saved_stat.speed > saved_stat.top) saved_stat.top = saved_stat.speed;
					if (saved_stat.offset > val + saved_stat.rollover) saved_stat.offset = 0;
					saved_stat.total = (val + saved_stat.rollover) - saved_stat.offset;
					saved_stat.last = val;

					//? Add values to graph
					bandwidth.push_back(saved_stat.speed);
					while (cmp_greater(bandwidth.size(), width * 2)) bandwidth.pop_front();

					//? Set counters for auto scaling
					if (net_auto and selected_iface == iface) {
						if (net_sync and saved_stat.speed < netif.stat.at(dir == "download" ? "upload" : "download").speed) continue;
						if (saved_stat.speed > graph_max[dir]) {
							++max_count[dir][0];
							if (max_count[dir][1] > 0) --max_count[dir][1];
						}
						else if (graph_max[dir] > 10 << 10 and saved_stat.speed < graph_max[dir] / 10) {
							++max_count[dir][1];
							if (max_count[dir][0] > 0) --max_count[dir][0];
						}

					}
				}
			}

			//? Clean up net map if needed
			if (net.size() > interfaces.size()) {
				for (auto it = net.begin(); it != net.end();) {
					if (not v_contains(interfaces, it->first))
						it = net.erase(it);
					else
						it++;
				}
			}

			timestamp = new_timestamp;
		}

		//? Return empty net_info struct if no interfaces was found
		if (net.empty())
			return empty_net;

		//? Find an interface to display if selected isn't set or valid
		if (selected_iface.empty() or not v_contains(interfaces, selected_iface)) {
			max_count["download"][0] = max_count["download"][1] = max_count["upload"][0] = max_count["upload"][1] = 0;
			redraw = true;
			if (net_auto) rescale = true;
			if (not config_iface.empty() and v_contains(interfaces, config_iface)) selected_iface = config_iface;
			else {
				//? Sort interfaces by total upload + download bytes
				auto sorted_interfaces = interfaces;
				rng::sort(sorted_interfaces, [&](const auto& a, const auto& b){
					return 	cmp_greater(net.at(a).stat["download"].total + net.at(a).stat["upload"].total,
										net.at(b).stat["download"].total + net.at(b).stat["upload"].total);
				});
				selected_iface.clear();
				//? Try to set to a connected interface
				for (const auto& iface : sorted_interfaces) {
					if (net.at(iface).connected) {
						selected_iface = iface;
						break;
					}
				}
				//? If no interface is connected set to first available
				if (selected_iface.empty() and not sorted_interfaces.empty()) selected_iface = sorted_interfaces.at(0);
				else if (sorted_interfaces.empty()) return empty_net;

			}
		}

		//? Calculate max scale for graphs if needed
		if (net_auto) {
			bool sync = false;
			for (const auto& dir: {"download", "upload"}) {
				for (const auto& sel : {0, 1}) {
					if (rescale or max_count[dir][sel] >= 5) {
						const long long avg_speed = (net[selected_iface].bandwidth[dir].size() > 5
							? std::accumulate(net.at(selected_iface).bandwidth.at(dir).rbegin(), net.at(selected_iface).bandwidth.at(dir).rbegin() + 5, 0ll) / 5
							: net[selected_iface].stat[dir].speed);
						graph_max[dir] = max(uint64_t(avg_speed * (sel == 0 ? 1.3 : 3.0)), (uint64_t)10 << 10);
						max_count[dir][0] = max_count[dir][1] = 0;
						redraw = true;
						if (net_sync) sync = true;
						break;
					}
				}
				//? Sync download/upload graphs if enabled
				if (sync) {
					const auto other = (string(dir) == "upload" ? "download" : "upload");
					graph_max[other] = graph_max[dir];
					max_count[other][0] = max_count[other][1] = 0;
					break;
				}
			}
		}

		rescale = false;
		return net.at(selected_iface);
	}
}

namespace Proc {

	vector<proc_info> current_procs;
	std::unordered_map<string, string> uid_user;
	string current_sort;
	string current_filter;
	bool current_rev{};
	bool is_tree_mode;

	fs::file_time_type passwd_time;

	uint64_t cputimes;
	int collapse = -1, expand = -1, toggle_children = -1, collapse_all = -1;
	uint64_t old_cputimes{};
	atomic<int> numpids{};
	int filter_found{};

	detail_container detailed;
	constexpr size_t KTHREADD = 2;
	static std::unordered_set<size_t> kernels_procs = {KTHREADD};
	static std::unordered_set<size_t> dead_procs;

	//* Get detailed info for selected process
	static void _collect_details(const size_t pid, const uint64_t uptime, vector<proc_info>& procs) {
		fs::path pid_path = Shared::procPath / std::to_string(pid);

		if (pid != detailed.last_pid) {
			detailed = {};
			detailed.last_pid = pid;
			detailed.skip_smaps = not Config::getB("proc_info_smaps");
		}

		//? Copy proc_info for process from proc vector
		auto p_info = rng::find(procs, pid, &proc_info::pid);
		detailed.entry = *p_info;

		//? Update cpu percent deque for process cpu graph
		if (not Config::getB("proc_per_core")) detailed.entry.cpu_p *= Shared::coreCount;
		detailed.cpu_percent.push_back(clamp((long long)round(detailed.entry.cpu_p), 0ll, 100ll));
		while (cmp_greater(detailed.cpu_percent.size(), width)) detailed.cpu_percent.pop_front();

		//? Process runtime
		if (detailed.entry.state != 'X') detailed.elapsed = sec_to_dhms(uptime - (detailed.entry.cpu_s / Shared::clkTck));
		else detailed.elapsed = sec_to_dhms(detailed.entry.death_time);
		if (detailed.elapsed.size() > 8) detailed.elapsed.resize(detailed.elapsed.size() - 3);

		//? Get parent process name
		if (detailed.parent.empty()) {
			auto p_entry = rng::find(procs, detailed.entry.ppid, &proc_info::pid);
			if (p_entry != procs.end()) detailed.parent = p_entry->name;
		}

		//? Expand process status from single char to explanative string
		detailed.status = (proc_states.contains(detailed.entry.state)) ? proc_states.at(detailed.entry.state) : "Unknown";

		ifstream d_read;
		string short_str;

		//? Try to get RSS mem from proc/[pid]/smaps
		detailed.memory.clear();
		if (not detailed.skip_smaps and fs::exists(pid_path / "smaps")) {
			d_read.open(pid_path / "smaps");
			uint64_t rss = 0;
			try {
				while (d_read.good()) {
					d_read.ignore(SSmax, 'R');
					if (d_read.peek() == 's') {
						d_read.ignore(SSmax, ':');
						getline(d_read, short_str, 'k');
						rss += stoull(short_str);
					}
				}
				if (rss == detailed.entry.mem >> 10)
					detailed.skip_smaps = true;
				else {
					detailed.mem_bytes.push_back(rss << 10);
					detailed.memory = floating_humanizer(rss, false, 1);
				}
			}
			catch (const std::invalid_argument&) {}
			catch (const std::out_of_range&) {}
			d_read.close();
		}
		if (detailed.memory.empty()) {
			detailed.mem_bytes.push_back(detailed.entry.mem);
			detailed.memory = floating_humanizer(detailed.entry.mem);
		}
		if (detailed.first_mem == -1 or detailed.first_mem < detailed.mem_bytes.back() / 2 or detailed.first_mem > detailed.mem_bytes.back() * 4) {
			detailed.first_mem = min((uint64_t)detailed.mem_bytes.back() * 2, Mem::get_totalMem());
			redraw = true;
		}

		while (cmp_greater(detailed.mem_bytes.size(), width)) detailed.mem_bytes.pop_front();

		//? Get bytes read and written from proc/[pid]/io
		if (fs::exists(pid_path / "io")) {
			d_read.open(pid_path / "io");
			try {
				string name;
				while (d_read.good()) {
					getline(d_read, name, ':');
					if (name.ends_with("read_bytes")) {
						getline(d_read, short_str);
						detailed.io_read = floating_humanizer(stoull(short_str));
					}
					else if (name.ends_with("write_bytes")) {
						getline(d_read, short_str);
						detailed.io_write = floating_humanizer(stoull(short_str));
						break;
					}
					else
						d_read.ignore(SSmax, '\n');
				}
			}
			catch (const std::invalid_argument&) {}
			catch (const std::out_of_range&) {}
			d_read.close();
		}
	}

	//* Collects and sorts process information from /proc
	auto collect(bool no_update) -> vector<proc_info>& {
		if (Runner::stopping) return current_procs;
		const auto& sorting = Config::getS("proc_sorting");
		auto reverse = Config::getB("proc_reversed");
		const auto& filter = Config::getS("proc_filter");
		auto per_core = Config::getB("proc_per_core");
		auto should_filter_kernel = Config::getB("proc_filter_kernel");
		auto tree = Config::getB("proc_tree");
		auto show_detailed = Config::getB("show_detailed");
		const auto pause_proc_list = Config::getB("pause_proc_list");
		const size_t detailed_pid = Config::getI("detailed_pid");
		bool should_filter = current_filter != filter;
		if (should_filter) current_filter = filter;
		bool sorted_change = (sorting != current_sort or reverse != current_rev or should_filter);
		bool tree_mode_change = tree != is_tree_mode;
		if (sorted_change) {
			current_sort = sorting;
			current_rev = reverse;
		}
		if (tree_mode_change) is_tree_mode = tree;
		ifstream pread;
		string long_string;
		string short_str;

		static vector<size_t> found;

		const double uptime = system_uptime();

		const int cmult = (per_core) ? Shared::coreCount : 1;
		bool got_detailed = false;

		static size_t proc_clear_count{};

		//* Use pids from last update if only changing filter, sorting or tree options
		if (no_update and not current_procs.empty()) {
			if (show_detailed and detailed_pid != detailed.last_pid) _collect_details(detailed_pid, round(uptime), current_procs);
		}
		//* ---------------------------------------------Collection start----------------------------------------------
		else {
			should_filter = true;
			found.clear();

			//? First make sure kernel proc cache is cleared.
			if (should_filter_kernel and ++proc_clear_count >= 256) {
				//? Clearing the cache is used in the event of a pid wrap around.
				//? In that event processes that acquire old kernel pids would also be filtered out so we need to manually clean the cache every now and then.
				kernels_procs.clear();
				kernels_procs.emplace(KTHREADD);
				proc_clear_count = 0;
			}

			auto totalMem = Mem::get_totalMem();
			int totalMem_len = to_string(totalMem >> 10).size();

			//? Update uid_user map if /etc/passwd changed since last run
			if (not Shared::passwd_path.empty() and fs::last_write_time(Shared::passwd_path) != passwd_time) {
				string r_uid, r_user;
				passwd_time = fs::last_write_time(Shared::passwd_path);
				uid_user.clear();
				pread.open(Shared::passwd_path);
				if (pread.good()) {
					while (pread.good()) {
						getline(pread, r_user, ':');
						pread.ignore(SSmax, ':');
						getline(pread, r_uid, ':');
						if (uid_user.contains(r_uid)) break;
						uid_user[r_uid] = r_user;
						pread.ignore(SSmax, '\n');
					}
				}
				else {
					Shared::passwd_path.clear();
				}
				pread.close();
			}

			//? Get cpu total times from /proc/stat up to the guest field
			cputimes = 0;
			pread.open(Shared::procPath / "stat");
			if (pread.good()) {
				pread.ignore(SSmax, ' ');
				int i = 0;
				for (uint64_t times; i < 8 and pread >> times; cputimes += times, i++);
			}
			else throw std::runtime_error("Failure to read /proc/stat");
			pread.close();

			//? Iterate over all pids in /proc
			for (const auto& d: fs::directory_iterator(Shared::procPath)) {
				if (Runner::stopping)
					return current_procs;

				if (pread.is_open()) pread.close();

				const string pid_str = d.path().filename();
				if (not isdigit(pid_str[0])) continue;

				const size_t pid = stoul(pid_str);

				if (should_filter_kernel and kernels_procs.contains(pid)) {
					continue;
				}

				found.push_back(pid);

				//? Check if pid already exists in current_procs
				auto find_old = rng::find(current_procs, pid, &proc_info::pid);
				bool no_cache{};
				//? Only add new processes if not paused
				if (find_old == current_procs.end()) {
					if (not pause_proc_list) {
						current_procs.push_back({pid});
						find_old = current_procs.end() - 1;
						no_cache = true;
					}
					else continue;
				}
				else if (dead_procs.contains(pid)) continue;

				auto& new_proc = *find_old;

				//? Get program name, command and username
				if (no_cache) {
					pread.open(d.path() / "comm");
					if (not pread.good()) continue;
					getline(pread, new_proc.name);
					pread.close();
					//? Check for whitespace characters in name and set offset to get correct fields from stat file
					new_proc.name_offset = rng::count(new_proc.name, ' ');

					pread.open(d.path() / "cmdline");
					if (not pread.good()) continue;
					long_string.clear();
					while(getline(pread, long_string, '\0')) {
						new_proc.cmd += long_string + ' ';
						if (new_proc.cmd.size() > 1000) {
							new_proc.cmd.resize(1000);
							break;
						}
					}
					pread.close();
					if (not new_proc.cmd.empty()) new_proc.cmd.pop_back();

					pread.open(d.path() / "status");
					if (not pread.good()) continue;
					string uid;
					string line;
					while (pread.good()) {
						getline(pread, line, ':');
						if (line == "Uid") {
							pread.ignore();
							getline(pread, uid, '\t');
							break;
						} else {
							pread.ignore(SSmax, '\n');
						}
					}
					pread.close();
					if (uid_user.contains(uid)) {
						new_proc.user = uid_user.at(uid);
					}
					else {
					#if !(defined(STATIC_BUILD) && defined(__GLIBC__))
						try {
							struct passwd* udet;
							udet = getpwuid(stoi(uid));
							if (udet != nullptr and udet->pw_name != nullptr) {
								new_proc.user = string(udet->pw_name);
							}
							else {
								new_proc.user = uid;
							}
						}
						catch (...) { new_proc.user = uid; }
					#else
						new_proc.user = uid;
					#endif
					}
				}

				//? Parse /proc/[pid]/stat
				pread.open(d.path() / "stat");
				if (not pread.good()) continue;

				const auto& offset = new_proc.name_offset;
				short_str.clear();
				int x = 0, next_x = 3;
				uint64_t cpu_t = 0;
				try {
					for (;;) {
						while (pread.good() and ++x < next_x + offset) pread.ignore(SSmax, ' ');
						if (not pread.good()) break;
						else getline(pread, short_str, ' ');

						switch (x-offset) {
							case 3: //? Process state
								new_proc.state = short_str.at(0);
								if (new_proc.ppid != 0) next_x = 14;
								continue;
							case 4: //? Parent pid
								new_proc.ppid = stoull(short_str);
								next_x = 14;
								continue;
							case 14: //? Process utime
								cpu_t = stoull(short_str);
								continue;
							case 15: //? Process stime
								cpu_t += stoull(short_str);
								next_x = 19;
								continue;
							case 19: //? Nice value
								new_proc.p_nice = stoll(short_str);
								continue;
							case 20: //? Number of threads
								new_proc.threads = stoull(short_str);
								if (new_proc.cpu_s == 0) {
									next_x = 22;
									new_proc.cpu_t = cpu_t;
								}
								else
									next_x = 24;
								continue;
							case 22: //? Get cpu seconds if missing
								new_proc.cpu_s = stoull(short_str);
								next_x = 24;
								continue;
							case 24: //? RSS memory (can be inaccurate, but parsing smaps increases total cpu usage by ~20x)
								if (cmp_greater(short_str.size(), totalMem_len))
									new_proc.mem = totalMem;
								else
									new_proc.mem = stoull(short_str) * Shared::pageSize;
						}
						break;
					}

				}
				catch (const std::invalid_argument&) { continue; }
				catch (const std::out_of_range&) { continue; }

				pread.close();

				if (should_filter_kernel and new_proc.ppid == KTHREADD) {
					kernels_procs.emplace(new_proc.pid);
					found.pop_back();
				}

				if (x-offset < 24) continue;

				//? Get RSS memory from /proc/[pid]/statm if value from /proc/[pid]/stat looks wrong
				if (new_proc.mem >= totalMem) {
					pread.open(d.path() / "statm");
					if (not pread.good()) continue;
					pread.ignore(SSmax, ' ');
					pread >> new_proc.mem;
					new_proc.mem *= Shared::pageSize;
					pread.close();
				}

				//? Process cpu usage since last update
				new_proc.cpu_p = clamp(round(cmult * 1000 * (cpu_t - new_proc.cpu_t) / max((uint64_t)1, cputimes - old_cputimes)) / 10.0, 0.0, 100.0 * Shared::coreCount);

				//? Process cumulative cpu usage since process start
				new_proc.cpu_c = (double)cpu_t / max(1.0, (uptime * Shared::clkTck) - new_proc.cpu_s);

				//? Update cached value with latest cpu times
				new_proc.cpu_t = cpu_t;

				if (show_detailed and not got_detailed and new_proc.pid == detailed_pid) {
					got_detailed = true;
				}
			}

			//? Clear dead processes from current_procs and remove kernel processes if enabled and not paused
			if (not pause_proc_list) {
				auto eraser = rng::remove_if(current_procs, [&](const auto& element){ return not v_contains(found, element.pid); });
				current_procs.erase(eraser.begin(), eraser.end());
				if (!dead_procs.empty()) dead_procs.clear();
			}
			//? Set correct state of dead processes if paused
			else {
				const bool keep_dead_proc_usage = Config::getB("keep_dead_proc_usage");
				for (auto& r : current_procs) {
					if (rng::find(found, r.pid) == found.end()) {
						if (r.state != 'X') r.death_time = round(uptime) - (r.cpu_s / Shared::clkTck);
						r.state = 'X';
						dead_procs.emplace(r.pid);
						//? Reset cpu usage for dead processes if paused and option is set
						if (!keep_dead_proc_usage) {
							r.cpu_p = 0.0;
							r.mem = 0;
						}
					}
				}
			}

			//? Update the details info box for process if active
			if (show_detailed and got_detailed) {
				_collect_details(detailed_pid, round(uptime), current_procs);
			}
			else if (show_detailed and not got_detailed and detailed.status != "Dead") {
				detailed.status = "Dead";
				redraw = true;
			}

			old_cputimes = cputimes;
		}
		//* ---------------------------------------------Collection done-----------------------------------------------

		//* Match filter if defined
		if (should_filter) {
			filter_found = 0;
			for (auto& p : current_procs) {
				if (not tree and not filter.empty()) {
					if (!matches_filter(p, filter)) {
						p.filtered = true;
						filter_found++;
					} else {
						p.filtered = false;
					}
				} else {
					p.filtered = false;
				}
			}
		}

		//* Sort processes
		if ((sorted_change or tree_mode_change) or (not no_update and not pause_proc_list)) {
			proc_sorter(current_procs, sorting, reverse, tree);
		}

		//* Generate tree view if enabled
		if (tree and (not no_update or should_filter or sorted_change)) {
			bool locate_selection = false;

			if (toggle_children != -1) {
				auto collapser = rng::find(current_procs, toggle_children, &proc_info::pid);
				if (collapser != current_procs.end()){
					for (auto& p : current_procs) {
						if (p.ppid == collapser->pid) {
							auto child = rng::find(current_procs, p.pid, &proc_info::pid);
							if (child != current_procs.end()){
								child->collapsed = not child->collapsed;
							}
						}
					}
					if (Config::ints.at("proc_selected") > 0) locate_selection = true;
				}
				toggle_children = -1;
			}

			if (auto find_pid = (collapse != -1 ? collapse : expand); find_pid != -1) {
				auto collapser = rng::find(current_procs, find_pid, &proc_info::pid);
				if (collapser != current_procs.end()) {
					if (collapse == expand) {
						collapser->collapsed = not collapser->collapsed;
					}
					else if (collapse > -1) {
						collapser->collapsed = true;
					}
					else if (expand > -1) {
						collapser->collapsed = false;
					}
					if (Config::ints.at("proc_selected") > 0) locate_selection = true;
				}
				collapse = expand = -1;
			}

			if (collapse_all != -1) {
				toggle_tree_collapse(current_procs);
				collapse_all = -1;
				if (Config::ints.at("proc_selected") > 0) locate_selection = true;
			}

			if (should_filter or not filter.empty()) filter_found = 0;

			vector<tree_proc> tree_procs;
			tree_procs.reserve(current_procs.size());

			if (!pause_proc_list) {
				for (auto& p : current_procs) {
					if (not v_contains(found, p.ppid)) p.ppid = 0;
				}
			}

			//? Stable sort to retain selected sorting among processes with the same parent
			rng::stable_sort(current_procs, rng::less{}, & proc_info::ppid);

			//? Auto-collapse processes with many children when entering tree mode
			_auto_collapse_oversized(current_procs, tree_mode_change);

			//? Start recursive iteration over processes with the lowest shared parent pids
			for (auto& p : rng::equal_range(current_procs, current_procs.at(0).ppid, rng::less{}, &proc_info::ppid)) {
				_tree_gen(p, current_procs, tree_procs, 0, false, filter, false, no_update, should_filter);
			}

			//? Recursive sort over tree structure to account for collapsed processes in the tree
			int index = 0;
			tree_sort(tree_procs, sorting, reverse, (pause_proc_list and not (sorted_change or tree_mode_change)), index, current_procs.size());

			//? Recursive construction of ASCII tree prefixes.
			for (auto t = tree_procs.begin(); t != tree_procs.end(); ++t) {
				_collect_prefixes(*t, t == tree_procs.end() - 1);
			}

			//? Final sort based on tree index
			rng::stable_sort(current_procs, rng::less {}, &proc_info::tree_index);

			//? Move current selection/view to the selected process when collapsing/expanding in the tree
			if (locate_selection) {
				int loc = rng::find(current_procs, Proc::selected_pid, &proc_info::pid)->tree_index;
				if (Config::ints.at("proc_start") >= loc or Config::ints.at("proc_start") <= loc - Proc::select_max)
					Config::ints.at("proc_start") = max(0, loc - 1);
				Config::ints.at("proc_selected") = loc - Config::ints.at("proc_start") + 1;
			}
		}

		numpids = (int)current_procs.size() - filter_found;

		return current_procs;
	}
}

namespace Tools {
	double system_uptime() {
		string upstr;
		ifstream pread(Shared::procPath / "uptime");
		if (pread.good()) {
			try {
				getline(pread, upstr, ' ');
				pread.close();
				return stod(upstr);
			}
			catch (const std::invalid_argument&) {}
			catch (const std::out_of_range&) {}
		}
        throw std::runtime_error(fmt::format("Failed to get uptime from {}", Shared::procPath / "uptime"));
	}
}
