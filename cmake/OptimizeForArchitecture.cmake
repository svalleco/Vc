# Determine the host CPU feature set and determine the best set of compiler
# flags to enable all supported SIMD relevant features. Alternatively, the
# target CPU can be explicitly selected (for generating more generic binaries
# or for targeting a different system).
# Compilers provide e.g. the -march=native flag to achieve a similar result.
# This fails to address the need for building for a different microarchitecture
# than the current host.
# The script tries to deduce all settings from the model and family numbers of
# the CPU instead of reading the CPUID flags from e.g. /proc/cpuinfo. This makes
# the detection more independent from the CPUID code in the kernel (e.g. avx2 is
# not listed on older kernels).
#
# Usage:
# OptimizeForArchitecture()
# If either of Vc_SSE_INTRINSICS_BROKEN, Vc_AVX_INTRINSICS_BROKEN,
# Vc_AVX2_INTRINSICS_BROKEN is defined and set, the OptimizeForArchitecture
# macro will consequently disable the relevant features via compiler flags.

#=============================================================================
# Copyright 2010-2015 Matthias Kretz <kretz@kde.org>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  * Neither the names of contributing organizations nor the
#    names of its contributors may be used to endorse or promote products
#    derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#=============================================================================

get_filename_component(_currentDir "${CMAKE_CURRENT_LIST_FILE}" PATH)
include("${_currentDir}/AddCompilerFlag.cmake")
include(CheckIncludeFile)

macro(_my_find _list _value _ret)
   list(FIND ${_list} "${_value}" _found)
   if(_found EQUAL -1)
      set(${_ret} FALSE)
   else(_found EQUAL -1)
      set(${_ret} TRUE)
   endif(_found EQUAL -1)
endmacro(_my_find)

macro(AutodetectHostArchitecture)
   set(TARGET_ARCHITECTURE "generic")
   set(Vc_ARCHITECTURE_FLAGS)
   set(_vendor_id)
   set(_cpu_family)
   set(_cpu_model)
   if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      file(READ "/proc/cpuinfo" _cpuinfo)
      string(REGEX REPLACE ".*vendor_id[ \t]*:[ \t]+([a-zA-Z0-9_-]+).*" "\\1" _vendor_id "${_cpuinfo}")
      string(REGEX REPLACE ".*cpu family[ \t]*:[ \t]+([a-zA-Z0-9_-]+).*" "\\1" _cpu_family "${_cpuinfo}")
      string(REGEX REPLACE ".*model[ \t]*:[ \t]+([a-zA-Z0-9_-]+).*" "\\1" _cpu_model "${_cpuinfo}")
      string(REGEX REPLACE ".*flags[ \t]*:[ \t]+([^\n]+).*" "\\1" _cpu_flags "${_cpuinfo}")
   elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
      exec_program("/usr/sbin/sysctl -n machdep.cpu.vendor" OUTPUT_VARIABLE _vendor_id)
      exec_program("/usr/sbin/sysctl -n machdep.cpu.model"  OUTPUT_VARIABLE _cpu_model)
      exec_program("/usr/sbin/sysctl -n machdep.cpu.family" OUTPUT_VARIABLE _cpu_family)
      exec_program("/usr/sbin/sysctl -n machdep.cpu.features" OUTPUT_VARIABLE _cpu_flags)
      string(TOLOWER "${_cpu_flags}" _cpu_flags)
      string(REPLACE "." "_" _cpu_flags "${_cpu_flags}")
   elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
      get_filename_component(_vendor_id "[HKEY_LOCAL_MACHINE\\Hardware\\Description\\System\\CentralProcessor\\0;VendorIdentifier]" NAME CACHE)
      get_filename_component(_cpu_id "[HKEY_LOCAL_MACHINE\\Hardware\\Description\\System\\CentralProcessor\\0;Identifier]" NAME CACHE)
      mark_as_advanced(_vendor_id _cpu_id)
      string(REGEX REPLACE ".* Family ([0-9]+) .*" "\\1" _cpu_family "${_cpu_id}")
      string(REGEX REPLACE ".* Model ([0-9]+) .*" "\\1" _cpu_model "${_cpu_id}")
   endif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
   if(_vendor_id STREQUAL "GenuineIntel")
      if(_cpu_family EQUAL 6)
         # taken from the Intel ORM
         # http://www.intel.com/content/www/us/en/processors/architectures-software-developer-manuals.html
         # CPUID Signature Values of Of Recent Intel Microarchitectures
         # 4E 5E       | Skylake microarchitecture
         # 3D 47 56    | Broadwell microarchitecture
         # 3C 45 46 3F | Haswell microarchitecture
         # 3A 3E       | Ivy Bridge microarchitecture
         # 2A 2D       | Sandy Bridge microarchitecture
         # 25 2C 2F    | Intel microarchitecture Westmere
         # 1A 1E 1F 2E | Intel microarchitecture Nehalem
         # 17 1D       | Enhanced Intel Core microarchitecture
         # 0F          | Intel Core microarchitecture
         #
         # Values from the Intel SDE:
         # 5C | Goldmont
         # 5A | Silvermont
         # 57 | Knights Landing
         # 66 | Cannonlake
         # 55 | Skylake Server
         # 4E | Skylake Client
         # 3C | Broadwell (likely a bug in the SDE)
         # 3C | Haswell
         if(_cpu_model EQUAL 0x57)
            set(TARGET_ARCHITECTURE "knl")  # Knights Landing
         elseif(_cpu_model EQUAL 0x5C)
            set(TARGET_ARCHITECTURE "goldmont")
         elseif(_cpu_model EQUAL 0x5A)
            set(TARGET_ARCHITECTURE "silvermont")
         elseif(_cpu_model EQUAL 0x66)
            set(TARGET_ARCHITECTURE "cannonlake")
         elseif(_cpu_model EQUAL 0x55)
            set(TARGET_ARCHITECTURE "skylake-xeon")
         elseif(_cpu_model EQUAL 0x4E OR _cpu_model EQUAL 0x5E)
            set(TARGET_ARCHITECTURE "skylake")
         elseif(_cpu_model EQUAL 0x3D OR _cpu_model EQUAL 0x47 OR _cpu_model EQUAL 0x56)
            set(TARGET_ARCHITECTURE "broadwell")
         elseif(_cpu_model EQUAL 0x3C OR _cpu_model EQUAL 0x45 OR _cpu_model EQUAL 0x46 OR _cpu_model EQUAL 0x3F)
            set(TARGET_ARCHITECTURE "haswell")
         elseif(_cpu_model EQUAL 0x3A OR _cpu_model EQUAL 0x3E)
            set(TARGET_ARCHITECTURE "ivy-bridge")
         elseif(_cpu_model EQUAL 0x2A OR _cpu_model EQUAL 0x2D)
            set(TARGET_ARCHITECTURE "sandy-bridge")
         elseif(_cpu_model EQUAL 0x25 OR _cpu_model EQUAL 0x2C OR _cpu_model EQUAL 0x2F)
            set(TARGET_ARCHITECTURE "westmere")
         elseif(_cpu_model EQUAL 0x1A OR _cpu_model EQUAL 0x1E OR _cpu_model EQUAL 0x1F OR _cpu_model EQUAL 0x2E)
            set(TARGET_ARCHITECTURE "nehalem")
         elseif(_cpu_model EQUAL 0x17 OR _cpu_model EQUAL 0x1D)
            set(TARGET_ARCHITECTURE "penryn")
         elseif(_cpu_model EQUAL 0x0F)
            set(TARGET_ARCHITECTURE "merom")
         elseif(_cpu_model EQUAL 28)
            set(TARGET_ARCHITECTURE "atom")
         elseif(_cpu_model EQUAL 14)
            set(TARGET_ARCHITECTURE "core")
         elseif(_cpu_model LESS 14)
            message(WARNING "Your CPU (family ${_cpu_family}, model ${_cpu_model}) is not known. Auto-detection of optimization flags failed and will use the generic CPU settings with SSE2.")
            set(TARGET_ARCHITECTURE "generic")
         else()
            message(WARNING "Your CPU (family ${_cpu_family}, model ${_cpu_model}) is not known. Auto-detection of optimization flags failed and will use the 65nm Core 2 CPU settings.")
            set(TARGET_ARCHITECTURE "merom")
         endif()
      elseif(_cpu_family EQUAL 7) # Itanium (not supported)
         message(WARNING "Your CPU (Itanium: family ${_cpu_family}, model ${_cpu_model}) is not supported by OptimizeForArchitecture.cmake.")
      elseif(_cpu_family EQUAL 15) # NetBurst
         list(APPEND _available_vector_units_list "sse" "sse2")
         if(_cpu_model GREATER 2) # Not sure whether this must be 3 or even 4 instead
            list(APPEND _available_vector_units_list "sse" "sse2" "sse3")
         endif(_cpu_model GREATER 2)
      endif(_cpu_family EQUAL 6)
   elseif(_vendor_id STREQUAL "AuthenticAMD")
      if(_cpu_family EQUAL 22) # 16h
         set(TARGET_ARCHITECTURE "AMD 16h")
      elseif(_cpu_family EQUAL 21) # 15h
         if(_cpu_model LESS 2)
            set(TARGET_ARCHITECTURE "bulldozer")
         else()
            set(TARGET_ARCHITECTURE "piledriver")
         endif()
      elseif(_cpu_family EQUAL 20) # 14h
         set(TARGET_ARCHITECTURE "AMD 14h")
      elseif(_cpu_family EQUAL 18) # 12h
      elseif(_cpu_family EQUAL 16) # 10h
         set(TARGET_ARCHITECTURE "barcelona")
      elseif(_cpu_family EQUAL 15)
         set(TARGET_ARCHITECTURE "k8")
         if(_cpu_model GREATER 64) # I don't know the right number to put here. This is just a guess from the hardware I have access to
            set(TARGET_ARCHITECTURE "k8-sse3")
         endif(_cpu_model GREATER 64)
      endif()
   endif(_vendor_id STREQUAL "GenuineIntel")
endmacro()

macro(OptimizeForArchitecture)
   set(TARGET_ARCHITECTURE "auto" CACHE STRING "CPU architecture to optimize for. \
Using an incorrect setting here can result in crashes of the resulting binary because of invalid instructions used. \
Setting the value to \"auto\" will try to optimize for the architecture where cmake is called. \
Other supported values are: \"none\", \"generic\", \"core\", \"merom\" (65nm Core2), \
\"penryn\" (45nm Core2), \"nehalem\", \"westmere\", \"sandy-bridge\", \"ivy-bridge\", \
\"haswell\", \"broadwell\", \"skylake\", \"skylake-xeon\", \"cannonlake\", \"silvermont\", \
\"goldmont\", \"knl\" (Knights Landing), \"atom\", \"k8\", \"k8-sse3\", \"barcelona\", \
\"istanbul\", \"magny-cours\", \"bulldozer\", \"interlagos\", \"piledriver\", \
\"AMD 14h\", \"AMD 16h\".")
   set(_force)
   if(NOT _last_target_arch STREQUAL "${TARGET_ARCHITECTURE}")
      message(STATUS "target changed from \"${_last_target_arch}\" to \"${TARGET_ARCHITECTURE}\"")
      set(_force FORCE)
   endif()
   set(_last_target_arch "${TARGET_ARCHITECTURE}" CACHE STRING "" FORCE)
   mark_as_advanced(_last_target_arch)
   string(TOLOWER "${TARGET_ARCHITECTURE}" TARGET_ARCHITECTURE)

   set(_march_flag_list)
   set(_available_vector_units_list)

   if(TARGET_ARCHITECTURE STREQUAL "auto")
      AutodetectHostArchitecture()
      message(STATUS "Detected CPU: ${TARGET_ARCHITECTURE}")
   endif(TARGET_ARCHITECTURE STREQUAL "auto")

   macro(_nehalem)
      list(APPEND _march_flag_list "nehalem")
      list(APPEND _march_flag_list "corei7")
      list(APPEND _march_flag_list "core2")
      list(APPEND _available_vector_units_list "sse" "sse2" "sse3" "ssse3" "sse4.1" "sse4.2")
   endmacro()
   macro(_westmere)
      list(APPEND _march_flag_list "westmere")
      _nehalem()
   endmacro()
   macro(_sandybridge)
      list(APPEND _march_flag_list "sandybridge")
      list(APPEND _march_flag_list "corei7-avx")
      _westmere()
      list(APPEND _available_vector_units_list "sse" "sse2" "sse3" "ssse3" "sse4.1" "sse4.2" "avx")
   endmacro()
   macro(_ivybridge)
      list(APPEND _march_flag_list "ivybridge")
      list(APPEND _march_flag_list "core-avx-i")
      _sandybridge()
      list(APPEND _available_vector_units_list "rdrnd" "f16c")
   endmacro()
   macro(_haswell)
      list(APPEND _march_flag_list "haswell")
      list(APPEND _march_flag_list "core-avx2")
      _ivybridge()
      list(APPEND _available_vector_units_list "avx2" "fma" "bmi" "bmi2")
   endmacro()
   macro(_broadwell)
      list(APPEND _march_flag_list "broadwell")
      _haswell()
   endmacro()
   macro(_skylake)
      list(APPEND _march_flag_list "skylake")
      _broadwell()
   endmacro()
   macro(_skylake_xeon)
      _skylake()
      list(APPEND _available_vector_units_list "avx512f" "avx512cd" "avx512dq" "avx512bw" "avx512vl")
   endmacro()
   macro(_cannonlake)
      list(APPEND _march_flag_list "cannonlake")
      _skylake_xeon()
      list(APPEND _available_vector_units_list "avx512ifma" "avx512vbmi")
   endmacro()
   macro(_knightslanding)
      list(APPEND _march_flag_list "knl")
      _broadwell()
      list(APPEND _available_vector_units_list "avx512f" "avx512pf" "avx512er" "avx512cd")
   endmacro()

   if(TARGET_ARCHITECTURE STREQUAL "core")
      list(APPEND _march_flag_list "core2")
      list(APPEND _available_vector_units_list "sse" "sse2" "sse3")
   elseif(TARGET_ARCHITECTURE STREQUAL "merom")
      list(APPEND _march_flag_list "merom")
      list(APPEND _march_flag_list "core2")
      list(APPEND _available_vector_units_list "sse" "sse2" "sse3" "ssse3")
   elseif(TARGET_ARCHITECTURE STREQUAL "penryn")
      list(APPEND _march_flag_list "penryn")
      list(APPEND _march_flag_list "core2")
      list(APPEND _available_vector_units_list "sse" "sse2" "sse3" "ssse3")
      message(STATUS "Sadly the Penryn architecture exists in variants with SSE4.1 and without SSE4.1.")
      if(_cpu_flags MATCHES "sse4_1")
         message(STATUS "SSE4.1: enabled (auto-detected from this computer's CPU flags)")
         list(APPEND _available_vector_units_list "sse4.1")
      else()
         message(STATUS "SSE4.1: disabled (auto-detected from this computer's CPU flags)")
      endif()
   elseif(TARGET_ARCHITECTURE STREQUAL "knl")
      _knightslanding()
   elseif(TARGET_ARCHITECTURE STREQUAL "cannonlake")
      _cannonlake()
   elseif(TARGET_ARCHITECTURE STREQUAL "skylake-xeon")
      _skylake_xeon()
   elseif(TARGET_ARCHITECTURE STREQUAL "skylake")
      _skylake()
   elseif(TARGET_ARCHITECTURE STREQUAL "broadwell")
      _broadwell()
   elseif(TARGET_ARCHITECTURE STREQUAL "haswell")
      _haswell()
   elseif(TARGET_ARCHITECTURE STREQUAL "ivy-bridge")
      _ivybridge()
   elseif(TARGET_ARCHITECTURE STREQUAL "sandy-bridge")
      _sandybridge()
   elseif(TARGET_ARCHITECTURE STREQUAL "westmere")
      _westmere()
   elseif(TARGET_ARCHITECTURE STREQUAL "nehalem")
      _nehalem()
   elseif(TARGET_ARCHITECTURE STREQUAL "atom")
      list(APPEND _march_flag_list "atom")
      list(APPEND _march_flag_list "core2")
      list(APPEND _available_vector_units_list "sse" "sse2" "sse3" "ssse3")
   elseif(TARGET_ARCHITECTURE STREQUAL "k8")
      list(APPEND _march_flag_list "k8")
      list(APPEND _available_vector_units_list "sse" "sse2")
   elseif(TARGET_ARCHITECTURE STREQUAL "k8-sse3")
      list(APPEND _march_flag_list "k8-sse3")
      list(APPEND _march_flag_list "k8")
      list(APPEND _available_vector_units_list "sse" "sse2" "sse3")
   elseif(TARGET_ARCHITECTURE STREQUAL "AMD 16h")
      list(APPEND _march_flag_list "btver2")
      list(APPEND _march_flag_list "btver1")
      list(APPEND _available_vector_units_list "sse" "sse2" "sse3" "ssse3" "sse4a" "sse4.1" "sse4.2" "avx" "f16c")
   elseif(TARGET_ARCHITECTURE STREQUAL "AMD 14h")
      list(APPEND _march_flag_list "btver1")
      list(APPEND _available_vector_units_list "sse" "sse2" "sse3" "ssse3" "sse4a")
   elseif(TARGET_ARCHITECTURE STREQUAL "piledriver")
      list(APPEND _march_flag_list "bdver2")
      list(APPEND _march_flag_list "bdver1")
      list(APPEND _march_flag_list "bulldozer")
      list(APPEND _march_flag_list "barcelona")
      list(APPEND _march_flag_list "core2")
      list(APPEND _available_vector_units_list "sse" "sse2" "sse3" "ssse3" "sse4a" "sse4.1" "sse4.2" "avx" "xop" "fma4" "fma" "f16c")
   elseif(TARGET_ARCHITECTURE STREQUAL "interlagos")
      list(APPEND _march_flag_list "bdver1")
      list(APPEND _march_flag_list "bulldozer")
      list(APPEND _march_flag_list "barcelona")
      list(APPEND _march_flag_list "core2")
      list(APPEND _available_vector_units_list "sse" "sse2" "sse3" "ssse3" "sse4a" "sse4.1" "sse4.2" "avx" "xop" "fma4")
   elseif(TARGET_ARCHITECTURE STREQUAL "bulldozer")
      list(APPEND _march_flag_list "bdver1")
      list(APPEND _march_flag_list "bulldozer")
      list(APPEND _march_flag_list "barcelona")
      list(APPEND _march_flag_list "core2")
      list(APPEND _available_vector_units_list "sse" "sse2" "sse3" "ssse3" "sse4a" "sse4.1" "sse4.2" "avx" "xop" "fma4")
   elseif(TARGET_ARCHITECTURE STREQUAL "barcelona")
      list(APPEND _march_flag_list "barcelona")
      list(APPEND _march_flag_list "core2")
      list(APPEND _available_vector_units_list "sse" "sse2" "sse3" "sse4a")
   elseif(TARGET_ARCHITECTURE STREQUAL "istanbul")
      list(APPEND _march_flag_list "barcelona")
      list(APPEND _march_flag_list "core2")
      list(APPEND _available_vector_units_list "sse" "sse2" "sse3" "sse4a")
   elseif(TARGET_ARCHITECTURE STREQUAL "magny-cours")
      list(APPEND _march_flag_list "barcelona")
      list(APPEND _march_flag_list "core2")
      list(APPEND _available_vector_units_list "sse" "sse2" "sse3" "sse4a")
   elseif(TARGET_ARCHITECTURE STREQUAL "generic")
      list(APPEND _march_flag_list "generic")
   elseif(TARGET_ARCHITECTURE STREQUAL "none")
      # add this clause to remove it from the else clause
   else(TARGET_ARCHITECTURE STREQUAL "core")
      message(FATAL_ERROR "Unknown target architecture: \"${TARGET_ARCHITECTURE}\". Please set TARGET_ARCHITECTURE to a supported value.")
   endif(TARGET_ARCHITECTURE STREQUAL "core")

   if(NOT TARGET_ARCHITECTURE STREQUAL "none")
      set(_disable_vector_unit_list)
      set(_enable_vector_unit_list)
      if(DEFINED Vc_AVX_INTRINSICS_BROKEN AND Vc_AVX_INTRINSICS_BROKEN)
         UserWarning("AVX disabled per default because of old/broken toolchain")
         set(_avx_broken true)
         set(_avx2_broken true)
         set(_fma4_broken true)
         set(_xop_broken true)
      else()
         set(_avx_broken false)
         if(DEFINED Vc_FMA4_INTRINSICS_BROKEN AND Vc_FMA4_INTRINSICS_BROKEN)
            UserWarning("FMA4 disabled per default because of old/broken toolchain")
            set(_fma4_broken true)
         else()
            set(_fma4_broken false)
         endif()
         if(DEFINED Vc_XOP_INTRINSICS_BROKEN AND Vc_XOP_INTRINSICS_BROKEN)
            UserWarning("XOP disabled per default because of old/broken toolchain")
            set(_xop_broken true)
         else()
            set(_xop_broken false)
         endif()
         if(DEFINED Vc_AVX2_INTRINSICS_BROKEN AND Vc_AVX2_INTRINSICS_BROKEN)
            UserWarning("AVX2 disabled per default because of old/broken toolchain")
            set(_avx2_broken true)
         else()
            set(_avx2_broken false)
         endif()
      endif()

      macro(_enable_or_disable _name _flag _documentation _broken)
         if(_broken)
            set(_found false)
         else()
            _my_find(_available_vector_units_list "${_flag}" _found)
         endif()
         set(USE_${_name} ${_found} CACHE BOOL "${documentation}" ${_force})
         mark_as_advanced(USE_${_name})
         if(USE_${_name})
            list(APPEND _enable_vector_unit_list "${_flag}")
         else()
            list(APPEND _disable_vector_unit_list "${_flag}")
         endif()
      endmacro()
      _enable_or_disable(SSE2 "sse2" "Use SSE2. If SSE2 instructions are not enabled the SSE implementation will be disabled." false)
      _enable_or_disable(SSE3 "sse3" "Use SSE3. If SSE3 instructions are not enabled they will be emulated." false)
      _enable_or_disable(SSSE3 "ssse3" "Use SSSE3. If SSSE3 instructions are not enabled they will be emulated." false)
      _enable_or_disable(SSE4_1 "sse4.1" "Use SSE4.1. If SSE4.1 instructions are not enabled they will be emulated." false)
      _enable_or_disable(SSE4_2 "sse4.2" "Use SSE4.2. If SSE4.2 instructions are not enabled they will be emulated." false)
      _enable_or_disable(SSE4a "sse4a" "Use SSE4a. If SSE4a instructions are not enabled they will be emulated." false)
      _enable_or_disable(AVX "avx" "Use AVX. This will all floating-point vector sizes relative to SSE." _avx_broken)
      _enable_or_disable(FMA "fma" "Use FMA." _avx_broken)
      _enable_or_disable(BMI2 "bmi2" "Use BMI2." _avx_broken)
      _enable_or_disable(AVX2 "avx2" "Use AVX2. This will double all of the vector sizes relative to SSE." _avx2_broken)
      _enable_or_disable(XOP "xop" "Use XOP." _xop_broken)
      _enable_or_disable(FMA4 "fma4" "Use FMA4." _fma4_broken)
      _enable_or_disable(AVX512F "avx512f" "Use AVX512F. This will double all floating-point vector sizes relative to AVX2." false)
      _enable_or_disable(AVX512VL "avx512vl" "Use AVX512VL. This enables 128- and 256-bit vector length instructions with EVEX coding (improved write-masking & more vector registers)." _avx2_broken)
      _enable_or_disable(AVX512PF "avx512pf" "Use AVX512PF. This enables prefetch instructions for gathers and scatters." false)
      _enable_or_disable(AVX512ER "avx512er" "Use AVX512ER. This enables exponential and reciprocal instructions." false)
      _enable_or_disable(AVX512CD "avx512cd" "Use AVX512CD." false)
      _enable_or_disable(AVX512DQ "avx512dq" "Use AVX512DQ." false)
      _enable_or_disable(AVX512BW "avx512bw" "Use AVX512BW." false)
      _enable_or_disable(AVX512IFMA "avx512ifma" "Use AVX512IFMA." false)
      _enable_or_disable(AVX512VBMI "avx512vbmi" "Use AVX512VBMI." false)

      if(MSVC)
         # MSVC on 32 bit can select /arch:SSE2 (since 2010 also /arch:AVX)
         # MSVC on 64 bit cannot select anything (should have changed with MSVC 2010)
         _my_find(_enable_vector_unit_list "avx" _avx)
         set(_avx_flag FALSE)
         if(_avx)
            AddCompilerFlag("/arch:AVX" CXX_FLAGS Vc_ARCHITECTURE_FLAGS CXX_RESULT _avx_flag)
         endif()
         if(NOT _avx_flag)
            _my_find(_enable_vector_unit_list "sse2" _found)
            if(_found)
               AddCompilerFlag("/arch:SSE2" CXX_FLAGS Vc_ARCHITECTURE_FLAGS)
            endif()
         endif()
         foreach(_flag ${_enable_vector_unit_list})
            string(TOUPPER "${_flag}" _flag)
            string(REPLACE "." "_" _flag "__${_flag}__")
            add_definitions("-D${_flag}")
         endforeach(_flag)
      elseif(CMAKE_CXX_COMPILER MATCHES "/(icpc|icc)$") # ICC (on Linux)
         _my_find(_available_vector_units_list "avx2"    _found)
         if(_found)
            AddCompilerFlag("-xCORE-AVX2" CXX_FLAGS Vc_ARCHITECTURE_FLAGS)
         else(_found)
            _my_find(_available_vector_units_list "f16c"    _found)
            if(_found)
               AddCompilerFlag("-xCORE-AVX-I" CXX_FLAGS Vc_ARCHITECTURE_FLAGS)
            else(_found)
               _my_find(_available_vector_units_list "avx"    _found)
               if(_found)
                  AddCompilerFlag("-xAVX" CXX_FLAGS Vc_ARCHITECTURE_FLAGS)
               else(_found)
                  _my_find(_available_vector_units_list "sse4.2" _found)
                  if(_found)
                     AddCompilerFlag("-xSSE4.2" CXX_FLAGS Vc_ARCHITECTURE_FLAGS)
                  else(_found)
                     _my_find(_available_vector_units_list "sse4.1" _found)
                     if(_found)
                        AddCompilerFlag("-xSSE4.1" CXX_FLAGS Vc_ARCHITECTURE_FLAGS)
                     else(_found)
                        _my_find(_available_vector_units_list "ssse3"  _found)
                        if(_found)
                           AddCompilerFlag("-xSSSE3" CXX_FLAGS Vc_ARCHITECTURE_FLAGS)
                        else(_found)
                           _my_find(_available_vector_units_list "sse3"   _found)
                           if(_found)
                              # If the target host is an AMD machine then we still want to use -xSSE2 because the binary would refuse to run at all otherwise
                              _my_find(_march_flag_list "barcelona" _found)
                              if(NOT _found)
                                 _my_find(_march_flag_list "k8-sse3" _found)
                              endif(NOT _found)
                              if(_found)
                                 AddCompilerFlag("-xSSE2" CXX_FLAGS Vc_ARCHITECTURE_FLAGS)
                              else(_found)
                                 AddCompilerFlag("-xSSE3" CXX_FLAGS Vc_ARCHITECTURE_FLAGS)
                              endif(_found)
                           else(_found)
                              _my_find(_available_vector_units_list "sse2"   _found)
                              if(_found)
                                 AddCompilerFlag("-xSSE2" CXX_FLAGS Vc_ARCHITECTURE_FLAGS)
                              endif(_found)
                           endif(_found)
                        endif(_found)
                     endif(_found)
                  endif(_found)
               endif(_found)
            endif(_found)
         endif(_found)
      else() # not MSVC and not ICC => GCC, Clang, Open64
         foreach(_flag ${_march_flag_list})
            AddCompilerFlag("-march=${_flag}" CXX_RESULT _good CXX_FLAGS Vc_ARCHITECTURE_FLAGS)
            if(_good)
               break()
            endif(_good)
         endforeach(_flag)
         foreach(_flag ${_enable_vector_unit_list})
            AddCompilerFlag("-m${_flag}" CXX_RESULT _result)
            if(_result)
               set(_header FALSE)
               if(_flag STREQUAL "sse3")
                  set(_header "pmmintrin.h")
               elseif(_flag STREQUAL "ssse3")
                  set(_header "tmmintrin.h")
               elseif(_flag STREQUAL "sse4.1")
                  set(_header "smmintrin.h")
               elseif(_flag STREQUAL "sse4.2")
                  set(_header "smmintrin.h")
               elseif(_flag STREQUAL "sse4a")
                  set(_header "ammintrin.h")
               elseif(_flag STREQUAL "avx")
                  set(_header "immintrin.h")
               elseif(_flag STREQUAL "avx2")
                  set(_header "immintrin.h")
               elseif(_flag STREQUAL "fma4")
                  set(_header "x86intrin.h")
               elseif(_flag STREQUAL "xop")
                  set(_header "x86intrin.h")
               endif()
               set(_resultVar "HAVE_${_header}")
               string(REPLACE "." "_" _resultVar "${_resultVar}")
               if(_header)
                  CHECK_INCLUDE_FILE("${_header}" ${_resultVar} "-m${_flag}")
                  if(NOT ${_resultVar})
                     set(_useVar "USE_${_flag}")
                     string(TOUPPER "${_useVar}" _useVar)
                     string(REPLACE "." "_" _useVar "${_useVar}")
                     message(STATUS "disabling ${_useVar} because ${_header} is missing")
                     set(${_useVar} FALSE)
                     list(APPEND _disable_vector_unit_list "${_flag}")
                  endif()
               endif()
               if(NOT _header OR ${_resultVar})
                  list(APPEND Vc_ARCHITECTURE_FLAGS "-m${_flag}")
               endif()
            endif()
         endforeach(_flag)
         foreach(_flag ${_disable_vector_unit_list})
            AddCompilerFlag("-mno-${_flag}" CXX_FLAGS Vc_ARCHITECTURE_FLAGS)
         endforeach(_flag)
      endif()
   endif()
endmacro(OptimizeForArchitecture)
