# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/dahu/AscendProjects/Atlas200DK

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dahu/AscendProjects/Atlas200DK/build/intermediates

# Include any dependencies generated for this target.
include src/CMakeFiles/main.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/main.dir/flags.make

src/CMakeFiles/main.dir/utils.cpp.o: src/CMakeFiles/main.dir/flags.make
src/CMakeFiles/main.dir/utils.cpp.o: ../../src/utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dahu/AscendProjects/Atlas200DK/build/intermediates/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/main.dir/utils.cpp.o"
	cd /home/dahu/AscendProjects/Atlas200DK/build/intermediates/src && /usr/bin/aarch64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/utils.cpp.o -c /home/dahu/AscendProjects/Atlas200DK/src/utils.cpp

src/CMakeFiles/main.dir/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/utils.cpp.i"
	cd /home/dahu/AscendProjects/Atlas200DK/build/intermediates/src && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dahu/AscendProjects/Atlas200DK/src/utils.cpp > CMakeFiles/main.dir/utils.cpp.i

src/CMakeFiles/main.dir/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/utils.cpp.s"
	cd /home/dahu/AscendProjects/Atlas200DK/build/intermediates/src && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dahu/AscendProjects/Atlas200DK/src/utils.cpp -o CMakeFiles/main.dir/utils.cpp.s

src/CMakeFiles/main.dir/utils.cpp.o.requires:

.PHONY : src/CMakeFiles/main.dir/utils.cpp.o.requires

src/CMakeFiles/main.dir/utils.cpp.o.provides: src/CMakeFiles/main.dir/utils.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/main.dir/build.make src/CMakeFiles/main.dir/utils.cpp.o.provides.build
.PHONY : src/CMakeFiles/main.dir/utils.cpp.o.provides

src/CMakeFiles/main.dir/utils.cpp.o.provides.build: src/CMakeFiles/main.dir/utils.cpp.o


src/CMakeFiles/main.dir/model_process.cpp.o: src/CMakeFiles/main.dir/flags.make
src/CMakeFiles/main.dir/model_process.cpp.o: ../../src/model_process.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dahu/AscendProjects/Atlas200DK/build/intermediates/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/main.dir/model_process.cpp.o"
	cd /home/dahu/AscendProjects/Atlas200DK/build/intermediates/src && /usr/bin/aarch64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/model_process.cpp.o -c /home/dahu/AscendProjects/Atlas200DK/src/model_process.cpp

src/CMakeFiles/main.dir/model_process.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/model_process.cpp.i"
	cd /home/dahu/AscendProjects/Atlas200DK/build/intermediates/src && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dahu/AscendProjects/Atlas200DK/src/model_process.cpp > CMakeFiles/main.dir/model_process.cpp.i

src/CMakeFiles/main.dir/model_process.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/model_process.cpp.s"
	cd /home/dahu/AscendProjects/Atlas200DK/build/intermediates/src && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dahu/AscendProjects/Atlas200DK/src/model_process.cpp -o CMakeFiles/main.dir/model_process.cpp.s

src/CMakeFiles/main.dir/model_process.cpp.o.requires:

.PHONY : src/CMakeFiles/main.dir/model_process.cpp.o.requires

src/CMakeFiles/main.dir/model_process.cpp.o.provides: src/CMakeFiles/main.dir/model_process.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/main.dir/build.make src/CMakeFiles/main.dir/model_process.cpp.o.provides.build
.PHONY : src/CMakeFiles/main.dir/model_process.cpp.o.provides

src/CMakeFiles/main.dir/model_process.cpp.o.provides.build: src/CMakeFiles/main.dir/model_process.cpp.o


src/CMakeFiles/main.dir/classify_process.cpp.o: src/CMakeFiles/main.dir/flags.make
src/CMakeFiles/main.dir/classify_process.cpp.o: ../../src/classify_process.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dahu/AscendProjects/Atlas200DK/build/intermediates/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/CMakeFiles/main.dir/classify_process.cpp.o"
	cd /home/dahu/AscendProjects/Atlas200DK/build/intermediates/src && /usr/bin/aarch64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/classify_process.cpp.o -c /home/dahu/AscendProjects/Atlas200DK/src/classify_process.cpp

src/CMakeFiles/main.dir/classify_process.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/classify_process.cpp.i"
	cd /home/dahu/AscendProjects/Atlas200DK/build/intermediates/src && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dahu/AscendProjects/Atlas200DK/src/classify_process.cpp > CMakeFiles/main.dir/classify_process.cpp.i

src/CMakeFiles/main.dir/classify_process.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/classify_process.cpp.s"
	cd /home/dahu/AscendProjects/Atlas200DK/build/intermediates/src && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dahu/AscendProjects/Atlas200DK/src/classify_process.cpp -o CMakeFiles/main.dir/classify_process.cpp.s

src/CMakeFiles/main.dir/classify_process.cpp.o.requires:

.PHONY : src/CMakeFiles/main.dir/classify_process.cpp.o.requires

src/CMakeFiles/main.dir/classify_process.cpp.o.provides: src/CMakeFiles/main.dir/classify_process.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/main.dir/build.make src/CMakeFiles/main.dir/classify_process.cpp.o.provides.build
.PHONY : src/CMakeFiles/main.dir/classify_process.cpp.o.provides

src/CMakeFiles/main.dir/classify_process.cpp.o.provides.build: src/CMakeFiles/main.dir/classify_process.cpp.o


src/CMakeFiles/main.dir/main.cpp.o: src/CMakeFiles/main.dir/flags.make
src/CMakeFiles/main.dir/main.cpp.o: ../../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dahu/AscendProjects/Atlas200DK/build/intermediates/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/CMakeFiles/main.dir/main.cpp.o"
	cd /home/dahu/AscendProjects/Atlas200DK/build/intermediates/src && /usr/bin/aarch64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/main.cpp.o -c /home/dahu/AscendProjects/Atlas200DK/src/main.cpp

src/CMakeFiles/main.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/main.cpp.i"
	cd /home/dahu/AscendProjects/Atlas200DK/build/intermediates/src && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dahu/AscendProjects/Atlas200DK/src/main.cpp > CMakeFiles/main.dir/main.cpp.i

src/CMakeFiles/main.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/main.cpp.s"
	cd /home/dahu/AscendProjects/Atlas200DK/build/intermediates/src && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dahu/AscendProjects/Atlas200DK/src/main.cpp -o CMakeFiles/main.dir/main.cpp.s

src/CMakeFiles/main.dir/main.cpp.o.requires:

.PHONY : src/CMakeFiles/main.dir/main.cpp.o.requires

src/CMakeFiles/main.dir/main.cpp.o.provides: src/CMakeFiles/main.dir/main.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/main.dir/build.make src/CMakeFiles/main.dir/main.cpp.o.provides.build
.PHONY : src/CMakeFiles/main.dir/main.cpp.o.provides

src/CMakeFiles/main.dir/main.cpp.o.provides.build: src/CMakeFiles/main.dir/main.cpp.o


# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/utils.cpp.o" \
"CMakeFiles/main.dir/model_process.cpp.o" \
"CMakeFiles/main.dir/classify_process.cpp.o" \
"CMakeFiles/main.dir/main.cpp.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

../../out/main: src/CMakeFiles/main.dir/utils.cpp.o
../../out/main: src/CMakeFiles/main.dir/model_process.cpp.o
../../out/main: src/CMakeFiles/main.dir/classify_process.cpp.o
../../out/main: src/CMakeFiles/main.dir/main.cpp.o
../../out/main: src/CMakeFiles/main.dir/build.make
../../out/main: src/CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dahu/AscendProjects/Atlas200DK/build/intermediates/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable ../../../out/main"
	cd /home/dahu/AscendProjects/Atlas200DK/build/intermediates/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/main.dir/build: ../../out/main

.PHONY : src/CMakeFiles/main.dir/build

src/CMakeFiles/main.dir/requires: src/CMakeFiles/main.dir/utils.cpp.o.requires
src/CMakeFiles/main.dir/requires: src/CMakeFiles/main.dir/model_process.cpp.o.requires
src/CMakeFiles/main.dir/requires: src/CMakeFiles/main.dir/classify_process.cpp.o.requires
src/CMakeFiles/main.dir/requires: src/CMakeFiles/main.dir/main.cpp.o.requires

.PHONY : src/CMakeFiles/main.dir/requires

src/CMakeFiles/main.dir/clean:
	cd /home/dahu/AscendProjects/Atlas200DK/build/intermediates/src && $(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/main.dir/clean

src/CMakeFiles/main.dir/depend:
	cd /home/dahu/AscendProjects/Atlas200DK/build/intermediates && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dahu/AscendProjects/Atlas200DK /home/dahu/AscendProjects/Atlas200DK/src /home/dahu/AscendProjects/Atlas200DK/build/intermediates /home/dahu/AscendProjects/Atlas200DK/build/intermediates/src /home/dahu/AscendProjects/Atlas200DK/build/intermediates/src/CMakeFiles/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/main.dir/depend

