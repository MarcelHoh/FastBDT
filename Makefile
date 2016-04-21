# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


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
CMAKE_SOURCE_DIR = /home/thomas/FastBDT

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/thomas/FastBDT

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target install
install: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Install the project..."
	/usr/bin/cmake -P cmake_install.cmake
.PHONY : install

# Special rule for the target install
install/fast: preinstall/fast
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Install the project..."
	/usr/bin/cmake -P cmake_install.cmake
.PHONY : install/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target install/local
install/local: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Installing only the local directory..."
	/usr/bin/cmake -DCMAKE_INSTALL_LOCAL_ONLY=1 -P cmake_install.cmake
.PHONY : install/local

# Special rule for the target install/local
install/local/fast: install/local

.PHONY : install/local/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# Special rule for the target install/strip
install/strip: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Installing the project stripped..."
	/usr/bin/cmake -DCMAKE_INSTALL_DO_STRIP=1 -P cmake_install.cmake
.PHONY : install/strip

# Special rule for the target install/strip
install/strip/fast: install/strip

.PHONY : install/strip/fast

# Special rule for the target list_install_components
list_install_components:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Available install components are: \"Unspecified\""
.PHONY : list_install_components

# Special rule for the target list_install_components
list_install_components/fast: list_install_components

.PHONY : list_install_components/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/thomas/FastBDT/CMakeFiles /home/thomas/FastBDT/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/thomas/FastBDT/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named FastBDT_shared

# Build rule for target.
FastBDT_shared: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 FastBDT_shared
.PHONY : FastBDT_shared

# fast build rule for target.
FastBDT_shared/fast:
	$(MAKE) -f CMakeFiles/FastBDT_shared.dir/build.make CMakeFiles/FastBDT_shared.dir/build
.PHONY : FastBDT_shared/fast

#=============================================================================
# Target rules for targets named FastBDT_static

# Build rule for target.
FastBDT_static: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 FastBDT_static
.PHONY : FastBDT_static

# fast build rule for target.
FastBDT_static/fast:
	$(MAKE) -f CMakeFiles/FastBDT_static.dir/build.make CMakeFiles/FastBDT_static.dir/build
.PHONY : FastBDT_static/fast

#=============================================================================
# Target rules for targets named FastBDTMain

# Build rule for target.
FastBDTMain: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 FastBDTMain
.PHONY : FastBDTMain

# fast build rule for target.
FastBDTMain/fast:
	$(MAKE) -f CMakeFiles/FastBDTMain.dir/build.make CMakeFiles/FastBDTMain.dir/build
.PHONY : FastBDTMain/fast

#=============================================================================
# Target rules for targets named unittests

# Build rule for target.
unittests: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 unittests
.PHONY : unittests

# fast build rule for target.
unittests/fast:
	$(MAKE) -f CMakeFiles/unittests.dir/build.make CMakeFiles/unittests.dir/build
.PHONY : unittests/fast

#=============================================================================
# Target rules for targets named FastBDT_CInterface

# Build rule for target.
FastBDT_CInterface: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 FastBDT_CInterface
.PHONY : FastBDT_CInterface

# fast build rule for target.
FastBDT_CInterface/fast:
	$(MAKE) -f CMakeFiles/FastBDT_CInterface.dir/build.make CMakeFiles/FastBDT_CInterface.dir/build
.PHONY : FastBDT_CInterface/fast

examples/FastBDTMain.o: examples/FastBDTMain.cxx.o

.PHONY : examples/FastBDTMain.o

# target to build an object file
examples/FastBDTMain.cxx.o:
	$(MAKE) -f CMakeFiles/FastBDTMain.dir/build.make CMakeFiles/FastBDTMain.dir/examples/FastBDTMain.cxx.o
.PHONY : examples/FastBDTMain.cxx.o

examples/FastBDTMain.i: examples/FastBDTMain.cxx.i

.PHONY : examples/FastBDTMain.i

# target to preprocess a source file
examples/FastBDTMain.cxx.i:
	$(MAKE) -f CMakeFiles/FastBDTMain.dir/build.make CMakeFiles/FastBDTMain.dir/examples/FastBDTMain.cxx.i
.PHONY : examples/FastBDTMain.cxx.i

examples/FastBDTMain.s: examples/FastBDTMain.cxx.s

.PHONY : examples/FastBDTMain.s

# target to generate assembly for a file
examples/FastBDTMain.cxx.s:
	$(MAKE) -f CMakeFiles/FastBDTMain.dir/build.make CMakeFiles/FastBDTMain.dir/examples/FastBDTMain.cxx.s
.PHONY : examples/FastBDTMain.cxx.s

src/FastBDT.o: src/FastBDT.cxx.o

.PHONY : src/FastBDT.o

# target to build an object file
src/FastBDT.cxx.o:
	$(MAKE) -f CMakeFiles/FastBDT_shared.dir/build.make CMakeFiles/FastBDT_shared.dir/src/FastBDT.cxx.o
	$(MAKE) -f CMakeFiles/FastBDT_static.dir/build.make CMakeFiles/FastBDT_static.dir/src/FastBDT.cxx.o
	$(MAKE) -f CMakeFiles/FastBDT_CInterface.dir/build.make CMakeFiles/FastBDT_CInterface.dir/src/FastBDT.cxx.o
.PHONY : src/FastBDT.cxx.o

src/FastBDT.i: src/FastBDT.cxx.i

.PHONY : src/FastBDT.i

# target to preprocess a source file
src/FastBDT.cxx.i:
	$(MAKE) -f CMakeFiles/FastBDT_shared.dir/build.make CMakeFiles/FastBDT_shared.dir/src/FastBDT.cxx.i
	$(MAKE) -f CMakeFiles/FastBDT_static.dir/build.make CMakeFiles/FastBDT_static.dir/src/FastBDT.cxx.i
	$(MAKE) -f CMakeFiles/FastBDT_CInterface.dir/build.make CMakeFiles/FastBDT_CInterface.dir/src/FastBDT.cxx.i
.PHONY : src/FastBDT.cxx.i

src/FastBDT.s: src/FastBDT.cxx.s

.PHONY : src/FastBDT.s

# target to generate assembly for a file
src/FastBDT.cxx.s:
	$(MAKE) -f CMakeFiles/FastBDT_shared.dir/build.make CMakeFiles/FastBDT_shared.dir/src/FastBDT.cxx.s
	$(MAKE) -f CMakeFiles/FastBDT_static.dir/build.make CMakeFiles/FastBDT_static.dir/src/FastBDT.cxx.s
	$(MAKE) -f CMakeFiles/FastBDT_CInterface.dir/build.make CMakeFiles/FastBDT_CInterface.dir/src/FastBDT.cxx.s
.PHONY : src/FastBDT.cxx.s

src/FastBDT_C_API.o: src/FastBDT_C_API.cxx.o

.PHONY : src/FastBDT_C_API.o

# target to build an object file
src/FastBDT_C_API.cxx.o:
	$(MAKE) -f CMakeFiles/unittests.dir/build.make CMakeFiles/unittests.dir/src/FastBDT_C_API.cxx.o
	$(MAKE) -f CMakeFiles/FastBDT_CInterface.dir/build.make CMakeFiles/FastBDT_CInterface.dir/src/FastBDT_C_API.cxx.o
.PHONY : src/FastBDT_C_API.cxx.o

src/FastBDT_C_API.i: src/FastBDT_C_API.cxx.i

.PHONY : src/FastBDT_C_API.i

# target to preprocess a source file
src/FastBDT_C_API.cxx.i:
	$(MAKE) -f CMakeFiles/unittests.dir/build.make CMakeFiles/unittests.dir/src/FastBDT_C_API.cxx.i
	$(MAKE) -f CMakeFiles/FastBDT_CInterface.dir/build.make CMakeFiles/FastBDT_CInterface.dir/src/FastBDT_C_API.cxx.i
.PHONY : src/FastBDT_C_API.cxx.i

src/FastBDT_C_API.s: src/FastBDT_C_API.cxx.s

.PHONY : src/FastBDT_C_API.s

# target to generate assembly for a file
src/FastBDT_C_API.cxx.s:
	$(MAKE) -f CMakeFiles/unittests.dir/build.make CMakeFiles/unittests.dir/src/FastBDT_C_API.cxx.s
	$(MAKE) -f CMakeFiles/FastBDT_CInterface.dir/build.make CMakeFiles/FastBDT_CInterface.dir/src/FastBDT_C_API.cxx.s
.PHONY : src/FastBDT_C_API.cxx.s

src/FastBDT_IO.o: src/FastBDT_IO.cxx.o

.PHONY : src/FastBDT_IO.o

# target to build an object file
src/FastBDT_IO.cxx.o:
	$(MAKE) -f CMakeFiles/FastBDT_shared.dir/build.make CMakeFiles/FastBDT_shared.dir/src/FastBDT_IO.cxx.o
	$(MAKE) -f CMakeFiles/FastBDT_static.dir/build.make CMakeFiles/FastBDT_static.dir/src/FastBDT_IO.cxx.o
	$(MAKE) -f CMakeFiles/FastBDT_CInterface.dir/build.make CMakeFiles/FastBDT_CInterface.dir/src/FastBDT_IO.cxx.o
.PHONY : src/FastBDT_IO.cxx.o

src/FastBDT_IO.i: src/FastBDT_IO.cxx.i

.PHONY : src/FastBDT_IO.i

# target to preprocess a source file
src/FastBDT_IO.cxx.i:
	$(MAKE) -f CMakeFiles/FastBDT_shared.dir/build.make CMakeFiles/FastBDT_shared.dir/src/FastBDT_IO.cxx.i
	$(MAKE) -f CMakeFiles/FastBDT_static.dir/build.make CMakeFiles/FastBDT_static.dir/src/FastBDT_IO.cxx.i
	$(MAKE) -f CMakeFiles/FastBDT_CInterface.dir/build.make CMakeFiles/FastBDT_CInterface.dir/src/FastBDT_IO.cxx.i
.PHONY : src/FastBDT_IO.cxx.i

src/FastBDT_IO.s: src/FastBDT_IO.cxx.s

.PHONY : src/FastBDT_IO.s

# target to generate assembly for a file
src/FastBDT_IO.cxx.s:
	$(MAKE) -f CMakeFiles/FastBDT_shared.dir/build.make CMakeFiles/FastBDT_shared.dir/src/FastBDT_IO.cxx.s
	$(MAKE) -f CMakeFiles/FastBDT_static.dir/build.make CMakeFiles/FastBDT_static.dir/src/FastBDT_IO.cxx.s
	$(MAKE) -f CMakeFiles/FastBDT_CInterface.dir/build.make CMakeFiles/FastBDT_CInterface.dir/src/FastBDT_IO.cxx.s
.PHONY : src/FastBDT_IO.cxx.s

src/test_FastBDT.o: src/test_FastBDT.cxx.o

.PHONY : src/test_FastBDT.o

# target to build an object file
src/test_FastBDT.cxx.o:
	$(MAKE) -f CMakeFiles/unittests.dir/build.make CMakeFiles/unittests.dir/src/test_FastBDT.cxx.o
.PHONY : src/test_FastBDT.cxx.o

src/test_FastBDT.i: src/test_FastBDT.cxx.i

.PHONY : src/test_FastBDT.i

# target to preprocess a source file
src/test_FastBDT.cxx.i:
	$(MAKE) -f CMakeFiles/unittests.dir/build.make CMakeFiles/unittests.dir/src/test_FastBDT.cxx.i
.PHONY : src/test_FastBDT.cxx.i

src/test_FastBDT.s: src/test_FastBDT.cxx.s

.PHONY : src/test_FastBDT.s

# target to generate assembly for a file
src/test_FastBDT.cxx.s:
	$(MAKE) -f CMakeFiles/unittests.dir/build.make CMakeFiles/unittests.dir/src/test_FastBDT.cxx.s
.PHONY : src/test_FastBDT.cxx.s

src/test_FastBDT_C_API.o: src/test_FastBDT_C_API.cxx.o

.PHONY : src/test_FastBDT_C_API.o

# target to build an object file
src/test_FastBDT_C_API.cxx.o:
	$(MAKE) -f CMakeFiles/unittests.dir/build.make CMakeFiles/unittests.dir/src/test_FastBDT_C_API.cxx.o
.PHONY : src/test_FastBDT_C_API.cxx.o

src/test_FastBDT_C_API.i: src/test_FastBDT_C_API.cxx.i

.PHONY : src/test_FastBDT_C_API.i

# target to preprocess a source file
src/test_FastBDT_C_API.cxx.i:
	$(MAKE) -f CMakeFiles/unittests.dir/build.make CMakeFiles/unittests.dir/src/test_FastBDT_C_API.cxx.i
.PHONY : src/test_FastBDT_C_API.cxx.i

src/test_FastBDT_C_API.s: src/test_FastBDT_C_API.cxx.s

.PHONY : src/test_FastBDT_C_API.s

# target to generate assembly for a file
src/test_FastBDT_C_API.cxx.s:
	$(MAKE) -f CMakeFiles/unittests.dir/build.make CMakeFiles/unittests.dir/src/test_FastBDT_C_API.cxx.s
.PHONY : src/test_FastBDT_C_API.cxx.s

src/test_FastBDT_IO.o: src/test_FastBDT_IO.cxx.o

.PHONY : src/test_FastBDT_IO.o

# target to build an object file
src/test_FastBDT_IO.cxx.o:
	$(MAKE) -f CMakeFiles/unittests.dir/build.make CMakeFiles/unittests.dir/src/test_FastBDT_IO.cxx.o
.PHONY : src/test_FastBDT_IO.cxx.o

src/test_FastBDT_IO.i: src/test_FastBDT_IO.cxx.i

.PHONY : src/test_FastBDT_IO.i

# target to preprocess a source file
src/test_FastBDT_IO.cxx.i:
	$(MAKE) -f CMakeFiles/unittests.dir/build.make CMakeFiles/unittests.dir/src/test_FastBDT_IO.cxx.i
.PHONY : src/test_FastBDT_IO.cxx.i

src/test_FastBDT_IO.s: src/test_FastBDT_IO.cxx.s

.PHONY : src/test_FastBDT_IO.s

# target to generate assembly for a file
src/test_FastBDT_IO.cxx.s:
	$(MAKE) -f CMakeFiles/unittests.dir/build.make CMakeFiles/unittests.dir/src/test_FastBDT_IO.cxx.s
.PHONY : src/test_FastBDT_IO.cxx.s

src/test_all.o: src/test_all.cxx.o

.PHONY : src/test_all.o

# target to build an object file
src/test_all.cxx.o:
	$(MAKE) -f CMakeFiles/unittests.dir/build.make CMakeFiles/unittests.dir/src/test_all.cxx.o
.PHONY : src/test_all.cxx.o

src/test_all.i: src/test_all.cxx.i

.PHONY : src/test_all.i

# target to preprocess a source file
src/test_all.cxx.i:
	$(MAKE) -f CMakeFiles/unittests.dir/build.make CMakeFiles/unittests.dir/src/test_all.cxx.i
.PHONY : src/test_all.cxx.i

src/test_all.s: src/test_all.cxx.s

.PHONY : src/test_all.s

# target to generate assembly for a file
src/test_all.cxx.s:
	$(MAKE) -f CMakeFiles/unittests.dir/build.make CMakeFiles/unittests.dir/src/test_all.cxx.s
.PHONY : src/test_all.cxx.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... install"
	@echo "... FastBDT_shared"
	@echo "... FastBDT_static"
	@echo "... FastBDTMain"
	@echo "... unittests"
	@echo "... FastBDT_CInterface"
	@echo "... rebuild_cache"
	@echo "... install/local"
	@echo "... edit_cache"
	@echo "... install/strip"
	@echo "... list_install_components"
	@echo "... examples/FastBDTMain.o"
	@echo "... examples/FastBDTMain.i"
	@echo "... examples/FastBDTMain.s"
	@echo "... src/FastBDT.o"
	@echo "... src/FastBDT.i"
	@echo "... src/FastBDT.s"
	@echo "... src/FastBDT_C_API.o"
	@echo "... src/FastBDT_C_API.i"
	@echo "... src/FastBDT_C_API.s"
	@echo "... src/FastBDT_IO.o"
	@echo "... src/FastBDT_IO.i"
	@echo "... src/FastBDT_IO.s"
	@echo "... src/test_FastBDT.o"
	@echo "... src/test_FastBDT.i"
	@echo "... src/test_FastBDT.s"
	@echo "... src/test_FastBDT_C_API.o"
	@echo "... src/test_FastBDT_C_API.i"
	@echo "... src/test_FastBDT_C_API.s"
	@echo "... src/test_FastBDT_IO.o"
	@echo "... src/test_FastBDT_IO.i"
	@echo "... src/test_FastBDT_IO.s"
	@echo "... src/test_all.o"
	@echo "... src/test_all.i"
	@echo "... src/test_all.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

