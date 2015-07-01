INCDIR     = /usr/local/cuda/samples/common/inc/ /usr/local/cuda/targets/x86_64-linux/include/
INCLUDE    = $(addprefix -I,$(INCDIR))

CPPFLAGS   = -Xcompiler -fopenmp -arch=sm_20 $(INCLUDE) -m64 -D_DEBUG # -std=c++11
DEBUG      = -g -G

CPP        = nvcc
MKDEP      = g++ $(INCLUDE) -MD
CUDAMKDEP  = nvcc $(INCLUDE) -M

# Check if debug mode is on, if not use optimization flags
ifeq ($(mode),debug)
	CPPFLAGS += $(DEBUG)
else
	CPPFLAGS += --optimize 2
endif

.PHONY: clean distclean doc docshow

main: $(patsubst %.cu,%.o,$(wildcard *.cu)) $(patsubst %.c,%.o,$(wildcard *.c)) $(patsubst %.cc,%.o,$(wildcard *.cc))
	$(CPP) $(CPPFLAGS) $+ -o $@
	
-include main.d

# want to build again modules even if (only) header is changed
%.o: %.cc %.h Makefile
	$(CUDAMKDEP) $< -o $(patsubst %.o,%.d,$@)
	$(CPP) $(CPPFLAGS) --x c++ -dc $< -o $@

%.o: %.c Makefile
	$(CUDAMKDEP) $< -o $(patsubst %.o,%.d,$@) --x c++
	$(CPP) $(CPPFLAGS) --x cu -dc $< -o $@

%.o: %.cu Makefile
	$(CUDAMKDEP) $< -o $(patsubst %.o,%.d,$@) --x cu
	$(CPP) $(CPPFLAGS) --x cu -dc $< -o $@

evaluateMean:%: %.C Makefile
	g++ -O2 -Wall $< -o $@

# generate documentation with doxygen
doc:
	doxygen && make -C latex &

# generate documentation with doxygen
# and call evince to display it in background
docshow: doc
	evince latex/refman.pdf &

clean:
	rm --recursive --verbose --force *.o *.d

distclean: clean
	@echo
	rm --recursive --verbose --force main evaluateMean
