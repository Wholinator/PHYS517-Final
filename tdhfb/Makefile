## Build TDHFB
##   make            -- optimized build
##   make debug      -- debug build with assertions
##   make clean
##   make hfb-only   -- standalone static-HFB exe (preserves original main)

CXX      ?= g++
EIGEN    ?= /usr/include/eigen3
HDF5_INC ?= /usr/include/hdf5/serial
HDF5_LIB ?= /usr/lib/x86_64-linux-gnu/hdf5/serial

CXXFLAGS_COMMON := -std=c++17 -I$(EIGEN) -I$(HDF5_INC)
CXXFLAGS_OPT   := -O3 -march=native -DNDEBUG -DEIGEN_NO_DEBUG
CXXFLAGS_DBG   := -O0 -g

LDFLAGS  := -L$(HDF5_LIB) -lhdf5

SRCS := hfb_numeric_current.cpp data_output.cpp tdhfb.cpp tdhfb_main.cpp
HDRS := data_output.h hfb_api.h tdhfb.h

all: tdhfb

tdhfb: $(SRCS) $(HDRS)
	$(CXX) $(CXXFLAGS_COMMON) $(CXXFLAGS_OPT) $(SRCS) $(LDFLAGS) -o $@

debug: $(SRCS) $(HDRS)
	$(CXX) $(CXXFLAGS_COMMON) $(CXXFLAGS_DBG) $(SRCS) $(LDFLAGS) -o tdhfb_debug

hfb-only: hfb_numeric_current.cpp data_output.cpp $(HDRS)
	$(CXX) $(CXXFLAGS_COMMON) $(CXXFLAGS_OPT) -DHFB_STANDALONE_MAIN \
	    hfb_numeric_current.cpp data_output.cpp $(LDFLAGS) -o hfb

clean:
	rm -f tdhfb tdhfb_debug hfb *.o

.PHONY: all clean debug hfb-only
