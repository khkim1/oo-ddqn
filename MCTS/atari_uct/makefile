# This will likely need to be changed to suit your installation.
CXX := g++ -std=c++11
ALE := ${HOME}/subsystem/xitari
CAFFE := ${HOME}/subsystem/caffe-dev
FLAGS := -I$(ALE) -I$(CAFFE)/include -I$(CAFFE)/src -I$(CAFFE)/build/src -I$(CAFFE)/tools -I/usr/local/cuda/include -I/opt/intel/mkl/include -L$(ALE) -L/opt/intel/mkl/lib/intel64 -L/opt/intel/mkl/lib  -L/usr/local/lib
LDFLAGS := -lxitari -O2 -lgflags -lglog -lopencv_core -lopencv_highgui  -lmkl_rt -lboost_system -DUSE_MKL -lopencv_videoio -lopencv_photo -lopencv_imgcodecs -lopencv_video -lopencv_ml -lopencv_imgproc -lopencv_core
# -lSDL -lSDL_gfx -lSDL_image 

all: uctPlanner atariUCTPlanner 
atariUCTPlanner: atariUCTPlanner.o uct.o atari.hpp
	$(CXX) $(FLAGS) atariUCTPlanner.o uct.o -o atariUCTPlanner -pthread -fPIC -DNDEBUG $(LDFLAGS)
atariUCTPlanner.o: atariUCTPlanner.cpp uct.hpp atari.hpp
	$(CXX) $(FLAGS) -c atariUCTPlanner.cpp -o atariUCTPlanner.o -pthread -fPIC -DNDEBUG $(LDFLAGS)
uctPlanner:	uctPlanner.o uct.o
	$(CXX) $(FLAGS) uctPlanner.o uct.o  -o uctPlanner 
uct.o:	uct.cpp	uct.hpp 
	$(CXX)  uct.cpp -c -o uct.o
uctPlanner.o:	uctPlanner.cpp  uct.hpp
	$(CXX)  uctPlanner.cpp -c -o uctPlanner.o 

clean: 
	rm  uctPlanner uctPlanner.o uct.o atariUCTPlanner atariUCTPlanner.o 
