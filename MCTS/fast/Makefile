UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
	CXX := g++ -std=c++11
	XITARI_ROOT := xitari/linux
	TF_ROOT := tflinux
	TF_FLAGS := -Wl,-rpath='$ORIGIN/lib' -I$(TF_ROOT)/include -L$(TF_ROOT)/lib 
	TF_LDFLAGS := -ltensorflow_cc
endif

ifeq ($(UNAME), Darwin)
	CXX := clang++ -std=c++11
	XITARI_ROOT := xitari/osx
	TF_ROOT := tfmac
	TF_FLAGS := -Wl,-rpath,'$ORIGIN/lib' -I$(TF_ROOT)/include -L$(TF_ROOT)/lib 
	TF_LDFLAGS := -ltensorflow_cc -ltensorflow_framework
endif

FLAGS := -I$(XITARI_ROOT)/include/xitari -L$(XITARI_ROOT)/lib -L/usr/local/lib
LDFLAGS := -lxitari -O2 -lgflags -lglog \
	-lopencv_core -lopencv_highgui -lopencv_videoio -lopencv_photo \
	-lopencv_imgcodecs -lopencv_video -lopencv_ml -lopencv_imgproc -lopencv_core

# Old code by Xiaoxiao Guo
atariUCTPlanner: atariUCTPlanner.o uct.o atari.hpp
	$(CXX) $(FLAGS) atariUCTPlanner.o uct.o -o atariUCTPlanner \
		-pthread -fPIC -DNDEBUG $(LDFLAGS)
atariUCTPlanner.o: atariUCTPlanner.cpp uct.hpp atari.hpp
	$(CXX) $(FLAGS) -c atariUCTPlanner.cpp -o atariUCTPlanner.o \
		-pthread -fPIC -DNDEBUG $(LDFLAGS)
uctPlanner:	uctPlanner.o uct.o
	$(CXX) $(FLAGS) uctPlanner.o uct.o -o uctPlanner 
uct.o:	uct.cpp	uct.hpp 
	$(CXX)  uct.cpp -c -o uct.o
uctPlanner.o:	uctPlanner.cpp  uct.hpp
	$(CXX)  uctPlanner.cpp -c -o uctPlanner.o 

# New code for oodqn
.PHONY all: objectPlanner model_test atariUCTPlanner
objectPlanner: objectPlanner.o mcts.o tf_util.o util.o obj_sim.h
	$(CXX) $(FLAGS) $(TF_FLAGS) objectPlanner.o tf_util.o mcts.o \
		-o objectPlanner -pthread -fPIC -DNDEBUG $(TF_LDFLAGS) $(LDFLAGS)
		# -o objectPlanner -pthread -fPIC $(TF_LDFLAGS) $(LDFLAGS)
objectPlanner.o: objectPlanner.cc mcts.h atari.h util.h
	$(CXX) $(FLAGS) $(TF_FLAGS) -c objectPlanner.cc \
		-o objectPlanner.o -pthread -fPIC -DNDEBUG $(TF_LDFLAGS) $(LDFLAGS)
		# -o objectPlanner.o -pthread -fPIC $(TF_LDFLAGS) $(LDFLAGS)
obj_sim.o: tf_util.o obj_sim.h
	$(CXX) $(FLAGS) $(TF_FLAGS) -c obj_sim.h -o obj_sim.o $(TF_LDFLAGS)
model_test: tf_util.o model_test.o
	$(CXX) $(FLAGS) $(TF_FLAGS) tf_util.o model_test.o \
		-o model_test $(TF_LDFLAGS) $(LDFLAGS)
model_test.o: tf_util.o model_test.cc
	$(CXX) $(FLAGS) $(TF_FLAGS) -c model_test.cc -o model_test.o
tf_util.o: tf_util.cc tf_util.h
	$(CXX) $(FLAGS) $(TF_FLAGS) -c tf_util.cc -o tf_util.o
util.o: util.h
	$(CXX) $(FLAGS) -c util.h -o util.o

mcts.o: mcts.cc mcts.h constants.h
	$(CXX) mcts.cc -c -o mcts.o

atari_sim.o:
	$(CXX) $(FLAGS) -c atari_sim.cc -o atari_sim.o 
atariPlanner: atariPlanner.o mcts.o tf_util.o util.o atari_sim.o
	$(CXX) $(FLAGS) $(TF_FLAGS) atariPlanner.o tf_util.o atari_sim.o mcts.o \
		-o atariPlanner -pthread -fPIC -DNDEBUG $(TF_LDFLAGS) $(LDFLAGS)
atariPlanner.o: mcts.h atari_sim.h util.h tf_util.h
	$(CXX) $(FLAGS) $(TF_FLAGS) -c atariPlanner.cc \
		-o atariPlanner.o -pthread -fPIC -DNDEBUG $(TF_LDFLAGS) $(LDFLAGS)
planner: planner.o mcts.o tf_util.o util.o atari_sim.o atari_obj_sim.o
	$(CXX) $(FLAGS) $(TF_FLAGS) \
		planner.o tf_util.o atari_sim.o atari_obj_sim.o mcts.o \
		-o planner -pthread -fPIC -DNDEBUG $(TF_LDFLAGS) $(LDFLAGS)
planner.o: mcts.h atari_sim.h util.h tf_util.h
	$(CXX) $(FLAGS) $(TF_FLAGS) -c planner.cc \
		-o planner.o -pthread -fPIC -DNDEBUG $(TF_LDFLAGS) $(LDFLAGS)
atari_obj_sim.o:
	$(CXX) $(FLAGS) -c atari_obj_sim.cc -o atari_obj_sim.o 
model_sim.o:
	$(CXX) $(FLAGS) $(TF_FLAGS) -c model_sim.cc -o model_sim.o


clean:
	rm -f uctPlanner uctPlanner.o uct.o atariUCTPlanner atariUCTPlanner.o \
		util.o mcts.o tf_util.o tf_util model_test.o model_test \
		objectPlanner.o objectPlanner \
		atari_sim.o mcts.o atariPlanner.o atariPlanner \
		planner.o planner \
		atari_obj_sim.o
