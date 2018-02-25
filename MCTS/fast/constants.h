#ifndef __CONSTANTS_H__
#define __CONSTANTS_H__

#include <vector>

namespace oodqn {

// Mapping from ale::Action enum to human-readable name.
// Note: this requires C++11 to compile.
const char * const ALE_ACTION_NAMES[] = {
  "NOOP"           ,
  "FIRE"           ,
  "UP"             ,
  "RIGHT"          ,
  "LEFT"           ,
  "DOWN"           ,
  "UPRIGHT"        ,
  "UPLEFT"         ,
  "DOWNRIGHT"      ,
  "DOWNLEFT"       ,
  "UPFIRE"         ,
  "RIGHTFIRE"      ,
  "LEFTFIRE"       ,
  "DOWNFIRE"       ,
  "UPRIGHTFIRE"    ,
  "UPLEFTFIRE"     ,
  "DOWNRIGHTFIRE"  ,
  "DOWNLEFTFIRE"  
};

}  // namespace oodqn

// XXX: Put the following inside oodqn namespace.

// Convenient typedefs.
typedef std::vector<float> Vec;
typedef unsigned char Pixel;

// Float error threshold for comparing state vectors.
const float kEps = 1e-8;

// Size of history to be fed into the state model.
const int HISTORY_SIZE = 12;

// Constants for the trained object-level dynamics model.
const char kInputPlaceholderBall[] = "Model_ball_input/x_single";
const char kInputPlaceholderPro[] = "Model_pro_input/x_single";
const char kInputPlaceholderAnt[] = "Model_ant_input/x_single";

const char kNextStateTensorBall[] = "Model_ballM_network/next_state/Tanh";
const char kNextStateTensorPro[] = "Model_proM_network/next_state/Tanh";
const char kNextStateTensorAnt[] = "Model_antM_network/next_state/Tanh";

const char kInputPlaceholderReward[] = "input";
const char kRewardTensor[] = "fc3/BiasAdd";

const float kInitialState[] = {  0.,           0.,           0.,
                                 0.,           0.,           0.,
                                 0.77987421,  -0.13836478,   1. };
const float kResetState[] =   { -0.77987421,   0.10062893,   1.,
                                -0.02515723,   0.03773585,   1.,
                                 0.77987421,   0.01257862,   1. };

#endif

/* First 20 frames in object space:
    [[[ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.77987421]
     [-0.13836478]
     [ 1.        ]]]
    [[[ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.77987421]
     [-0.13836478]
     [ 1.        ]]]
    [[[ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.77987421]
     [-0.13836478]
     [ 1.        ]]]
    [[[ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.77987421]
     [-0.13836478]
     [ 1.        ]]]
    [[[ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.77987421]
     [-0.13836478]
     [ 1.        ]]]
    [[[ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.77987421]
     [-0.13836478]
     [ 1.        ]]]
    [[[ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.77987421]
     [-0.13836478]
     [ 1.        ]]]
    [[[ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.77987421]
     [-0.13836478]
     [ 1.        ]]]
    [[[ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.77987421]
     [-0.13836478]
     [ 1.        ]]]
    [[[ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.77987421]
     [-0.03773585]
     [ 1.        ]]]
    [[[ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.77987421]
     [ 0.08805031]
     [ 1.        ]]]
    [[[ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.77987421]
     [ 0.13836478]
     [ 1.        ]]]
    [[[ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.77987421]
     [ 0.13836478]
     [ 1.        ]]]
    [[[-0.77987421]
     [ 0.10062893]
     [ 1.        ]
     [-0.02515723]
     [ 0.03773585]
     [ 1.        ]
     [ 0.77987421]
     [ 0.01257862]
     [ 1.        ]]]
    [[[-0.77987421]
     [ 0.1509434 ]
     [ 1.        ]
     [-0.0754717 ]
     [ 0.08805031]
     [ 1.        ]
     [ 0.77987421]
     [-0.23899371]
     [ 1.        ]]]
    [[[-0.77987421]
     [ 0.20125786]
     [ 1.        ]
     [-0.12578616]
     [ 0.13836478]
     [ 1.        ]
     [ 0.77987421]
     [-0.51572327]
     [ 1.        ]]]
    [[[-0.77987421]
     [ 0.25157233]
     [ 1.        ]
     [-0.17610063]
     [ 0.18867925]
     [ 1.        ]
     [ 0.77987421]
     [-0.79245283]
     [ 1.        ]]]
    [[[-0.77987421]
     [ 0.30188679]
     [ 1.        ]
     [-0.22641509]
     [ 0.23899371]
     [ 1.        ]
     [ 0.77987421]
     [-0.94968553]
     [ 1.        ]]]
    [[[-0.77987421]
     [ 0.35220126]
     [ 1.        ]
     [-0.27672956]
     [ 0.28930818]
     [ 1.        ]
     [ 0.77987421]
     [-0.96226415]
     [ 1.        ]]]
    [[[-0.77987421]
     [ 0.40251572]
     [ 1.        ]
     [-0.32704403]
     [ 0.33962264]
     [ 1.        ]
     [ 0.77987421]
     [-0.97484277]
     [ 1.        ]]]
 */
