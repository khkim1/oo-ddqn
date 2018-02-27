#include <algorithm>
#include <vector>
#include <ale_interface.hpp>
#include <numeric>
#include "constants.h"
#include <stdio.h>

using namespace std;

namespace oodqn {

vector<Pixel> CropScreen(const ale::ALEScreen& screen){
  const vector<Pixel> array = screen.getArray();
  assert(array.size() == 210 * 160);

  vector<Pixel> output;

  std::copy(array.begin() + 35*160,
            array.begin() + 35*160 + 160*160,
            std::back_inserter(output));
  assert(output.size() == 160 * 160);
  return output;
}

Vec AleScreenToObjState(const ale::ALEScreen& screen){
  vector<Pixel> cropped = CropScreen(screen);

  // Initialize the object vector
  Vec object_coord({0., 0., 0., 0., 0., 0., 0., 0., 0.});

  // Detected coordinate vectors
  Vec ant_coord_x;
  Vec ant_coord_y;
  Vec ball_coord_x;
  Vec ball_coord_y;
  Vec pro_coord_x;
  Vec pro_coord_y;

  // Detect objects by color
  for (int i = 0; i < 160; i++){
    for (int j = 0; j < 160; j++){
      unsigned char r, g, b;
      ale::ALEInterface::getRGB(cropped[160*i+j], r, g, b);
      // if (r != 144 && r != 213 && r != 236 && r != 0) {
      //   // cout << "\n   R value: " << (int)r << endl;
      //   printf("\n  R Value(%d, %d): %d", j, i, (int)r);
      // }

      switch(r) {
        // Color value of opponent paddle
        case 92 :
          ant_coord_x.push_back(i);
          ant_coord_y.push_back(j);
          break;

        // Color value of ball
        case 236 :
          ball_coord_x.push_back(i);
          ball_coord_y.push_back(j);
          break;

        // Color value of your paddle
        case 213 :
          pro_coord_x.push_back(i);
          pro_coord_y.push_back(j);
          break;
      }
    }
  }

  if (!ant_coord_x.empty()){
    object_coord[0] = (std::accumulate(ant_coord_x.begin(), ant_coord_x.end(), 0.0)/ant_coord_x.size()) / 79.5 - 1.0;
    object_coord[1] = (std::accumulate(ant_coord_y.begin(), ant_coord_y.end(), 0.0)/ant_coord_y.size()) / 79.5 - 1.0;
    object_coord[2] = 1;
  }

  if (!ball_coord_x.empty()) {
    object_coord[3] = (std::accumulate(ball_coord_x.begin(), ball_coord_x.end(), 0.0)/ball_coord_x.size()) / 79.5 - 1.0;
    object_coord[4] = (std::accumulate(ball_coord_y.begin(), ball_coord_y.end(), 0.0)/ball_coord_y.size()) / 79.5 - 1.0;
    object_coord[5] = 1;
  }

  if (!pro_coord_x.empty()) {
    object_coord[6] = (std::accumulate(pro_coord_x.begin(), pro_coord_x.end(), 0.0)/pro_coord_x.size()) / 79.5 - 1.0;
    object_coord[7] = (std::accumulate(pro_coord_y.begin(), pro_coord_y.end(), 0.0)/pro_coord_y.size()) / 79.5 - 1.0;
    object_coord[8] = 1;
  }

  return object_coord;
}

}  // namespace oodqn
