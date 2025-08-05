#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <random>
#include <tuple>
#include <cstring>
#include <utility>
#include <stdexcept>
#include <cmath>

namespace py = pybind11;

std::tuple<py::list, int> propagate_fire_cpp(
    py::array_t<int> forest_np,
    double pb,
    const std::vector<std::pair<int,int>>& neighbours,
    py::array_t<int> neighboursBoolTensor_np,
    bool saveHistory)
{
    // Request input forest array
    auto buf = forest_np.request();
    if (buf.ndim != 2) throw std::runtime_error("forest must be 2D");
    int H = buf.shape[0];
    int W = buf.shape[1];
    int *f_ptr = static_cast<int*>(buf.ptr);
    std::vector<int> forest(f_ptr, f_ptr + H*W);

    // Request neighbour mask tensor (N x H x W)
    auto buf2 = neighboursBoolTensor_np.request();
    if (buf2.ndim != 3) throw std::runtime_error("neighboursBoolTensor must be 3D");
    int N = buf2.shape[0];
    int *nb_ptr = static_cast<int*>(buf2.ptr);

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    py::list history;
    int propagationTime = 0;

    // Check if there are any burning trees initially
    bool thereIsFire = false;
    for (int i = 0; i < H*W; i++) {
        if (forest[i] == 2) {
            thereIsFire = true;
            break;
        }
    }

    if (saveHistory) {
        py::array_t<int> arr({H, W});
        auto out = arr.request();
        std::memcpy(static_cast<int*>(out.ptr), forest.data(), H*W*sizeof(int));
        history.append(arr);
    }

    while (thereIsFire) {
        propagationTime++;
        thereIsFire = false;
        std::vector<int> new_forest = forest;
        std::vector<int> burningNeighbours(H*W, 0);

        // Count burning neighbours for each cell
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                int idx = i*W + j;
                int count = 0;
                for (int k = 0; k < N; ++k) {
                    int mask = nb_ptr[k*(H*W) + idx];
                    if (mask == 0) continue;

                    int dx = neighbours[k].first;
                    int dy = neighbours[k].second;
                    int ni = i + dy;
                    int nj = j + dx;
                    
                    // Proper boundary check: fire stops at edges
                    if (ni < 0 || ni >= H || nj < 0 || nj >= W) {
                        continue;
                    }

                    if (forest[ni*W + nj] == 2) {
                        ++count;
                    }
                }
                burningNeighbours[idx] = count;
            }
        }

        // Update forest based on fire and spread probability
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                int idx = i*W + j;
                if (forest[idx] == 2) {
                    // burned tree becomes ash
                    new_forest[idx] = 3;
                } else if (forest[idx] == 1 && burningNeighbours[idx] > 0) {
                    // tree can catch fire
                    double p_rand = dist(rng);
                    if (p_rand < 1 - std::pow(1 - pb, burningNeighbours[idx])) {
                        new_forest[idx] = 2;
                        thereIsFire = true;
                    }
                }
            }
        }

        forest.swap(new_forest);

        if (saveHistory) {
            py::array_t<int> arr({H, W});
            auto out = arr.request();
            std::memcpy(static_cast<int*>(out.ptr), forest.data(), H*W*sizeof(int));
            history.append(arr);
        }
    }

    return std::make_tuple(history, propagationTime);
}

PYBIND11_MODULE(fire_plus_loop, m) {
    m.doc() = "Fire propagation with nested loops";
    m.def("propagate_fire_cpp", &propagate_fire_cpp,
          py::arg("forest"), py::arg("pb"), py::arg("neighbours"),
          py::arg("neighboursBoolTensor"), py::arg("saveHistory") = false,
          "Simulate fire spread and return history and steps");
}