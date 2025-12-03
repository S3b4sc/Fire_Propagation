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

// Helper: binary Shannon entropy in nats
inline double binary_shannon(double p) {
    if (p <= 0.0 || p >= 1.0) return 0.0;
    return -(p * std::log(p) + (1.0 - p) * std::log(1.0 - p));
}

// Helper: multinomial Shannon entropy contribution in nats
inline double multinomial_term(double p) {
    if (p <= 0.0) return 0.0;
    return -p * std::log(p);
}

/*
  State codes:
    - healthy / tree  == 1
    - burning         == 2
    - burned / ash    == 3
    - empty            == 0  <-- IGNORED from entropy computations

  Returns (history_list, propagationTime, entropies_array)
  entropies_array.shape == (3,) and entries are the time-averaged entropies:
    [ mean_t S_burned(t), mean_t S_burning(t), mean_t S_healthy(t) ]
*/
std::tuple<py::list, int, py::array_t<double>> propagate_fire_cpp(
    py::array_t<int> forest_np,
    double pb,
    const std::vector<std::pair<int,int>>& neighbours,
    py::array_t<int> neighboursBoolTensor_np,
    bool saveHistory,
    bool multinomial // For the entropy nature
)
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

    // Instead of storing per-timestep entropies, we will accumulate sums and count snapshots
    double sum_S_burned = 0.0;
    double sum_S_burning = 0.0;
    double sum_S_healthy = 0.0;
    ssize_t snapshot_count = 0;

    // Function to compute entropies from current forest vector and accumulate them
    auto compute_and_accumulate_entropy = [&](const std::vector<int>& fvec) {
        int total_nonempty = 0;
        int cnt_burned = 0;
        int cnt_burning = 0;
        int cnt_healthy = 0;

        for (int k = 0; k < H * W; ++k) {
            int v = fvec[k];
            if (v == 0) {
                continue; // IGNORE empty cells entirely
            }
            ++total_nonempty;
            if (v == 1) ++cnt_healthy;
            else if (v == 2) ++cnt_burning;
            else if (v == 3) ++cnt_burned;
        }

        double S_burned = 0.0, S_burning = 0.0, S_healthy = 0.0;
        if (total_nonempty > 0) {
            double p_burned  = static_cast<double>(cnt_burned)  / static_cast<double>(total_nonempty);
            double p_burning = static_cast<double>(cnt_burning) / static_cast<double>(total_nonempty);
            double p_healthy = static_cast<double>(cnt_healthy) / static_cast<double>(total_nonempty);

            if (multinomial) {
                // multinomial Shannon entropy contributions
                S_burned  = multinomial_term(p_burned);
                S_burning = multinomial_term(p_burning);
                S_healthy = multinomial_term(p_healthy);
            } else {
                // binary Shannon entropy contributions
                S_burned  = binary_shannon(p_burned);
                S_burning = binary_shannon(p_burning);
                S_healthy = binary_shannon(p_healthy);
            }
        }

        // accumulate
        sum_S_burned  += S_burned;
        sum_S_burning += S_burning;
        sum_S_healthy += S_healthy;
        ++snapshot_count;
    };

    // Save initial snapshot & accumulate entropy for it
    if (saveHistory) {
        py::array_t<int> arr({H, W});
        auto out = arr.request();
        std::memcpy(static_cast<int*>(out.ptr), forest.data(), H*W*sizeof(int));
        history.append(arr);
    }
    compute_and_accumulate_entropy(forest); // initial snapshot

    while (thereIsFire) {
        propagationTime++;
        thereIsFire = false;
        std::vector<int> new_forest = forest;
        std::vector<int> burningNeighbours(H*W, 0);

        // Count burning neighbours
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

        // Update forest
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                int idx = i*W + j;
                if (forest[idx] == 2) {
                    new_forest[idx] = 3;
                } else if (forest[idx] == 1 && burningNeighbours[idx] > 0) {
                    double p_rand = dist(rng);
                    if (p_rand < 1.0 - std::pow(1.0 - pb, static_cast<double>(burningNeighbours[idx]))) {
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

        compute_and_accumulate_entropy(forest);
    }

    // Compute time-averaged entropies
    double avg_S_burned = 0.0;
    double avg_S_burning = 0.0;
    double avg_S_healthy = 0.0;
    if (snapshot_count > 0) {
        avg_S_burned  = sum_S_burned  / static_cast<double>(snapshot_count);
        avg_S_burning = sum_S_burning / static_cast<double>(snapshot_count);
        avg_S_healthy = sum_S_healthy / static_cast<double>(snapshot_count);
    }

    // Prepare entropies numpy array
    py::array_t<double> ent_arr(py::array::ShapeContainer{(size_t)3});
    auto ent_buf = ent_arr.request();
    double *ent_ptr = static_cast<double*>(ent_buf.ptr);

    ent_ptr[0] = avg_S_burned;
    ent_ptr[1] = avg_S_burning;
    ent_ptr[2] = avg_S_healthy;

    return std::make_tuple(history, propagationTime, ent_arr);
}

PYBIND11_MODULE(fire_plus_loop, m) {
    m.doc() = "Fire propagation with nested loops; returns history, steps, and averaged entropies (3,). Supports multinomial or binary entropies.";
    m.def("propagate_fire_cpp", &propagate_fire_cpp,
          py::arg("forest"), py::arg("pb"), py::arg("neighbours"),
          py::arg("neighboursBoolTensor"), py::arg("saveHistory") = false,
          py::arg("multinomial") = false,   // <--- default false
          "Simulate fire spread and return history, steps, and per-simulation averaged entropies (3,). If multinomial=True, computes Shannon entropy contributions.");
}
