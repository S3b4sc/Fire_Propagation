#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>            // For converting py::list to STL containers
#include <vector>
#include <utility>
#include <stdexcept>
#include <cmath>
#include <cstring>                   // for memcpy
#include <typeinfo>
#include <iostream>

// xtensor headers:
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xstrided_view.hpp>
#include <xtensor/xio.hpp>
#include <xtensor-python/pytensor.hpp>

namespace py = pybind11;

// Define alias types for easier reading.
using xt_int  = xt::xarray<int>;
using xt_bool = xt::xarray<bool>;
using xt_float = xt::xarray<double>;

//-------------------------------------------------------------------------
// Function 1: createNeighbourTensor
//
// This function replicates the behavior of your Python createNeighbourTensor.
// It takes the current forest (2D array), a vector of neighbor offsets,
// and a 3D tensor (neighboursBoolTensor). For each neighbor offset, it uses xt::roll
// (i.e. vectorized shifting) and then zeroes out wrapped borders as in your original code.
// Finally, it multiplies by neighboursBoolTensor to mask out the invalid ones.
xt_int createNeighbourTensor(const xt_int& forest,
                             const std::vector<std::pair<int,int>>& neighbours,
                             const xt_int& neighboursBoolTensor) {

    auto forest_shape = forest.shape(); // forest_shape: [height, width]
    size_t height = forest_shape[0];
    size_t width  = forest_shape[1];
    size_t num_neigh = neighbours.size();

    // Create a tensor of zeros with shape (num_neigh, height, width)
    xt_int tensor = xt::zeros<int>({num_neigh, height, width});
    
    for (size_t i = 0; i < num_neigh; ++i) {
        int x = neighbours[i].first;
        int y = neighbours[i].second;
        
        // Here we roll along axis 0 by y and along axis 1 by -x.
        xt_int rolled = xt::roll(forest, static_cast<long>(y), 0);
        rolled = xt::roll(rolled, static_cast<long>(-x), 1);
        
        // Correct the borders (vectorized loop over rows or columns).
        if (x == 1) {
            for (size_t r = 0; r < height; r++) {
                rolled(r, width - 1) = 0;
            }
        } else if (x == -1) {
            for (size_t r = 0; r < height; r++) {
                rolled(r, 0) = 0;
            }
        }
        if (y == 1) {
            for (size_t c = 0; c < width; c++) {
                rolled(0, c) = 0;
            }
        } else if (y == -1) {
            for (size_t c = 0; c < width; c++) {
                rolled(height - 1, c) = 0;
            }
        }
        // Multiply elementwise with the corresponding slice from neighboursBoolTensor.
        // We assume neighboursBoolTensor has shape (num_neigh, height, width).
        xt::view(tensor, i, xt::all(), xt::all()) = xt::eval(rolled * xt::view(neighboursBoolTensor, i, xt::all(), xt::all()));

    }
    return tensor;
}

//-------------------------------------------------------------------------
// Function 2: propagate_fire_cpp
//
// This function replicates your propagateFire method in a vectorized way.
// (For clarity, we assume that "Apply_occupation_proba" has already been applied
// to the forest in Python before calling this C++ function.)
//
// Parameters:
// - forest_np: the forest as a 2D NumPy array (converted to int32)
// - pb: the probability threshold (double)
// - py_neighbours: a Python list of (x,y) pairs for neighbor offsets
// - neighboursBoolTensor_np: a 3D NumPy array (int) of shape (num_neigh, height, width)
// - saveHistory: boolean flag; if true, history is recorded (here we do not return it, but you could extend this)
// Returns: a pair containing the final forest (as a py::array) and the propagation time (int)
std::tuple<py::list, int, py::array_t<int>> propagate_fire_cpp(py::array_t<int> forest_np,
                                             double pb,
                                             py::list py_neighbours,
                                             py::array_t<int> neighboursBoolTensor_np,
                                             bool saveHistory) {

    // Convert forest_np to xtensor (non-owning view):
    auto buf = forest_np.request();
    if (buf.ndim != 2)
        throw std::runtime_error("Forest must be 2D");

    size_t height = buf.shape[0];
    size_t width  = buf.shape[1];
    int* ptr = static_cast<int*>(buf.ptr);
    std::vector<size_t> shape {height, width};
    xt_int forest = xt::adapt(ptr, height * width, xt::no_ownership(), shape);

    // Convert neighboursBoolTensor_np to xtensor.
    auto buf2 = neighboursBoolTensor_np.request();
    if (buf2.ndim != 3)
        throw std::runtime_error("neighboursBoolTensor must be 3D");

    std::vector<size_t> shape2 { (size_t)buf2.shape[0], (size_t)buf2.shape[1], (size_t)buf2.shape[2] };
    xt_int neighboursBoolTensor = xt::adapt(static_cast<int*>(buf2.ptr), buf2.size, xt::no_ownership(), shape2);

    // Convert py_neighbours to std::vector<std::pair<int,int>>
    std::vector<std::pair<int,int>> neighbours;
    for (size_t i = 0; i < py_neighbours.size(); i++){
        py::tuple tup = py_neighbours[i].cast<py::tuple>();
        int a = tup[0].cast<int>();
        int b = tup[1].cast<int>();
        neighbours.push_back({a, b});
    }
    
    int propagationTime = 0;
    bool thereIsFire = true;
    
    // (Optional) Create a history container.
    std::vector<xt_int> history;

    // Main simulation loop:
    while (thereIsFire > 0) {
        // Optionally record history.
        if (saveHistory) { history.push_back(forest); }
        propagationTime++;
        
        
        // Create neighbor tensor using our helper function.
        xt_int neighboursTensor = createNeighbourTensor(forest, neighbours, neighboursBoolTensor);
        
        // Compute "couldPropagate": elementwise check if neighboursTensor equals 2.
        auto couldPropagate = xt::equal(neighboursTensor,2);

        
        // Sum along the neighbor axis (axis 0) to obtain burning neighbor counts for each cell.
        xt_int amountOfBurningNeighbours = xt::sum(couldPropagate, {0});
        

        //std::cout << "Size: " << amountOfBurningNeighbours.size() << std::endl;


        
        // Determine which cells have at least one burning neighbor.
        auto cellsToEvaluate = xt::greater(amountOfBurningNeighbours, 0);
        auto burningTrees = xt::equal(forest,2);
        //std::cout << "burning: "<<xt::sum(burningTrees) << std::endl;
        
        
        // Generate a random probability matrix of shape (height, width). (Using xtensor’s random module.)
        xt_float probabilityMatrixForest = xt::random::rand<double>({height, width});
        
        // For cells to evaluate, update the probability:
        // new_p = 1 - (1 - p)^(1 / (amountOfBurningNeighbours))
        // Use xt::where to vectorize: if cell is to be evaluated, do the power calculation;
        // otherwise, leave the original probability.
        xt_float updatedProb = xt::where(cellsToEvaluate, 1.0 - xt::pow(1.0 - probabilityMatrixForest, 1.0 / amountOfBurningNeighbours),
            probabilityMatrixForest);
        

        auto couldBurn = (updatedProb <= pb);
        
        // Reduce the couldPropagate array along neighbor axis with a logical OR.
        //xt::xarray<std::size_t> axes = {0};
//
        //xt_bool anyCouldPropagate = xt::reduce(couldPropagate,
        //    [](bool a, bool b) { return a || b; }, false, axes);

        
        // Determine new burning trees:
        // New burning trees are those cells that are trees (forest == 1), can burn (couldBurn)
        // and have at least one neighbor that can propagate (anyCouldPropagate).
        auto newBurningTrees = (xt::equal(forest,1) & couldBurn & cellsToEvaluate);
        //std::cout<<"new: "<<xt::sum(newBurningTrees)<<std::endl;
        // Update forest:
        // Burning trees become burnt (3) and new burning trees become burning (2).
        forest = xt::where(burningTrees, 3, forest);
        
        forest = xt::where(newBurningTrees, 2, forest);
        //std::cout<<"new Forest: "<<xt::sum(xt::equal(forest,2))<<std::endl;
        
        
        
        
        // Continue loop if there are any new burning trees.
        
        thereIsFire = (xt::sum(xt::equal(forest, 2))() > 0);

        //std::cout<<"new Forest: "<<xt::sum(xt::equal(forest, 2))()<<std::endl;
        
    }
    // Convert history of forests to a py::list of py::array objects
    
    py::list py_history;
    for (auto& state : history) {
        //std::cout<<state<<std::endl;
        py_history.append(py::cast(state));

    }

    
    
    // After simulation, convert updated forest back to a NumPy array.
    //std::cout << "Size: " << forest.size() << std::endl;
    //py::array final_forest = py::cast(forest);
    
    
    return std::make_tuple(py_history, propagationTime, py::cast(forest));
}

//-------------------------------------------------------------------------
// Pybind11 Module Definition
PYBIND11_MODULE(fire_plus, m) {
    m.doc() = "Fire propagation module using vectorized xtensor operations";
    
    m.def("propagate_fire_cpp", &propagate_fire_cpp,
          "Simulate fire propagation and return final forest and propagation time",
          py::arg("forest"),
          py::arg("pb"),
          py::arg("neighbours"),
          py::arg("neighboursBoolTensor"),
          py::arg("saveHistory") = false);
    
    // Expose createNeighbourTensor as a separate function, if needed.
    m.def("createNeighbourTensor", [](py::array_t<int> forest,
                                      py::list py_neighbours,
                                      py::array_t<int> neighboursBoolTensor) {
        auto buf = forest.request();
        size_t height = buf.shape[0];
        size_t width  = buf.shape[1];
        int* ptr = static_cast<int*>(buf.ptr);
        std::vector<size_t> shape{height, width};
        xt_int forest_xt = xt::adapt(ptr, height * width, xt::no_ownership(), shape);
        
        auto buf2 = neighboursBoolTensor.request();
        std::vector<size_t> shape2{(size_t)buf2.shape[0], (size_t)buf2.shape[1], (size_t)buf2.shape[2]};
        xt_int neighboursBoolTensor_xt = xt::adapt(static_cast<int*>(buf2.ptr), buf2.size, xt::no_ownership(), shape2);
        
        std::vector<std::pair<int,int>> neighbours;
        for (size_t i = 0; i < py_neighbours.size(); i++){
            py::tuple tup = py_neighbours[i].cast<py::tuple>();
            int a = tup[0].cast<int>();
            int b = tup[1].cast<int>();
            neighbours.push_back({a, b});
        }
        
        xt_int tensor = createNeighbourTensor(forest_xt, neighbours, neighboursBoolTensor_xt);
        return py::cast(tensor);
    }, "Create the neighbor tensor using vectorized xtensor operations");
}

// Compile: python setup.py build_ext --inplace
