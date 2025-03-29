#include <torch/extension.h>

#include "gs_core.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("initialize", &gs::initialize,
        "Initialize GaussianMapper with configuration and model path",
        py::arg("gaussian_cfg_path"), py::arg("result_path"));

  m.def("render_from_pose", &gs::renderFromPose,
        "Render image from a given pose (4x4 transformation matrix)",
        py::arg("pose_tensor"), py::arg("width"), py::arg("height"));

  m.def("cleanup", &gs::cleanup, "Clean up resources before program exit");
}