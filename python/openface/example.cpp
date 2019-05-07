#include <pybind11/pybind11.h>

namespace py = pybind11;

int add(int i, int j){
  return i+j;
}

//Macro that creates a function that will be called when an "import" statement is issued withing Python.The module name (example) is given as the first macro argument (not in quotes!). The second argument (m) defines a variable of type py::module which is the main interface for creating bindings. The method module::def() generates binding code that exposes the add() function to Python.


PYBIND11_MODULE(example, m){
  m.doc() = "pybind11 example plugin"; //optional module docstring
  m.def("add", &add, "A function which adds two numbers");
}
