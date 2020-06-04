#include <pybind11/pybind11.h>

#include <LandmarkCoreIncludes.h>
#include <FaceAnalyser.h>
#include <GazeEstimation.h>
#include <RecorderOpenFace.h>
#include <RecorderOpenFaceParameters.h>
#include <SequenceCapture.h>
#include <Visualizer.h>
#include <VisualizationUtils.h>
#include "opencv2/objdetect.hpp"
#include <dlib/image_processing/frontal_face_detector.h>
#include <ImageCapture.h>
//Convert
#include <pybind11/stl.h>
#include "util/conversions.h"
#include <iostream>

namespace py = pybind11;
using namespace LandmarkDetector;
using namespace FaceAnalysis;
using namespace GazeAnalysis;
using namespace Utilities;

cv::Mat read_image(std::string image_name)
{
        cv::Mat image = cv::imread(image_name, CV_LOAD_IMAGE_COLOR);
            return image;
}

void receiveVec6f(cv::Vec6f v)
{
  std::cout << "Point: " << v << std::endl;
}


cv::Scalar test_scalar(cv::Scalar s)
{
    return s;
}

cv::Point3f test_point3f(cv::Point3f p)
{
    return p;
}

cv::Vec6f sendVec6f()
{
    cv::Vec6f v(1, 2, 3, 4, 5, 6);
    return v;
}

PYBIND11_MODULE(pyopenface, m){
    
    //General initializing functions
     
    m.doc() = "OpenFace python wrapper";
    NDArrayConverter::init_numpy();
    
    m.def("read_image", &read_image, "A function that read an image", py::arg("image"));
    m.def("receiveVec6f", &receiveVec6f);
    m.def("test_scalar", &test_scalar);
    m.def("test_point3f", &test_point3f);
    m.def("sendVec6f", &sendVec6f);



    //General functions to create face detectors from opencv and dlib.
    py::class_<cv::CascadeClassifier>(m, "cvCascadeClassifier")
        .def(py::init<>())
        .def(py::init<std::string &>());

    py::class_<dlib::frontal_face_detector>(m, "dlibFrontalFaceDetector")
        .def(py::init<>());

    m.def("dlib_get_frontal_face_detector", &dlib::get_frontal_face_detector);



    //OpenFace class wrappers
    //LandmarkDetector::FaceModelParameters
    py::class_<FaceModelParameters> face_mod_params(m, "FaceModelParameters");
    py::enum_<FaceModelParameters::LandmarkDetector>(face_mod_params, "LandmarkDetector")
        .value("CLM_DETECTOR", FaceModelParameters::CLM_DETECTOR)
        .value("CLNF_DETECTOR", FaceModelParameters::CLNF_DETECTOR)
        .value("CECLM_DETECTOR", FaceModelParameters::CECLM_DETECTOR)
        .export_values();
    py::enum_<FaceModelParameters::FaceDetector>(face_mod_params, "FaceDetector")
        .value("HAAR_DETECTOR", FaceModelParameters::HAAR_DETECTOR)
        .value("HOG_SVM_DETECTOR", FaceModelParameters::HOG_SVM_DETECTOR)
        .value("MTCNN_DETECTOR", FaceModelParameters::MTCNN_DETECTOR)
        .export_values();

    face_mod_params.def(py::init<vector<std::string> &>());
    face_mod_params.def(py::init<>());
    face_mod_params.def_readwrite("curr_landmark_detector", &FaceModelParameters::curr_landmark_detector);
    face_mod_params.def_readwrite("curr_face_detector", &FaceModelParameters::curr_face_detector);
    face_mod_params.def_readwrite("num_optimisation_iteration", &FaceModelParameters::num_optimisation_iteration);
    face_mod_params.def_readwrite("limit_pose", &FaceModelParameters::limit_pose);
    face_mod_params.def_readwrite("validate_detections", &FaceModelParameters::validate_detections);
    face_mod_params.def_readwrite("validation_boundary", &FaceModelParameters::validation_boundary);
    face_mod_params.def_readwrite("window_sizes_small", &FaceModelParameters::window_sizes_small);
    face_mod_params.def_readwrite("window_sizes_init", &FaceModelParameters::window_sizes_init);
    face_mod_params.def_readwrite("window_sizes_current", &FaceModelParameters::window_sizes_current);
    face_mod_params.def_readwrite("face_template_scale", &FaceModelParameters::face_template_scale);
    face_mod_params.def_readwrite("use_face_template", &FaceModelParameters::use_face_template);
    face_mod_params.def_readwrite("model_location", &FaceModelParameters::model_location);
    face_mod_params.def_readwrite("sigma", &FaceModelParameters::sigma);
    face_mod_params.def_readwrite("reg_factor", &FaceModelParameters::reg_factor);
    face_mod_params.def_readwrite("weight_factor", &FaceModelParameters::weight_factor);
    face_mod_params.def_readwrite("multi_view", &FaceModelParameters::multi_view);
    face_mod_params.def_readwrite("reinit_video_every", &FaceModelParameters::reinit_video_every);
    face_mod_params.def_readwrite("haar_face_detector_location", &FaceModelParameters::haar_face_detector_location);
    face_mod_params.def_readwrite("mtcnn_face_detector_location", &FaceModelParameters::mtcnn_face_detector_location);
    face_mod_params.def_readwrite("refine_hierarchical", &FaceModelParameters::refine_hierarchical);
    face_mod_params.def_readwrite("refine_parameters", &FaceModelParameters::refine_parameters);
    

    //LandmarkDetector::FaceDetectorMTCNN
    py::class_<FaceDetectorMTCNN>(m, "FaceDetectorMTCNN")
        .def(py::init<>())
        .def(py::init<std::string &>())
        .def(py::init<const FaceDetectorMTCNN &>())
        .def("DetectFaces", &FaceDetectorMTCNN::DetectFaces)
        .def("Read", &FaceDetectorMTCNN::Read)
        .def("empty", &FaceDetectorMTCNN::empty);


    //LandmarkDetector::PDM
    py::class_<PDM>(m, "PDM")
        .def(py::init<>())
        .def(py::init<const PDM &>())
        .def("Read", &PDM::Read)
        .def("NumberOfPoints", &PDM::NumberOfPoints)
        .def("NumberOfModes", &PDM::NumberOfModes)
        .def("Clamp", &PDM::Clamp)
        .def("CalcShape2D", &PDM::CalcShape2D)
        .def("CalcParams", (void (PDM::*)(cv::Vec6f&, const cv::Rect_<float>&, const cv::Mat_<float>&, const cv::Vec3f)) &PDM::CalcParams, "Provided the bounding box of a face and the local parameters (with optional rotation), generates the global parameters that can generate the face with the provided bounding box")
        .def("CalcParams", (void (PDM::*)(cv::Vec6f&, cv::Mat_<float>&, const cv::Mat_<float>&, const cv::Vec3f)) &PDM::CalcParams, "Provided the landmark location compute global and local parameters best fitting it (can provide optional rotation for potentially better results)")
        .def("CalcBoundingBox", &PDM::CalcBoundingBox)
        .def("ComputeRigidJacobian", &PDM::ComputeRigidJacobian)
        .def("ComputeJacobian", &PDM::ComputeJacobian)
        .def("UpdateModelParameters", &PDM::UpdateModelParameters)
        .def_readwrite("mean_shape", &PDM::mean_shape)
        .def_readwrite("princ_comp", &PDM::princ_comp)
        .def_readwrite("eigen_values", &PDM::eigen_values);

    //LandmarkDetector::SVR_patch_expert
    py::class_<SVR_patch_expert>(m, "SVR_patch_expert")
        .def(py::init<>())
        .def(py::init<const SVR_patch_expert &>())
        .def("Read", &SVR_patch_expert::Read)
        .def("Response", &SVR_patch_expert::Response)
        .def("ResponseDepth", &SVR_patch_expert::ResponseDepth)
        .def_readwrite("type", &SVR_patch_expert::type)
        .def_readwrite("bias", &SVR_patch_expert::bias)
        .def_readwrite("weights", &SVR_patch_expert::weights)
        .def_readwrite("weights_dfts", &SVR_patch_expert::weights_dfts)
        .def_readwrite("confidence", &SVR_patch_expert::confidence);


    //LandmarkDetector::Multi_SVR_patch_expert
    py::class_<Multi_SVR_patch_expert>(m, "Multi_SVR_patch_expert")
        .def(py::init<>())
        .def(py::init<const Multi_SVR_patch_expert &>())
        .def("Read", &Multi_SVR_patch_expert::Read)
        .def("Response", &Multi_SVR_patch_expert::Response)
        .def("ResponseDepth", &Multi_SVR_patch_expert::ResponseDepth);


    //Landmark::CCNF_patch_expert
    py::class_<CCNF_patch_expert>(m, "CCNF_patch_expert")
        .def(py::init<>())
        .def(py::init<const CCNF_patch_expert &>())
        .def("Read", &CCNF_patch_expert::Read)
        .def("Response", &CCNF_patch_expert::Response)
        .def("ResponseOpenBlas", &CCNF_patch_expert::ResponseOpenBlas)
        .def("ComputeSigmas", &CCNF_patch_expert::ComputeSigmas)
        .def_readwrite("width", &CCNF_patch_expert::width)
        .def_readwrite("height", &CCNF_patch_expert::height)
        .def_readwrite("window_sizes", &CCNF_patch_expert::window_sizes)
        .def_readwrite("Sigmas", &CCNF_patch_expert::Sigmas)
        .def_readwrite("betas", &CCNF_patch_expert::betas)
        .def_readwrite("weight_matrix", &CCNF_patch_expert::weight_matrix)
        .def_readwrite("patch_confidence", &CCNF_patch_expert::patch_confidence);


    //Landmark::CEN_patch_expert
    py::class_<CEN_patch_expert>(m, "CEN_patch_expert")
        .def(py::init<>())
        .def(py::init<const CEN_patch_expert &>())
        .def("Read", &CEN_patch_expert::Read)
        .def("Response", &CEN_patch_expert::Response)
        .def("ResponseInternal", &CEN_patch_expert::ResponseInternal)
        .def("ResponseSparse", &CEN_patch_expert::ResponseSparse)
        .def_readwrite("width_support", &CEN_patch_expert::width_support)
        .def_readwrite("height_support", &CEN_patch_expert::height_support)
        .def_readwrite("biases", &CEN_patch_expert::biases)
        .def_readwrite("weights", &CEN_patch_expert::weights)
        .def_readwrite("activation_function", &CEN_patch_expert::activation_function)
        .def_readwrite("confidence", &CEN_patch_expert::confidence);

    //LandmarkDetector::Patch_experts
    py::class_<Patch_experts>(m, "Patch_experts")
        .def(py::init<>())
        .def(py::init<const Patch_experts &>())
        .def("Response", &Patch_experts::Response)
        .def("GetViewIdx", &Patch_experts::GetViewIdx)
        .def("nViews", &Patch_experts::nViews)
        .def("Read", &Patch_experts::Read);
    
    
    //LandmarkDetector::CLNF
    py::class_<CLNF> clnf(m, "CLNF");
    clnf.def(py::init<>());
    clnf.def(py::init<std::string>());
    clnf.def(py::init<const CLNF &>());
    clnf.def("DetectLandmarks", &CLNF::DetectLandmarks);
    clnf.def("GetShape", &CLNF::GetShape);
    clnf.def("GetVisibilities", &CLNF::GetVisibilities);
    clnf.def("Reset", (void (CLNF::*)()) &CLNF::Reset, "Reset the model (useful if we want to completelly reinitialise, or we want to track another video)");
    clnf.def("Reset", (void (CLNF::*)(double, double)) &CLNF::Reset, "Reset the model, choosing the face nearest (x,y) where x and y are between 0 and 1.");
    clnf.def("Read", &CLNF::Read);
    clnf.def_readwrite("pdm", &CLNF::pdm);
    clnf.def_readwrite("patch_experts", &CLNF::patch_experts);
    clnf.def_readwrite("params_local", &CLNF::params_local);
    clnf.def_readwrite("params_global", &CLNF::params_global);
    clnf.def_readwrite("hierarchical_models", &CLNF::hierarchical_models);
    clnf.def_readwrite("hierarchical_model_names", &CLNF::hierarchical_model_names);
    clnf.def_readwrite("hierarchical_mapping", &CLNF::hierarchical_mapping);
    clnf.def_readwrite("hierarchical_params", &CLNF::hierarchical_params);
    clnf.def_readwrite("face_detector_HAAR", &CLNF::face_detector_HAAR);
    clnf.def_readwrite("haar_face_detector_location", &CLNF::haar_face_detector_location);
    clnf.def_readwrite("face_detector_HOG", &CLNF::face_detector_HOG);
    clnf.def_readwrite("face_detector_MTCNN", &CLNF::face_detector_MTCNN);


    clnf.def_readwrite("detection_success", &CLNF::detection_success);
    clnf.def_readwrite("tracking_initialised", &CLNF::tracking_initialised);
    clnf.def_readwrite("detection_certainty", &CLNF::detection_certainty);
    clnf.def_readwrite("eye_model", &CLNF::eye_model);
    clnf.def_readwrite("triangulations", &CLNF::triangulations);
    clnf.def_readwrite("detected_landmarks", &CLNF::detected_landmarks);
    clnf.def_readwrite("model_likelihood", &CLNF::model_likelihood);
    clnf.def_readwrite("landmark_likelihoods", &CLNF::landmark_likelihoods);
    clnf.def_readwrite("failures_in_a_row", &CLNF::failures_in_a_row);
    clnf.def_readwrite("face_template", &CLNF::face_template);
    clnf.def_readwrite("preference_det", &CLNF::preference_det);
    clnf.def_readwrite("view_used", &CLNF::view_used);
    clnf.def_readwrite("loaded_successfully", &CLNF::loaded_successfully);

    //LandmarkDetectior::LandmarkDetectorFunc
    
    m.def("DetectLandmarksInVideo", (bool (*) (const cv::Mat &, CLNF&, FaceModelParameters&, cv::Mat &)) &DetectLandmarksInVideo, "Detect landmarks without giving face bounding box");
    m.def("DetectLandmarksInVideo", (bool (*) (const cv::Mat &, const cv::Rect_<double>, CLNF&, FaceModelParameters&, cv::Mat &)) &DetectLandmarksInVideo, "Detect landmarks with face bounding box");
    m.def("DetectLandmarksInImage", (bool (*) (const cv::Mat &, CLNF&, FaceModelParameters&, cv::Mat &)) &DetectLandmarksInVideo, "Detect landmarks without face bounding box");
    m.def("DetectLandmarksInImage", (bool (*) (const cv::Mat &, const cv::Rect_<double>, CLNF&, FaceModelParameters&, cv::Mat &)) &DetectLandmarksInVideo, "Detect landmarks with face bounding box");
    m.def("GetPose", &GetPose);
    m.def("GetPoseWRTCamera", &GetPoseWRTCamera);

    //FaceAnalysis::FaceAnalyserParameters

    py::class_<FaceAnalyserParameters>(m, "FaceAnalyserParameters")
        .def(py::init<>())
        .def(py::init<std::string>())
        .def(py::init<std::vector<std::string> &>())
        .def("setAlignedOutput", &FaceAnalyserParameters::setAlignedOutput)
        .def("OptimizeForVideos", &FaceAnalyserParameters::OptimizeForVideos)
        .def("OptimizeForImages", &FaceAnalyserParameters::OptimizeForImages)
        .def("getAlignMask", &FaceAnalyserParameters::getAlignMask)
        .def("getSimScaleOut", &FaceAnalyserParameters::getSimScaleOut)
        .def("getSimSizeOut", &FaceAnalyserParameters::getSimSizeOut)
        .def("getDynamic", &FaceAnalyserParameters::getDynamic)
        .def("getModelLoc", &FaceAnalyserParameters::getModelLoc)
        .def("getOrientationBins", &FaceAnalyserParameters::getOrientationBins)
        .def("setAlignedOutput", &FaceAnalyserParameters::setAlignedOutput)
        .def_readwrite("grayscale", &FaceAnalyserParameters::grayscale);


    //FaceAnalysis::FaceAnalyser
    py::class_<FaceAnalyser> face_analyser(m, "FaceAnalyser");
    py::enum_<FaceAnalyser::RegressorType>(face_analyser, "RegressorType")
        .value("SVR_appearance_static_linear", FaceAnalyser::SVR_appearance_static_linear)
        .value("SVR_appearance_dynamic_linear", FaceAnalyser::SVR_appearance_dynamic_linear)
        .value("SVR_dynamic_geom_linear", FaceAnalyser::SVR_dynamic_geom_linear)
        .value("SVR_combined_linear", FaceAnalyser::SVR_combined_linear)
        .value("SVM_linear_stat", FaceAnalyser::SVM_linear_stat)
        .value("SVM_linear_dyn", FaceAnalyser::SVR_combined_linear)
        .value("SVR_linear_static_seg", FaceAnalyser::SVR_linear_static_seg)
        .value("SVR_linear_dynamic_seg", FaceAnalyser::SVR_linear_dynamic_seg)
        .export_values();

    face_analyser.def(py::init<FaceAnalyserParameters &>())
        .def("AddNextFrame", &FaceAnalyser::AddNextFrame)
        .def("GetCurrentTimeSeconds", &FaceAnalyser::AddNextFrame)
        .def("GetCurrentAUsClass", &FaceAnalyser::AddNextFrame)
        .def("GetCurrentAUsReg", &FaceAnalyser::AddNextFrame)
        .def("PredictStaticAUsAndComputeFeatures", &FaceAnalyser::AddNextFrame)
        .def("Reset", &FaceAnalyser::AddNextFrame)
        .def("GetLatestHOG", &FaceAnalyser::AddNextFrame)
        .def("GetLatestNeutralHOG", &FaceAnalyser::AddNextFrame)
        .def("GetTriangulation", &FaceAnalyser::AddNextFrame)
        .def("GetGeomDescriptor", &FaceAnalyser::AddNextFrame)
        .def("GetAUClassNames", &FaceAnalyser::AddNextFrame)
        .def("GetAURegNames", &FaceAnalyser::AddNextFrame)
        .def("GetDynamicAUClass", &FaceAnalyser::AddNextFrame)
        .def("GetDynamicAUReg", &FaceAnalyser::AddNextFrame)
        .def("ExtractAllPredictionsOfflineReg", &FaceAnalyser::AddNextFrame)
        .def("ExtractAllPredictionsOfflineClass", &FaceAnalyser::AddNextFrame)
        .def("PostprocessOutputFile", &FaceAnalyser::AddNextFrame)
        .def("GetCurrentAUsCombined", &FaceAnalyser::AddNextFrame)
        .def("GetLatestAlignedFace", &FaceAnalyser::AddNextFrame);
    
    m.def("EstimateGaze", [](LandmarkDetector::CLNF clnf_model, cv::Point3f gaze_absolute, float fx, float fy, float cx, float cy, bool left_eye){
		    cv::Point3f gaze_abs = gaze_absolute;

		    GazeAnalysis::EstimateGaze(clnf_model, gaze_abs, fx, fy, cx, cy, left_eye);

		    return gaze_absolute;
		    }, "Returns a 3d gaze vector");
    m.def("GetGazeAngle", &GetGazeAngle);
    m.def("GetPupilPosition", &GetPupilPosition);

    //Utilities::SequenceCapture
    py::class_<SequenceCapture>(m, "SequenceCapture")
        .def(py::init<>())
        .def("Open", &SequenceCapture::Open)
        .def("OpenWebcam", &SequenceCapture::OpenWebcam)
        .def("OpenImageSequence", &SequenceCapture::OpenImageSequence)
        .def("OpenVideoFile", &SequenceCapture::OpenVideoFile)
        .def("IsWebcam", &SequenceCapture::IsWebcam)
        .def("GetNextFrame", &SequenceCapture::GetNextFrame)
        .def("GetGrayFrame", &SequenceCapture::GetGrayFrame)
        .def("GetProgress", &SequenceCapture::GetProgress)
        .def("GetFrameNumber", &SequenceCapture::GetFrameNumber)
        .def("IsOpened", &SequenceCapture::IsOpened)
        .def("Close", &SequenceCapture::Close)
        .def_readwrite("frame_width", &SequenceCapture::frame_width)
        .def_readwrite("frame_height", &SequenceCapture::frame_height)
        .def_readwrite("fx", &SequenceCapture::fx)
        .def_readwrite("fy", &SequenceCapture::fy)
        .def_readwrite("cx", &SequenceCapture::cx)
        .def_readwrite("cy", &SequenceCapture::cy)
        .def_readwrite("fps", &SequenceCapture::fps)
        .def_readwrite("time_stamp", &SequenceCapture::time_stamp)
        .def_readwrite("name", &SequenceCapture::name)
        .def_readwrite("no_input_specified", &SequenceCapture::no_input_specified);
        
    //Utilities::ImageCapture
    py::class_<ImageCapture>(m, "ImageCapture")
        .def(py::init<>())
        .def("Open", &ImageCapture::Open)
        .def("OpenDirectory", &ImageCapture::OpenDirectory)
        .def("OpenImageFiles", &ImageCapture::OpenImageFiles)
        .def("GetNextImage", &ImageCapture::GetNextImage)
        .def("GetGrayFrame", &ImageCapture::GetGrayFrame)
        .def("GetBoundingBoxes", &ImageCapture::GetBoundingBoxes)
        .def("GetProgress", &ImageCapture::GetProgress)
        .def_readwrite("image_width", &ImageCapture::image_width)
        .def_readwrite("image_height", &ImageCapture::image_height)
        .def_readwrite("fx", &ImageCapture::fx)
        .def_readwrite("fy", &ImageCapture::fy)
        .def_readwrite("cx", &ImageCapture::cx)
        .def_readwrite("cy", &ImageCapture::cy)
        .def_readwrite("name", &ImageCapture::name)
        .def_readwrite("has_bounding_boxes", &ImageCapture::has_bounding_boxes);
    
    //Utilities::Visualizer
    py::class_<Visualizer>(m, "Visualizer")
        .def(py::init<std::vector<std::string>>())
        .def(py::init<bool, bool, bool, bool>())
        .def("SetImage", &Visualizer::SetImage)
        .def("SetObservationLandmarks", &Visualizer::SetObservationLandmarks)
        .def("SetObservationPose", &Visualizer::SetObservationPose)
        .def("SetObservationActionUnits", &Visualizer::SetObservationActionUnits)
        .def("SetObservationGaze", &Visualizer::SetObservationGaze)
        .def("SetObservationFaceAlign", &Visualizer::SetObservationFaceAlign)
        .def("SetObservationHOG", &Visualizer::SetObservationHOG)
        .def("SetFps", &Visualizer::SetFps)
        .def("ShowObservation", &Visualizer::ShowObservation)
        .def("GetVisImage", &Visualizer::GetVisImage)
        .def("GetHOGVis", &Visualizer::GetHOGVis)
        .def_readwrite("vis_track", &Visualizer::vis_track)
        .def_readwrite("vis_hog", &Visualizer::vis_hog)
        .def_readwrite("vis_align", &Visualizer::vis_align)
        .def_readwrite("vis_aus", &Visualizer::vis_aus)
        .def_readwrite("visualisation_boundary", &Visualizer::visualisation_boundary);

    //Utilities::VisualizationUtils
    m.def("DrawBox", (void (*)(cv::Mat, cv::Vec6f, cv::Scalar, int, float, float, float, float)) &DrawBox);
    m.def("DrawBox", (void (*)(const std::vector<std::pair<cv::Point2f, cv::Point2f>>&, cv::Mat, cv::Scalar, int)) &DrawBox);
    
}
