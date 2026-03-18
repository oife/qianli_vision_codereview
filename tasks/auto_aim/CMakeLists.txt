cmake_minimum_required(VERSION 3.16)

add_subdirectory(planner/tinympc)

add_library(auto_aim OBJECT 
    armor/armor.cpp
    classifier/classifier.cpp 
    detector/detector.cpp
    solver/solver.cpp
    aimer/aimer.cpp
    target/target.cpp
    tracker/tracker.cpp
    voter/voter.cpp
    shooter/shooter.cpp
    yolos/yolo.cpp
    yolos/yolov5.cpp
    yolos/yolov8.cpp
    yolos/yolo11.cpp
    yolos/yolo26n.cpp
    multithread/commandgener.cpp
    multithread/mt_detector.cpp
    planner/planner.cpp
    planner/quintic_polynomial/quintic_polynomial.cpp
)

target_link_libraries(auto_aim io backend tinympcstatic)