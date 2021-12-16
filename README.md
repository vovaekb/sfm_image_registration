# sfm_image_registration

This repository provides a solution to the task of image registration: given two images of the same object captured at different viewpoints estimate the pose of image A given the pose of B.
First compile executable:

    g++ test_sfm.cpp -o test_sfm `pkg-config --cflags --libs opencv`

Then run compiled executable:
        
    ./test_sfm <image1_path> <image2_path>
