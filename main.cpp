//
//  main.cpp
//  interpolation
//
//  Created by Gary Chu on 04/04/2019.
//  Copyright © 2019 Gary Chu. All rights reserved.
//

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <OpenGL/gl.h>
#include <OpenGl/glu.h>

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <iterator>
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

MatrixXd ori_pose(214,2);
MatrixXd aft_pose(214,2);
MatrixXd ori_aft_dif(214,2);
MatrixXd face(332,3);
int frames = 100;
//int frames = 10;
int v_count = 0, f_count = 0, aft_count = 0, frame_count = 0;
int interpolation_mode = 0;

MatrixXd final_b(332*4,1);
MatrixXd A1(332*4,1);
MatrixXd A2(332*4,1);
MatrixXd final_A(332*4,214*2);
//MatrixXd precomputed_A(214*2,332*4);
MatrixXd precomputed_A(213*2,332*4);

MatrixXd A_trans[332];
//MatrixXd R_gamma[332];
double R_gamma_theta[332];
MatrixXd S[332];
MatrixXd I(2,2);

MatrixXd A_gamma(double theta, MatrixXd S_t, double t) {
//    cerr << R_gamma_t << endl;
    
    MatrixXd R_gamma_t(2,2);
    double theta_t = theta * t;
//    if (theta_t < 0) {
//        theta_t = -theta_t;
//    }
    R_gamma_t << cos(theta_t), -sin(theta_t),
                 sin(theta_t), cos(theta_t);

    return R_gamma_t * ((1 - t) * I + t * S_t);
}

void fill_b(double t, double first_x, double first_y) {
    for (int i = 0; i < 332; i++) {
        MatrixXd temp = A_gamma(R_gamma_theta[i], S[i], t);
//        cerr << "bibi" << endl;
        final_b(4*i) = temp(0,0);
        final_b(4*i+1) = temp(0,1);
        final_b(4*i+2) = temp(1,0);
        final_b(4*i+3) = temp(1,1);
    }
    final_b = final_b - first_x * A1 - first_y * A2;
}

void decompose() {
    for (int i = 0; i < 332; i++) {
        JacobiSVD<MatrixXd> svd(A_trans[i], ComputeFullU | ComputeFullV);
//        cerr << "cici" << endl;
        MatrixXd R_alpha = svd.matrixU();
//        cerr << A_trans[i] << endl;
        MatrixXd D = svd.singularValues().asDiagonal();
        MatrixXd R_beta = svd.matrixV().transpose();
//        MatrixXd R_beta = svd.matrixV();
        
//        R_gamma[i] = R_alpha * R_beta;
        
        MatrixXd R_gamma = R_alpha * R_beta;
//        R_gamma_theta[i] = acos(R_gamma(0,0));
        R_gamma_theta[i] = asin(R_gamma(1,0));
        cerr << R_gamma_theta[i] << endl;
        
        S[i] = R_beta.transpose() * D * R_beta;
//        cerr << D << endl;
//        exit(1);
    }
}

void calculate_A() {
    final_A << MatrixXd::Zero(final_A.rows(),final_A.cols());
    for (int i = 0; i < 332; i++) {
        int one = face(i,0);
        int two = face(i,1);
        int three = face(i,2);
        
        double x_1 = ori_pose(one,0);
        double y_1 = ori_pose(one,1);
        double x_1_p = aft_pose(one,0);
        double y_1_p = aft_pose(one,1);
        double x_2 = ori_pose(two,0);
        double y_2 = ori_pose(two,1);
        double x_2_p = aft_pose(two,0);
        double y_2_p = aft_pose(two,1);
        double x_3 = ori_pose(three,0);
        double y_3 = ori_pose(three,1);
        double x_3_p = aft_pose(three,0);
        double y_3_p = aft_pose(three,1);
        
        MatrixXd this_A(6,6);
        this_A << x_1, y_1, 1, 0, 0, 0,
                  0, 0, 0, x_1, y_1, 1,
                  x_2, y_2, 1, 0, 0, 0,
                  0, 0, 0, x_2, y_2, 1,
                  x_3, y_3, 1, 0, 0, 0,
                  0, 0, 0, x_3, y_3, 1;
        MatrixXd this_A_inv = this_A.inverse();
        
        final_A(i*4,2*one) = this_A_inv(0,0);
        final_A(i*4,2*one+1) = this_A_inv(0,1);
        final_A(i*4,2*two) = this_A_inv(0,2);
        final_A(i*4,2*two+1) = this_A_inv(0,3);
        final_A(i*4,2*three) = this_A_inv(0,4);
        final_A(i*4,2*three+1) = this_A_inv(0,5);
            
        final_A(i*4+1,2*one) = this_A_inv(1,0);
        final_A(i*4+1,2*one+1) = this_A_inv(1,1);
        final_A(i*4+1,2*two) = this_A_inv(1,2);
        final_A(i*4+1,2*two+1) = this_A_inv(1,3);
        final_A(i*4+1,2*three) = this_A_inv(1,4);
        final_A(i*4+1,2*three+1) = this_A_inv(1,5);
            
        final_A(i*4+2,2*one) = this_A_inv(3,0);
        final_A(i*4+2,2*one+1) = this_A_inv(3,1);
        final_A(i*4+2,2*two) = this_A_inv(3,2);
        final_A(i*4+2,2*two+1) = this_A_inv(3,3);
        final_A(i*4+2,2*three) = this_A_inv(3,4);
        final_A(i*4+2,2*three+1) = this_A_inv(3,5);
            
        final_A(i*4+3,2*one) = this_A_inv(4,0);
        final_A(i*4+3,2*one+1) = this_A_inv(4,1);
        final_A(i*4+3,2*two) = this_A_inv(4,2);
        final_A(i*4+3,2*two+1) = this_A_inv(4,3);
        final_A(i*4+3,2*three) = this_A_inv(4,4);
        final_A(i*4+3,2*three+1) = this_A_inv(4,5);
        
        MatrixXd this_B(6,1);
        this_B << x_1_p, y_1_p, x_2_p, y_2_p, x_3_p, y_3_p;
        MatrixXd this_x = this_A_inv * this_B;
        MatrixXd this_A_trans(2,2);
        this_A_trans << this_x(0), this_x(1),
                        this_x(3), this_x(4);
        A_trans[i] = this_A_trans;
    }
//    cerr << final_A.rows() << final_A.cols() << endl;
//    MatrixXd temp = (final_A.transpose() * final_A).inverse();
    
    MatrixXd new_A(332*4,214*2-2);
    
    for (int i = 0; i < A1.rows(); i++) {
        A1(i,0) = final_A(i,0);
        A2(i,0) = final_A(i,1);
        for (int j = 0; j < new_A.cols(); j++) {
            new_A(i,j) = final_A(i,j+2);
        }
    }
    
    
//    precomputed_A = (final_A.transpose() * final_A).inverse() * final_A.transpose();
    
    precomputed_A = (new_A.transpose() * new_A).inverse() * new_A.transpose();
//    cerr << temp(0,0) << endl;
}

void parse_obj() {
    string obj_line;
    ifstream obj_file("/Users/garychu/Desktop/cm50245cw/interpolation/interpolation/interpolation/man.obj");
    //    ifstream obj_file("man.obj");
    
    while (!obj_file.eof()) {
        getline(obj_file, obj_line);
        if (obj_line[0] != 'v' & obj_line[0] != 'f') {
            continue;
        } else {
            istringstream iss(obj_line);
            vector<string> results(istream_iterator<string>{iss}, istream_iterator<string>());
            if (results[0] == "v") {
                ori_pose(v_count,0) = stod(results[1]);
                ori_pose(v_count,1) = -1 * stod(results[2]);
                v_count++;
            } else if (results[0] == "f") {
                face(f_count,0) = stoi(results[1]) - 1;
                face(f_count,1) = stoi(results[2]) - 1;
                face(f_count,2) = stoi(results[3]) - 1;
                f_count++;
            }
        }
    }
    obj_file.close();
}

void reset() {
    frame_count = 0;
}

void keyboard(unsigned char key, int, int) {
    switch (key) {
        case 'q': exit(1);  break;
            //place control point
        case 'r': reset(); break;
        case 'z':
            interpolation_mode = 0;
            reset();
            break;
        case 'x':
            interpolation_mode = 1;
            reset();
            break;
    }
    glutPostRedisplay();
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);
    double t = double(frame_count) / frames;
    
    if (interpolation_mode == 0) {
        for (int i = 0; i < 332; i++) {
            glBegin(GL_LINE_LOOP);
            glVertex2f(ori_pose(face(i,0),0)+frame_count*ori_aft_dif(face(i,0),0), ori_pose(face(i,0),1)+frame_count*ori_aft_dif(face(i,0),1));
            glVertex2f(ori_pose(face(i,1),0)+frame_count*ori_aft_dif(face(i,1),0), ori_pose(face(i,1),1)+frame_count*ori_aft_dif(face(i,1),1));
            glVertex2f(ori_pose(face(i,2),0)+frame_count*ori_aft_dif(face(i,2),0), ori_pose(face(i,2),1)+frame_count*ori_aft_dif(face(i,2),1));
            glEnd();
        }
    } else {
        
        double cur_x1 = ori_pose(0,0) + ori_aft_dif(0,0)*frame_count;
        double cur_y1 = ori_pose(0,1) + ori_aft_dif(0,1)*frame_count;
        
        
        
        cerr << t << endl;
        fill_b(t, cur_x1, cur_y1);
//        cerr << "hihi" << endl;
//        cerr << precomputed_A(0,0);
        MatrixXd coor = precomputed_A * final_b;
        cerr << coor(0) << coor(1) << endl;
//        exit(1);
        for (int i = 0; i < 332; i++) {
            double x1 = 0;
            double y1 = 0;
            double x2 = 0;
            double y2 = 0;
            double x3 = 0;
            double y3 = 0;
            if (face(i,0) == 0) {
                x1 = cur_x1;
                y1 = cur_y1;
            } else {
                x1 = coor(2*face(i,0)-2);
                y1 = coor(2*face(i,0)-1);
            }
            if (face(i,1) == 0) {
                x2 = cur_x1;
                y2 = cur_y1;
            } else {
                x2 = coor(2*face(i,1)-2);
                y2 = coor(2*face(i,1)-1);
            }
            if (face(i,2) == 0) {
                x3 = cur_x1;
                y3 = cur_y1;
            } else {
                x3 = coor(2*face(i,2)-2);
                y3 = coor(2*face(i,2)-1);
            }
            glBegin(GL_LINE_LOOP);
//            glVertex2f(coor(2*face(i,0)), coor(2*face(i,0)+1));
//            glVertex2f(coor(2*face(i,1)), coor(2*face(i,1)+1));
//            glVertex2f(coor(2*face(i,2)), coor(2*face(i,2)+1));
            glVertex2f(x1,y1);
            glVertex2f(x2,y2);
            glVertex2f(x3,y3);
            glEnd();
        }
    }
    if (frame_count < frames) {
        frame_count++;
    }
    glutSwapBuffers();
}

void get_endpose() {
    string obj_line;
    ifstream obj_file("/Users/garychu/Desktop/cm50245cw/interpolation/interpolation/interpolation/keyframe.txt");
    
    while (!obj_file.eof()) {
        getline(obj_file, obj_line);
        istringstream iss(obj_line);
        vector<string> results(istream_iterator<string>{iss}, istream_iterator<string>());
        aft_pose(aft_count,0) = stod(results[0]);
        aft_pose(aft_count,1) = stod(results[1]);
//        cout << stod(results[0]) << endl;
        aft_count++;
    }
    obj_file.close();
}

//resizing the window
void reshape(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    //    glOrtho(-2.5, 2.5, 2.5, -2.5, -1.0, 1.0);
    glOrtho(-2, 2, 2, -2, -1.0, 1.0);
//    glOrtho(0, 10, 10, 0, -1.0, 1.0);
//    glOrtho(-5, 5, 5, -5, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
}

void timer(int) {
    glutPostRedisplay();
    glutTimerFunc(frames*2, timer, 0);
//    glutTimerFunc(1000, timer, 0);
}

void init() {
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    parse_obj();
    get_endpose();
    for (int i = 0; i < 214; i++) {
        ori_aft_dif(i,0) = (aft_pose(i,0) - ori_pose(i,0))/frames;
        ori_aft_dif(i,1) = (aft_pose(i,1) - ori_pose(i,1))/frames;
    }
    I << 1, 0,
         0, 1;
    calculate_A();
    decompose();
}

int main(int argc, char * argv[]) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA);
    glutInitWindowSize(512, 512);
    glutInitWindowPosition(300, 100);
    
    glutCreateWindow("ARAP Interpolation");
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    
    glutTimerFunc(0, timer, 0);
    
//    glutMouseFunc(mouse_click);
//    glutPassiveMotionFunc(mouse_motion);
//    glutMotionFunc(mouse_drag);
    
    init();
    glutMainLoop();
    
    return 0;
}
