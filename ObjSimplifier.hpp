//
//  ObjSimplifier.hpp
//  Assignment0
//
//  Created by Neil on 22/01/2017.
//  Copyright © 2017 Neil. All rights reserved.
//

#ifndef ObjSimplifier_hpp
#define ObjSimplifier_hpp

#include <stdio.h>
#include <vector>
#include <unordered_set>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/LU>
using namespace std;

class Facet;

class Point {
public:
    Eigen::Vector3d pos;
    Eigen::Vector3d normal;
    Eigen::Matrix4d Q;
    vector<shared_ptr<Point>> adjs;
    vector<shared_ptr<Facet>> facets;
};

class Facet {
public:
    weak_ptr<Point> v1;
    weak_ptr<Point> v2;
    weak_ptr<Point> v3;
    int n1, n2, n3, p1, p2, p3;
    bool operator==(const Facet& other) {
        return (v1.lock() == other.v1.lock()
                && v2.lock() == other.v2.lock()
                && v3.lock() == other.v3.lock())
        || (v1.lock() == other.v1.lock()
            && v2.lock() == other.v3.lock()
            && v3.lock() == other.v2.lock())
        || (v1.lock() == other.v2.lock()
            && v2.lock() == other.v1.lock()
            && v3.lock() == other.v3.lock())
        || (v1.lock() == other.v2.lock()
            && v2.lock() == other.v3.lock()
            && v3.lock() == other.v1.lock())
        || (v1.lock() == other.v3.lock()
            && v2.lock() == other.v2.lock()
            && v3.lock() == other.v1.lock())
        || (v1.lock() == other.v3.lock()
            && v2.lock() == other.v1.lock()
            && v3.lock() == other.v2.lock());
        
    }
    shared_ptr<Point> contains(const shared_ptr<Point>& p) {
        if (v1.lock() == p) return v1.lock();
        if (v2.lock() == p) return v2.lock();
        if (v3.lock() == p) return v3.lock();
        return nullptr;
    }
};

class PointPair {
public:
    pair<shared_ptr<Point>, shared_ptr<Point>> pr;
    float cost;
    shared_ptr<Point> cand;
    
};

class ObjSimplifier {
    vector<shared_ptr<Point>> pts;
    vector<Eigen::Vector3d> normals;
    vector<shared_ptr<PointPair>> pairs;
    float simplify_ratio = 0.2;
    
public:
    void initQ();
    float evaluateCost(shared_ptr<PointPair> pr);
    void simplify(const char* input, const char* output);
    void merge(shared_ptr<PointPair> pr);
};

#endif /* ObjSimplifier_hpp */
