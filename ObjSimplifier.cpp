//
//  ObjSimplifier.cpp
//  Assignment0
//
//  Created by Neil on 22/01/2017.
//  Copyright © 2017 Neil. All rights reserved.
//

#include "ObjSimplifier.hpp"

#ifndef MAX_BUFFER_SIZE
#define MAX_BUFFER_SIZE 1024
#endif



void ObjSimplifier::initQ() {
    for (int i = 0; i < pts.size(); i++) {
        auto p = pts[i];
        auto facets = p->facets;
        Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
        for (int j = 0; j < facets.size(); j++) {
            auto f = facets[j];
            Eigen::MatrixXd coord(3, 4);
            coord << f->v1.lock()->pos[0], f->v1.lock()->pos[1], f->v1.lock()->pos[2], 1,
            f->v2.lock()->pos[0], f->v2.lock()->pos[1], f->v2.lock()->pos[2], 1,
            f->v3.lock()->pos[0], f->v3.lock()->pos[1], f->v3.lock()->pos[2], 1;
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(coord, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Vector4d param = svd.matrixV().col(3);
            float normalizer = sqrt(param[0] * param[0] + param[1] * param[1] + param[2] * param[2]);
            param /= normalizer;
            
            auto K = param * param.transpose();
            Q += K;
        }
        p->Q = Q;
    }
}

float ObjSimplifier::evaluateCost(shared_ptr<PointPair> pr) {
    auto v1 = pr->pr.first;
    auto v2 = pr->pr.second;
    Eigen::Matrix4d Q = v1->Q + v2->Q;
    Eigen::Matrix4d dQ = Q;
    dQ.row(3).setZero();
    dQ(3, 3) = 1.;
    Eigen::FullPivLU<Eigen::Matrix4d> lu(dQ);
    Eigen::Vector4d v;
    if (lu.isInvertible()) {
        v = dQ.inverse() * Eigen::Vector4d(0, 0, 0, 1);
    } else {
        std::cout << "dQ is not invertible! Fallback not implemented!" << std::endl;
    }
    if (!pr->cand) {
        pr->cand = make_shared<Point>();
        pr->cand->Q = Q;
        pr->cand->pos = Eigen::Vector3d(v.x(), v.y(), v.z());
        
        // use least square to interpolate
        Eigen::MatrixXd coord(3, 2);
        coord <<v1->pos[0], v2->pos[0],
        v1->pos[1], v2->pos[1],
        v1->pos[2], v2->pos[2];
        
        Eigen::Vector2d coef = (coord.transpose() * coord).inverse() * coord.transpose() * pr->cand->pos;
        
        pr->cand->normal = coef[0] * v1->normal + coef[1] * v2->normal;
        pr->cand->normal.normalize();
    }
//    cout << "cost " <<  v.transpose() * Q * v << endl;
    return v.transpose() * Q * v;
}

void ObjSimplifier::merge(shared_ptr<PointPair> pr) {
    auto v1 = pr->pr.first;
    auto v2 = pr->pr.second;
    auto cand = pr->cand;
    auto s1 = pairs.size();
    pairs.erase(remove_if(pairs.begin(), pairs.end(), [&](shared_ptr<PointPair> pair){
        return pair->pr.first == v1 || pair->pr.second == v2 || pair->pr.first == v2 || pair->pr.second == v1;
    }), pairs.end());
    assert(pairs.size() < s1);
    
    unordered_set<shared_ptr<Point>> adjs;
    for (auto adj : v1->adjs) {
        assert(adj != v1);
        if (adj != v2) {
            adjs.insert(adj);
        }
    }
    for (auto adj : v2->adjs) {
        assert(adj != v2);
        if (adj != v1) {
            adjs.insert(adj);
        }
    }
    
    for (auto adj : adjs) {
        cand->adjs.push_back(adj);
        adj->adjs.erase(remove(adj->adjs.begin(), adj->adjs.end(), v1), adj->adjs.end());
        adj->adjs.erase(remove(adj->adjs.begin(), adj->adjs.end(), v2), adj->adjs.end());
        adj->adjs.push_back(cand);
        
        // add new pair
        auto pp = make_shared<PointPair>();
        pp->pr = make_pair(cand, adj);
        pp->cost = evaluateCost(pp);

        // find its position in sorted pairs
        auto it = pairs.begin();
        while (it != pairs.end() && (*it)->cost < pp->cost) {
            it++;
        }
        pairs.insert(it, pp);
    }
    
    unordered_set<shared_ptr<Facet>> facets;
    for (auto f : v1->facets) {
        facets.insert(f);
    }
    for (auto f : v2->facets) {
        facets.insert(f);
    }
    
    for (auto f : facets) {
        vector<shared_ptr<Point>> triangle;
        triangle.push_back(f->v1.lock());
        triangle.push_back(f->v2.lock());
        triangle.push_back(f->v3.lock());
        assert(triangle.size() == 3);
        
        if (f->contains(v1) && f->contains(v2)) { // facet should be removed
            triangle.erase(remove(triangle.begin(), triangle.end(), v1), triangle.end());
            triangle.erase(remove(triangle.begin(), triangle.end(), v2), triangle.end());
            assert(triangle.size() == 1);
            
            auto v = *triangle.begin();
            v->facets.erase(remove(v->facets.begin(), v->facets.end(), f), v->facets.end());
            if (v->facets.size() == 0) {
                cout << "Warning! Degenerate vertex..." << endl;
            }
        } else {
            if (f->contains(v1)) {
                auto v = f->contains(v1);
                if (v == f->v1.lock()) f->v1 = cand;
                if (v == f->v2.lock()) f->v2 = cand;
                if (v == f->v3.lock()) f->v3 = cand;
                cand->facets.push_back(f);
            }
            if (f->contains(v2)) {
                auto v = f->contains(v2);
                if (v == f->v1.lock()) f->v1 = cand;
                if (v == f->v2.lock()) f->v2 = cand;
                if (v == f->v3.lock()) f->v3 = cand;
                cand->facets.push_back(f);
            }
        }
    }
    
    assert(find(pts.begin(), pts.end(), v1) != pts.end());
    assert(find(pts.begin(), pts.end(), v2) != pts.end());
    pts.erase(remove(pts.begin(), pts.end(), v1), pts.end());
    pts.erase(remove(pts.begin(), pts.end(), v2), pts.end());
    pts.push_back(cand);
    
    normals.erase(remove(normals.begin(), normals.end(), v1->normal), normals.end());
    normals.erase(remove(normals.begin(), normals.end(), v2->normal), normals.end());
    normals.push_back(cand->normal);

}

void ObjSimplifier::simplify(const char *file, const char* output) {
    fstream fs;
    fs.open(file, fstream::in);
    if (fs.fail()) {
        return;
    }
    
    // load the OBJ file here
    char buffer[MAX_BUFFER_SIZE];
    int cnt = 0;
    while(fs.getline(buffer, MAX_BUFFER_SIZE))
    {
        stringstream ss(buffer);
        string s;
        ss >> s;
        
        
        if (s == "v") { // vertex
            Eigen::Vector3d v;
            ss >> v[0] >> v[1] >> v[2];
            auto p = make_shared<Point>();
            p->pos = v;
            pts.push_back(p);
        } else if (s == "vn") {
            Eigen::Vector3d n;
            ss >> n[0] >> n[1] >> n[2];
            normals.push_back(n);
            normals.back().normalize();
        } else if (s == "f") {
            cnt ++;
            int a, b, c, d, e, f, g, h, i;
            sscanf(buffer, "f %d/%d/%d %d/%d/%d %d/%d/%d", &a, &b, &c, &d, &e, &f, &g, &h, &i);
            a -= 1;
            b -= 1;
            c -= 1;
            d -= 1;
            e -= 1;
            f -= 1;
            g -= 1;
            h -= 1;
            i -= 1;
            
            pts[a]->normal = normals[c];
//            pts[a]->nindex = c;
            pts[d]->normal = normals[f];
//            pts[d]->nindex = f;
            pts[g]->normal = normals[i];
//            pts[g]->nindex = i;
            
            auto face = make_shared<Facet>();
            face->v1 = pts[a];
            face->v2 = pts[d];
            face->v3 = pts[g];
//            face->p1 = a;
//            face->p2 = d;
//            face->p3 = g;
//            face->n1 = c;
//            face->n2 = f;
//            face->n3 = i;
            pts[a]->facets.push_back(face);
            pts[d]->facets.push_back(face);
            pts[g]->facets.push_back(face);
            
            if (find(pts[a]->adjs.begin(), pts[a]->adjs.end(), pts[d]) == pts[a]->adjs.end()) { // not found
                pts[a]->adjs.push_back(pts[d]);
            }
            if (find(pts[a]->adjs.begin(), pts[a]->adjs.end(), pts[g]) == pts[a]->adjs.end()) { // not found
                pts[a]->adjs.push_back(pts[g]);
            }
            
            if (find(pts[d]->adjs.begin(), pts[d]->adjs.end(), pts[a]) == pts[d]->adjs.end()) { // not found
                pts[d]->adjs.push_back(pts[a]);
            }
            if (find(pts[d]->adjs.begin(), pts[d]->adjs.end(), pts[g]) == pts[d]->adjs.end()) { // not found
                pts[d]->adjs.push_back(pts[g]);
            }
            
            if (find(pts[g]->adjs.begin(), pts[g]->adjs.end(), pts[a]) == pts[g]->adjs.end()) { // not found
                pts[g]->adjs.push_back(pts[a]);
            }
            if (find(pts[g]->adjs.begin(), pts[g]->adjs.end(), pts[d]) == pts[g]->adjs.end()) { // not found
                pts[g]->adjs.push_back(pts[d]);
            }
        }
    }
    cout << "Original facet count " << cnt << endl;
    fs.close();
    
    initQ();
    int total_size = 0;
    for (auto p : pts) {
        total_size += p->adjs.size();
        for (auto adj : p->adjs) {
            // constructing initial pairs
            assert(find(pts.begin(), pts.end(), adj) != pts.end());
            if (find_if(pairs.begin(), pairs.end(), [&](const shared_ptr<PointPair>& pr) {
                return (pr->pr.first == adj && pr->pr.second == p) || (pr->pr.second == adj && pr->pr.first == p);
            }) == pairs.end()) {
                auto pp = make_shared<PointPair>();
                pp->pr = make_pair(p, adj);
                pp->cost = evaluateCost(pp);
                pairs.push_back(pp);
            }
        }
    }
    
    sort(pairs.begin(), pairs.end(), [](const shared_ptr<PointPair>& pr1, const shared_ptr<PointPair>& pr2) {
        return pr1->cost < pr2->cost;
    });
    
    for (auto pr : pairs) {
        assert(find(pts.begin(), pts.end(), pr->pr.first) != pts.end());
        assert(find(pts.begin(), pts.end(), pr->pr.second) != pts.end());
    }
    
    assert(pairs.size() == total_size / 2);
    int simp_size = simplify_ratio * pts.size();
    while (pts.size() > simp_size) {
        auto cand = pairs[0];
        merge(cand);
    }
    
    fs.open(output, fstream::out | fstream::trunc);
    if (fs.fail()) {
        return;
    }
    
    for (int i = 0; i < pts.size(); i++) {
        fs << "v " << pts[i]->pos.x() << " " << pts[i]->pos.y() << " " << pts[i]->pos.z() << endl;
    }
    for (int i = 0; i < normals.size(); i++) {
        fs << "vn " << normals[i].x() << " " << normals[i].y() << " " << normals[i].z() << endl;
    }
    
    vector<shared_ptr<Facet>> facets;
    for (int i = 0; i < pts.size(); i++) {
        for (auto f : pts[i]->facets) {
            if (find_if(facets.begin(), facets.end(), [&](shared_ptr<Facet> facet) {
                return *facet == *f;
            }) == facets.end()) {
                facets.push_back(f);
            }
        }
    }
    cout << "Simplified facet size " << facets.size() << endl;
    for (int i = 0; i < facets.size(); i++) {
        fs << "f ";
        auto v1 = facets[i]->v1.lock();
        auto v2 = facets[i]->v2.lock();
        auto v3 = facets[i]->v3.lock();
        
        size_t a = find(pts.begin(), pts.end(), v1) - pts.begin() + 1;
        size_t c = find(normals.begin(), normals.end(), pts[a-1]->normal) - normals.begin() + 1;
        size_t d = find(pts.begin(), pts.end(), v2) - pts.begin() + 1;
        size_t f = find(normals.begin(), normals.end(), pts[d-1]->normal) - normals.begin() + 1;
        size_t g = find(pts.begin(), pts.end(), v3) - pts.begin() + 1;
        size_t j = find(normals.begin(), normals.end(), pts[g-1]->normal) - normals.begin() + 1;
        
        fs << a << "/0/" << c << " " << d << "/0/" << f << " " << g << "/0/" << j << endl;
    }
    fs.close();
}
