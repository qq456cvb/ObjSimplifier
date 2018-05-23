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

Eigen::VectorXd ObjSimplifier::packAttr(shared_ptr<Point> p) {
    Eigen::VectorXd result;
    vector<double> attrs_vec;
    for (int m = 0; m < 3; m++) {
        attrs_vec.push_back(p->pos[m]);
    }
    if (has_normal) {
        for (int m = 0; m < 3; m++) {
            attrs_vec.push_back(p->normal[m]);
        }
    }
    if (has_texture) {
        for (int m = 0; m < 2; m++) {
            attrs_vec.push_back(p->uv[m]);
        }
    }
    result = Eigen::Map<Eigen::VectorXd>(attrs_vec.data(), attrs_vec.size());
    return result;
}

void ObjSimplifier::initQ() {
    for (int i = 0; i < pts.size(); i++) {
        auto p = pts[i];
        auto facets = p->facets;
        Eigen::MatrixXd Q;
        
        for (int j = 0; j < facets.size(); j++) {
            auto f = facets[j];
            Eigen::VectorXd v1_attr = packAttr(f->v1.lock());
            Eigen::VectorXd v2_attr = packAttr(f->v2.lock());
            Eigen::VectorXd v3_attr = packAttr(f->v3.lock());
            Eigen::VectorXd e1 = v1_attr - v2_attr;
            e1.normalize();
            Eigen::VectorXd e2 = v3_attr - v2_attr - e1.dot(v3_attr - v2_attr) * e1;
            e2.normalize();
            
            int n_dim = static_cast<int>(e1.size());
            
            Eigen::MatrixXd A = Eigen::MatrixXd::Identity(n_dim, n_dim) - e1 * e1.transpose() - e2 * e2.transpose();
            double v2e1 = v2_attr.dot(e1), v2e2 = v2_attr.dot(e2);
            Eigen::VectorXd b = v2e1 * e1 + v2e2 * e2 - v2_attr;
            double c = v2_attr.dot(v2_attr) - v2e1 * v2e1 - v2e2 * v2e2;
            
            Eigen::MatrixXd Ab(A.rows(), A.cols() + b.cols());
            Ab << A, b;
            Eigen::MatrixXd bc(b.size() + 1, 1);
            bc << b, c;
            Eigen::MatrixXd K(Ab.rows() + 1, Ab.cols());
            K << Ab, bc.transpose();
            if (Q.size() == 0) {
                Q = K;
            } else {
            Q += K;
        }
        }
        p->Q = Q;
    }
}

float ObjSimplifier::evaluateCost(shared_ptr<PointPair> pr) {
    auto v1 = pr->pr.first;
    auto v2 = pr->pr.second;
    Eigen::MatrixXd Q = v1->Q + v2->Q;
    Eigen::MatrixXd dQ = Q;
    dQ.row(Q.rows() - 1).setZero();
    dQ(Q.rows() - 1, Q.cols() - 1) = 1.;
    Eigen::FullPivLU<Eigen::MatrixXd> lu(dQ);
    Eigen::VectorXd v;
    if (lu.isInvertible()) {
        Eigen::VectorXd tmp(dQ.cols());
        tmp.fill(0);
        tmp[tmp.size() - 1] = 1.;
        v = dQ.inverse() * tmp;
    } else {
        std::cout << "dQ is not invertible! Fallback to find on segment v1v2" << std::endl;
        auto v1_attr = packAttr(v1);
        auto v2_attr = packAttr(v2);
        Eigen::VectorXd v1minusv2 = v1_attr - v2_attr;
        Eigen::VectorXd q = Q.topRightCorner(Q.rows() - 1, 1);

        // symmetric
        Eigen::MatrixXd Q_lr = Q.topLeftCorner(Q.rows() - 1, Q.cols() - 1);
        Eigen::VectorXd c = Q_lr * v1minusv2;
        double denom = c.dot(v1minusv2);
        if (abs(denom) < 1e-7) {
            std::cout << "not differentiable on segment v1v2! Fallback to find among v1, v2, mid(v1, v2)" << std::endl;
            Eigen::VectorXd v1_homo(v1_attr.size() + 1);
            v1_homo.fill(1);
            v1_homo.topRows(v1_attr.size()) = v1_attr;
            Eigen::VectorXd v2_homo(v2_attr.size() + 1);
            v2_homo.fill(1);
            v2_homo.topRows(v2_attr.size()) = v2_attr;
            Eigen::VectorXd vmid_homo = 0.5 * v1_homo + 0.5 * v2_homo;
            double cost1 = v1_homo.transpose() * Q * v1_homo;
            double cost2 = v2_homo.transpose() * Q * v2_homo;
            double cost_mid = vmid_homo.transpose() * Q * vmid_homo;
            if (cost1 < cost2) {
                if (cost1 < cost_mid) {
                    v = v1_homo;
                } else {
                    v = vmid_homo;
                }
            } else {
                if (cost2 < cost_mid) {
                    v = v2_homo;
                } else {
                    v = vmid_homo;
                }
            }
        } else {
            double qv = q.dot(v1minusv2);
            double k = c.dot(v2_attr) - qv;
            if (k > 1) k = 1;
            if (k < 0) k = 0;
            Eigen::VectorXd v3 = k * v1_attr + (1 - k) * v2_attr;
            v = Eigen::VectorXd(v3.size() + 1);
            v.fill(1);
            v.topRows(v3.size()) = v3;
        }
    }
    // force normalize
    v.segment(3, 3).normalize();
    if (!pr->cand) {
        pr->cand = shared_ptr<Point>(new Point());
        pr->cand->Q = Q;
        pr->cand->pos = v.topRows(3);
        if (has_normal) {
            pr->cand->normal = v.segment(3, 3);
        }
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
    Eigen::Vector3d pos_max(std::numeric_limits<double>::min(),
                            std::numeric_limits<double>::min(),
                            std::numeric_limits<double>::min()),
        pos_min(std::numeric_limits<double>::max(),
                std::numeric_limits<double>::max(),
                std::numeric_limits<double>::max());
    while(fs.getline(buffer, MAX_BUFFER_SIZE))
    {
        stringstream ss(buffer);
        string s;
        ss >> s;
        
        
        if (s == "v") { // vertex
            Eigen::Vector3d v;
            ss >> v[0] >> v[1] >> v[2];
            for (int i = 0; i < 3; i++) {
                pos_max[i] = max(pos_max[i], v[i]);
                pos_min[i] = min(pos_min[i], v[i]);
            }
            auto p = shared_ptr<Point>(new Point());
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
            ss >> s;
            auto cnt = count(s.begin(), s.end(), '/');
            if (cnt == 2) {
                sscanf(buffer, "f %d/%d/%d %d/%d/%d %d/%d/%d", &a, &b, &c, &d, &e, &f, &g, &h, &i);
                has_normal = true;
            } else if (cnt == 1) {
                
            } else {
                sscanf(buffer, "f %d %d %d", &a, &d, &g);
            }

            a -= 1;
            b -= 1;
            c -= 1;
            d -= 1;
            e -= 1;
            f -= 1;
            g -= 1;
            h -= 1;
            i -= 1;
            
            if (has_normal) {
                // assume each point has only one normal for all facets
                pts[a]->normal = normals[c];
                //            pts[a]->nindex = c;
                pts[d]->normal = normals[f];
                //            pts[d]->nindex = f;
                pts[g]->normal = normals[i];
                //            pts[g]->nindex = i;
            }
            
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
    
    // rescale the position coordinates
    Eigen::Vector3d pos_scale = pos_max - pos_min;
    for (auto pt : pts) {
        pt->pos = pt->pos - pos_min;
        pt->pos = pt->pos.cwiseQuotient(pos_scale);
    }
    
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
        fs << "v " << pts[i]->pos.x() * pos_scale.x() + pos_min.x() << " " << pts[i]->pos.y() * pos_scale.y() + pos_min.y() << " " << pts[i]->pos.z() * pos_scale.z() + pos_min.z() << endl;
    }
    if (has_normal) {
        for (int i = 0; i < normals.size(); i++) {
            fs << "vn " << normals[i].x() << " " << normals[i].y() << " " << normals[i].z() << endl;
        }
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
        size_t d = find(pts.begin(), pts.end(), v2) - pts.begin() + 1;
        size_t g = find(pts.begin(), pts.end(), v3) - pts.begin() + 1;
        
        if (has_normal) {
            size_t c = find(normals.begin(), normals.end(), pts[a-1]->normal) - normals.begin() + 1;
            size_t f = find(normals.begin(), normals.end(), pts[d-1]->normal) - normals.begin() + 1;
            size_t j = find(normals.begin(), normals.end(), pts[g-1]->normal) - normals.begin() + 1;
            fs << a << "/0/" << c << " " << d << "/0/" << f << " " << g << "/0/" << j << endl;
        } else {
            fs << a << " " << d << " " << g << endl;
        }
        
        
    }
    fs.close();
}
